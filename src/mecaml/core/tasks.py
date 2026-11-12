"""
MécaML : calcul dans un fil séparé, avec annulation et suivi, sans aucune dépendance d'interface.

L'interface ne bloque jamais : elle lance une `Tache`, interroge son état (`etat`,
`progression`, `evenements`) à intervalle régulier et peut demander l'annulation. Le calcul
vérifie le jeton d'annulation entre deux unités de travail (un modèle sur un pli, par exemple),
jamais au milieu d'un ajustement : scikit-learn ne se laisse pas interrompre proprement, et le
calcul ne passe pas par des processus multiples (un seul fil, donc des résultats reproductibles).

Un seul mécanisme d'annulation pour tout le cœur : le jeton `Annulation` est aussi un appelable
sans argument qui rend vrai quand l'arrêt est demandé, ce qui est la forme attendue par
`io.charger(..., annuler=jeton)` ; `ErreurAnnulation` est l'exception commune (`guards.ErreurAnnulee`
en est un alias), et la tâche dont le calcul s'interrompt par elle passe à l'état ANNULEE, non à ECHEC.

Un jeton, un fil, un résultat :

    tache = Tache(lambda annulation, avancer: calculer(annulation, avancer)).demarrer()
    ...                      # boucle de l'interface : tache.etat, tache.progression()
    tache.annuler()          # demande ; le fil s'arrête à la prochaine vérification
    valeur = tache.resultat()  # lève l'erreur du calcul, ou ErreurAnnulation si annulé

Les avertissements de scikit-learn (convergence, classes absentes d'un pli) sont rassemblés par
`collecter_avertissements`, qui ne modifie ni l'état global des avertissements d'un autre fil ni ses
filtres : un crochet unique, installé une fois, ne détourne que les avertissements du fil qui collecte.
"""

from __future__ import annotations

import threading
import traceback
import warnings
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Annulation",
    "ErreurAnnulation",
    "ErreurAnnulee",
    "EtatTache",
    "Progression",
    "Tache",
    "collecter_avertissements",
    "erreur_etrangere",
    "message_utilisateur",
    "precharger_scikit_learn",
]


class ErreurAnnulation(Exception):
    """Le calcul a été interrompu à la demande de l'utilisateur."""


ErreurAnnulee = ErreurAnnulation  # ancien nom de l'exception de lecture : une seule classe


class EtatTache(StrEnum):
    """Cycle de vie d'une tâche."""

    PRETE = "prete"
    EN_COURS = "en_cours"
    TERMINEE = "terminee"
    ANNULEE = "annulee"
    ECHEC = "echec"


def erreur_etrangere(erreur: BaseException) -> bool:
    """Vrai si l'erreur vient d'une bibliothèque tierce (scikit-learn, pandas, numpy...), non du cœur.

    Les erreurs du cœur disent le motif en français ; une exception levée plus bas, dans une bibliothèque,
    porte un texte anglais que l'utilisateur ne doit pas lire tel quel. Une exception définie par le paquet
    ne l'est jamais ; sinon le critère est le module du dernier cadre de la pile (celui qui a levé).
    """
    if type(erreur).__module__.split(".")[0] == "mecaml":
        return False
    maillon = erreur.__traceback__
    while maillon is not None and maillon.tb_next is not None:
        maillon = maillon.tb_next
    if maillon is not None:
        module = str(maillon.tb_frame.f_globals.get("__name__", ""))
    else:
        module = type(erreur).__module__
    return not (module == "mecaml" or module.startswith("mecaml.") or module == "__main__")


def message_utilisateur(erreur: BaseException, etape: str | None = None) -> str:
    """Message français pour une erreur imprévue d'une bibliothèque de calcul (voir `erreur_etrangere`)."""
    lieu = f"l'étape « {etape} »" if etape else "le calcul"
    return (
        f"Une erreur imprévue d'une bibliothèque de calcul ({type(erreur).__name__}) a interrompu {lieu}. "
        "Vérifier la colonne cible et le type des colonnes à l'étape précédente, puis relancer ; "
        "si l'erreur revient, la signaler avec ce nom d'erreur et le jeu de données employé."
    )


@dataclass(frozen=True)
class Progression:
    """Avancement d'un calcul : étape en cours, unités faites sur unités prévues."""

    etape: str
    fait: int
    total: int
    message: str = ""

    @property
    def fraction(self) -> float:
        """Part accomplie, entre 0 et 1 (0 si le total est inconnu)."""
        return 0.0 if self.total <= 0 else min(1.0, self.fait / self.total)


class Annulation:
    """Jeton d'annulation, partagé entre l'interface (qui demande) et le calcul (qui vérifie)."""

    def __init__(self) -> None:
        self._evenement = threading.Event()

    def demander(self) -> None:
        """Demande l'arrêt ; sans effet immédiat sur un ajustement en cours."""
        self._evenement.set()

    @property
    def demandee(self) -> bool:
        """Vrai si l'arrêt a été demandé."""
        return self._evenement.is_set()

    def __call__(self) -> bool:
        """Forme appelable du jeton : vrai si l'arrêt est demandé (argument `annuler` de `io.charger`)."""
        return self._evenement.is_set()

    def verifier(self) -> None:
        """Lève `ErreurAnnulation` si l'arrêt a été demandé ; à appeler entre deux unités de travail."""
        if self._evenement.is_set():
            raise ErreurAnnulation("Calcul annulé.")


type Fonction[T] = Callable[[Annulation, Callable[[Progression], None]], T]


class Tache[T]:
    """Calcul exécuté dans un fil dédié, annulable, dont l'état se lit sans bloquer."""

    def __init__(self, fonction: Fonction[T], nom: str = "calcul") -> None:
        self.nom = nom
        self.annulation = Annulation()
        self._fonction = fonction
        self._verrou = threading.Lock()
        self._fil = threading.Thread(target=self._executer, name=f"mecaml-{nom}", daemon=True)
        self._etat = EtatTache.PRETE
        self._valeur: T | None = None
        self._erreur: BaseException | None = None
        self._trace = ""
        self._derniere: Progression | None = None
        self._evenements: deque[Progression] = deque(maxlen=1000)

    # Lancement
    def demarrer(self) -> Tache[T]:
        """Lance le fil (une seule fois) ; rend la tâche pour chaîner l'appel."""
        with self._verrou:
            if self._etat is not EtatTache.PRETE:
                raise RuntimeError(f"Tâche {self.nom} déjà lancée.")
            self._etat = EtatTache.EN_COURS
        self._fil.start()
        return self

    def annuler(self) -> None:
        """Demande l'annulation ; la tâche passe à ANNULEE quand le calcul s'arrête."""
        self.annulation.demander()

    def _avancer(self, progression: Progression) -> None:
        with self._verrou:
            self._derniere = progression
            self._evenements.append(progression)

    def _executer(self) -> None:
        try:
            valeur = self._fonction(self.annulation, self._avancer)
        except ErreurAnnulation as erreur:
            self._terminer(EtatTache.ANNULEE, erreur=erreur)
        except Exception as erreur:  # le fil ne doit rien laisser filer : l'erreur est gardée et relue
            self._terminer(EtatTache.ECHEC, erreur=erreur, trace=traceback.format_exc())
        else:
            self._terminer(EtatTache.TERMINEE, valeur=valeur)

    def _terminer(
        self,
        etat: EtatTache,
        *,
        valeur: T | None = None,
        erreur: BaseException | None = None,
        trace: str = "",
    ) -> None:
        with self._verrou:
            self._valeur, self._erreur, self._trace, self._etat = valeur, erreur, trace, etat

    # Lecture
    @property
    def etat(self) -> EtatTache:
        """État courant."""
        with self._verrou:
            return self._etat

    @property
    def terminee(self) -> bool:
        """Vrai quand le fil a fini, quelle que soit l'issue."""
        return self.etat in (EtatTache.TERMINEE, EtatTache.ANNULEE, EtatTache.ECHEC)

    @property
    def erreur(self) -> BaseException | None:
        """Exception du calcul (ou d'annulation), si la tâche a échoué ou été annulée."""
        with self._verrou:
            return self._erreur

    @property
    def trace(self) -> str:
        """Trace de l'échec, pour le journal (jamais pour l'écran)."""
        with self._verrou:
            return self._trace

    def progression(self) -> Progression | None:
        """Dernier avancement signalé, ou None."""
        with self._verrou:
            return self._derniere

    def evenements(self) -> list[Progression]:
        """Avancements signalés depuis le dernier appel (la liste est vidée)."""
        with self._verrou:
            vus = list(self._evenements)
            self._evenements.clear()
            return vus

    def attendre(self, delai: float | None = None) -> bool:
        """Attend la fin du fil (au plus `delai` secondes) ; vrai si la tâche est terminée."""
        if self._fil.ident is not None:
            self._fil.join(delai)
        return self.terminee

    def resultat(self) -> T:
        """Valeur du calcul ; relève son erreur s'il a échoué, l'erreur d'annulation s'il a été annulé."""
        with self._verrou:
            etat, valeur, erreur = self._etat, self._valeur, self._erreur
        if etat is EtatTache.TERMINEE:
            return valeur  # type: ignore[return-value]
        if erreur is not None:
            raise erreur
        raise RuntimeError(f"Tâche {self.nom} non terminée (état {etat}).")


# ============================================================
# AVERTISSEMENTS DE SCIKIT-LEARN, RASSEMBLÉS FIL PAR FIL
# ============================================================
_LOCAL = threading.local()
_VERROU_CROCHET = threading.Lock()
_SUIVANT: list = []  # affichage remplacé par le crochet, pour les avertissements d'un fil qui ne collecte pas
MODULE_SURVEILLE = r"sklearn(\.|$)"  # les seuls avertissements rendus systématiques


def _crochet(message, category, filename, lineno, file=None, line=None) -> None:
    """Détourne l'avertissement vers le collecteur du fil courant, sinon l'affiche comme avant."""
    collecteur = getattr(_LOCAL, "collecteur", None)
    if collecteur is not None:
        collecteur.append(f"{category.__name__} : {message}")
    elif _SUIVANT:
        _SUIVANT[0](message, category, filename, lineno, file, line)


def _installer_crochet() -> None:
    """Pose le crochet s'il n'y est pas (un autre code peut avoir remis l'affichage qu'il remplace)."""
    with _VERROU_CROCHET:
        if warnings.showwarning is not _crochet:
            _SUIVANT[:] = [warnings.showwarning]
            warnings.showwarning = _crochet
    # Filtrage par module : chaque avertissement de scikit-learn est émis à chaque fois, en tête de liste,
    # sans toucher à ceux des autres modules ni restaurer d'état.
    warnings.filterwarnings("always", module=MODULE_SURVEILLE)


@contextmanager
def collecter_avertissements() -> Iterator[list[str]]:
    """Rassemble dans une liste les avertissements du fil courant (« Catégorie : message »).

    Remplace `warnings.catch_warnings`, qui n'est pas sûr entre fils : rien n'est sauvé ni restauré
    globalement, et les collecteurs de deux fils ne se mélangent pas (variable locale au fil).
    """
    _installer_crochet()
    precedent = getattr(_LOCAL, "collecteur", None)
    rassembles: list[str] = []
    _LOCAL.collecteur = rassembles
    try:
        yield rassembles
    finally:
        _LOCAL.collecteur = precedent


def precharger_scikit_learn() -> Tache[None]:
    """Importe en arrière-plan les modules de scikit-learn utilisés ; l'écran d'accueil reste réactif.

    Le premier import de scikit-learn, puis le premier ajustement d'un gradient boosting, prennent
    plusieurs secondes : tout cela se fait dans un fil dès le démarrage, pendant la lecture de l'accueil.
    """

    def importer(annulation: Annulation, avancer: Callable[[Progression], None]) -> None:
        etape = "Chargement des bibliothèques"
        annulation.verifier()
        from sklearn import compose, impute, pipeline, preprocessing  # noqa: F401

        avancer(Progression(etape, 1, 4, "prétraitement"))
        annulation.verifier()
        from sklearn import ensemble, linear_model  # noqa: F401

        avancer(Progression(etape, 2, 4, "modèles"))
        annulation.verifier()
        from sklearn import metrics, model_selection  # noqa: F401

        avancer(Progression(etape, 3, 4, "validation et métriques"))
        annulation.verifier()
        from mecaml.core.train import prechauffer

        prechauffer()
        avancer(Progression(etape, 4, 4, "calcul parallèle"))

    return Tache(importer, nom="prechargement").demarrer()
