"""
MécaML : entraînement et validation croisée des modèles comparés.

Garanties tenues ici :

- I2, ajustement confiné : tout ajustement passe par `Pipeline.fit` sur les seules lignes
  d'apprentissage d'un pli ; aucun transformateur n'est ajusté ailleurs ;
- I6, référence naïve : toute comparaison avec cible contient la référence sur la moyenne ou la classe
  majoritaire (ajoutée d'office si elle manque) et chaque score se lit par rapport à elle ; rien de la cible
  passée n'est jamais ajouté d'office : la persistance et la cible décalée se choisissent et se déclarent
  (`cible_connue`), et sont refusées sur une cible qui décompte (I3, voir `cible_passee`) ;
- I9, bornes : les techniques coûteuses (`EntreeCatalogue.coupee`) sont refusées au delà de
  `LIGNES_MAX_LENTES` lignes, le temps total est borné par un budget, l'annulation est
  vérifiée entre deux (modèle, pli) ;
- I1, scellé : seules les lignes d'apprentissage du coffre sont lues (`CoffreTest.apprentissage`).

La validation croisée est celle de la spécification (`SpecProjet.validation`, compatible avec sa
structure par construction) ; à défaut, elle se déduit de la structure par `split.plis_validation` :
lignes indépendantes (plis stratifiés en classification), groupes disjoints entre plis (stratifiés
en classification), fenêtre croissante pour une série chronologique. Aucun argument de ce module ne
permet de la changer : une validation qui place une unité des deux côtés d'un pli, ou le futur dans
l'apprentissage, rendrait le score optimiste.

La classe positive d'une classification binaire est celle de la spécification (`SpecProjet.classe_positive`) ;
sans elle, les métriques qui en dépendent sont refusées, et les autres colonnes du tableau reposent sur une
proposition (`Resultat.classe_positive_source` dit d'où elle vient).

Les ajustements ne passent pas par des processus multiples. Les avertissements de scikit-learn sont
rassemblés fil par fil (`tasks.collecter_avertissements`), sans état global à restaurer : plusieurs
tâches peuvent tourner à la fois.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mecaml.core import cible_passee, metrics, split
from mecaml.core.catalog import EntreeCatalogue, entree, entrees
from mecaml.core.catalog.entree import GRILLE_MAX_COMBINAISONS
from mecaml.core.cible_passee import PERSISTANCE
from mecaml.core.guards import technique_coupee
from mecaml.core.preprocess import colonnes_d_entree, etapes_persistance, etapes_pretraitement
from mecaml.core.tasks import Annulation, ErreurAnnulation, Progression, Tache, collecter_avertissements
from mecaml.core.types import (
    LIGNES_MAX_LENTES,
    REFERENCES_NAIVES,
    ChoixTechnique,
    CoffreTest,
    Probleme,
    SpecProjet,
    Structure,
    Table,
)

__all__ = [
    "BUDGET_PAR_DEFAUT_S",
    "LIGNES_MAX_LOURD",
    "ErreurBorne",
    "ErreurBudget",
    "ErreurEntrainement",
    "EvaluationTest",
    "ModeleAjuste",
    "Resultat",
    "ResultatModele",
    "ResultatRecherche",
    "ajuster_final",
    "avec_reference",
    "construire_pipeline",
    "entrainer",
    "evaluer_lignes",
    "lancer_entrainement",
    "modeles_disponibles",
    "plis_validation",
    "prechauffer",
    "proposer_modeles",
    "rechercher",
    "verifier_bornes",
]

LIGNES_MAX_LOURD = LIGNES_MAX_LENTES  # alias de la borne unique du socle (I9)
BUDGET_PAR_DEFAUT_S = 300.0  # temps maximal d'un entraînement complet (I9)
FAMILLES_MODELES = ("reference", "regression", "classification", "sans_cible")
REFERENCE = dict(REFERENCES_NAIVES)  # la moyenne et la classe majoritaire, source unique dans `types`
MODELES_PROPOSES = {
    Probleme.REGRESSION: ("regression.ridge", "regression.foret", "regression.gradient_histogramme"),
    Probleme.CLASSIFICATION: (
        "classification.logistique",
        "classification.foret",
        "classification.gradient_histogramme",
    ),
    Probleme.SANS_CIBLE: ("sans_cible.acp", "sans_cible.kmeans"),
}
ANOMALIES = ("sans_cible.foret_isolement", "sans_cible.facteur_local", "sans_cible.svm_une_classe")


MOTIF_CLASSE_POSITIVE_MULTICLASSE = (
    "La cible compte {n} classes : au delà de deux classes, précision, rappel et F1 sont des moyennes macro, "
    "sans classe positive ; regrouper les classes pour en désigner une, ou retirer la classe positive "
    "de la spécification."
)


class ErreurEntrainement(ValueError):
    """Entraînement impossible ou incohérent ; le message dit quoi corriger."""


class ErreurBorne(ErreurEntrainement):
    """Une borne de calcul (I9) est dépassée."""


class ErreurBudget(ErreurEntrainement):
    """Le budget de temps est épuisé avant la fin du calcul."""


# ============================================================
# RÉSULTATS
# ============================================================
def _valeurs_valides(valeurs: Iterable[float]) -> list[float]:
    return [v for v in valeurs if not math.isnan(v)]


@dataclass(frozen=True, eq=False)
class ResultatModele:
    """Résultat d'un modèle : scores par pli, prédictions hors pli, durée, avertissements ou erreur.

    Avec cible, `predictions` et `scores_continus` sont alignés sur les lignes d'apprentissage
    (`Resultat.positions`) ; une ligne jamais validée (fenêtre croissante) reste vide (NaN ou None).
    Sans cible, il n'y a qu'une valeur par score et `pipeline` est le pipeline ajusté.
    """

    choix: ChoixTechnique
    scores_plis: Mapping[str, tuple[float, ...]]
    duree_s: float
    erreur: str | None = None
    avertissements: tuple[str, ...] = ()
    predictions: np.ndarray | None = None
    scores_continus: np.ndarray | None = None
    etiquettes: np.ndarray | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    pipeline: Any = None

    @property
    def id(self) -> str:
        """Identifiant de la technique."""
        return self.choix.id

    @property
    def famille(self) -> str:
        """Famille de la technique (reference, regression...)."""
        return self.choix.famille

    @property
    def ok(self) -> bool:
        """Vrai si le modèle s'est ajusté sans erreur."""
        return self.erreur is None

    def moyenne(self, metrique: str) -> float:
        """Moyenne des plis valides d'une métrique ; NaN si aucun pli n'est défini."""
        valeurs = _valeurs_valides(self.scores_plis.get(metrique, ()))
        return float(np.mean(valeurs)) if valeurs else math.nan

    def plis_valides(self, metrique: str) -> int:
        """Nombre de plis où la métrique est définie (un pli sans classe positive n'en est pas)."""
        return len(_valeurs_valides(self.scores_plis.get(metrique, ())))

    def ecart_type(self, metrique: str) -> float:
        """Écart type (ddof 1) des plis valides ; 0 avec un seul pli, NaN sans pli."""
        valeurs = _valeurs_valides(self.scores_plis.get(metrique, ()))
        if not valeurs:
            return math.nan
        return float(np.std(valeurs, ddof=1)) if len(valeurs) > 1 else 0.0


@dataclass(frozen=True, eq=False)
class Resultat:
    """Comparaison des modèles de `spec.modeles` : ce que reçoit `Entrainer` (S5 vers S6).

    `spec` est la spécification réellement entraînée (référence naïve ajoutée si besoin) : c'est
    elle qu'il faut transmettre à `Entrainer`. `plis` donne, pour chaque pli, les positions locales
    (0 à n-1 dans `positions`) d'apprentissage et de validation. `classe_positive_source` : « specification »
    quand l'utilisateur l'a fixée, sinon l'origine d'une proposition non confirmée (« zero_un », « lexique »,
    « rarete ») ; None hors classification binaire.
    """

    spec: SpecProjet
    metrique: str
    meilleur: str
    modeles: tuple[ResultatModele, ...]
    validation: str
    plis: tuple[tuple[np.ndarray, np.ndarray], ...]
    positions: np.ndarray
    y_vrai: np.ndarray | None
    classes: tuple[Any, ...] | None
    classe_positive: Any
    classe_positive_source: str | None
    n_sans_cible: int
    duree_s: float

    def modele(self, id_modele: str) -> ResultatModele:
        """Résultat d'un modèle par identifiant ; `KeyError` explicite sinon."""
        for m in self.modeles:
            if m.id == id_modele:
                return m
        raise KeyError(f"Modèle absent de la comparaison : {id_modele!r}.")

    def lignes_validees(self) -> np.ndarray:
        """Masque des lignes validées au moins une fois (une fenêtre croissante ne valide pas le début)."""
        masque = np.zeros(len(self.positions), dtype=bool)
        for _, val in self.plis:
            masque[val] = True
        return masque

    def reference(self) -> ResultatModele | None:
        """Référence naïve (moyenne ou classe majoritaire), s'il y en a une.

        La persistance, de la famille `reference` elle aussi, n'en est jamais une : elle lit la cible
        passée, et les gains se lisent contre la moyenne quel que soit l'ordre des modèles.
        """
        return next((m for m in self.modeles if m.id in REFERENCES_NAIVES.values()), None)

    def valeur(self, m: ResultatModele) -> float:
        """Moyenne de la métrique du projet pour un modèle."""
        return m.moyenne(self.metrique)

    def gain(self, id_modele: str) -> float | None:
        """Amélioration sur la référence naïve, positive quand le modèle est meilleur (None si indéfinie)."""
        ref = self.reference()
        valeur, base = self.valeur(self.modele(id_modele)), (self.valeur(ref) if ref else math.nan)
        if math.isnan(valeur) or math.isnan(base) or self.meilleur == metrics.NEUTRE:
            return None
        return valeur - base if self.meilleur == metrics.HAUT else base - valeur

    def classement(self) -> tuple[ResultatModele, ...]:
        """Modèles du meilleur au moins bon ; échecs et scores indéfinis en fin de liste."""
        valides = [m for m in self.modeles if m.ok and not math.isnan(self.valeur(m))]
        autres = [m for m in self.modeles if m not in valides]
        if self.meilleur != metrics.NEUTRE:
            valides.sort(key=self.valeur, reverse=self.meilleur == metrics.HAUT)
        return (*valides, *autres)

    def tableau(self) -> pd.DataFrame:
        """Tableau comparatif de S6 : une ligne par modèle, le meilleur d'abord."""
        lignes = []
        for m in self.classement():
            statut = (
                "échec : " + m.erreur if m.erreur else ("référence" if m.famille == "reference" else "ajusté")
            )
            lignes.append(
                {
                    "modele": m.id,
                    "famille": m.famille,
                    "moyenne": self.valeur(m),
                    "ecart_type": m.ecart_type(self.metrique),
                    "gain_vs_reference": self.gain(m.id),
                    "plis_valides": m.plis_valides(self.metrique),
                    "duree_s": m.duree_s,
                    "statut": statut,
                }
            )
        return pd.DataFrame(lignes)


@dataclass(frozen=True, eq=False)
class ResultatRecherche:
    """Essais d'une recherche d'hyperparamètres sur les plis du projet.

    Le meilleur essai est choisi sur les mêmes plis que ceux qui le notent : son score est donc
    optimiste. Le jeu de test reste scellé ; seule la validation croisée est consultée.
    """

    choix_initial: ChoixTechnique
    mode: str
    metrique: str
    meilleur_sens: str
    essais: tuple[ResultatModele, ...]

    def classement(self) -> tuple[ResultatModele, ...]:
        """Essais du meilleur au moins bon (les échecs et scores indéfinis à la fin)."""
        valides = [m for m in self.essais if m.ok and not math.isnan(m.moyenne(self.metrique))]
        valides.sort(key=lambda m: m.moyenne(self.metrique), reverse=self.meilleur_sens == metrics.HAUT)
        return (*valides, *(m for m in self.essais if m not in valides))

    @property
    def meilleur(self) -> ChoixTechnique | None:
        """Choix du meilleur essai valide, ou None si tous ont échoué."""
        classement = self.classement()
        return classement[0].choix if classement and classement[0].ok else None


@dataclass(frozen=True, eq=False)
class ModeleAjuste:
    """Pipeline ajusté sur toutes les lignes d'apprentissage, prêt à prédire (S7 et S8)."""

    choix: ChoixTechnique
    pipeline: Any
    classes: tuple[Any, ...] | None
    classe_positive: Any
    avertissements: tuple[str, ...] = ()


# ============================================================
# BORNES ET SPÉCIFICATION
# ============================================================
def _entree(id_technique: str, registre: tuple[EntreeCatalogue, ...] | None) -> EntreeCatalogue:
    try:
        return entree(id_technique, registre)
    except KeyError as erreur:
        raise ErreurEntrainement(str(erreur.args[0])) from None


def verifier_bornes(
    spec: SpecProjet, n_lignes: int, registre: Iterable[EntreeCatalogue] | None = None
) -> None:
    """Refuse les techniques coupées (`EntreeCatalogue.coupee`) au delà de `LIGNES_MAX_LENTES` lignes (I9).

    Sont examinées les techniques du prétraitement, des modèles et de la validation de la spécification ;
    le message est celui de `guards.technique_coupee`, le seul du cœur.
    """
    registre = None if registre is None else tuple(registre)
    ids = [c.id for c in (*spec.pretraitement, *spec.modeles)] + (
        [spec.validation] if spec.validation else []
    )
    for id_technique in ids:
        _entree(id_technique, registre)  # une technique inconnue est refusée ici, avec son motif
        refus = technique_coupee(id_technique, n_lignes, registre)
        if refus is not None:
            raise ErreurBorne(refus)


def avec_reference(spec: SpecProjet, registre: Iterable[EntreeCatalogue] | None = None) -> SpecProjet:
    """Spécification munie de la référence sur la moyenne (ou la classe majoritaire) en tête des modèles (I6).

    Elle s'ajoute dès que la spécification n'en porte pas. Rien de la cible passée n'est ajouté d'office : la
    persistance est un choix du projet, qui déclare la cible passée connue à la prévision (`cible_connue`) ;
    elle ne remplace pas la référence sur la moyenne.
    """
    if not spec.probleme.avec_cible or any(c.id in REFERENCE.values() for c in spec.modeles):
        return spec
    return spec.avec(modeles=(ChoixTechnique.creer(REFERENCE[spec.probleme]), *spec.modeles))


def modeles_disponibles(
    probleme: Probleme, registre: Iterable[EntreeCatalogue] | None = None
) -> tuple[EntreeCatalogue, ...]:
    """Catalogue filtré de S6 : références et modèles applicables au problème."""
    return tuple(e for e in entrees(registre, probleme=probleme) if e.famille in FAMILLES_MODELES)


def proposer_modeles(
    probleme: Probleme, structure: Structure = Structure.INDEPENDANTE
) -> tuple[ChoixTechnique, ...]:
    """Comparaison proposée par défaut en S6 : la référence naïve, puis trois modèles courants.

    `structure` est gardée pour l'appelant : aucune référence ne dépend plus de la structure, car la
    persistance ne se propose jamais sans que le projet déclare la cible passée connue.
    """
    base = tuple(ChoixTechnique.creer(i) for i in MODELES_PROPOSES[probleme])
    if not probleme.avec_cible:
        return base
    return (ChoixTechnique.creer(REFERENCE[probleme]), *base)


def _exiger_un_modele(spec: SpecProjet) -> None:
    """La comparaison porte sur au moins un modèle choisi, la référence naïve ne comptant pas."""
    choisis = [c for c in spec.modeles if c.famille != "reference"]
    if not spec.probleme.avec_cible:
        choisis = list(spec.modeles)
    if not choisis:
        raise ErreurEntrainement("Aucun modèle à comparer : choisir un modèle en S6, hors référence naïve.")


def _verifier_modeles(spec: SpecProjet, registre: tuple[EntreeCatalogue, ...] | None) -> None:
    if not spec.modeles:
        raise ErreurEntrainement("Aucun modèle à comparer : choisir au moins un modèle en S6.")
    for choix in spec.modeles:
        e = _entree(choix.id, registre)
        if e.famille not in FAMILLES_MODELES or spec.probleme not in e.problemes:
            raise ErreurEntrainement(
                f"{choix.id} n'est pas un modèle applicable à un problème de {spec.probleme}."
            )
    if any(c.id == PERSISTANCE for c in spec.modeles) and not spec.structure.chronologique:
        raise ErreurEntrainement(
            "La référence par persistance suppose des lignes ordonnées dans le temps : "
            "choisir une structure chronologique en S3."
        )
    if spec.probleme.avec_cible:
        metrics.valider(spec.metrique or "", spec.probleme)
    elif spec.metrique:
        metrics.valider(spec.metrique, spec.probleme)


def construire_pipeline(
    spec: SpecProjet,
    choix: ChoixTechnique,
    registre: Iterable[EntreeCatalogue] | None = None,
    contexte: pd.DataFrame | None = None,
) -> Any:
    """`Pipeline` non ajusté : prétraitement par rôle, sélection éventuelle, puis le modèle `choix`.

    La référence par persistance fait exception : elle ignore le prétraitement du projet et ne lit que
    la cible décalée d'un pas.
    """
    from sklearn.pipeline import Pipeline

    registre = None if registre is None else tuple(registre)
    modele = _entree(choix.id, registre).construire(spec.graine, **choix.parametres_dict())
    fabrique_etapes = etapes_persistance if choix.id == PERSISTANCE else etapes_pretraitement
    etapes = fabrique_etapes(spec, registre, contexte)
    return Pipeline([*etapes, ("modele", modele)])


# ============================================================
# DONNÉES ET PLIS
# ============================================================
@dataclass(frozen=True, eq=False)
class _Contexte:
    spec: SpecProjet
    X: pd.DataFrame
    y: np.ndarray | None
    donnees: pd.DataFrame
    positions: np.ndarray
    plis: tuple[tuple[np.ndarray, np.ndarray], ...]
    validation: str
    classes: tuple[Any, ...] | None
    positive: Any
    positive_source: str | None
    n_sans_cible: int
    contexte: pd.DataFrame | None
    registre: tuple[EntreeCatalogue, ...] | None


def _lignes_d_apprentissage(
    spec: SpecProjet, table: Table, coffre: CoffreTest | None
) -> tuple[pd.DataFrame, np.ndarray]:
    if not spec.probleme.avec_cible:
        return table.donnees, np.arange(table.n_lignes)
    if coffre is None:
        raise ErreurEntrainement(
            "Aucun jeu de test mis sous scellé : découper les données (S4) avant d'entraîner."
        )
    if not coffre.repose_sur(table):
        raise ErreurEntrainement("Le coffre de test n'a pas été construit sur cette table.")
    if coffre.decoupage != spec.decoupage():
        raise ErreurEntrainement(
            "Le coffre de test correspond à un autre découpage que celui de la spécification."
        )
    return coffre.apprentissage(), coffre.positions_apprentissage()


def _colonnes_presentes(spec: SpecProjet, donnees: pd.DataFrame) -> None:
    attendues = [*colonnes_d_entree(spec), *([spec.cible] if spec.cible else [])]
    absentes = [c for c in attendues if c not in donnees.columns]
    if absentes:
        raise ErreurEntrainement(f"Colonnes absentes des données : {absentes[:5]}.")
    if not spec.variables():
        raise ErreurEntrainement(
            "Aucune variable d'entrée : donner le rôle numérique ou catégoriel à des colonnes."
        )


def _cible_et_classes(
    spec: SpecProjet, donnees: pd.DataFrame, positions: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Sépare les lignes sans cible (comptées, gardées en contexte) et contrôle que la cible convient."""
    presentes = donnees[spec.cible].notna().to_numpy()
    avec, sans = split.separer_sans_cible(donnees, spec.cible)
    positions = positions[presentes]
    serie = avec[spec.cible]
    if spec.probleme is Probleme.REGRESSION:
        if not pd.api.types.is_numeric_dtype(serie):
            raise ErreurEntrainement(
                f"La cible {spec.cible!r} n'est pas numérique : choisir la classification."
            )
        return avec, serie.to_numpy(dtype=float), positions, sans
    return avec, serie.to_numpy(), positions, sans


def _refuser_la_cible_passee(spec: SpecProjet, donnees: pd.DataFrame) -> None:
    """Refuse la cible décalée ou la persistance sans déclaration, ou sur une cible qui décompte (I3).

    Seules les lignes d'apprentissage qui ont une cible sont lues. Rien n'est ajusté avant ce refus.
    """
    refus = cible_passee.constats(spec, donnees)
    if refus:
        raise ErreurEntrainement(" ".join(f"{c.message} {c.action}" for c in refus))


def plis_validation(
    spec: SpecProjet, donnees: pd.DataFrame, registre: Iterable[EntreeCatalogue] | None = None
) -> tuple[str, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Plis (positions locales d'apprentissage et de validation) de la validation du projet.

    Source unique : `split.plis_validation`, qui suit `spec.validation` quand elle est fixée (le socle a
    refusé toute méthode incompatible avec la structure) et déduit sinon la validation de la structure.
    Rend l'identifiant de la validation employée et les plis. Aucun paramètre ne permet de la contourner.
    """
    try:
        plis = split.plis_validation(donnees, spec)
    except split.ErreurDecoupage as erreur:
        raise ErreurEntrainement(f"Validation croisée impossible : {erreur}") from None
    return split.validation_appliquee(spec), tuple(plis)


def _classe_positive(spec: SpecProjet, y: np.ndarray, classes: tuple[Any, ...]) -> tuple[Any, str | None]:
    """Classe positive et son origine ; refus si la métrique en dépend et que la spécification l'omet.

    Au delà de deux classes il n'y a pas de classe positive : précision, rappel et F1 sont des moyennes macro,
    et une classe positive déclarée est refusée plutôt que ignorée en silence.
    """
    if len(classes) > 2 and spec.classe_positive is not None:
        raise ErreurEntrainement(MOTIF_CLASSE_POSITIVE_MULTICLASSE.format(n=len(classes)))
    if len(classes) != 2:
        return None, None
    if spec.classe_positive is not None:
        if spec.classe_positive not in classes:
            raise ErreurEntrainement(
                f"La classe positive {spec.classe_positive!r} n'est pas une classe de la cible "
                f"{list(classes)}."
            )
        return classes[classes.index(spec.classe_positive)], "specification"
    proposee, origine = metrics.proposer_classe_positive(y)
    if spec.exige_classe_positive():
        raise ErreurEntrainement(
            f"La métrique {spec.metrique} dépend de la classe positive : la fixer en S3 parmi "
            f"{list(classes)} "
            f"(proposition : {proposee!r}, d'après {origine})."
        )
    return proposee, origine


def _preparer(
    spec: SpecProjet,
    table: Table,
    coffre: CoffreTest | None,
    registre: tuple[EntreeCatalogue, ...] | None,
    avec_plis: bool = True,
) -> _Contexte:
    donnees, positions = _lignes_d_apprentissage(spec, table, coffre)
    _colonnes_presentes(spec, donnees)
    y, classes, positive, origine, sans = None, None, None, None, None
    if spec.probleme.avec_cible:
        donnees, y, positions, sans = _cible_et_classes(spec, donnees, positions)
        _refuser_la_cible_passee(spec, donnees)
        if spec.probleme is Probleme.CLASSIFICATION:
            classes = tuple(np.unique(y).tolist())
            if len(classes) < 2:
                raise ErreurEntrainement(
                    "La cible n'a qu'une classe dans l'apprentissage : classification impossible."
                )
            positive, origine = _classe_positive(spec, y, classes)
    verifier_bornes(spec, len(donnees), registre)
    identifiant, plis = ("aucune", ())
    if spec.probleme.avec_cible and avec_plis:
        identifiant, plis = plis_validation(spec, donnees, registre)
    n_sans_cible = 0 if sans is None else len(sans)
    # Les lignes sans cible complètent les séries des variables temporelles ; elles ne servent à rien d'autre.
    contexte = None if sans is None or not n_sans_cible else sans[list(colonnes_d_entree(spec))]
    return _Contexte(
        spec,
        donnees[list(colonnes_d_entree(spec))],
        y,
        donnees,
        positions,
        plis,
        identifiant,
        classes,
        positive,
        origine,
        n_sans_cible,
        contexte,
        registre,
    )


# ============================================================
# RYTHME : ANNULATION, BUDGET, PROGRESSION
# ============================================================
class _Rythme:
    """Vérifie l'annulation et le budget entre deux unités de travail et signale l'avancement."""

    def __init__(
        self, annulation: Annulation | None, progression: Any, budget_s: float, total: int, etape: str
    ) -> None:
        self.annulation, self.progression, self.budget_s = annulation, progression, budget_s
        self.total, self.etape, self.fait = total, etape, 0
        self.debut = time.perf_counter()

    def avant(self) -> None:
        if self.annulation is not None:
            self.annulation.verifier()
        ecoule = time.perf_counter() - self.debut
        if ecoule > self.budget_s:
            raise ErreurBudget(
                f"Budget de {self.budget_s:.0f} s épuisé après {ecoule:.0f} s : "
                "comparer moins de modèles, moins de plis ou une grille plus petite."
            )

    def apres(self, message: str, unites: int = 1) -> None:
        self.fait += unites
        if self.progression is not None:
            self.progression(Progression(self.etape, self.fait, self.total, message))


# ============================================================
# VALIDATION CROISÉE D'UN MODÈLE
# ============================================================
def _scores_des_predictions(
    spec: SpecProjet,
    y_vrai: np.ndarray,
    predictions: np.ndarray,
    scores: Any,
    classes: tuple[Any, ...] | None,
    positive: Any,
    notes: list[str],
) -> dict[str, float]:
    """Métriques standard du problème et métrique du projet pour des lignes prédites."""
    donnees = {
        "y_vrai": y_vrai,
        "y_pred": predictions,
        "scores": scores,
        "classes": classes,
        "classe_positive": positive,
    }
    valeurs = metrics.calculer_standard(spec.probleme, notes, **donnees)
    if spec.metrique and spec.metrique not in valeurs:
        try:
            valeurs[spec.metrique] = metrics.calculer(spec.metrique, **donnees)
        except metrics.ErreurMetrique as erreur:
            valeurs[spec.metrique] = math.nan
            notes.append(f"{spec.metrique} : {erreur}")
    return valeurs


def _un_pli(
    ctx: _Contexte, pipeline: Any, app: np.ndarray, val: np.ndarray, notes: list[str]
) -> tuple[dict[str, float], np.ndarray, Any]:
    """Ajuste une copie du pipeline sur le pli d'apprentissage ; note la validation (I2)."""
    from sklearn.base import clone

    modele = clone(pipeline)
    x_val = ctx.X.iloc[val]
    with collecter_avertissements() as captees:
        modele.fit(ctx.X.iloc[app], ctx.y[app])
        predictions = modele.predict(x_val)
        scores = None
        if ctx.classes is not None:
            scores = metrics.scores_du_modele(modele, x_val, ctx.classes, ctx.positive)
    notes.extend(sorted(set(captees)))
    return (
        _scores_des_predictions(ctx.spec, ctx.y[val], predictions, scores, ctx.classes, ctx.positive, notes),
        predictions,
        scores,
    )


def _vide(ctx: _Contexte) -> tuple[np.ndarray, np.ndarray | None]:
    """Tableaux hors pli, complétés pli après pli : prédictions au type de la cible, scores continus en nan.

    Une classification garde le type des étiquettes (entiers, booléens ou textes) : un tableau d'objets
    ferait croire à scikit-learn que la cible est inconnue. Les lignes jamais validées sont données par
    `Resultat.lignes_validees`, non par une valeur vide.
    """
    n = len(ctx.y)
    if ctx.classes is None:
        return np.full(n, np.nan), None
    return np.full(n, ctx.classes[0], dtype=np.asarray(ctx.classes).dtype), np.full(
        (n, len(ctx.classes)) if ctx.positive is None else n, np.nan
    )


def _valider_modele(ctx: _Contexte, choix: ChoixTechnique, rythme: _Rythme) -> ResultatModele:
    """Validation croisée d'un modèle : un ajustement par pli, prédictions hors pli conservées."""
    debut = time.perf_counter()
    pipeline = construire_pipeline(ctx.spec, choix, ctx.registre, ctx.contexte)
    scores_plis: dict[str, list[float]] = {}
    notes: list[str] = []
    predites, continus = _vide(ctx)
    for k, (app, val) in enumerate(ctx.plis, start=1):
        rythme.avant()
        valeurs, predictions, scores = _un_pli(ctx, pipeline, app, val, notes)
        for nom, valeur in valeurs.items():
            scores_plis.setdefault(nom, []).append(valeur)
        predites[val] = predictions
        if continus is not None and scores is not None:
            continus[val] = scores
        rythme.apres(f"{choix.id}, pli {k} sur {len(ctx.plis)}")
    return ResultatModele(
        choix=choix,
        scores_plis={nom: tuple(v) for nom, v in scores_plis.items()},
        duree_s=time.perf_counter() - debut,
        avertissements=tuple(dict.fromkeys(notes)),
        predictions=predites,
        scores_continus=continus,
    )


def _valider_ou_consigner(
    ctx: _Contexte, choix: ChoixTechnique, rythme: _Rythme, sans_cible: Any = None
) -> ResultatModele:
    """Un modèle qui échoue est consigné et la comparaison continue, sauf pour la référence naïve."""
    debut, avant = time.perf_counter(), rythme.fait
    try:
        if ctx.spec.probleme.avec_cible:
            return _valider_modele(ctx, choix, rythme)
        return _ajuster_sans_cible(ctx, choix, rythme, sans_cible)
    except (ErreurAnnulation, ErreurBudget):
        raise
    except Exception as erreur:  # un modèle défaillant ne doit pas faire perdre les autres
        if choix.id in REFERENCES_NAIVES.values():
            raise ErreurEntrainement(
                f"La référence naïve a échoué ({erreur}) : vérifier le prétraitement et les données."
            ) from erreur
        restant = (len(ctx.plis) if ctx.spec.probleme.avec_cible else 1) - (rythme.fait - avant)
        rythme.apres(f"{choix.id} : échec", max(restant, 0))
        return ResultatModele(
            choix, {}, time.perf_counter() - debut, erreur=f"{type(erreur).__name__} : {erreur}"
        )


# ============================================================
# SANS CIBLE
# ============================================================
def _scores_sans_cible(
    final: Any, etiquettes: Any, transformees: Any, ref: Any, graine: int, anomalie: bool, notes: list[str]
) -> dict[str, float]:
    choix_metriques = []
    if etiquettes is not None:
        choix_metriques.append("taux_anomalies" if anomalie else "silhouette")
    if hasattr(final, "explained_variance_ratio_"):
        choix_metriques.append("variance_expliquee")
    if ref is not None and etiquettes is not None:
        choix_metriques.append("rand_ajuste")
    valeurs: dict[str, float] = {}
    for nom in choix_metriques:
        try:
            valeurs[nom] = metrics.calculer(
                nom, modele=final, etiquettes=etiquettes, x=transformees, etiquettes_ref=ref, graine=graine
            )
        except metrics.ErreurMetrique as erreur:
            valeurs[nom] = math.nan
            notes.append(f"{nom} : {erreur}")
    return valeurs


def _details_sans_cible(final: Any, etiquettes: np.ndarray | None, anomalie: bool) -> dict[str, Any]:
    """Faits utiles à l'écran : groupes et bruit, part d'anomalies, variance par composante."""
    if hasattr(final, "explained_variance_ratio_"):
        return {"variance_par_composante": [float(v) for v in final.explained_variance_ratio_]}
    if etiquettes is None:
        return {}
    if anomalie:
        return {"part_anomalies": float(np.mean(etiquettes == -1))}
    return {
        "n_groupes": len(set(etiquettes.tolist()) - {-1}),
        "part_bruit": float(np.mean(etiquettes == -1)),
    }


def _ajuster_sans_cible(ctx: _Contexte, choix: ChoixTechnique, rythme: _Rythme, ref: Any) -> ResultatModele:
    """Ajuste un modèle sans cible sur toutes les lignes : une valeur par score, étiquettes conservées."""
    rythme.avant()
    debut = time.perf_counter()
    pipeline = construire_pipeline(ctx.spec, choix, ctx.registre, ctx.contexte)
    with collecter_avertissements() as captees:
        etiquettes = None
        if hasattr(pipeline, "fit_predict"):
            etiquettes = np.asarray(pipeline.fit_predict(ctx.X))
        else:
            pipeline.fit(ctx.X)
    notes = sorted(set(captees))
    final, anomalie = pipeline[-1], choix.id in ANOMALIES
    transformees = pipeline[:-1].transform(ctx.X) if etiquettes is not None and not anomalie else None
    valeurs = _scores_sans_cible(final, etiquettes, transformees, ref, ctx.spec.graine, anomalie, notes)
    rythme.apres(choix.id)
    return ResultatModele(
        choix=choix,
        scores_plis={nom: (v,) for nom, v in valeurs.items()},
        duree_s=time.perf_counter() - debut,
        avertissements=tuple(notes),
        etiquettes=etiquettes,
        details=_details_sans_cible(final, etiquettes, anomalie),
        pipeline=pipeline,
    )


# ============================================================
# POINTS D'ENTRÉE
# ============================================================
def _metrique_du_projet(spec: SpecProjet) -> tuple[str, str]:
    texte = spec.metrique or "silhouette"
    return texte, metrics.analyser(texte)[0].meilleur


def entrainer(
    spec: SpecProjet,
    table: Table,
    coffre: CoffreTest | None = None,
    *,
    annulation: Annulation | None = None,
    progression: Any = None,
    budget_s: float = BUDGET_PAR_DEFAUT_S,
    registre: Iterable[EntreeCatalogue] | None = None,
    etiquettes_reference: Any = None,
) -> Resultat:
    """Valide chaque modèle de `spec.modeles` par validation croisée sur les lignes d'apprentissage.

    Avec cible, une référence naïve est ajoutée si elle manque (I6) et seules les lignes
    d'apprentissage du coffre sont lues (I1). Sans cible, chaque modèle est ajusté une fois sur
    toutes les lignes ; `etiquettes_reference` (un groupe connu par ligne) donne l'indice de Rand.
    `progression(Progression)` reçoit l'avancement ; `annulation.verifier()` est appelée entre deux
    (modèle, pli). Rend un `Resultat` ; transmettre `resultat.spec` à `Entrainer`.
    """
    debut = time.perf_counter()
    registre = None if registre is None else tuple(registre)
    _exiger_un_modele(spec)
    spec = avec_reference(spec, registre)
    _verifier_modeles(spec, registre)
    ctx = _preparer(spec, table, coffre, registre)
    ref = _etiquettes_de_reference(etiquettes_reference, len(ctx.X))
    unites = len(spec.modeles) * (len(ctx.plis) if spec.probleme.avec_cible else 1)
    rythme = _Rythme(annulation, progression, budget_s, unites, "Entraînement")
    modeles = tuple(_valider_ou_consigner(ctx, choix, rythme, ref) for choix in spec.modeles)
    texte, sens = _metrique_du_projet(spec)
    return Resultat(
        spec=spec,
        metrique=texte,
        meilleur=sens,
        modeles=modeles,
        validation=ctx.validation,
        plis=ctx.plis,
        positions=ctx.positions,
        y_vrai=ctx.y,
        classes=ctx.classes,
        classe_positive=ctx.positive,
        classe_positive_source=ctx.positive_source,
        n_sans_cible=ctx.n_sans_cible,
        duree_s=time.perf_counter() - debut,
    )


def _etiquettes_de_reference(etiquettes: Any, n: int) -> np.ndarray | None:
    if etiquettes is None:
        return None
    ref = np.asarray(etiquettes)
    if ref.ndim != 1 or len(ref) != n:
        raise ErreurEntrainement(f"Les étiquettes de référence ont {len(ref)} valeurs pour {n} lignes.")
    return ref


def _candidats(
    choix: ChoixTechnique, e: EntreeCatalogue, mode: str, n_iter: int, graine: int
) -> list[ChoixTechnique]:
    from sklearn.model_selection import ParameterGrid, ParameterSampler

    if mode not in ("grille", "aleatoire"):
        raise ErreurEntrainement(f"Mode de recherche inconnu : {mode!r} (grille ou aleatoire).")
    if not e.grille_bornee:
        raise ErreurEntrainement(f"{choix.id} n'a aucun hyperparamètre à rechercher.")
    if e.combinaisons > GRILLE_MAX_COMBINAISONS:
        raise ErreurBorne(f"Grille de {e.combinaisons} combinaisons, borne {GRILLE_MAX_COMBINAISONS}.")
    if mode == "grille":
        essais = list(ParameterGrid(e.grille()))
    else:
        if not 1 <= n_iter <= 50:
            raise ErreurBorne(f"Recherche aléatoire : de 1 à 50 tirages, {n_iter} demandés.")
        essais = list(ParameterSampler(e.grille(), n_iter=min(n_iter, e.combinaisons), random_state=graine))
    base = choix.parametres_dict()
    return [ChoixTechnique.creer(choix.id, colonnes=choix.colonnes, **{**base, **essai}) for essai in essais]


def rechercher(
    spec: SpecProjet,
    choix: ChoixTechnique,
    table: Table,
    coffre: CoffreTest | None = None,
    *,
    mode: str = "grille",
    n_iter: int = 20,
    annulation: Annulation | None = None,
    progression: Any = None,
    budget_s: float = BUDGET_PAR_DEFAUT_S,
    registre: Iterable[EntreeCatalogue] | None = None,
) -> ResultatRecherche:
    """Recherche d'hyperparamètres d'un modèle sur la grille bornée de son entrée (S6).

    Chaque combinaison est validée sur les mêmes plis que la comparaison ; la grille complète
    (`mode="grille"`) ou un tirage de `n_iter` combinaisons (`mode="aleatoire"`). Annulation et
    budget sont vérifiés entre deux (combinaison, pli).
    """
    if not spec.probleme.avec_cible:
        raise ErreurEntrainement(
            "La recherche d'hyperparamètres se fait avec une cible, sur la validation croisée."
        )
    registre = None if registre is None else tuple(registre)
    spec = spec.avec(modeles=(choix,))
    _verifier_modeles(spec, registre)
    ctx = _preparer(spec, table, coffre, registre)
    candidats = _candidats(choix, _entree(choix.id, registre), mode, n_iter, spec.graine)
    rythme = _Rythme(annulation, progression, budget_s, len(candidats) * len(ctx.plis), "Recherche")
    essais = tuple(_valider_ou_consigner(ctx, c, rythme) for c in candidats)
    texte, sens = _metrique_du_projet(spec)
    return ResultatRecherche(choix, mode, texte, sens, essais)


def ajuster_final(
    spec: SpecProjet,
    choix: ChoixTechnique,
    table: Table,
    coffre: CoffreTest | None = None,
    *,
    registre: Iterable[EntreeCatalogue] | None = None,
) -> ModeleAjuste:
    """Ajuste le modèle retenu sur toutes les lignes d'apprentissage (jamais sur le test)."""
    registre = None if registre is None else tuple(registre)
    ctx = _preparer(spec, table, coffre, registre, avec_plis=False)
    pipeline = construire_pipeline(spec, choix, registre, ctx.contexte)
    with collecter_avertissements() as captees:
        pipeline.fit(ctx.X, ctx.y)
    notes = tuple(sorted(set(captees)))
    return ModeleAjuste(choix, pipeline, ctx.classes, ctx.positive, notes)


class EvaluationTest(dict):
    """Métriques sur des lignes (nom de métrique vers valeur), avec le compte des lignes retirées.

    `n_lignes` : lignes évaluées ; `n_sans_cible` : lignes retirées faute de cible, que le rapport doit citer.
    """

    def __init__(self, scores: dict[str, float], n_lignes: int, n_sans_cible: int) -> None:
        super().__init__(scores)
        self.n_lignes = n_lignes
        self.n_sans_cible = n_sans_cible


def evaluer_lignes(modele: ModeleAjuste, spec: SpecProjet, lignes: pd.DataFrame) -> EvaluationTest:
    """Métriques du modèle ajusté sur des lignes (le test ouvert en S8, par exemple) ; NaN si indéfinies.

    Les lignes sans cible sont retirées et comptées (`n_sans_cible`).
    """
    if not spec.probleme.avec_cible:
        raise ErreurEntrainement("L'évaluation sur des lignes de test suppose une cible.")
    presentes = lignes[spec.cible].notna().to_numpy()
    n_sans_cible = int((~presentes).sum())
    variables = lignes[list(colonnes_d_entree(spec))]
    # Toutes les lignes sont prédites (les variables temporelles ont besoin de celles sans cible),
    # mais seules celles qui ont une cible sont notées.
    predictions = np.asarray(modele.pipeline.predict(variables))[presentes]
    y = lignes.loc[presentes, spec.cible].to_numpy(
        dtype=float if spec.probleme is Probleme.REGRESSION else None
    )
    scores = None
    if modele.classes is not None:
        brut = metrics.scores_du_modele(modele.pipeline, variables, modele.classes, modele.classe_positive)
        scores = None if brut is None else np.asarray(brut)[presentes]
    valeurs = _scores_des_predictions(
        spec, y, predictions, scores, modele.classes, modele.classe_positive, []
    )
    return EvaluationTest(valeurs, int(presentes.sum()), n_sans_cible)


def prechauffer() -> None:
    """Premier ajustement factice d'un gradient boosting sur histogrammes, sans rapport avec le projet.

    À froid, le premier ajustement de ce modèle prend plusieurs secondes (initialisation du calcul
    parallèle de la bibliothèque) ; les suivants, quelques centièmes de seconde. L'application le
    lance en arrière-plan à l'ouverture pour que le premier entraînement de l'utilisateur ne l'attende pas.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline

    echantillon = np.random.default_rng(0).normal(size=(20, 3))
    pipeline = Pipeline([("modele", HistGradientBoostingRegressor(max_iter=2, random_state=0))])
    pipeline.fit(echantillon, echantillon[:, 0])


def lancer_entrainement(
    spec: SpecProjet, table: Table, coffre: CoffreTest | None = None, **options: Any
) -> Tache:
    """Lance `entrainer` dans un fil annulable ; `tache.resultat()` rend le `Resultat`."""

    def calcul(annulation: Annulation, avancer: Any) -> Resultat:
        return entrainer(spec, table, coffre, annulation=annulation, progression=avancer, **options)

    return Tache(calcul, nom="entrainement").demarrer()
