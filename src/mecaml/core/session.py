"""
MécaML : session du projet, machine à dix états S0 à S9.

L'interface n'applique aucune règle : elle envoie des actions, la session les
accepte ou les refuse avec un motif actionnable. Une session est gelée ; chaque
action acceptée rend une nouvelle session qui partage la table par référence.

Trois règles de transition :

1. Avance. L'action qui produit l'état Sk est acceptée depuis S(k-1). Refaire une
   étape déjà franchie (Sk atteint ou dépassé) vaut retour à S(k-1) puis avance.
2. Retour arrière. `Revenir(Sk)` invalide tout ce qui a été produit après Sk,
   spécification comprise : prétraitement effacé sous S5, modèles effacés sous S6.
   Avant l'ouverture du test, tout état antérieur est permis ; après, seuls S5 à S8
   et S0 (nouveau projet) : le découpage n'est jamais refait au vu du test.
3. Rejeu de l'audit. L'audit est rejoué quand l'objectif (cible, rôles, métrique)
   ou le découpage change, à chaque nouveau coffre et à chaque retour en S2 ou S3 ;
   les acquittements des alertes encore présentes sont conservés.

Garde des constats : quitter S4 (et toute avance ensuite) exige qu'aucun constat
BLOQUANT ne subsiste et que chaque ALERTE soit acquittée. `FixerObjectif` et
`Decouper` n'y sont pas soumis : ce sont les actions qui corrigent (changer un
rôle, écarter une colonne, changer de découpage), et elles rejouent l'audit.

Tests déjà vus : la session garde les empreintes des données dont le test a été
ouvert (`tests_vus`), y compris à travers « nouveau projet » et le chargement de la
même table ; toute ouverture sur une empreinte déjà vue donne un score optimiste.

Branche sans cible (`Probleme.SANS_CIBLE`) : S4 sans coffre, S5 à S7 portent la
réduction de dimension, le regroupement ou les anomalies, S8 est sans objet et
l'export se fait depuis S7.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, ClassVar, TypeVar

import pandas as pd

from mecaml.core.types import (
    REFERENCES_NAIVES,
    CoffreTest,
    Constat,
    Decision,
    Niveau,
    Probleme,
    SpecProjet,
    Table,
)

__all__ = [
    "ACTIONS",
    "Acquitter",
    "Action",
    "Auditer",
    "Auditeur",
    "Charger",
    "Decider",
    "Decouper",
    "Entrainer",
    "Etat",
    "Exporter",
    "FixerObjectif",
    "Pretraiter",
    "Revenir",
    "Session",
    "Tester",
    "TransitionRefusee",
    "auditeur_neutre",
]

journal = logging.getLogger(__name__)
_T = TypeVar("_T")


# ============================================================
# ÉTATS
# ============================================================
class Etat(IntEnum):
    """États du projet ; l'écran k mène de S(k-1) à Sk."""

    VIDE = 0
    DONNEES = 1
    AUDIT = 2
    OBJECTIF = 3
    DECOUPE = 4
    PRETRAITEMENT = 5
    MODELES = 6
    DIAGNOSTIC = 7
    TESTE = 8
    EXPORTE = 9


# États que l'on peut refaire après l'ouverture du test.
_REPRISES_APRES_TEST = frozenset({Etat.PRETRAITEMENT, Etat.MODELES, Etat.DIAGNOSTIC})


class TransitionRefusee(Exception):
    """Action refusée dans l'état courant ; le message dit quoi faire."""

    def __init__(self, etat: Etat, action: Action, motif: str) -> None:
        self.etat = etat
        self.action = action
        self.motif = motif
        super().__init__(f"{type(action).__name__} refusée en S{int(etat)} {etat.name} : {motif}")


Auditeur = Callable[[Table, SpecProjet | None, CoffreTest | None], Sequence[Constat]]
"""Auditeur : (table, spécification ou None, coffre ou None) vers constats.

En S2 et S3, le coffre est None : le seul usage permis de la cible sur toutes les
lignes est la détection de fuite à seuil élevé. Dès que le coffre est fourni (S4),
tout calcul qui fait intervenir la cible se restreint à `positions_apprentissage()`.
Les codes rendus sont uniques.
"""


def auditeur_neutre(table: Table, spec: SpecProjet | None, coffre: CoffreTest | None) -> tuple[Constat, ...]:
    """Auditeur par défaut : aucun constat."""
    return ()


# ============================================================
# ACTIONS
# ============================================================
@dataclass(frozen=True)
class Action:
    """Action envoyée par l'interface ; `produit` est l'état qu'elle fait atteindre."""

    produit: ClassVar[Etat | None] = None


@dataclass(frozen=True)
class Charger(Action):
    """S0 vers S1 : table chargée et bornée."""

    table: Table
    produit: ClassVar[Etat | None] = Etat.DONNEES


@dataclass(frozen=True)
class Auditer(Action):
    """S1 vers S2 : profil et constats sans cible."""

    produit: ClassVar[Etat | None] = Etat.AUDIT


@dataclass(frozen=True)
class Acquitter(Action):
    """Acquitte des alertes, en S2, S3 ou S4 ; ne change pas d'état."""

    codes: frozenset[str]


@dataclass(frozen=True)
class FixerObjectif(Action):
    """S2 vers S3 : problème, cible, rôles, métrique, classe positive, structure ; audit rejoué.

    En classification, la classe positive doit être une valeur de la cible ; elle est
    exigée quand la cible a deux classes et que la métrique en dépend
    (`SpecProjet.exige_classe_positive`) : l'interface la propose, l'utilisateur la confirme.
    Au delà de deux classes il n'y en a pas (moyennes macro) : en désigner une est refusé.
    """

    spec: SpecProjet
    produit: ClassVar[Etat | None] = Etat.OBJECTIF


@dataclass(frozen=True)
class Decouper(Action):
    """S3 vers S4 : coffre de test scellé, construit pour le découpage de la spécification.

    `spec` facultative : si elle est donnée, seul le découpage peut y changer. L'audit
    est rejoué avec le coffre (contrôles de groupes disjoints et de dates ordonnées).
    """

    coffre: CoffreTest | None
    spec: SpecProjet | None = None
    produit: ClassVar[Etat | None] = Etat.DECOUPE


@dataclass(frozen=True)
class Pretraiter(Action):
    """S4 vers S5 : prétraitement retenu (seul `pretraitement` peut changer)."""

    spec: SpecProjet
    produit: ClassVar[Etat | None] = Etat.PRETRAITEMENT


@dataclass(frozen=True)
class Entrainer(Action):
    """S5 vers S6 : résultats de validation croisée des modèles de `spec.modeles`.

    Avec cible, la comparaison contient une référence naïve (`REFERENCES_NAIVES`).
    """

    resultats: Any
    spec: SpecProjet | None = None
    produit: ClassVar[Etat | None] = Etat.MODELES


@dataclass(frozen=True)
class Decider(Action):
    """S6 vers S7 : diagnostic hors pli et décision motivée sur un modèle comparé."""

    decision: Decision
    produit: ClassVar[Etat | None] = Etat.DIAGNOSTIC


@dataclass(frozen=True)
class Tester(Action):
    """S7 vers S8 : ouverture du coffre ; `evaluer` reçoit une copie des lignes de test.

    L'ouverture est comptée avant l'appel d'`evaluer` : si `evaluer` lève une
    exception, la session reste en S7 mais l'ouverture compte, et la suivante
    donnera un score optimiste. C'est le choix prudent : le test a pu être vu.
    """

    evaluer: Callable[[pd.DataFrame], Any]
    produit: ClassVar[Etat | None] = Etat.TESTE


@dataclass(frozen=True)
class Exporter(Action):
    """S8 (ou S7 sans cible) vers S9 : rapport, modèle, script."""

    export: Any
    produit: ClassVar[Etat | None] = Etat.EXPORTE


@dataclass(frozen=True)
class Revenir(Action):
    """Retour arrière vers un état antérieur ; l'aval est invalidé."""

    etat: Etat


ACTIONS: tuple[type[Action], ...] = (
    Charger,
    Auditer,
    Acquitter,
    FixerObjectif,
    Decouper,
    Pretraiter,
    Entrainer,
    Decider,
    Tester,
    Exporter,
    Revenir,
)


def _purger_spec(spec: SpecProjet | None, etat: Etat) -> SpecProjet | None:
    """Spécification ramenée à `etat` : prétraitement effacé sous S5, modèles sous S6."""
    if spec is None:
        return None
    if etat < Etat.PRETRAITEMENT and (spec.pretraitement or spec.modeles):
        return spec.avec(pretraitement=(), modeles=())
    if etat < Etat.MODELES and spec.modeles:
        return spec.avec(modeles=())
    return spec


# ============================================================
# SESSION
# ============================================================
@dataclass(frozen=True, eq=False)
class Session:
    """État gelé du projet. Chaque champ est produit par un état précis.

    `table` (S1), `constats` et `acquittes` (S2, rejoués en S3 et S4), `spec` (S3,
    complétée en S4 à S6), `coffre` (S4), `resultats` (S6), `decision` (S7),
    `resultat_test` et `test_optimiste` (S8), `export` (S9). `tests_vus` et
    `historique` traversent tous les états, nouveau projet compris.

    Limite : une ancienne session gardée par référence n'est pas protégée (elle
    ignore les ouvertures faites depuis) ; l'interface ne garde que la courante.
    """

    etat: Etat = Etat.VIDE
    table: Table | None = None
    constats: tuple[Constat, ...] = ()
    acquittes: frozenset[str] = frozenset()
    cle_audit: tuple[Any, ...] | None = None
    spec: SpecProjet | None = None
    coffre: CoffreTest | None = None
    resultats: Any = None
    decision: Decision | None = None
    resultat_test: Any = None
    test_optimiste: bool = False
    export: Any = None
    tests_vus: frozenset[str] = frozenset()
    historique: tuple[tuple[str, Etat], ...] = ()
    auditeur: Auditeur = field(default=auditeur_neutre, repr=False)

    # Lecture
    @property
    def test_vu(self) -> bool:
        """Vrai si le coffre de test courant a déjà été ouvert."""
        return self.coffre is not None and self.coffre.ouvertures > 0

    @property
    def score_test_optimiste(self) -> bool:
        """Vrai si le résultat de test courant vient d'un test déjà vu."""
        return self.test_optimiste

    @property
    def sans_cible(self) -> bool:
        """Vrai si l'objectif fixé est sans cible."""
        return self.spec is not None and not self.spec.probleme.avec_cible

    def bloquants(self) -> tuple[Constat, ...]:
        """Constats bloquants courants."""
        return tuple(c for c in self.constats if c.niveau is Niveau.BLOQUANT)

    def alertes_en_attente(self) -> tuple[Constat, ...]:
        """Alertes non encore acquittées."""
        return tuple(c for c in self.constats if c.niveau is Niveau.ALERTE and c.code not in self.acquittes)

    def peut(self, action: Action) -> bool:
        """Vrai si l'état courant admet l'action (règles d'état seules, contenu non examiné)."""
        try:
            _gestionnaire(action)
            self._admettre(action)
        except (TransitionRefusee, TypeError):
            return False
        return True

    # Application
    def appliquer(self, action: Action) -> Session:
        """Applique une action ; `TransitionRefusee` si elle est interdite, `TypeError` si inconnue."""
        gestionnaire = _gestionnaire(action)
        self._admettre(action)
        suivante = gestionnaire(self, action)
        suivante = replace(suivante, historique=(*self.historique, (type(action).__name__, suivante.etat)))
        journal.info("%s : S%d vers S%d", type(action).__name__, self.etat, suivante.etat)
        return suivante

    def _refus(self, action: Action, motif: str) -> TransitionRefusee:
        return TransitionRefusee(self.etat, action, motif)

    def _exiger(self, valeur: _T | None, action: Action, quoi: str) -> _T:
        """Valeur présente, sinon refus explicite (jamais d'`assert`)."""
        if valeur is None:
            raise self._refus(action, f"{quoi} absent dans cet état.")
        return valeur

    def _admettre(self, action: Action) -> None:
        """Règles d'état communes ; les gestionnaires vérifient ensuite le contenu."""
        if isinstance(action, Revenir):
            self._admettre_retour(action)
        elif isinstance(action, Acquitter):
            if self.etat not in (Etat.AUDIT, Etat.OBJECTIF, Etat.DECOUPE):
                raise self._refus(action, "les alertes s'acquittent en S2, S3 ou S4.")
        else:
            self._admettre_avance(action)

    def _admettre_retour(self, action: Revenir) -> None:
        cible = action.etat
        if not isinstance(cible, Etat):
            raise self._refus(action, "l'état visé doit être un `Etat`.")
        if cible >= self.etat:
            raise self._refus(action, "on ne revient que vers un état antérieur.")
        if self.test_vu and cible is not Etat.VIDE and cible < Etat.PRETRAITEMENT:
            raise self._refus(
                action,
                "le test a été ouvert : le découpage ne se refait pas au vu du test. Revenir en S5, "
                "S6 ou S7 ; un nouveau projet (S0) sur les mêmes données gardera le test pour déjà vu.",
            )

    def _admettre_avance(self, action: Action) -> None:
        k = type(action).produit
        if k is None:
            raise TypeError(f"Action inconnue : {type(action).__name__}.")
        e = self.etat
        sans_cible = self.sans_cible
        if isinstance(action, Tester) and sans_cible:
            raise self._refus(action, "un projet sans cible n'a pas de jeu de test ; exporter depuis S7.")
        if isinstance(action, Exporter) and e is Etat.DIAGNOSTIC and sans_cible:
            return
        if isinstance(action, Exporter) and e is Etat.DIAGNOSTIC:
            raise self._refus(action, "évaluer d'abord le modèle choisi sur le jeu de test (S8).")
        if k == e + 1:
            return
        if k > e + 1:
            raise self._refus(action, f"étape sautée : terminer d'abord S{int(e) + 1}.")
        if not self.test_vu:
            return
        if k in _REPRISES_APRES_TEST or (k is Etat.EXPORTE and e is Etat.EXPORTE):
            return
        if k is Etat.TESTE:
            raise self._refus(action, "le test est déjà ouvert ; modifier S5, S6 ou S7 avant de le rouvrir.")
        raise self._refus(
            action,
            "le test a été ouvert : données, objectif et découpage sont figés. Un nouveau projet "
            "(Revenir en S0) permet de les changer ; le test restera compté pour déjà vu.",
        )

    # Outils internes
    def _tronquer(self, etat: Etat) -> Session:
        """Session ramenée à `etat` : tout ce qui a été produit après est invalidé."""
        s = replace(self, etat=etat, spec=_purger_spec(self.spec, etat))
        if etat < Etat.DONNEES:
            s = replace(s, table=None)
        if etat < Etat.AUDIT:
            s = replace(s, constats=(), acquittes=frozenset(), cle_audit=None)
        if etat < Etat.OBJECTIF:
            s = replace(s, spec=None)
        if etat < Etat.DECOUPE:
            s = replace(s, coffre=None)
        if etat < Etat.MODELES:
            s = replace(s, resultats=None)
        if etat < Etat.DIAGNOSTIC:
            s = replace(s, decision=None)
        if etat < Etat.TESTE:
            s = replace(s, resultat_test=None, test_optimiste=False)
        if etat < Etat.EXPORTE:
            s = replace(s, export=None)
        return s

    def _auditer(self, spec: SpecProjet | None, *, forcer: bool = False) -> Session:
        """Rejoue l'audit si sa clé change (ou si `forcer`) ; garde les acquittements valables."""
        if self.table is None:
            raise RuntimeError("Audit impossible : aucune table chargée.")
        cle = (None if spec is None else spec.cle_audit(), self.coffre is not None)
        if not forcer and cle == self.cle_audit:
            return self
        constats = tuple(self.auditeur(self.table, spec, self.coffre))
        if any(not isinstance(c, Constat) for c in constats):
            raise TypeError("L'auditeur doit rendre des `Constat`.")
        codes = [c.code for c in constats]
        doubles = sorted({c for c in codes if codes.count(c) > 1})
        if doubles:
            raise ValueError(f"L'auditeur a rendu des codes de constat en double : {doubles[:5]}.")
        alertes = {c.code for c in constats if c.niveau is Niveau.ALERTE}
        journal.info("audit rejoué : %d constats", len(constats))
        return replace(self, constats=constats, acquittes=self.acquittes & alertes, cle_audit=cle)

    def _garde_constats(self, action: Action) -> None:
        if self.bloquants():
            premiers = self.bloquants()
            codes = ", ".join(c.code for c in premiers[:3])
            geste = f" Pour le premier : {premiers[0].action}" if premiers[0].action else ""
            raise self._refus(action, f"constats bloquants à lever d'abord ({codes}) : voir l'audit.{geste}")
        if self.alertes_en_attente():
            codes = ", ".join(c.code for c in self.alertes_en_attente()[:3])
            raise self._refus(action, f"alertes à acquitter d'abord ({codes}).")

    def _verifier_spec(self, action: Action, spec: object) -> SpecProjet:
        if not isinstance(spec, SpecProjet):
            raise self._refus(action, "une `SpecProjet` est attendue.")
        table = self._exiger(self.table, action, "table")
        absentes = sorted(spec.colonnes_citees() - set(table.colonnes))
        if absentes:
            raise self._refus(action, f"colonnes absentes de la table : {absentes[:5]}.")
        return spec


def _gestionnaire(action: object) -> Callable[[Session, Any], Session]:
    """Gestionnaire de l'action ; `TypeError` pour un objet qui n'est pas une action connue."""
    gestionnaire = _GESTIONNAIRES.get(type(action))
    if gestionnaire is None:
        raise TypeError(
            f"Action inconnue : {type(action).__name__} ; actions permises : "
            + ", ".join(a.__name__ for a in ACTIONS)
            + "."
        )
    return gestionnaire


# ============================================================
# GESTIONNAIRES D'ACTIONS
# ============================================================
def _nouvelle_session(s: Session) -> Session:
    """Session vide qui garde l'auditeur, l'historique et les tests déjà vus."""
    return Session(auditeur=s.auditeur, historique=s.historique, tests_vus=s.tests_vus)


def _charger(s: Session, a: Charger) -> Session:
    if not isinstance(a.table, Table):
        raise s._refus(a, "une `Table` est attendue.")
    return replace(_nouvelle_session(s), etat=Etat.DONNEES, table=a.table)


def _auditer(s: Session, a: Auditer) -> Session:
    return replace(s._tronquer(Etat.DONNEES)._auditer(None), etat=Etat.AUDIT)


def _acquitter(s: Session, a: Acquitter) -> Session:
    codes = frozenset(a.codes)
    alertes = {c.code for c in s.constats if c.niveau is Niveau.ALERTE}
    inconnus = sorted(codes - alertes)
    if inconnus:
        raise s._refus(a, f"codes sans alerte correspondante : {inconnus[:5]}.")
    return replace(s, acquittes=s.acquittes | codes)


def _classes_de_la_cible(table: Table, cible: str) -> list[Any]:
    """Valeurs distinctes non manquantes de la cible, en objets Python."""
    return pd.unique(table.donnees[cible].dropna()).tolist()


def _verifier_classe_positive(s: Session, a: FixerObjectif, spec: SpecProjet) -> None:
    """Classe positive confirmée en S3 : exigée en binaire si la métrique en dépend, prise dans la cible."""
    if spec.probleme is not Probleme.CLASSIFICATION or spec.cible is None:
        return
    classes = _classes_de_la_cible(s._exiger(s.table, a, "table"), spec.cible)
    apercu = sorted(map(str, classes))[:10]
    if spec.classe_positive is not None and spec.classe_positive not in classes:
        raise s._refus(
            a, f"la classe positive {spec.classe_positive!r} n'est pas une valeur de la cible : {apercu}."
        )
    if spec.classe_positive is not None and len(classes) > 2:
        raise s._refus(
            a,
            f"la cible compte {len(classes)} classes : au delà de deux classes, précision, rappel et F1 sont "
            "des moyennes macro, sans classe positive ; regrouper les classes pour en désigner une, ou ne "
            "pas fixer de classe positive.",
        )
    if spec.classe_positive is None and spec.exige_classe_positive() and len(classes) == 2:
        raise s._refus(
            a,
            f"classification à deux classes évaluée par « {spec.metrique} » : désigner la classe positive "
            f"(la classe que l'on cherche à détecter, défaut ou panne par exemple) parmi {apercu} ; "
            "elle n'est jamais déduite.",
        )


def _fixer_objectif(s: Session, a: FixerObjectif) -> Session:
    spec = _purger_spec(s._verifier_spec(a, a.spec), Etat.OBJECTIF)
    _verifier_classe_positive(s, a, spec)
    base = replace(s._tronquer(Etat.AUDIT), spec=spec)
    return replace(base._auditer(spec), etat=Etat.OBJECTIF)


def _decouper(s: Session, a: Decouper) -> Session:
    courante = s._exiger(s.spec, a, "objectif")
    table = s._exiger(s.table, a, "table")
    spec = courante if a.spec is None else s._verifier_spec(a, a.spec)
    if spec.objectif() != courante.objectif():
        raise s._refus(a, "l'objectif se change en S3, pas au découpage.")
    spec = _purger_spec(spec, Etat.DECOUPE)
    if spec.probleme.avec_cible:
        if not isinstance(a.coffre, CoffreTest):
            raise s._refus(a, "un problème avec cible exige un coffre de test.")
        if not a.coffre.repose_sur(table):
            raise s._refus(a, "le coffre n'a pas été construit sur la table chargée.")
        if a.coffre.decoupage != spec.decoupage():
            raise s._refus(
                a, "le coffre a été construit pour un autre découpage que celui de la spécification."
            )
        if a.coffre.ouvertures:
            raise s._refus(a, "le coffre fourni a déjà été ouvert.")
    elif a.coffre is not None:
        raise s._refus(a, "un projet sans cible ne met pas de test sous scellé.")
    base = replace(s._tronquer(Etat.OBJECTIF), spec=spec, coffre=a.coffre)
    return replace(base._auditer(spec, forcer=True), etat=Etat.DECOUPE)


def _pretraiter(s: Session, a: Pretraiter) -> Session:
    courante = s._exiger(s.spec, a, "objectif")
    spec = s._verifier_spec(a, a.spec)
    if spec.objectif() != courante.objectif() or spec.decoupage() != courante.decoupage():
        raise s._refus(a, "seul le prétraitement change ici ; objectif et découpage se changent en S3 et S4.")
    base = s._tronquer(Etat.DECOUPE)._auditer(spec, forcer=True)  # l'audit lit le prétraitement choisi
    base._garde_constats(a)
    return replace(base, spec=_purger_spec(spec, Etat.PRETRAITEMENT), etat=Etat.PRETRAITEMENT)


def _entrainer(s: Session, a: Entrainer) -> Session:
    courante = s._exiger(s.spec, a, "objectif")
    spec = courante if a.spec is None else s._verifier_spec(a, a.spec)
    if (spec.objectif(), spec.decoupage(), spec.pretraitement) != (
        courante.objectif(),
        courante.decoupage(),
        courante.pretraitement,
    ):
        raise s._refus(a, "seuls les modèles changent ici ; le prétraitement se change en S5.")
    if not spec.modeles:
        raise s._refus(a, "aucun modèle comparé : renseigner `spec.modeles`.")
    if spec.probleme.avec_cible and not any(c.id in REFERENCES_NAIVES.values() for c in spec.modeles):
        raise s._refus(
            a,
            "ajouter une référence naïve (la moyenne ou la classe majoritaire, `reference.dummy_*`) aux "
            "modèles comparés : tout score se lit par rapport à elle, et la persistance n'en tient pas lieu.",
        )
    if a.resultats is None:
        raise s._refus(a, "résultats de validation croisée absents.")
    base = s._tronquer(Etat.PRETRAITEMENT)._auditer(spec, forcer=True)  # l'audit lit aussi les modèles
    base._garde_constats(a)
    return replace(base, spec=spec, resultats=a.resultats, etat=Etat.MODELES)


def _decider(s: Session, a: Decider) -> Session:
    spec = s._exiger(s.spec, a, "objectif")
    if not isinstance(a.decision, Decision):
        raise s._refus(a, "une `Decision` est attendue.")
    if spec.probleme.avec_cible and a.decision.metrique != spec.metrique:
        raise s._refus(
            a,
            f"la décision doit porter sur la métrique fixée en S3 ({spec.metrique}), "
            f"pas sur {a.decision.metrique}.",
        )
    ids_compares = [c.id for c in spec.modeles]
    if a.decision.modele_id not in ids_compares:
        raise s._refus(
            a,
            f"le modèle choisi ({a.decision.modele_id}) n'est pas parmi les modèles comparés en S6 : "
            f"{ids_compares}.",
        )
    base = s._tronquer(Etat.MODELES)
    base._garde_constats(a)
    return replace(base, decision=a.decision, etat=Etat.DIAGNOSTIC)


def _tester(s: Session, a: Tester) -> Session:
    coffre = s._exiger(s.coffre, a, "coffre de test")
    decision = s._exiger(s.decision, a, "décision")
    if not callable(a.evaluer):
        raise s._refus(a, "`evaluer` doit être une fonction des lignes de test.")
    s._garde_constats(a)
    cle = coffre.empreinte_test
    deja_vu = cle in s.tests_vus or coffre.ouvertures > 0
    lignes = coffre.ouvrir(decision)
    resultat = a.evaluer(lignes)
    return replace(
        s,
        resultat_test=resultat,
        test_optimiste=deja_vu,
        tests_vus=s.tests_vus | {cle},
        etat=Etat.TESTE,
    )


def _exporter(s: Session, a: Exporter) -> Session:
    if a.export is None:
        raise s._refus(a, "rien à exporter.")
    s._garde_constats(a)
    return replace(s, export=a.export, etat=Etat.EXPORTE)


def _revenir(s: Session, a: Revenir) -> Session:
    if a.etat is Etat.VIDE:
        return _nouvelle_session(s)
    base = s._tronquer(a.etat)
    if a.etat is Etat.AUDIT:
        base = base._auditer(None)  # profil seul, sans objectif
    elif a.etat is Etat.OBJECTIF:
        base = base._auditer(base.spec)  # objectif sans coffre
    return base


_GESTIONNAIRES: dict[type[Action], Callable[[Session, Any], Session]] = {
    Charger: _charger,
    Auditer: _auditer,
    Acquitter: _acquitter,
    FixerObjectif: _fixer_objectif,
    Decouper: _decouper,
    Pretraiter: _pretraiter,
    Entrainer: _entrainer,
    Decider: _decider,
    Tester: _tester,
    Exporter: _exporter,
    Revenir: _revenir,
}
