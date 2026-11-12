"""
MécaML : découpage apprentissage et test, coffre scellé, plis de validation.

Le découpage se déduit de la `Structure` des lignes :

- indépendante : tirage stratifié sur la cible en classification, aléatoire sinon ;
- groupée, et groupée chronologique : unités entières d'un seul côté, aucune unité
  à cheval, classes équilibrées entre les côtés en classification (le test réunit
  m parts d'unités sur k, pour une part des lignes à 5 points au plus de la part
  visée) ; dans chaque unité, l'ordre du temps est ensuite suivi par les variables
  construites (`features_ts`) ;
- chronologique : coupure entre deux dates distinctes, toutes les lignes de date
  supérieure ou égale à la coupure en test, aucune date à cheval.

Le découpage est déclaré (`Decoupage`) : méthode, part de test, graine, colonne de
groupe ou de date, nombre de lignes de test et empreinte SHA-256 de leurs
positions, sous la forme du champ `base_test` du dépôt d'exercices. Les quatre
méthodes de ce contrat y suivent l'algorithme du dépôt, sur les positions comptées
à partir de 0 dans la table lue sans filtrage : relire un `base_test` recalcule les
positions et vérifie l'empreinte (mode challenge).

Les plis de validation croisée suivent la même structure, ou la validation imposée
par la spécification si elle lui convient. Trois contrôles, destinés à l'audit rejoué
en S4, signalent une unité ou une date à cheval (I4), une classe absente du test ou
d'un pli (I7) et, par un constat d'information, la part de test obtenue quand elle
s'écarte de la part visée (`controler_part_test`). Seules les colonnes qui définissent
le découpage (cible, groupe, date) sont lues sur toutes les lignes ; le reste du test
ne sort du coffre qu'à son ouverture.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from mecaml.core import renvois
from mecaml.core.guards import decimale, technique_coupee
from mecaml.core.types import (
    GRAINE_CONTRAT,
    CoffreTest,
    Constat,
    Niveau,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
    validations_permises,
)

__all__ = [
    "CHAMPS_BASE_TEST",
    "GRAINE_CONTRAT",
    "METHODES_CONTRAT",
    "Decoupage",
    "ErreurDecoupage",
    "Methode",
    "classes_absentes",
    "controler_classes",
    "controler_decoupage",
    "controler_part_test",
    "controler_structure",
    "empreinte_positions",
    "methode_deduite",
    "methodes_permises",
    "part_obtenue",
    "plis_au_plus",
    "plis_validation",
    "positions_test",
    "sceller",
    "sceller_base_test",
    "sceller_fichier",
    "separer_sans_cible",
    "spec_pour_base_test",
    "validation_appliquee",
    "validation_deduite",
]

CHAMPS_BASE_TEST = (
    "methode",
    "test_size",
    "random_state",
    "colonne_groupe",
    "lignes_test",
    "empreinte_sha256",
)
_RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXEMPLES = 3  # valeurs citées dans un message

# Unités stratifiées : le test réunit m parts d'unités sur k, avec k au plus PARTS_MAX et une part de lignes
# à ECART_PART_MAX au plus de la part visée ; l'audit de S4 dit la part obtenue dès qu'elle s'en écarte de
# ECART_PART_INFO.
PARTS_MAX = 10
ECART_PART_MAX = 0.05
ECART_PART_INFO = 0.01

# Renvois au livre, lus dans la table unique (`mecaml.core.renvois`).
REF_MORCELLEMENT = renvois.theme("decoupage.morcellement")
REF_VALIDATION_CROISEE = renvois.theme("decoupage.plis")
REF_VALIDATION_GLISSANTE = renvois.theme("decoupage.dates")


# ============================================================
# MÉTHODES ET DÉCOUPAGE DÉCLARÉ
# ============================================================
class Methode(StrEnum):
    """Méthode de découpage ; `METHODES_CONTRAT` sont celles du champ `base_test`."""

    ALEATOIRE = "aleatoire"
    STRATIFIE = "stratifie"
    UNITES = "unites"
    CHRONOLOGIQUE = "chronologique"
    UNITES_STRATIFIEES = "unites_stratifiees"  # unités entières, classes équilibrées
    FICHIER = "fichier"  # test fourni à part (feuille ou fichier)


METHODES_CONTRAT = frozenset({Methode.ALEATOIRE, Methode.STRATIFIE, Methode.UNITES, Methode.CHRONOLOGIQUE})
_METHODES_A_COLONNE = frozenset({Methode.UNITES, Methode.UNITES_STRATIFIEES, Methode.CHRONOLOGIQUE})


class ErreurDecoupage(ValueError):
    """Découpage impossible ou non conforme ; le message dit quoi faire."""


def empreinte_positions(positions: Sequence[int] | np.ndarray) -> str:
    """SHA-256 des positions croissantes, écrites en décimal, séparées par des virgules (ASCII)."""
    valeurs = sorted(int(p) for p in np.asarray(positions).ravel())
    return hashlib.sha256(",".join(map(str, valeurs)).encode("ascii")).hexdigest()


def _est_entier(valeur: object) -> bool:
    return isinstance(valeur, (int, np.integer)) and not isinstance(valeur, (bool, np.bool_))


@dataclass(frozen=True)
class Decoupage:
    """Découpage déclaré : de quoi recalculer la base de test et la reconnaître.

    `colonne_groupe` porte la colonne des unités (`unites`, `unites_stratifiees`), de
    date (`chronologique`) ou de strate si elle n'est pas la cible (`stratifie`). En
    mode `fichier`, l'empreinte est celle du contenu du fichier de test.
    """

    methode: Methode
    test_size: float
    random_state: int
    colonne_groupe: str | None
    lignes_test: int
    empreinte_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "methode", _methode(self.methode))
        ts = self.test_size
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not 0.0 < float(ts) < 1.0:
            raise ErreurDecoupage(f"Découpage : `test_size` = {ts!r} hors de l'intervalle ouvert (0, 1).")
        object.__setattr__(self, "test_size", float(ts))
        if not _est_entier(self.random_state) or self.random_state < 0:
            raise ErreurDecoupage("Découpage : `random_state` doit être un entier positif ou nul.")
        object.__setattr__(self, "random_state", int(self.random_state))
        col = self.colonne_groupe
        if col is not None and (not isinstance(col, str) or not col.strip()):
            raise ErreurDecoupage("Découpage : `colonne_groupe` doit être un nom de colonne ou null.")
        if col is None and self.methode in _METHODES_A_COLONNE:
            raise ErreurDecoupage(f"Découpage : la méthode {self.methode} exige `colonne_groupe`.")
        if not _est_entier(self.lignes_test) or self.lignes_test <= 0:
            raise ErreurDecoupage("Découpage : `lignes_test` doit être un entier positif.")
        object.__setattr__(self, "lignes_test", int(self.lignes_test))
        if not isinstance(self.empreinte_sha256, str) or not _RE_SHA256.match(self.empreinte_sha256):
            raise ErreurDecoupage(
                "Découpage : `empreinte_sha256` attend 64 caractères hexadécimaux minuscules."
            )

    @property
    def conforme_contrat(self) -> bool:
        """Vrai si le découpage s'écrit en `base_test` du dépôt (méthode du contrat, graine 42)."""
        return self.methode in METHODES_CONTRAT and self.random_state == GRAINE_CONTRAT

    def vers_base_test(self) -> dict[str, Any]:
        """Champ `base_test` du dépôt d'exercices, champs dans l'ordre du contrat."""
        if self.methode is Methode.FICHIER:
            raise ErreurDecoupage(
                "Découpage : un test fourni à part ne s'écrit pas en `base_test` ; déclarer un découpage "
                "de la table (aléatoire, stratifié, par unités ou chronologique)."
            )
        if self.methode not in METHODES_CONTRAT:
            raise ErreurDecoupage(
                "Découpage : les unités stratifiées ne s'écrivent pas en `base_test` ; pour un défi, "
                "découper par unités (méthode « unites » du dépôt)."
            )
        valeurs = (self.methode.value, self.test_size, self.random_state, self.colonne_groupe)
        return dict(zip(CHAMPS_BASE_TEST, (*valeurs, self.lignes_test, self.empreinte_sha256), strict=True))

    @classmethod
    def depuis_base_test(cls, base_test: Mapping[str, Any]) -> Decoupage:
        """Lit un champ `base_test` : champs exacts, méthode du contrat, valeurs vérifiées."""
        if not isinstance(base_test, Mapping):
            raise ErreurDecoupage("Base de test : un objet (dictionnaire) est attendu.")
        manquants = [c for c in CHAMPS_BASE_TEST if c not in base_test]
        inconnus = sorted(set(base_test) - set(CHAMPS_BASE_TEST))
        if manquants or inconnus:
            raise ErreurDecoupage(f"Base de test : champs manquants {manquants}, champs inconnus {inconnus}.")
        if base_test["methode"] not in {m.value for m in METHODES_CONTRAT}:
            raise ErreurDecoupage(
                f"Base de test : la méthode {base_test['methode']!r} n'appartient pas au contrat ; "
                f"permises : {', '.join(sorted(m.value for m in METHODES_CONTRAT))}."
            )
        return cls(**{c: base_test[c] for c in CHAMPS_BASE_TEST})


def _methode(valeur: object) -> Methode:
    try:
        return Methode(valeur)
    except ValueError:
        permises = ", ".join(m.value for m in Methode)
        raise ErreurDecoupage(f"Méthode de découpage inconnue {valeur!r} ; permises : {permises}.") from None


# ============================================================
# POSITIONS DE TEST
# ============================================================
def positions_test(
    donnees: pd.DataFrame,
    methode: Methode,
    test_size: float,
    graine: int,
    colonne: str | None,
    *,
    strate: str | None = None,
) -> np.ndarray:
    """Positions de test, croissantes et comptées à partir de 0.

    `colonne` : strate (`stratifie`), unités (`unites`, `unites_stratifiees`) ou date
    (`chronologique`) ; ignorée pour `aleatoire`. `strate` : classes à équilibrer
    entre unités (`unites_stratifiees`). Les méthodes du contrat suivent
    l'algorithme du dépôt d'exercices.
    """
    methode = _methode(methode)
    n = len(donnees)
    if n < 2:
        raise ErreurDecoupage("Découpage impossible : la table compte moins de deux lignes.")
    positions = np.arange(n)
    if methode is Methode.ALEATOIRE:
        test = _tirage(positions, test_size, graine, None)
    elif methode is Methode.STRATIFIE:
        test = _tirage(positions, test_size, graine, _strates(donnees, colonne))
    elif methode is Methode.UNITES:
        test = _par_unites(donnees, positions, test_size, graine, colonne)
    elif methode is Methode.UNITES_STRATIFIEES:
        test = _par_unites_stratifiees(donnees, positions, test_size, graine, colonne, strate)
    elif methode is Methode.CHRONOLOGIQUE:
        test = _par_date(donnees, positions, test_size, colonne)
    else:
        raise ErreurDecoupage("La méthode « fichier » ne découpe pas une table : fournir le jeu de test.")
    test = np.sort(np.asarray(test, dtype=np.int64))
    if test.size == 0 or test.size >= n:
        raise ErreurDecoupage(
            "Découpage dégénéré : le test doit compter au moins une ligne, sans les prendre toutes."
        )
    return test


def _colonne_presente(
    donnees: pd.DataFrame, colonne: str | None, quoi: str, *, complete: bool = True
) -> pd.Series:
    """Colonne qui définit le découpage ; `complete` : aucune valeur manquante tolérée."""
    if colonne is None or colonne not in donnees.columns:
        raise ErreurDecoupage(f"Colonne {quoi} absente de la table : {colonne!r}.")
    serie = donnees[colonne]
    manquantes = int(serie.isna().sum())
    if manquantes and complete:
        raise ErreurDecoupage(
            f"Colonne {quoi} « {colonne} » incomplète ({manquantes} valeurs manquantes) : compléter ces "
            "lignes ou les retirer du fichier avant de découper."
        )
    return serie


def _codes(serie: pd.Series, *, manque_a_part: bool) -> np.ndarray:
    """Codes des classes dans l'ordre trié ; une cible manquante forme la dernière strate.

    Une strate manquante d'une seule ligne, ou `manque_a_part` faux, rejoint la classe
    la plus fréquente : elle ne peut pas être répartie et ne doit pas faire échouer le tirage.
    """
    codes, _ = pd.factorize(serie, sort=True)
    manque = codes < 0
    if not manque.any():
        return codes
    if manque_a_part and manque.sum() >= 2:
        codes[manque] = codes.max() + 1
    else:
        codes[manque] = np.bincount(codes[~manque]).argmax()
    return codes


def _strates(donnees: pd.DataFrame, colonne: str | None) -> pd.Series | np.ndarray:
    """Strates du tirage : la colonne telle quelle (algorithme du dépôt), ou ses codes si manques."""
    serie = _colonne_presente(donnees, colonne, "de strate", complete=False)
    return serie if not serie.isna().any() else _codes(serie, manque_a_part=True)


def _tirage(positions: np.ndarray, test_size: float, graine: int, strates: Any) -> np.ndarray:
    from sklearn.model_selection import train_test_split

    try:
        _, test = train_test_split(positions, test_size=test_size, random_state=graine, stratify=strates)
    except ValueError as erreur:
        if strates is None:
            raise ErreurDecoupage(
                "Tirage impossible : la part de test ne laisse aucune ligne d'un côté ; ajuster la part "
                "de test."
            ) from erreur
        raise ErreurDecoupage(
            "Stratification impossible : une classe compte moins de deux lignes, ou le test compte moins "
            "de lignes que de classes. Regrouper les classes rares, augmenter la part de test, ou choisir "
            "un découpage aléatoire."
        ) from erreur
    return test


def _par_unites(
    donnees: pd.DataFrame, positions: np.ndarray, test_size: float, graine: int, colonne: str | None
) -> np.ndarray:
    from sklearn.model_selection import GroupShuffleSplit

    unites = _colonne_presente(donnees, colonne, "des unités")
    if unites.nunique() < 2:
        raise ErreurDecoupage(f"Une seule unité dans « {colonne} » : rien ne reste pour l'apprentissage.")
    tirage = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=graine)
    try:
        _, test = next(tirage.split(positions, groups=unites))
    except ValueError as erreur:
        raise ErreurDecoupage(
            f"Tirage des unités impossible : {unites.nunique()} unités ne se partagent pas selon une part de "
            f"test de {test_size} ; ajuster la part de test au nombre d'unités."
        ) from erreur
    return test


def _exiger_effectifs(classes: np.ndarray, k: int, quoi: str) -> None:
    effectifs = np.bincount(classes)
    effectifs = effectifs[effectifs > 0]
    if effectifs.min() < k:
        raise ErreurDecoupage(
            f"La classe la plus rare compte {effectifs.min()} lignes pour {k} {quoi} : regrouper les "
            "classes rares ou réduire le nombre de parts."
        )


def _exiger_unites(n_unites: int, k: int, test_size: float) -> None:
    if n_unites < k:
        raise ErreurDecoupage(
            f"{n_unites} unités pour une part de test de {test_size} : il en faut au moins {k} ; "
            "augmenter la part de test, ou découper par unités sans équilibrer les classes."
        )


def _fractions(part_visee: float, ecart_max: float) -> list[tuple[int, int]]:
    """Réunions de m parts sur k (2 <= k <= PARTS_MAX) dont la part m/k est à `ecart_max` de la part visée.

    Triées de la plus proche à la plus éloignée, puis par k croissant (peu de parts : peu d'unités exigées).
    """
    proches = [
        (m, k)
        for k in range(2, PARTS_MAX + 1)
        for m in range(1, k)
        if abs(m / k - part_visee) <= ecart_max + 1e-9
    ]
    return sorted(proches, key=lambda mk: (round(abs(mk[0] / mk[1] - part_visee), 9), mk[1]))


def _part_impossible(part_visee: float) -> ErreurDecoupage:
    """Refus actionnable : les parts de test possibles les plus proches de la part visée."""
    possibles: dict[float, tuple[int, int]] = {}
    for k in range(2, PARTS_MAX + 1):
        for m in range(1, k):
            possibles.setdefault(round(m / k, 9), (m, k))
    proches = sorted(possibles, key=lambda p: abs(p - part_visee))[:4]
    liste = ", ".join(
        f"{decimale(p, 2)} ({possibles[p][0]} part sur {possibles[p][1]})" for p in sorted(proches)
    )
    return ErreurDecoupage(
        f"Part de test de {decimale(part_visee, 2)} impossible par unités stratifiées : les unités entières "
        f"se répartissent en parts de m sur k, avec k au plus {PARTS_MAX}, et aucune n'est à moins de "
        f"{round(100 * ECART_PART_MAX)} points de la part visée. Parts les plus proches : {liste}. Choisir "
        "l'une d'elles, ou découper par unités sans équilibrer les classes."
    )


def _choisir_parts(part_visee: float, unites: pd.Series, classes: np.ndarray) -> tuple[int, int]:
    """(m, k) : le test réunit m parts d'unités sur k, avec m/k à 5 points au plus de la part visée.

    La plus proche des fractions que les données permettent (assez d'unités, assez de lignes de la classe la
    plus rare pour k parts) ; sinon refus, qui dit pourquoi la meilleure fraction ne convient pas.
    """
    candidats = _fractions(part_visee, ECART_PART_MAX)
    if not candidats:
        raise _part_impossible(part_visee)
    refus: list[ErreurDecoupage] = []
    for m, k in candidats:
        try:
            _exiger_unites(unites.nunique(), k, part_visee)
            _exiger_effectifs(classes, k, "parts du tirage")
        except ErreurDecoupage as erreur:
            refus.append(erreur)
            continue
        return m, k
    raise refus[0]


def _par_unites_stratifiees(
    donnees: pd.DataFrame,
    positions: np.ndarray,
    test_size: float,
    graine: int,
    colonne: str | None,
    strate: str | None,
) -> np.ndarray:
    """Unités entières, classes équilibrées entre k parts d'unités ; le test en réunit m (m/k ≈ part visée).

    Les unités se partagent en k parts de classes équilibrées ; le test réunit m de ces parts, de sorte que sa
    part des lignes soit à 5 points au plus de la part visée (`_choisir_parts`). Parmi les réunions de m
    parts, le test est celle qui contient le plus de classes, puis dont la taille est la plus proche de la
    part visée (la première en cas d'égalité : tirage reproductible).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    unites = _colonne_presente(donnees, colonne, "des unités")
    classes = _codes(_colonne_presente(donnees, strate, "de strate", complete=False), manque_a_part=False)
    m, k = _choisir_parts(test_size, unites, classes)
    tirage = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=graine)
    parts = [val for _, val in tirage.split(positions, classes, groups=unites)]
    reunions = (np.concatenate(combinaison) for combinaison in itertools.combinations(parts, m))
    return max(reunions, key=lambda val: _qualite_de_part(val, classes, test_size))


def _qualite_de_part(val: np.ndarray, classes: np.ndarray, test_size: float) -> tuple[int, float]:
    """Test retenu : le plus de classes présentes, puis la taille la plus proche de la visée."""
    presentes = np.count_nonzero(np.bincount(classes[val], minlength=int(classes.max()) + 1))
    return int(presentes), -abs(val.size - test_size * classes.size)


def _par_date(
    donnees: pd.DataFrame, positions: np.ndarray, test_size: float, colonne: str | None
) -> np.ndarray:
    """Coupure à la date de rang n moins ceil(part × n) : toute date au moins égale va en test."""
    valeurs = _colonne_presente(donnees, colonne, "de date").to_numpy()
    n = len(valeurs)
    coupure = np.sort(valeurs, kind="stable")[n - math.ceil(test_size * n)]
    test = positions[valeurs >= coupure]
    if test.size == n:
        raise ErreurDecoupage(
            f"Coupure impossible dans « {colonne} » : la date de coupure est la première ; réduire la part "
            "de test ou vérifier la colonne de date."
        )
    return test


# ============================================================
# MÉTHODE DÉDUITE DE LA STRUCTURE
# ============================================================
def methodes_permises(spec: SpecProjet) -> frozenset[Methode]:
    """Méthodes compatibles avec la structure (et le problème, pour la stratification)."""
    classification = spec.probleme is Probleme.CLASSIFICATION
    if spec.structure.groupee:
        return frozenset({Methode.UNITES_STRATIFIEES, Methode.UNITES} if classification else {Methode.UNITES})
    if spec.structure.chronologique:
        return frozenset({Methode.CHRONOLOGIQUE})
    return frozenset({Methode.STRATIFIE, Methode.ALEATOIRE} if classification else {Methode.ALEATOIRE})


def methode_deduite(spec: SpecProjet) -> Methode:
    """Méthode par défaut : unités (stratifiées en classification), date, strates, ou hasard."""
    classification = spec.probleme is Probleme.CLASSIFICATION
    if spec.structure.groupee:
        return Methode.UNITES_STRATIFIEES if classification else Methode.UNITES
    if spec.structure.chronologique:
        return Methode.CHRONOLOGIQUE
    return Methode.STRATIFIE if classification else Methode.ALEATOIRE


def _colonne_de(spec: SpecProjet, methode: Methode) -> str | None:
    """Colonne déclarée avec la méthode (unités ou date) ; la strate par défaut est la cible."""
    if methode in (Methode.UNITES, Methode.UNITES_STRATIFIEES):
        return spec.colonne_groupe
    if methode is Methode.CHRONOLOGIQUE:
        return spec.colonne_date
    return None


def _exiger_cible(spec: SpecProjet) -> None:
    if not isinstance(spec, SpecProjet):
        raise TypeError("Découpage : une `SpecProjet` est attendue.")
    if not spec.probleme.avec_cible:
        raise ErreurDecoupage("Un projet sans cible ne met pas de test sous scellé.")


def _exiger_colonnes(table: Table, spec: SpecProjet) -> None:
    if not isinstance(table, Table):
        raise TypeError("Découpage : une `Table` est attendue.")
    absentes = sorted(spec.colonnes_citees() - set(table.colonnes))
    if absentes:
        raise ErreurDecoupage(f"Colonnes de la spécification absentes de la table : {absentes[:5]}.")


# ============================================================
# SCELLÉ
# ============================================================
def sceller(
    table: Table, spec: SpecProjet, *, methode: Methode | str | None = None
) -> tuple[Decoupage, CoffreTest]:
    """Découpe la table selon la structure de la spécification et met le test sous scellé.

    Le coffre est construit pour `spec.decoupage()`, la graine est `spec.graine`.
    """
    _exiger_cible(spec)
    _exiger_colonnes(table, spec)
    methode = methode_deduite(spec) if methode is None else _methode(methode)
    permises = methodes_permises(spec)
    if methode not in permises:
        noms = ", ".join(sorted(m.value for m in permises))
        raise ErreurDecoupage(
            f"Méthode {methode} incompatible avec la structure {spec.structure} du problème : choisir parmi "
            f"{noms}, ou changer la structure en S3."
        )
    colonne = _colonne_de(spec, methode)
    strate = spec.cible if methode is Methode.STRATIFIE else colonne
    positions = positions_test(table.donnees, methode, spec.part_test, spec.graine, strate, strate=spec.cible)
    decoupage = Decoupage(
        methode, spec.part_test, spec.graine, colonne, positions.size, empreinte_positions(positions)
    )
    coffre = CoffreTest.par_positions(table, positions, decoupage=spec.decoupage())
    _echouer_si(controler_structure(table, spec, coffre))
    return decoupage, coffre


def spec_pour_base_test(spec: SpecProjet, declare: Decoupage) -> SpecProjet:
    """Spécification dont le découpage (structure, part, graine) est celui de la base déclarée.

    Les rôles ne changent pas : la colonne déclarée doit déjà porter le rôle GROUPE
    (`unites`) ou DATE (`chronologique`). Une validation imposée doit convenir à la
    structure qui en résulte.
    """
    _exiger_cible(spec)
    col = declare.colonne_groupe
    if declare.methode in (Methode.UNITES, Methode.UNITES_STRATIFIEES):
        _exiger_role(spec, col, Role.GROUPE)
        structure = Structure.GROUPEE_CHRONOLOGIQUE if spec.structure.chronologique else Structure.GROUPEE
    elif declare.methode is Methode.CHRONOLOGIQUE:
        _exiger_role(spec, col, Role.DATE)
        structure = Structure.CHRONOLOGIQUE
    elif declare.methode is Methode.FICHIER:
        raise ErreurDecoupage("Un test fourni à part se scelle par `sceller_fichier`.")
    else:
        structure = Structure.INDEPENDANTE
    permises = validations_permises(structure, spec.probleme)
    if spec.validation is not None and spec.validation not in permises:
        raise ErreurDecoupage(
            f"La validation imposée {spec.validation} ne convient pas à la structure {structure} de la base "
            f"déclarée : en S3, la laisser déduire, ou choisir parmi {sorted(permises)}."
        )
    return spec.avec(structure=structure, part_test=declare.test_size, graine=declare.random_state)


def _exiger_role(spec: SpecProjet, colonne: str | None, role: Role) -> None:
    if spec.colonnes_de(role) != (colonne,):
        raise ErreurDecoupage(
            f"La base de test déclarée s'appuie sur « {colonne} » : en S3, donner le rôle {role.name} à "
            f"« {colonne} », et à elle seule."
        )


def sceller_base_test(
    table: Table, spec: SpecProjet, base_test: Mapping[str, Any] | Decoupage
) -> tuple[SpecProjet, Decoupage, CoffreTest]:
    """Mode challenge : relit une base de test déclarée, la recalcule, vérifie son empreinte.

    Rend la spécification au découpage déclaré (à transmettre avec le coffre à
    `Decouper`), le découpage relu et le coffre scellé.
    """
    declare = base_test if isinstance(base_test, Decoupage) else Decoupage.depuis_base_test(base_test)
    spec_declaree = spec_pour_base_test(spec, declare)
    _exiger_colonnes(table, spec_declaree)
    colonne = declare.colonne_groupe
    if declare.methode is Methode.STRATIFIE and colonne is None:
        colonne = spec.cible
    positions = positions_test(
        table.donnees, declare.methode, declare.test_size, declare.random_state, colonne, strate=spec.cible
    )
    empreinte = empreinte_positions(positions)
    if positions.size != declare.lignes_test or empreinte != declare.empreinte_sha256:
        raise ErreurDecoupage(
            f"La base de test recalculée ({positions.size} lignes, empreinte {empreinte[:12]}) diffère de la "
            f"base déclarée ({declare.lignes_test} lignes, empreinte {declare.empreinte_sha256[:12]}) : "
            "charger le fichier publié tel quel, première feuille, sans filtrer ni trier de lignes."
        )
    coffre = CoffreTest.par_positions(table, positions, decoupage=spec_declaree.decoupage())
    _echouer_si(controler_structure(table, spec_declaree, coffre))
    return spec_declaree, declare, coffre


def sceller_fichier(
    table_apprentissage: Table, table_test: Table, spec: SpecProjet
) -> tuple[Decoupage, CoffreTest]:
    """Mode challenge à deux fichiers : le test est fourni à part ; la structure est vérifiée avant."""
    _exiger_cible(spec)
    _exiger_colonnes(table_apprentissage, spec)
    coffre = CoffreTest.par_fichier(table_apprentissage, table_test, decoupage=spec.decoupage())
    _echouer_si(_constats_structure(spec, table_apprentissage.donnees, table_test.donnees))
    total = coffre.n_apprentissage + coffre.n_test
    colonne = spec.colonne_groupe or spec.colonne_date
    decoupage = Decoupage(
        Methode.FICHIER, coffre.n_test / total, spec.graine, colonne, coffre.n_test, table_test.empreinte
    )
    return decoupage, coffre


def _echouer_si(constats: Sequence[Constat]) -> None:
    bloquants = [c for c in constats if c.bloquant]
    if bloquants:
        raise ErreurDecoupage(" ".join(f"{c.message} {c.action}" for c in bloquants))


def separer_sans_cible(donnees: pd.DataFrame, cible: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(lignes avec cible, lignes sans cible), dans l'ordre reçu.

    Les lignes sans cible sont retirées du modèle et comptées ; en structure
    chronologique, elles restent le contexte des variables temporelles.
    """
    presentes = donnees[cible].notna().to_numpy()
    return donnees[presentes], donnees[~presentes]


# ============================================================
# CONTRÔLES POUR L'AUDIT DE S4 : STRUCTURE (I4) ET CLASSES (I7)
# ============================================================
def _positions_de_test(table: Table, coffre: CoffreTest) -> np.ndarray:
    masque = np.ones(table.n_lignes, dtype=bool)
    masque[coffre.positions_apprentissage()] = False
    return np.flatnonzero(masque)


def controler_decoupage(
    table: Table, spec: SpecProjet | None, coffre: CoffreTest | None
) -> tuple[Constat, ...]:
    """Contrôles bloquants du découpage pour l'auditeur au rejeu de S4 : structure, puis classes.

    La part de test obtenue (`controler_part_test`) est une information, que l'audit ajoute à part.
    """
    return controler_structure(table, spec, coffre) + controler_classes(table, spec, coffre)


def part_obtenue(coffre: CoffreTest) -> float:
    """Part des lignes mises en test : lignes de test sur lignes de la table, test compris."""
    return coffre.n_test / (coffre.n_test + coffre.n_apprentissage)


def controler_part_test(spec: SpecProjet | None, coffre: CoffreTest | None) -> tuple[Constat, ...]:
    """Constat INFO qui dit la part de test obtenue quand elle s'écarte d'un point au moins de la part visée.

    Les unités et les dates se découpent entières : la part des lignes en test n'est pas exactement celle
    qu'on a demandée (`part_obtenue` la donne aussi au rapport). Un test fourni à part n'a pas de part visée.
    """
    if spec is None or coffre is None or coffre.mode_challenge:
        return ()
    obtenue = part_obtenue(coffre)
    if abs(obtenue - spec.part_test) < ECART_PART_INFO:
        return ()
    total = coffre.n_test + coffre.n_apprentissage
    return (
        Constat(
            Niveau.INFO,
            "DECOUPAGE.part_test_obtenue",
            f"Part de test : {decimale(100 * spec.part_test)} % visés, {decimale(100 * obtenue)} % obtenus "
            f"({coffre.n_test} lignes sur {total}) : le découpage garde entières les unités ou les dates, "
            "si bien que la part des lignes ne suit pas la part demandée à la ligne près.",
            "",
            REF_MORCELLEMENT,
        ),
    )


def controler_structure(
    table: Table, spec: SpecProjet | None, coffre: CoffreTest | None
) -> tuple[Constat, ...]:
    """Constats BLOQUANT si une unité ou une date est à cheval entre apprentissage et test.

    Ne lit que les colonnes de groupe et de date. Un coffre à fichier de test séparé
    est contrôlé à sa construction (`sceller_fichier`) : il ne se rouvre pas ici.
    """
    if spec is None or coffre is None or coffre.mode_challenge:
        return ()
    colonnes = [c for c in (spec.colonne_groupe, spec.colonne_date) if c is not None]
    if not colonnes:
        return ()
    utiles = table.donnees[colonnes]
    app = coffre.positions_apprentissage()
    return _constats_structure(spec, utiles.iloc[app], utiles.iloc[_positions_de_test(table, coffre)])


def _constats_structure(spec: SpecProjet, app: pd.DataFrame, test: pd.DataFrame) -> tuple[Constat, ...]:
    constats: list[Constat] = []
    groupe = spec.colonne_groupe
    if groupe is not None:
        constats += _constat_groupes(groupe, app[groupe], test[groupe])
    if spec.structure is Structure.CHRONOLOGIQUE:
        constats += _constat_dates(spec.colonne_date, app[spec.colonne_date], test[spec.colonne_date])
    return tuple(constats)


def _constat_groupes(colonne: str, app: pd.Series, test: pd.Series) -> list[Constat]:
    if app.isna().any() or test.isna().any():
        return [_incomplet("I4.unites_incompletes", colonne)]
    communes = sorted(set(app.unique()) & set(test.unique()), key=str)
    if not communes:
        return []
    exemples = ", ".join(map(str, communes[:_EXEMPLES]))
    return [
        Constat(
            Niveau.BLOQUANT,
            "I4.unites_a_cheval",
            f"{len(communes)} unité(s) de « {colonne} » à la fois en apprentissage et en test ({exemples}) : "
            "le score mesurerait la mémoire de ces unités, non la généralisation à une unité nouvelle.",
            "Refaire le découpage en S4 par unités entières.",
            REF_MORCELLEMENT,
        )
    ]


def _constat_dates(colonne: str, app: pd.Series, test: pd.Series) -> list[Constat]:
    if app.isna().any() or test.isna().any():
        return [_incomplet("I4.dates_incompletes", colonne)]
    if app.max() < test.min():
        return []
    return [
        Constat(
            Niveau.BLOQUANT,
            "I4.dates_a_cheval",
            f"Dates à cheval dans « {colonne} » : l'apprentissage va jusqu'à {app.max()}, le test commence "
            f"à {test.min()} ; le modèle apprendrait sur le futur de ce qu'il prédit.",
            "Refaire le découpage en S4 par une coupure chronologique.",
            REF_VALIDATION_GLISSANTE,
        )
    ]


def _incomplet(code: str, colonne: str) -> Constat:
    return Constat(
        Niveau.BLOQUANT,
        code,
        f"La colonne de structure « {colonne} » a des valeurs manquantes : le découpage ne peut pas "
        "être vérifié.",
        "Compléter ou retirer ces lignes, puis refaire le découpage.",
        REF_MORCELLEMENT,
    )


_ACTION_CLASSES = "Changer la part de test ou regrouper les classes rares, puis refaire le découpage (S4)."


def controler_classes(
    table: Table, spec: SpecProjet | None, coffre: CoffreTest | None
) -> tuple[Constat, ...]:
    """Constats BLOQUANT, en classification, si une classe manque au test, à l'apprentissage ou à un pli.

    Lit la cible de toutes les lignes, comme la stratification, et n'en rend que les
    classes absentes. Les plis sont ceux de `plis_validation` sur les lignes
    d'apprentissage qui ont une cible. Un coffre à fichier de test séparé n'est pas
    rouvert ici : son test est imposé.
    """
    if (
        spec is None
        or coffre is None
        or coffre.mode_challenge
        or spec.probleme is not Probleme.CLASSIFICATION
    ):
        return ()
    cible = table.donnees[spec.cible]
    y_app = cible.iloc[coffre.positions_apprentissage()]
    y_test = cible.iloc[_positions_de_test(table, coffre)].dropna()
    constats = _constats_test(spec.cible, y_app.dropna(), y_test)
    lignes = coffre.apprentissage()[y_app.notna().to_numpy()]
    try:
        plis = plis_validation(lignes, spec)
    except ErreurDecoupage as erreur:
        return (
            *constats,
            Constat(
                Niveau.BLOQUANT,
                "I7.plis_impossibles",
                f"Plis de validation impossibles : {erreur}",
                "Changer le nombre de plis ou la validation, ou regrouper les classes rares.",
                REF_VALIDATION_CROISEE,
            ),
        )
    if validation_appliquee(spec) == "validation.exclusion_unique":
        return constats  # une ligne par pli : la mesure se fait sur l'ensemble des prédictions hors pli
    return (*constats, *_constats_plis(lignes[spec.cible].to_numpy(), plis))


def _liste(classes: Sequence[Any]) -> str:
    return ", ".join(f"« {c} »" for c in sorted(classes, key=str))


def _constats_test(cible: str, y_app: pd.Series, y_test: pd.Series) -> tuple[Constat, ...]:
    vues_app, vues_test = set(y_app.unique()), set(y_test.unique())
    constats = []
    if vues_app - vues_test:
        constats.append(
            Constat(
                Niveau.BLOQUANT,
                "I7.classe_absente_test",
                f"Classe(s) {_liste(vues_app - vues_test)} de « {cible} » absente(s) du test ({len(y_test)} "
                "lignes notées) : précision, rappel et F1 de ces classes y seraient indéfinis.",
                _ACTION_CLASSES,
                REF_MORCELLEMENT,
            )
        )
    if vues_test - vues_app:
        constats.append(
            Constat(
                Niveau.BLOQUANT,
                "I7.classe_absente_apprentissage",
                f"Classe(s) {_liste(vues_test - vues_app)} de « {cible} » présente(s) au test seulement : "
                "le modèle ne peut pas les apprendre.",
                _ACTION_CLASSES,
                REF_MORCELLEMENT,
            )
        )
    return tuple(constats)


def classes_absentes(y: np.ndarray, plis: Plis) -> tuple[tuple[frozenset[Any], frozenset[Any]], ...]:
    """Pour chaque pli : (classes absentes de son apprentissage, classes absentes de sa validation).

    Une métrique propre à une classe absente de la validation d'un pli est indéfinie
    pour ce pli (et non nulle).
    """
    y = np.asarray(y)
    toutes = frozenset(pd.unique(y).tolist())
    return tuple(
        (toutes - frozenset(pd.unique(y[app]).tolist()), toutes - frozenset(pd.unique(y[val]).tolist()))
        for app, val in plis
    )


def _constats_plis(y: np.ndarray, plis: Plis) -> tuple[Constat, ...]:
    defauts = []
    for numero, (sans_app, sans_val) in enumerate(classes_absentes(y, plis), 1):
        if sans_val:
            defauts.append(f"pli {numero}, validation sans {_liste(sans_val)}")
        if sans_app:
            defauts.append(f"pli {numero}, apprentissage sans {_liste(sans_app)}")
    if not defauts:
        return ()
    return (
        Constat(
            Niveau.BLOQUANT,
            "I7.classe_absente_pli",
            f"Classes manquantes dans les plis de validation ({'; '.join(defauts[:_EXEMPLES])}"
            f"{' ...' if len(defauts) > _EXEMPLES else ''}) : les mesures de ces plis seraient indéfinies.",
            "Réduire le nombre de plis, regrouper les classes rares, ou changer la part de test.",
            REF_VALIDATION_CROISEE,
        ),
    )


# ============================================================
# PLIS DE VALIDATION CROISÉE
# ============================================================
Plis = tuple[tuple[np.ndarray, np.ndarray], ...]


def validation_deduite(spec: SpecProjet) -> str:
    """Validation par défaut de la structure : plis par unités (stratifiés en classification),
    validation glissante, ou plis aléatoires (stratifiés en classification)."""
    classification = spec.probleme is Probleme.CLASSIFICATION
    if spec.structure.groupee:
        return "validation.plis_groupes_stratifies" if classification else "validation.plis_groupes"
    if spec.structure.chronologique:
        return "validation.serie_temporelle"
    return "validation.kfold_stratifie" if classification else "validation.kfold"


def validation_appliquee(spec: SpecProjet) -> str:
    """Validation imposée par la spécification, sinon celle de la structure."""
    return spec.validation if spec.validation is not None else validation_deduite(spec)


def plis_validation(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    """Plis (apprentissage, validation) en positions de `donnees`, selon `validation_appliquee`.

    `donnees` : lignes d'apprentissage (celles du coffre, privées des lignes sans
    cible). Mêmes constructeurs et même graine que les entrées du registre ; en
    chronologique, fenêtre croissante sur les dates distinctes (aucune date à cheval).
    """
    if not isinstance(spec, SpecProjet):
        raise TypeError("Plis : une `SpecProjet` est attendue.")
    if not spec.probleme.avec_cible:
        return _plis_hasard(donnees, spec)  # sans cible : plis aléatoires, aucune validation imposée
    validation = validation_appliquee(spec)
    permises = validations_permises(spec.structure, spec.probleme)
    if validation not in permises:
        raise ErreurDecoupage(
            f"La validation {validation} ne convient pas à la structure {spec.structure} : choisir parmi "
            f"{sorted(permises)}."
        )
    return _PLIS[validation](donnees, spec)


def plis_au_plus(donnees: pd.DataFrame, spec: SpecProjet) -> int:
    """Plus grand nombre de plis que la validation de la structure permet sur ces lignes d'apprentissage.

    Jamais plus de plis que d'unités (structure groupée), de dates distinctes moins une (série ordonnée), de
    lignes, ni que la classe la plus rare n'a de lignes (classification). Sert aux propositions : un jeu de
    quatre unités ne se voit pas proposer cinq plis.
    """
    limites = [len(donnees)]
    colonne_groupe, colonne_date = spec.colonne_groupe, spec.colonne_date
    if colonne_groupe is not None and colonne_groupe in donnees.columns:
        limites.append(int(donnees[colonne_groupe].nunique()))
    elif colonne_date is not None and colonne_date in donnees.columns:
        limites.append(int(donnees[colonne_date].nunique()) - 1)
    chronologique_seule = colonne_groupe is None and colonne_date is not None
    if spec.probleme is Probleme.CLASSIFICATION and not chronologique_seule and spec.cible in donnees.columns:
        effectifs = donnees[spec.cible].dropna().value_counts()
        if len(effectifs):
            limites.append(int(effectifs.min()))
    return min(limites)


def _en_plis(decoupes: Any) -> Plis:
    return tuple((np.asarray(a, dtype=np.int64), np.asarray(v, dtype=np.int64)) for a, v in decoupes)


def _unites(donnees: pd.DataFrame, spec: SpecProjet) -> pd.Series:
    unites = _colonne_presente(donnees, spec.colonne_groupe, "des unités")
    if unites.nunique() < spec.n_plis:
        raise ErreurDecoupage(
            f"{unites.nunique()} unités pour {spec.n_plis} plis : réduire le nombre de plis à "
            f"{unites.nunique()} au plus."
        )
    return unites


def _classes_de(donnees: pd.DataFrame, spec: SpecProjet) -> np.ndarray:
    classes = _codes(_colonne_presente(donnees, spec.cible, "cible", complete=False), manque_a_part=True)
    _exiger_effectifs(classes, spec.n_plis, "plis")
    return classes


def _plis_unites(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import GroupKFold

    unites = _unites(donnees, spec)
    pli = GroupKFold(n_splits=spec.n_plis, shuffle=True, random_state=spec.graine)
    return _en_plis(pli.split(np.zeros(len(unites)), groups=unites))


def _plis_unites_stratifiees(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import StratifiedGroupKFold

    unites = _unites(donnees, spec)
    classes = _classes_de(donnees, spec)
    pli = StratifiedGroupKFold(n_splits=spec.n_plis, shuffle=True, random_state=spec.graine)
    return _en_plis(pli.split(np.zeros(len(unites)), classes, groups=unites))


def _plis_unites_aleatoires(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import GroupShuffleSplit

    unites = _unites(donnees, spec)
    pli = GroupShuffleSplit(n_splits=spec.n_plis, test_size=0.2, random_state=spec.graine)
    return _en_plis(pli.split(np.zeros(len(unites)), groups=unites))


def _plis_dates(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import TimeSeriesSplit

    k = spec.n_plis
    valeurs = _colonne_presente(donnees, spec.colonne_date, "de date").to_numpy()
    distinctes = np.unique(valeurs)
    if distinctes.size < k + 1:
        raise ErreurDecoupage(
            f"{distinctes.size} dates distinctes pour {k} plis chronologiques : réduire le nombre de plis "
            f"à {max(distinctes.size - 1, 1)} au plus."
        )
    rang = np.searchsorted(distinctes, valeurs)
    plis = []
    for app, val in TimeSeriesSplit(n_splits=k).split(distinctes):
        plis.append(
            (np.flatnonzero(rang <= app.max()), np.flatnonzero((rang >= val.min()) & (rang <= val.max())))
        )
    return _en_plis(plis)


def _plis_strates(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import StratifiedKFold

    classes = _classes_de(donnees, spec)
    pli = StratifiedKFold(n_splits=spec.n_plis, shuffle=True, random_state=spec.graine)
    return _en_plis(pli.split(np.zeros(len(classes)), classes))


def _plis_hasard(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import KFold

    n, k = len(donnees), spec.n_plis
    if n < k:
        raise ErreurDecoupage(f"{n} lignes pour {k} plis : réduire le nombre de plis.")
    return _en_plis(KFold(n_splits=k, shuffle=True, random_state=spec.graine).split(np.zeros(n)))


def _plis_exclusion(donnees: pd.DataFrame, spec: SpecProjet) -> Plis:
    from sklearn.model_selection import LeaveOneOut

    n = len(donnees)
    coupee = technique_coupee("validation.exclusion_unique", n)  # critère et message uniques du registre
    if coupee is not None:
        raise ErreurDecoupage(coupee)
    if n < 2:
        raise ErreurDecoupage(f"{n} ligne pour une exclusion d'une ligne : il en faut au moins deux.")
    return _en_plis(LeaveOneOut().split(np.zeros(n)))


_PLIS: dict[str, Callable[[pd.DataFrame, SpecProjet], Plis]] = {
    "validation.kfold": _plis_hasard,
    "validation.kfold_stratifie": _plis_strates,
    "validation.exclusion_unique": _plis_exclusion,
    "validation.plis_groupes": _plis_unites,
    "validation.plis_groupes_stratifies": _plis_unites_stratifiees,
    "validation.groupes_aleatoires": _plis_unites_aleatoires,
    "validation.serie_temporelle": _plis_dates,
}  # une entrée par validation de `types.VALIDATIONS` (vérifié par les tests)
