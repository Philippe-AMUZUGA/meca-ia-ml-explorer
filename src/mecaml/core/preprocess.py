"""
MécaML : prétraitement construit par rôle, tout entier dans le `Pipeline` (I2).

La spécification du projet (`SpecProjet.pretraitement`) liste des choix de techniques ; ce
module en tire des étapes scikit-learn, sans rien ajuster : l'ajustement se fait plus tard,
dans `Pipeline.fit`, donc à l'intérieur de chaque pli de validation croisée.

Principe de construction :

- la première étape de chaque branche, sans ajustement, ramène les marqueurs de valeur manquante de
  pandas (`None`, `pd.NA` des types `string` et `Int64`) à `NaN`, que les imputations savent lire
  (dans la branche, non en tête du pipeline : le `ColumnTransformer` choisit les colonnes par nom,
  si bien que l'ordre des colonnes reçues à la prédiction ne change rien) ;
- chaque variable (rôle NUMERIQUE ou CATEGORIELLE) reçoit les techniques qui la visent, dans un
  ordre fixe : valeurs manquantes, encodage, transformations ; un choix sans colonnes vise toutes
  les variables de son rôle habituel (`STADES`) ;
- les variables qui reçoivent la même suite d'étapes forment un groupe, traité par un
  `Pipeline` dans une branche du `ColumnTransformer` ;
- les variables temporelles (famille `chronologique`, structures chronologiques seulement) forment
  une branche à part qui reçoit la date, l'unité et les mesures : retards et fenêtres commencent
  par des valeurs manquantes au début de chaque série, imputées par la médiane de l'apprentissage ;
- la cible décalée (autorégression) et la référence par persistance lisent les valeurs passées de la
  cible : la colonne cible entre alors dans les colonnes reçues, mais seule sa version décalée d'au
  moins un pas sort de la branche, jamais la valeur courante ; ces deux choix ne sont permis que si le
  projet déclare la cible passée connue à la prévision (`cible_connue=True`, voir `cible_passee`) ;
- la sélection de variables (famille `selection`) vient ensuite, sur l'ensemble des colonnes
  produites ; le modèle est ajouté par `train`.

Les dates, groupes, identifiants et colonnes ignorées ne sont jamais des entrées du modèle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mecaml.core.catalog import EntreeCatalogue, entree
from mecaml.core.cible_passee import (
    CIBLE_DECALEE,
    PARAMETRE,
    PERSISTANCE,
    declaree,
    lit_la_cible_passee,
    message_non_declaree,
)
from mecaml.core.types import ChoixTechnique, Role, SpecProjet

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer

__all__ = [
    "STADES",
    "ErreurPretraitement",
    "colonnes_d_entree",
    "controles_pretraitement",
    "etapes_persistance",
    "etapes_pretraitement",
    "normaliser_manquants",
    "proposer_pretraitement",
]

NUM, CAT = "numerique", "categorielle"
LIBELLES_ROLES = {NUM: "numériques", CAT: "catégorielles"}
COLONNES_SORTIE_MAX = 5000  # garde-fou : expansion polynomiale démesurée


class ErreurPretraitement(ValueError):
    """Prétraitement incohérent avec les rôles ou le problème ; le message dit quoi corriger."""


@dataclass(frozen=True)
class Stade:
    """Place d'une technique : rang dans l'ordre d'application, rôle visé par défaut, rôles permis."""

    rang: int
    defaut: str
    permis: frozenset[str]


_NUM_SEUL = frozenset({NUM})
_CAT_SEUL = frozenset({CAT})
_LES_DEUX = frozenset({NUM, CAT})

STADES: dict[str, Stade] = {
    "manquants.simple": Stade(10, NUM, _LES_DEUX),
    "manquants.knn": Stade(10, NUM, _NUM_SEUL),
    "manquants.indicateur": Stade(10, NUM, _LES_DEUX),
    "encodage.one_hot": Stade(20, CAT, _CAT_SEUL),
    "encodage.ordinal": Stade(20, CAT, _CAT_SEUL),
    "transformation.ecretage_tukey": Stade(30, NUM, _NUM_SEUL),
    "transformation.puissance": Stade(31, NUM, _NUM_SEUL),
    "transformation.polynomiale": Stade(32, NUM, _NUM_SEUL),
    "transformation.standard": Stade(33, NUM, _NUM_SEUL),
    "transformation.minmax": Stade(33, NUM, _NUM_SEUL),
    "transformation.robuste": Stade(33, NUM, _NUM_SEUL),
    "transformation.discretisation": Stade(34, NUM, _NUM_SEUL),
}
FAMILLES_SELECTION = "selection"
FAMILLE_CHRONOLOGIQUE = "chronologique"
ECHELLES = ("transformation.standard", "transformation.minmax", "transformation.robuste")
ORDRE_SELECTION = {
    "selection.variance": 1,
    "selection.k_meilleures": 2,
    "selection.depuis_modele": 3,
    "selection.rfecv": 4,
}


# ============================================================
# AFFECTATION DES TECHNIQUES AUX COLONNES
# ============================================================
def _colonnes_par_role(spec: SpecProjet) -> dict[str, tuple[str, ...]]:
    return {NUM: spec.colonnes_de(Role.NUMERIQUE), CAT: spec.colonnes_de(Role.CATEGORIELLE)}


def _entree_de(
    choix: ChoixTechnique, spec: SpecProjet, registre: Iterable[EntreeCatalogue] | None
) -> EntreeCatalogue:
    try:
        e = entree(choix.id, registre)
    except KeyError:
        raise ErreurPretraitement(f"Technique inconnue du catalogue : {choix.id}.") from None
    if spec.probleme not in e.problemes:
        raise ErreurPretraitement(f"{choix.id} ne s'applique pas à un problème de {spec.probleme}.")
    return e


def _cibles_du_choix(
    choix: ChoixTechnique, stade: Stade, par_role: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Colonnes visées par un choix : celles qu'il nomme, ou toutes celles de son rôle habituel."""
    if not choix.colonnes:
        return par_role[stade.defaut]
    role_de = {c: r for r, colonnes in par_role.items() for c in colonnes}
    inconnues = [c for c in choix.colonnes if c not in role_de]
    if inconnues:
        raise ErreurPretraitement(
            f"{choix.id} vise {inconnues[:3]}, qui ne sont pas des variables du modèle "
            "(rôle numérique ou catégoriel)."
        )
    interdites = [c for c in choix.colonnes if role_de[c] not in stade.permis]
    if interdites:
        raise ErreurPretraitement(
            f"{choix.id} ne s'applique pas à {interdites[:3]} : "
            f"réservé aux colonnes {' ou '.join(LIBELLES_ROLES[r] for r in sorted(stade.permis))}."
        )
    return tuple(choix.colonnes)


def _suites_par_colonne(
    spec: SpecProjet, registre: Iterable[EntreeCatalogue] | None
) -> dict[str, list[tuple[int, int, ChoixTechnique]]]:
    """Pour chaque variable, les choix qui la visent, triés par (rang, position dans la spécification)."""
    par_role = _colonnes_par_role(spec)
    suites: dict[str, list[tuple[int, int, ChoixTechnique]]] = {
        c: [] for cols in par_role.values() for c in cols
    }
    for position, choix in enumerate(spec.pretraitement):
        e = _entree_de(choix, spec, registre)
        if e.famille in (FAMILLES_SELECTION, FAMILLE_CHRONOLOGIQUE):
            continue
        stade = STADES.get(choix.id)
        if stade is None:
            raise ErreurPretraitement(f"{choix.id} n'est pas une technique de prétraitement par colonne.")
        for colonne in _cibles_du_choix(choix, stade, par_role):
            suites[colonne].append((stade.rang, position, choix))
    for colonne, suite in suites.items():
        suite.sort(key=lambda t: (t[0], t[1]))
        _refuser_doublons(colonne, suite)
    return suites


def _refuser_doublons(colonne: str, suite: list[tuple[int, int, ChoixTechnique]]) -> None:
    """Deux techniques de même rang sur une colonne, sauf imputation avec indicateur, sont incohérentes."""
    vus: dict[int, str] = {}
    for rang, _, choix in suite:
        deja = vus.get(rang)
        paire = {deja, choix.id}
        indicateur_et_imputation = rang == 10 and "manquants.indicateur" in paire and len(paire) == 2
        if deja is not None and not indicateur_et_imputation:
            raise ErreurPretraitement(
                f"La colonne {colonne!r} reçoit deux techniques de la même étape : {deja} et {choix.id}."
            )
        vus[rang] = choix.id


# ============================================================
# CONSTRUCTION
# ============================================================
def _parametres(choix: ChoixTechnique) -> dict[str, Any]:
    return choix.parametres_dict()


def _bloc_manquants(suite: list[ChoixTechnique], graine: int, registre: Any) -> Any:
    """Imputation, éventuellement accompagnée d'un indicateur de valeur manquante (union de colonnes)."""
    from sklearn.pipeline import FeatureUnion

    imputation = next((c for c in suite if c.id != "manquants.indicateur"), None)
    indicateur = next((c for c in suite if c.id == "manquants.indicateur"), None)
    base = (
        entree(imputation.id, registre).construire(graine, **_parametres(imputation)) if imputation else None
    )
    if indicateur is None:
        return base
    marque = entree(indicateur.id, registre).construire(graine, **_parametres(indicateur))
    return FeatureUnion([("valeurs", base if base is not None else "passthrough"), ("indicateur", marque)])


def _normalisation() -> tuple[str, Any]:
    from sklearn.preprocessing import FunctionTransformer

    return "normalisation", FunctionTransformer(normaliser_manquants, feature_names_out="one-to-one")


def _etapes_du_groupe(suite: list[ChoixTechnique], graine: int, registre: Any) -> list[tuple[str, Any]]:
    """Étapes d'un groupe dans l'ordre des rangs ; l'imputation et l'indicateur forment une seule étape."""
    etapes: list[tuple[str, Any]] = [_normalisation()]
    manquants = [c for c in suite if c.id.startswith("manquants.")]
    for choix in suite:
        if choix.id.startswith("manquants."):
            if not any(nom == "manquants" for nom, _ in etapes):
                etapes.append(("manquants", _bloc_manquants(manquants, graine, registre)))
            continue
        objet = entree(choix.id, registre).construire(graine, **_parametres(choix))
        etapes.append((choix.id.replace(".", "_"), objet))
    return etapes


def _verifier_expansion(colonnes: tuple[str, ...], suite: list[ChoixTechnique]) -> None:
    for choix in suite:
        if choix.id != "transformation.polynomiale":
            continue
        degre = int(_parametres(choix).get("degree", 2))
        sorties = comb(len(colonnes) + degre, degre) - 1
        if sorties > COLONNES_SORTIE_MAX:
            raise ErreurPretraitement(
                f"L'expansion polynomiale de degré {degre} sur {len(colonnes)} colonnes "
                f"produirait {sorties} colonnes "
                f"(borne {COLONNES_SORTIE_MAX}) : réduire le degré ou restreindre les colonnes."
            )


def _groupes(suites: dict[str, list[tuple[int, int, ChoixTechnique]]]) -> dict[Any, list[str]]:
    """Variables regroupées par suite identique d'étapes (clé : identifiant et paramètres de chaque choix)."""
    groupes: dict[Any, list[str]] = {}
    for colonne, suite in suites.items():
        groupes.setdefault(tuple((c.id, c.parametres) for _, _, c in suite), []).append(colonne)
    return groupes


def _controler_groupe(colonnes: list[str], suite: list[ChoixTechnique], role: str) -> None:
    _verifier_expansion(tuple(colonnes), suite)
    if role == CAT and not any(c.id.startswith("encodage.") for c in suite):
        raise ErreurPretraitement(
            f"La colonne catégorielle {colonnes[0]!r} n'est pas encodée : "
            "ajouter un encodage (one hot ou ordinal)."
        )


def _construire_branches(spec: SpecProjet, registre: Any) -> list[tuple[str, Any, list[str]]]:
    from sklearn.pipeline import Pipeline

    categorielles = set(spec.colonnes_de(Role.CATEGORIELLE))
    suites = _suites_par_colonne(spec, registre)
    branches: list[tuple[str, Any, list[str]]] = []
    for k, colonnes in enumerate(_groupes(suites).values()):
        suite = [c for _, _, c in suites[colonnes[0]]]
        role = CAT if colonnes[0] in categorielles else NUM
        _controler_groupe(colonnes, suite, role)
        etapes = _etapes_du_groupe(suite, spec.graine, registre)
        branches.append((f"{role}_{k}", Pipeline(etapes), colonnes))
    return branches


def colonnes_d_entree(spec: SpecProjet) -> tuple[str, ...]:
    """Colonnes que reçoit le pipeline.

    Les variables, plus la date et l'unité si l'on construit des variables temporelles, plus la cible si
    l'on en lit le passé : le transformeur en a besoin, et aucune de ces colonnes n'entre telle quelle dans
    le modèle.
    """
    colonnes = list(spec.variables())
    passe = lit_la_cible_passee(spec)
    if passe or any(c.id.startswith(FAMILLE_CHRONOLOGIQUE + ".") for c in spec.pretraitement):
        colonnes += [c for c in (spec.colonne_date, spec.colonne_groupe) if c and c not in colonnes]
    if passe and spec.cible and spec.cible not in colonnes:
        colonnes.append(spec.cible)
    return tuple(colonnes)


def _mesures_de_la_branche(choix: ChoixTechnique, spec: SpecProjet) -> tuple[str, ...]:
    if choix.id == CIBLE_DECALEE:
        if not declaree(choix.parametres_dict()):
            message, action = message_non_declaree(CIBLE_DECALEE)
            raise ErreurPretraitement(f"{message} {action}")
        if not spec.cible or tuple(choix.colonnes) not in ((), (spec.cible,)):
            raise ErreurPretraitement(f"{choix.id} décale la cible du projet, et elle seule.")
        return (spec.cible,)
    mesures = tuple(choix.colonnes) or spec.colonnes_de(Role.NUMERIQUE)
    non_numeriques = [c for c in mesures if spec.role_de(c) is not Role.NUMERIQUE]
    if not mesures or non_numeriques:
        raise ErreurPretraitement(
            f"{choix.id} s'applique à des colonnes numériques ; reçu {non_numeriques or 'aucune'}."
        )
    return mesures


def _branche_chronologique(
    choix: ChoixTechnique,
    e: EntreeCatalogue,
    spec: SpecProjet,
    k: int,
    registre: Any,
    avec_echelle: bool = True,
    contexte: pd.DataFrame | None = None,
) -> tuple[str, Any, list[str]]:
    """Branche des variables temporelles : date, unité et mesures en entrée ; manquants du début imputés.

    `contexte` : lignes d'apprentissage sans cible, retirées avant le modèle ; elles complètent les séries
    du transformeur (le retard d'une ligne est la valeur de la ligne précédente, notée ou non), sans jamais
    être rendues.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if not spec.structure.chronologique:
        raise ErreurPretraitement(
            f"{choix.id} suppose des lignes ordonnées dans le temps : "
            "choisir une structure chronologique en S3."
        )
    mesures = _mesures_de_la_branche(choix, spec)
    colonnes = [c for c in (spec.colonne_date, spec.colonne_groupe, *mesures) if c]
    reglages = {
        **_parametres(choix),
        "colonne_date": spec.colonne_date,
        "colonne_groupe": spec.colonne_groupe,
        "colonnes": mesures,
        "contexte": _contexte_de_la_branche(contexte, colonnes),
    }
    etapes = [
        _normalisation(),
        ("variables", e.construire(spec.graine, **reglages)),
        ("manquants", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    echelle = next((c for c in spec.pretraitement if c.id in ECHELLES), None) if avec_echelle else None
    if echelle is not None:
        etapes.append(
            ("echelle", entree(echelle.id, registre).construire(spec.graine, **_parametres(echelle)))
        )
    return f"chronologique_{k}", Pipeline(etapes), colonnes


def _contexte_de_la_branche(contexte: pd.DataFrame | None, colonnes: list[str]) -> pd.DataFrame | None:
    """Colonnes du contexte utiles à la branche (aucune ligne : pas de contexte)."""
    if contexte is None or len(contexte) == 0:
        return None
    return contexte[colonnes].reset_index(drop=True)


def _branches_chronologiques(
    spec: SpecProjet, registre: Any, contexte: pd.DataFrame | None = None
) -> list[tuple[str, Any, list[str]]]:
    branches = []
    for choix in spec.pretraitement:
        e = _entree_de(choix, spec, registre)
        if e.famille == FAMILLE_CHRONOLOGIQUE:
            branches.append(
                _branche_chronologique(choix, e, spec, len(branches), registre, contexte=contexte)
            )
    return branches


def etapes_persistance(
    spec: SpecProjet,
    registre: Iterable[EntreeCatalogue] | None = None,
    contexte: pd.DataFrame | None = None,
) -> list[tuple[str, Any]]:
    """Étapes avant la référence par persistance : la seule cible décalée d'un pas, sans mise à l'échelle.

    Ce modèle ignore le prétraitement du projet : il prédit la dernière valeur connue de la cible de
    la même unité. La spécification doit porter la persistance avec la déclaration `cible_connue=True`
    (`mecaml.core.cible_passee`) ; sinon `ErreurPretraitement`.
    """
    from sklearn.compose import ColumnTransformer

    registre = None if registre is None else tuple(registre)
    if not any(c.id == PERSISTANCE and declaree(c.parametres_dict()) for c in spec.modeles):
        message, action = message_non_declaree(PERSISTANCE)
        raise ErreurPretraitement(f"{message} {action}")
    choix = ChoixTechnique.creer(CIBLE_DECALEE, retards=(1,), **{PARAMETRE: True})
    branche = _branche_chronologique(
        choix, _entree_de(choix, spec, registre), spec, 0, registre, avec_echelle=False, contexte=contexte
    )
    transformateur = ColumnTransformer(
        [branche], remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False
    )
    return [("pretraitement", transformateur)]


def normaliser_manquants(x: pd.DataFrame) -> pd.DataFrame:
    """Ramène à `NaN` les marqueurs de valeur manquante (`None`, `pd.NA`), sans rien apprendre.

    Les colonnes de texte et de catégories deviennent des `object` avec `NaN`, les colonnes numériques à
    marqueur `pd.NA` (`Int64`, `Float64`) des `float64`. Les autres colonnes sont rendues telles quelles.
    """
    sortie = x.copy(deep=False)
    for colonne in x.columns:
        serie = x[colonne]
        if pd.api.types.is_bool_dtype(serie.dtype) or pd.api.types.is_datetime64_any_dtype(serie.dtype):
            continue
        if pd.api.types.is_numeric_dtype(serie.dtype):
            if not isinstance(serie.dtype, np.dtype):
                sortie[colonne] = serie.astype("float64")
        else:
            sortie[colonne] = serie.astype(object).where(serie.notna(), np.nan)
    return sortie


def _transformateur_de_colonnes(
    spec: SpecProjet, registre: Any, contexte: pd.DataFrame | None = None
) -> ColumnTransformer:
    from sklearn.compose import ColumnTransformer

    branches = [*_construire_branches(spec, registre), *_branches_chronologiques(spec, registre, contexte)]
    if not branches:
        raise ErreurPretraitement(
            "Aucune variable d'entrée : donner le rôle numérique ou catégoriel à au moins une colonne."
        )
    return ColumnTransformer(
        [(nom, objet, colonnes) for nom, objet, colonnes in branches],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _etapes_de_selection(spec: SpecProjet, registre: Any) -> list[tuple[str, Any]]:
    choix_selection = []
    for choix in spec.pretraitement:
        e = _entree_de(choix, spec, registre)
        if e.famille == FAMILLES_SELECTION:
            choix_selection.append((ORDRE_SELECTION.get(choix.id, 9), choix, e))
    etapes = []
    for k, (_, choix, e) in enumerate(sorted(choix_selection, key=lambda t: t[0])):
        parametres = {**_parametres(choix), "probleme": spec.probleme.value}
        etapes.append((f"selection_{k}_{choix.id.split('.', 1)[1]}", e.construire(spec.graine, **parametres)))
    return etapes


def etapes_pretraitement(
    spec: SpecProjet,
    registre: Iterable[EntreeCatalogue] | None = None,
    contexte: pd.DataFrame | None = None,
) -> list[tuple[str, Any]]:
    """Étapes du `Pipeline` avant le modèle : transformateur par colonnes, puis sélections ; non ajustées."""
    registre = None if registre is None else tuple(registre)
    return [
        ("pretraitement", _transformateur_de_colonnes(spec, registre, contexte)),
        *_etapes_de_selection(spec, registre),
    ]


# ============================================================
# PROPOSITION ET CONTRÔLES (lecture des données d'apprentissage seulement)
# ============================================================
def _a_des_manquants(donnees: pd.DataFrame, colonnes: tuple[str, ...]) -> bool:
    return bool(colonnes) and bool(donnees[list(colonnes)].isna().to_numpy().any())


def proposer_pretraitement(spec: SpecProjet, donnees: pd.DataFrame) -> tuple[ChoixTechnique, ...]:
    """Prétraitement proposé en S5, modifiable : imputer, encoder, standardiser selon rôles et données.

    `donnees` ne doit contenir que les lignes d'apprentissage (`CoffreTest.apprentissage()`).
    """
    par_role = _colonnes_par_role(spec)
    numeriques, categorielles = par_role[NUM], par_role[CAT]
    choix: list[ChoixTechnique] = []
    if _a_des_manquants(donnees, numeriques):
        choix.append(ChoixTechnique.creer("manquants.simple", colonnes=numeriques, strategy="median"))
    if categorielles:
        if _a_des_manquants(donnees, categorielles):
            choix.append(
                ChoixTechnique.creer("manquants.simple", colonnes=categorielles, strategy="most_frequent")
            )
        choix.append(ChoixTechnique.creer("encodage.one_hot", colonnes=categorielles))
    if numeriques:
        choix.append(ChoixTechnique.creer("transformation.standard", colonnes=numeriques))
        if spec.structure.chronologique:
            choix.append(
                ChoixTechnique.creer("chronologique.variables_temporelles", retards=(1, 2), fenetres=(5,))
            )
    return tuple(choix)


def controles_pretraitement(spec: SpecProjet, donnees: pd.DataFrame) -> list[str]:
    """Avertissements actionnables sur le prétraitement choisi, sans rien ajuster ni refuser."""
    avis: list[str] = []
    try:
        suites = _suites_par_colonne(spec, None)
    except ErreurPretraitement as erreur:
        return [str(erreur)]
    absentes = [c for c in suites if c not in donnees.columns]
    if absentes:
        return [f"Colonnes absentes des données : {absentes[:5]}."]
    for colonne, suite in suites.items():
        imputee = any(c.id.startswith("manquants.") and c.id != "manquants.indicateur" for _, _, c in suite)
        if not imputee and donnees[colonne].isna().any():
            avis.append(
                f"La colonne {colonne!r} contient des valeurs manquantes sans imputation : "
                "ajouter une imputation, sauf si le modèle les accepte (gradient boosting sur histogrammes)."
            )
    return avis
