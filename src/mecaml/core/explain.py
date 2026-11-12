"""
MécaML : ce qui explique un modèle, lu sans jamais toucher au test avant son ouverture.

- importance par permutation : baisse du score quand les valeurs d'une variable sont mélangées. Avant le
  test (S7), elle se mesure plis par plis : pour chaque pli, le modèle est ajusté sur l'apprentissage du pli
  et la variable mélangée dans le pli de validation ; sur le test (S8), elle se mesure sur les lignes
  ouvertes avec le modèle ajusté sur tout l'apprentissage. L'algorithme est celui de l'entrée
  `explication.permutation` du catalogue ; seules les variables du modèle sont mélangées, la date, l'unité
  et la cible qu'un transformeur temporel lit restent fixes ; - importance par impureté, arbre en texte,
  coefficients d'un modèle linéaire : lus sur le modèle ajusté sur tout l'apprentissage
  (`train.ajuster_final`).

Une importance dit ce dont le modèle se sert, non ce qui cause la cible : deux variables corrélées se
partagent l'importance, et une variable très importante peut être un indice de fuite à vérifier.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mecaml.core import metrics, train
from mecaml.core.catalog import EntreeCatalogue, entree
from mecaml.core.preprocess import colonnes_d_entree
from mecaml.core.tasks import Annulation, collecter_avertissements
from mecaml.core.types import CoffreTest, Probleme, SpecProjet, Table

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "ErreurExplication",
    "arbre_texte",
    "coefficients",
    "figure_importances",
    "importance_par_impurete",
    "importances_sur_lignes",
    "importances_validation",
]

IMPORTANCE_PERMUTATION = "explication.permutation"
IMPORTANCE_IMPURETE = "explication.impurete"
ARBRE_TEXTE = "explication.arbre_texte"
COLONNES_IMPORTANCES = ("variable", "moyenne", "ecart_type", "plis_valides")


class ErreurExplication(ValueError):
    """Explication impossible pour ce modèle ou ces données ; le message dit quoi faire."""


# ============================================================
# MODÈLE VU À TRAVERS SES VARIABLES
# ============================================================
class _SurLesVariables:
    """Pipeline dont on ne présente que les variables : les autres colonnes d'entrée restent d'origine.

    Un transformeur temporel lit la date et l'unité en plus des mesures ; mélanger ces colonnes n'a pas de
    sens. Les lignes sont alignées par position, ce que préserve le mélange d'une colonne.
    """

    def __init__(self, pipeline: Any, fixes: pd.DataFrame, colonnes: list[str]) -> None:
        self._pipeline = pipeline
        self._fixes = fixes.reset_index(drop=True)
        self._colonnes = colonnes

    def _complet(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([x.reset_index(drop=True), self._fixes], axis=1)[self._colonnes]

    def fit(self, *_: Any, **__: Any) -> _SurLesVariables:
        """Ne refait aucun ajustement : scikit-learn demande seulement que l'objet en ait un."""
        return self

    def predict(self, x: pd.DataFrame) -> Any:
        return self._pipeline.predict(self._complet(x))

    def __getattr__(self, nom: str) -> Any:
        if nom in ("predict_proba", "decision_function"):
            methode = getattr(self._pipeline, nom)  # AttributeError si le modèle n'en a pas
            return lambda x: methode(self._complet(x))
        if nom == "classes_":
            return self._pipeline.classes_
        raise AttributeError(nom)


def _vue(pipeline: Any, x: pd.DataFrame, variables: list[str]) -> tuple[Any, pd.DataFrame]:
    """(modèle à évaluer, colonnes à mélanger) : sans colonne fixe, le pipeline lui-même."""
    autres = [c for c in x.columns if c not in variables]
    if not autres:
        return pipeline, x[variables]
    return _SurLesVariables(pipeline, x[autres], list(x.columns)), x[variables]


def _evaluateur(texte: str, classes: Any, positive: Any) -> Any:
    """Évaluateur de scikit-learn (plus grand est meilleur) qui ignore les lignes sans cible."""
    m, _ = metrics.analyser(texte)
    signe = -1.0 if m.meilleur == metrics.BAS else 1.0

    def evaluateur(modele: Any, x: pd.DataFrame, y: Any) -> float:
        y = np.asarray(y)
        presentes = ~pd.isna(y)
        predictions = np.asarray(modele.predict(x))[presentes]
        donnees: dict[str, Any] = {"y_vrai": y[presentes], "y_pred": predictions}
        if classes is not None:
            continus = metrics.scores_du_modele(modele, x, classes, positive)
            donnees.update(classes=classes, classe_positive=positive)
            donnees["scores"] = None if continus is None else np.asarray(continus)[presentes]
        try:
            return signe * metrics.calculer(texte, **donnees)
        except metrics.ErreurMetrique:
            return math.nan

    return evaluateur


def _permuter(
    spec: SpecProjet,
    modele: Any,
    x: pd.DataFrame,
    y: Any,
    texte: str,
    classes: Any,
    positive: Any,
    repetitions: int,
    registre: Iterable[EntreeCatalogue] | None,
) -> pd.DataFrame:
    variables = list(spec.variables())
    calculer = entree(IMPORTANCE_PERMUTATION, registre).construire(spec.graine, n_repeats=repetitions)
    vue, a_melanger = _vue(modele, x, variables)
    with collecter_avertissements():
        return calculer(vue, a_melanger, y, scoring=_evaluateur(texte, classes, positive))


def _agreger(parties: list[pd.DataFrame], variables: list[str]) -> pd.DataFrame:
    """Moyenne des plis (ceux où l'importance est définie) et dispersion entre plis, la plus forte d'abord."""
    par_pli = pd.DataFrame(
        {k: p.set_index("variable")["moyenne"].reindex(variables) for k, p in enumerate(parties)}
    )
    valides = par_pli.notna().sum(axis=1)
    tableau = pd.DataFrame(
        {
            "variable": variables,
            "moyenne": par_pli.mean(axis=1).to_numpy(),
            "ecart_type": par_pli.std(axis=1, ddof=1).where(valides > 1, 0.0).to_numpy(),
            "plis_valides": valides.to_numpy(),
        }
    )
    return tableau.sort_values("moyenne", ascending=False, ignore_index=True, na_position="last")


# ============================================================
# IMPORTANCE PAR PERMUTATION
# ============================================================
def importances_validation(
    resultat: train.Resultat,
    id_modele: str,
    table: Table,
    coffre: CoffreTest,
    *,
    repetitions: int = 5,
    annulation: Annulation | None = None,
    registre: Iterable[EntreeCatalogue] | None = None,
) -> pd.DataFrame:
    """Importance par permutation d'un modèle, mesurée pli par pli sur les lignes d'apprentissage.

    `moyenne` : baisse moyenne du score de la métrique du projet (dans le sens « plus grand est meilleur »)
    quand la variable est mélangée ; `ecart_type` : dispersion entre plis ; `plis_valides` : plis où le score
    est défini. Les lignes du test ne sont pas lues.
    """
    from sklearn.base import clone

    from mecaml.core import evaluate

    spec = resultat.spec
    if not spec.probleme.avec_cible:
        raise ErreurExplication("L'importance par permutation suppose une cible.")
    donnees = evaluate.apprentissage_du_projet(spec, table, coffre)
    choix = next((c for c in spec.modeles if c.id == id_modele), None)
    if choix is None:
        raise ErreurExplication(f"Le modèle {id_modele} ne figure pas parmi les modèles comparés.")
    pipeline = train.construire_pipeline(spec, choix, registre, donnees.contexte)
    parties = []
    for app, val in resultat.plis:
        if annulation is not None:
            annulation.verifier()
        modele = clone(pipeline)
        with collecter_avertissements():
            modele.fit(donnees.X.iloc[app], donnees.y[app])
        parties.append(
            _permuter(
                spec,
                modele,
                donnees.X.iloc[val],
                donnees.y[val],
                resultat.metrique,
                resultat.classes,
                resultat.classe_positive,
                repetitions,
                registre,
            )
        )
    return _agreger(parties, list(spec.variables()))


def importances_sur_lignes(
    modele: train.ModeleAjuste,
    spec: SpecProjet,
    lignes: pd.DataFrame,
    *,
    repetitions: int = 5,
    registre: Iterable[EntreeCatalogue] | None = None,
) -> pd.DataFrame:
    """Importance par permutation du modèle ajusté, sur des lignes qu'il n'a pas vues (le test ouvert)."""
    variables = lignes[list(colonnes_d_entree(spec))]
    cible = lignes[spec.cible].to_numpy(dtype=float if spec.probleme is Probleme.REGRESSION else None)
    tableau = _permuter(
        spec,
        modele.pipeline,
        variables,
        cible,
        spec.metrique or "",
        modele.classes,
        modele.classe_positive,
        repetitions,
        registre,
    )
    tableau = tableau.assign(plis_valides=np.where(tableau["moyenne"].notna(), 1, 0))
    return tableau.reset_index(drop=True)[list(COLONNES_IMPORTANCES)]


# ============================================================
# LECTURE DU MODÈLE AJUSTÉ SUR TOUT L'APPRENTISSAGE
# ============================================================
def importance_par_impurete(
    modele: train.ModeleAjuste, registre: Iterable[EntreeCatalogue] | None = None
) -> pd.DataFrame:
    """Importance par impureté (arbre, forêt, arbres extrêmes, gradient boosting) ; `ErreurExplication` sinon.

    Calculée sur l'apprentissage, elle favorise les variables à nombreuses valeurs : la confirmer par la
    permutation.
    """
    calculer = entree(IMPORTANCE_IMPURETE, registre).construire()
    try:
        return calculer(modele.pipeline)
    except ValueError as erreur:
        raise ErreurExplication(str(erreur)) from erreur


def arbre_texte(
    modele: train.ModeleAjuste, profondeur: int = 4, registre: Iterable[EntreeCatalogue] | None = None
) -> str:
    """Règles d'un arbre de décision ; `ErreurExplication` si le modèle final n'est pas un arbre."""
    calculer = entree(ARBRE_TEXTE, registre).construire(max_depth=profondeur)
    try:
        return calculer(modele.pipeline)
    except ValueError as erreur:
        raise ErreurExplication(str(erreur)) from erreur


def coefficients(modele: train.ModeleAjuste) -> pd.DataFrame | None:
    """Coefficients d'un modèle linéaire (une ligne par variable produite), ou None si le modèle n'en a pas.

    Les variables sont celles que le modèle reçoit, après encodage et mise à l'échelle : des coefficients de
    variables standardisées se comparent entre eux, ceux de variables en unités d'origine non.
    """
    final = modele.pipeline[-1]
    brut = getattr(final, "coef_", None)
    if brut is None:
        return None
    valeurs = np.asarray(brut, dtype=float)
    try:
        noms = [str(n) for n in modele.pipeline[:-1].get_feature_names_out()]
    except (AttributeError, ValueError, TypeError):
        noms = [f"x{k}" for k in range(valeurs.shape[-1])]
    if valeurs.ndim == 1:
        return pd.DataFrame({"variable": noms, "coefficient": valeurs})
    if valeurs.shape[0] == 1:
        return pd.DataFrame({"variable": noms, "coefficient": valeurs[0]})
    classes = [str(c) for c in getattr(final, "classes_", range(valeurs.shape[0]))]
    return pd.DataFrame(valeurs.T, columns=classes).assign(variable=noms)[["variable", *classes]]


# ============================================================
# FIGURE
# ============================================================
def figure_importances(
    tableau: pd.DataFrame, titre: str = "Importance des variables", maximum: int = 15
) -> Figure:
    """Barres horizontales des variables les plus importantes, avec la dispersion entre plis."""
    from matplotlib.figure import Figure

    haut = tableau.dropna(subset=["moyenne"]).head(maximum).iloc[::-1]
    fig = Figure(figsize=(6.4, 0.35 * max(len(haut), 3) + 1.2), layout="constrained", dpi=100)
    axe = fig.subplots()
    axe.barh(
        haut["variable"].astype(str),
        haut["moyenne"],
        xerr=haut["ecart_type"],
        color="#1f4e79",
        ecolor="#7f7f7f",
    )
    axe.axvline(0.0, color="black", linewidth=0.8)
    axe.set(xlabel="baisse du score quand la variable est mélangée", title=titre)
    axe.spines[["top", "right"]].set_visible(False)
    axe.grid(True, axis="x", color="#e5e5e5", linewidth=0.6)
    axe.set_axisbelow(True)
    return fig
