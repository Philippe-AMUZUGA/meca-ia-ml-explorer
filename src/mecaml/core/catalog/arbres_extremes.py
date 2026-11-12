"""Arbres extrêmement aléatoires, en régression et en classification."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_ARBRES_EXTREMES = "regression.arbres_extremes"
CLASSIFICATION_ARBRES_EXTREMES = "classification.arbres_extremes"


def _fabrique_regression_arbres_extremes(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        **preparer(REGRESSION_ARBRES_EXTREMES, {"n_estimators": 100, "n_jobs": 1}, parametres, graine)
    )


def _fabrique_classification_arbres_extremes(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        **preparer(CLASSIFICATION_ARBRES_EXTREMES, {"n_estimators": 100, "n_jobs": 1}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_ARBRES_EXTREMES,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_arbres_extremes,
        grille_bornee=(("n_estimators", (50, 100, 200)), ("max_depth", (5, 10, None))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_ARBRES_EXTREMES),
        texte_id=REGRESSION_ARBRES_EXTREMES,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_ARBRES_EXTREMES,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_arbres_extremes,
        grille_bornee=(
            ("n_estimators", (50, 100, 200)),
            ("max_depth", (5, 10, None)),
            ("class_weight", (None, "balanced")),
        ),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_ARBRES_EXTREMES),
        texte_id=CLASSIFICATION_ARBRES_EXTREMES,
    ),
)
