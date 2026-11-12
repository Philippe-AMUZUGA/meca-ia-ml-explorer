"""Gradient boosting classique d'arbres courts, en régression et en classification."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_GRADIENT_BOOSTING = "regression.gradient_boosting"
CLASSIFICATION_GRADIENT_BOOSTING = "classification.gradient_boosting"


def _fabrique_regression_gradient_boosting(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        **preparer(REGRESSION_GRADIENT_BOOSTING, {"n_estimators": 100}, parametres, graine)
    )


def _fabrique_classification_gradient_boosting(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        **preparer(CLASSIFICATION_GRADIENT_BOOSTING, {"n_estimators": 100}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_GRADIENT_BOOSTING,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_gradient_boosting,
        grille_bornee=(
            ("learning_rate", (0.05, 0.1, 0.2)),
            ("n_estimators", (50, 100, 200)),
            ("max_depth", (2, 3, 5)),
        ),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_GRADIENT_BOOSTING),
        texte_id=REGRESSION_GRADIENT_BOOSTING,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_GRADIENT_BOOSTING,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_gradient_boosting,
        grille_bornee=(
            ("learning_rate", (0.05, 0.1, 0.2)),
            ("n_estimators", (50, 100, 200)),
            ("max_depth", (2, 3, 5)),
        ),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_GRADIENT_BOOSTING),
        texte_id=CLASSIFICATION_GRADIENT_BOOSTING,
    ),
)
