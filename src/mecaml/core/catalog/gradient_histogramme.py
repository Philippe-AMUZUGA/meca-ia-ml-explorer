"""Gradient boosting sur histogrammes, tolérant aux valeurs manquantes."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_GRADIENT_HISTOGRAMME = "regression.gradient_histogramme"
CLASSIFICATION_GRADIENT_HISTOGRAMME = "classification.gradient_histogramme"


def _fabrique_regression_gradient_histogramme(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        **preparer(REGRESSION_GRADIENT_HISTOGRAMME, {"max_iter": 100}, parametres, graine)
    )


def _fabrique_classification_gradient_histogramme(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        **preparer(CLASSIFICATION_GRADIENT_HISTOGRAMME, {"max_iter": 100}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_GRADIENT_HISTOGRAMME,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_gradient_histogramme,
        grille_bornee=(("learning_rate", (0.05, 0.1, 0.2)), ("max_iter", (50, 100, 200))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_GRADIENT_HISTOGRAMME),
        texte_id=REGRESSION_GRADIENT_HISTOGRAMME,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_GRADIENT_HISTOGRAMME,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_gradient_histogramme,
        grille_bornee=(
            ("learning_rate", (0.05, 0.1, 0.2)),
            ("max_iter", (50, 100, 200)),
            ("class_weight", (None, "balanced")),
        ),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_GRADIENT_HISTOGRAMME),
        texte_id=CLASSIFICATION_GRADIENT_HISTOGRAMME,
    ),
)
