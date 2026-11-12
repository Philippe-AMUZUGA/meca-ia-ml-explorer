"""Bagging : moyenne ou vote de modèles ajustés sur des tirages avec remise."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_BAGGING = "regression.bagging"
CLASSIFICATION_BAGGING = "classification.bagging"


def _fabrique_regression_bagging(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import BaggingRegressor

    return BaggingRegressor(
        **preparer(REGRESSION_BAGGING, {"n_estimators": 20, "n_jobs": 1}, parametres, graine)
    )


def _fabrique_classification_bagging(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import BaggingClassifier

    return BaggingClassifier(
        **preparer(CLASSIFICATION_BAGGING, {"n_estimators": 20, "n_jobs": 1}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_BAGGING,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_bagging,
        grille_bornee=(("n_estimators", (10, 20, 50)), ("max_samples", (0.5, 0.8, 1.0))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_BAGGING),
        texte_id=REGRESSION_BAGGING,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_BAGGING,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_bagging,
        grille_bornee=(("n_estimators", (10, 20, 50)), ("max_samples", (0.5, 0.8, 1.0))),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_BAGGING),
        texte_id=CLASSIFICATION_BAGGING,
    ),
)
