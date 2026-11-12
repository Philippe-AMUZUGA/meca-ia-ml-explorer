"""AdaBoost, en régression et en classification : arbres courts ajustés l'un après l'autre."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_ADABOOST = "regression.adaboost"
CLASSIFICATION_ADABOOST = "classification.adaboost"


def _fabrique_regression_adaboost(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import AdaBoostRegressor

    return AdaBoostRegressor(**preparer(REGRESSION_ADABOOST, {"n_estimators": 50}, parametres, graine))


def _fabrique_classification_adaboost(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import AdaBoostClassifier

    return AdaBoostClassifier(**preparer(CLASSIFICATION_ADABOOST, {"n_estimators": 50}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_ADABOOST,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_adaboost,
        grille_bornee=(("n_estimators", (25, 50, 100)), ("learning_rate", (0.1, 0.5, 1.0))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_ADABOOST),
        texte_id=REGRESSION_ADABOOST,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_ADABOOST,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_adaboost,
        grille_bornee=(("n_estimators", (25, 50, 100)), ("learning_rate", (0.1, 0.5, 1.0))),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_ADABOOST),
        texte_id=CLASSIFICATION_ADABOOST,
    ),
)
