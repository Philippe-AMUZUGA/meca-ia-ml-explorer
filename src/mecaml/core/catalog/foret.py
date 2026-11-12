"""Forêt aléatoire, en régression et en classification."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_FORET = "regression.foret"
CLASSIFICATION_FORET = "classification.foret"


def _fabrique_regression_foret(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        **preparer(REGRESSION_FORET, {"n_estimators": 100, "n_jobs": 1}, parametres, graine)
    )


def _fabrique_classification_foret(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        **preparer(CLASSIFICATION_FORET, {"n_estimators": 100, "n_jobs": 1}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_FORET,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_foret,
        grille_bornee=(("n_estimators", (50, 100, 200)), ("max_depth", (5, 10, None))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_FORET),
        texte_id=REGRESSION_FORET,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_FORET,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_foret,
        grille_bornee=(
            ("n_estimators", (50, 100, 200)),
            ("max_depth", (5, 10, None)),
            ("class_weight", (None, "balanced")),
        ),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_FORET),
        texte_id=CLASSIFICATION_FORET,
    ),
)
