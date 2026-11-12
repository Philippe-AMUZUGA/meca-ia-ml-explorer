"""Arbre de décision, en régression et en classification."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_ARBRE = "regression.arbre"
CLASSIFICATION_ARBRE = "classification.arbre"


def _fabrique_regression_arbre(graine: int, **parametres: Any) -> Any:
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(**preparer(REGRESSION_ARBRE, {"max_depth": 5}, parametres, graine))


def _fabrique_classification_arbre(graine: int, **parametres: Any) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(**preparer(CLASSIFICATION_ARBRE, {"max_depth": 5}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_ARBRE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_arbre,
        grille_bornee=(("max_depth", (2, 3, 5, 8, None)), ("min_samples_leaf", (1, 5, 10))),
        cout=Cout.LEGER,
        ref_livre=ref(REGRESSION_ARBRE),
        texte_id=REGRESSION_ARBRE,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_ARBRE,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_arbre,
        grille_bornee=(
            ("max_depth", (2, 3, 5, 8, None)),
            ("min_samples_leaf", (1, 5, 10)),
            ("class_weight", (None, "balanced")),
        ),
        cout=Cout.LEGER,
        ref_livre=ref(CLASSIFICATION_ARBRE),
        texte_id=CLASSIFICATION_ARBRE,
    ),
)
