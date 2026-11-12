"""Perceptron multicouche, en régression et en classification, taille du réseau bornée."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_RESEAU_NEURONES = "regression.reseau_neurones"
CLASSIFICATION_RESEAU_NEURONES = "classification.reseau_neurones"


def _fabrique_regression_reseau_neurones(graine: int, **parametres: Any) -> Any:
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(
        **preparer(
            REGRESSION_RESEAU_NEURONES, {"hidden_layer_sizes": (50,), "max_iter": 500}, parametres, graine
        )
    )


def _fabrique_classification_reseau_neurones(graine: int, **parametres: Any) -> Any:
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        **preparer(
            CLASSIFICATION_RESEAU_NEURONES, {"hidden_layer_sizes": (50,), "max_iter": 500}, parametres, graine
        )
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_RESEAU_NEURONES,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_reseau_neurones,
        grille_bornee=(("hidden_layer_sizes", ((50,), (100,), (50, 50))), ("alpha", (0.0001, 0.001, 0.01))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_RESEAU_NEURONES),
        texte_id=REGRESSION_RESEAU_NEURONES,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_RESEAU_NEURONES,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_reseau_neurones,
        grille_bornee=(("hidden_layer_sizes", ((50,), (100,), (50, 50))), ("alpha", (0.0001, 0.001, 0.01))),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_RESEAU_NEURONES),
        texte_id=CLASSIFICATION_RESEAU_NEURONES,
    ),
)
