"""k plus proches voisins, en régression et en classification."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_PLUS_PROCHES_VOISINS = "regression.plus_proches_voisins"
CLASSIFICATION_PLUS_PROCHES_VOISINS = "classification.plus_proches_voisins"


def _fabrique_regression_plus_proches_voisins(graine: int, **parametres: Any) -> Any:
    from sklearn.neighbors import KNeighborsRegressor

    return KNeighborsRegressor(**preparer(REGRESSION_PLUS_PROCHES_VOISINS, {"n_neighbors": 5}, parametres))


def _fabrique_classification_plus_proches_voisins(graine: int, **parametres: Any) -> Any:
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(
        **preparer(CLASSIFICATION_PLUS_PROCHES_VOISINS, {"n_neighbors": 5}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_PLUS_PROCHES_VOISINS,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_plus_proches_voisins,
        grille_bornee=(("n_neighbors", (3, 5, 9, 15)), ("weights", ("uniform", "distance"))),
        cout=Cout.LEGER,
        ref_livre=ref(REGRESSION_PLUS_PROCHES_VOISINS),
        texte_id=REGRESSION_PLUS_PROCHES_VOISINS,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_PLUS_PROCHES_VOISINS,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_plus_proches_voisins,
        grille_bornee=(("n_neighbors", (3, 5, 9, 15)), ("weights", ("uniform", "distance"))),
        cout=Cout.LEGER,
        ref_livre=ref(CLASSIFICATION_PLUS_PROCHES_VOISINS),
        texte_id=CLASSIFICATION_PLUS_PROCHES_VOISINS,
    ),
)
