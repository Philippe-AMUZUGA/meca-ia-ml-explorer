"""Imputation des valeurs manquantes par les plus proches voisins."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

MANQUANTS_KNN = "manquants.knn"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.impute import KNNImputer

    return KNNImputer(**preparer(MANQUANTS_KNN, {"n_neighbors": 5}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=MANQUANTS_KNN,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_neighbors", (3, 5, 10)),),
        cout=Cout.MOYEN,
        ref_livre=ref(MANQUANTS_KNN),
        texte_id=MANQUANTS_KNN,
    ),
)
