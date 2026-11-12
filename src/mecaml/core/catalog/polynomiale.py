"""Expansion polynomiale : puissances et produits des variables, degré borné."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_POLYNOMIALE = "transformation.polynomiale"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import PolynomialFeatures

    return PolynomialFeatures(
        **preparer(TRANSFORMATION_POLYNOMIALE, {"degree": 2, "include_bias": False}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_POLYNOMIALE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("degree", (1, 2, 3)),),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_POLYNOMIALE),
        texte_id=TRANSFORMATION_POLYNOMIALE,
    ),
)
