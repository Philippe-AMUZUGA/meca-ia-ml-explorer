"""Standardisation : moyenne nulle et écart type unité."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_STANDARD = "transformation.standard"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import StandardScaler

    return StandardScaler(**preparer(TRANSFORMATION_STANDARD, {}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_STANDARD,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_STANDARD),
        texte_id=TRANSFORMATION_STANDARD,
    ),
)
