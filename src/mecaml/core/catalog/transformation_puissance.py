"""Transformation de puissance de Yeo et Johnson pour symétriser une distribution."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_PUISSANCE = "transformation.puissance"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import PowerTransformer

    return PowerTransformer(**preparer(TRANSFORMATION_PUISSANCE, {"method": "yeo-johnson"}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_PUISSANCE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_PUISSANCE),
        texte_id=TRANSFORMATION_PUISSANCE,
    ),
)
