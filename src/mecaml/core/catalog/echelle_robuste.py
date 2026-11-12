"""Mise à l'échelle robuste : médiane et écart interquartile, peu sensible aux valeurs extrêmes."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_ROBUSTE = "transformation.robuste"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import RobustScaler

    return RobustScaler(**preparer(TRANSFORMATION_ROBUSTE, {}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_ROBUSTE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_ROBUSTE),
        texte_id=TRANSFORMATION_ROBUSTE,
    ),
)
