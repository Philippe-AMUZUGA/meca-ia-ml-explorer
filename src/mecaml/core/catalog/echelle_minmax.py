"""Mise à l'échelle min max : valeurs ramenées dans l'intervalle de 0 à 1."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_MINMAX = "transformation.minmax"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import MinMaxScaler

    return MinMaxScaler(**preparer(TRANSFORMATION_MINMAX, {}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_MINMAX,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_MINMAX),
        texte_id=TRANSFORMATION_MINMAX,
    ),
)
