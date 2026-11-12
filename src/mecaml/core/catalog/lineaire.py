"""Régression linéaire par moindres carrés ordinaires."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_LINEAIRE = "regression.lineaire"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.linear_model import LinearRegression

    return LinearRegression(**preparer(REGRESSION_LINEAIRE, {}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_LINEAIRE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(REGRESSION_LINEAIRE),
        texte_id=REGRESSION_LINEAIRE,
    ),
)
