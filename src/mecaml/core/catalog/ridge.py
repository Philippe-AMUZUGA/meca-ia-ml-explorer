"""Régression de crête (Ridge) : pénalisation L2 des coefficients."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_RIDGE = "regression.ridge"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(**preparer(REGRESSION_RIDGE, {"alpha": 1.0}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_RIDGE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(("alpha", (0.01, 0.1, 1.0, 10.0, 100.0)),),
        cout=Cout.LEGER,
        ref_livre=ref(REGRESSION_RIDGE),
        texte_id=REGRESSION_RIDGE,
    ),
)
