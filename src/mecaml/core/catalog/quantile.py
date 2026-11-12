"""Régression quantile : modélise un quantile de la cible (la médiane par défaut)."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_QUANTILE = "regression.quantile"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.linear_model import QuantileRegressor

    return QuantileRegressor(
        **preparer(REGRESSION_QUANTILE, {"quantile": 0.5, "alpha": 0.0, "solver": "highs"}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_QUANTILE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(("quantile", (0.1, 0.5, 0.9)), ("alpha", (0.0, 0.01, 0.1))),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_QUANTILE),
        texte_id=REGRESSION_QUANTILE,
    ),
)
