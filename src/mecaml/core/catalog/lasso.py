"""Régression Lasso : pénalisation L1, certains coefficients deviennent nuls."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_LASSO = "regression.lasso"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.linear_model import Lasso

    return Lasso(**preparer(REGRESSION_LASSO, {"alpha": 0.1}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_LASSO,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(("alpha", (0.001, 0.01, 0.1, 1.0)),),
        cout=Cout.LEGER,
        ref_livre=ref(REGRESSION_LASSO),
        texte_id=REGRESSION_LASSO,
    ),
)
