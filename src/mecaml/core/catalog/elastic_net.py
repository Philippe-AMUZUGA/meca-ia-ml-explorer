"""Régression Elastic Net : pénalisation mixte L1 et L2 des coefficients."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_ELASTIC_NET = "regression.elastic_net"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.linear_model import ElasticNet

    return ElasticNet(**preparer(REGRESSION_ELASTIC_NET, {"alpha": 0.1, "l1_ratio": 0.5}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_ELASTIC_NET,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(("alpha", (0.001, 0.01, 0.1, 1.0)), ("l1_ratio", (0.2, 0.5, 0.8))),
        cout=Cout.LEGER,
        ref_livre=ref(REGRESSION_ELASTIC_NET),
        texte_id=REGRESSION_ELASTIC_NET,
    ),
)
