"""Discrétisation d'une variable continue en classes."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_DISCRETISATION = "transformation.discretisation"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import KBinsDiscretizer

    return KBinsDiscretizer(
        **preparer(
            TRANSFORMATION_DISCRETISATION,
            {"n_bins": 5, "encode": "ordinal", "strategy": "quantile"},
            parametres,
            graine,
        )
    )


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_DISCRETISATION,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_bins", (3, 5, 8, 10)), ("strategy", ("quantile", "uniform"))),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_DISCRETISATION),
        texte_id=TRANSFORMATION_DISCRETISATION,
    ),
)
