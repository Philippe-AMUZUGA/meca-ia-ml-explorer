"""Encodage « one hot » : une colonne 0 ou 1 par modalité, modalités rares regroupées."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

ENCODAGE_ONE_HOT = "encodage.one_hot"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder(
        **preparer(
            ENCODAGE_ONE_HOT,
            {"handle_unknown": "infrequent_if_exist", "sparse_output": False, "max_categories": 50},
            parametres,
        )
    )


ENTREES = (
    EntreeCatalogue(
        id=ENCODAGE_ONE_HOT,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(ENCODAGE_ONE_HOT),
        texte_id=ENCODAGE_ONE_HOT,
    ),
)
