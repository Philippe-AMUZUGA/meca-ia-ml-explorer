"""Encodage ordinal : un entier par modalité, modalité inconnue codée par -1."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

ENCODAGE_ORDINAL = "encodage.ordinal"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.preprocessing import OrdinalEncoder

    return OrdinalEncoder(
        **preparer(
            ENCODAGE_ORDINAL,
            {"handle_unknown": "use_encoded_value", "unknown_value": -1, "encoded_missing_value": -1},
            parametres,
        )
    )


ENTREES = (
    EntreeCatalogue(
        id=ENCODAGE_ORDINAL,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(ENCODAGE_ORDINAL),
        texte_id=ENCODAGE_ORDINAL,
    ),
)
