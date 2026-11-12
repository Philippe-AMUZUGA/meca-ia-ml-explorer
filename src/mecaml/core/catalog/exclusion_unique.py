"""Validation par exclusion d'une ligne (leave one out), réservée aux petits jeux."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_EXCLUSION_UNIQUE = "validation.exclusion_unique"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import LeaveOneOut

    return LeaveOneOut(**preparer(VALIDATION_EXCLUSION_UNIQUE, {}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_EXCLUSION_UNIQUE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LOURD,
        ref_livre=ref(VALIDATION_EXCLUSION_UNIQUE),
        texte_id=VALIDATION_EXCLUSION_UNIQUE,
    ),
)
