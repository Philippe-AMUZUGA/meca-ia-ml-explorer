"""Validation en fenêtre croissante : l'apprentissage précède toujours la validation."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_SERIE_TEMPORELLE = "validation.serie_temporelle"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import TimeSeriesSplit

    return TimeSeriesSplit(**preparer(VALIDATION_SERIE_TEMPORELLE, {"n_splits": 5}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_SERIE_TEMPORELLE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("n_splits", (3, 5, 10)), ("gap", (0, 5, 10))),
        cout=Cout.LEGER,
        ref_livre=ref(VALIDATION_SERIE_TEMPORELLE),
        texte_id=VALIDATION_SERIE_TEMPORELLE,
    ),
)
