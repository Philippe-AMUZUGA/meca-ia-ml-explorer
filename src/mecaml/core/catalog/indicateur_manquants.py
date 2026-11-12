"""Indicateur de valeur manquante : une colonne 0 ou 1 par variable."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

MANQUANTS_INDICATEUR = "manquants.indicateur"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.impute import MissingIndicator

    return MissingIndicator(**preparer(MANQUANTS_INDICATEUR, {"features": "all"}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=MANQUANTS_INDICATEUR,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(MANQUANTS_INDICATEUR),
        texte_id=MANQUANTS_INDICATEUR,
    ),
)
