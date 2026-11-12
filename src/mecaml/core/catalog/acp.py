"""Analyse en composantes principales : réduction de dimension."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_ACP = "sans_cible.acp"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.decomposition import PCA

    return PCA(**preparer(SANS_CIBLE_ACP, {}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_ACP,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_components", (2, 3, 5, 0.9, 0.95)),),
        cout=Cout.LEGER,
        ref_livre=ref(SANS_CIBLE_ACP),
        texte_id=SANS_CIBLE_ACP,
    ),
)
