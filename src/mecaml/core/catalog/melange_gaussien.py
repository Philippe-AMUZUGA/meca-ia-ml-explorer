"""Mélange de lois normales : regroupement souple, un groupe par loi."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_MELANGE_GAUSSIEN = "sans_cible.melange_gaussien"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(**preparer(SANS_CIBLE_MELANGE_GAUSSIEN, {"n_components": 3}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_MELANGE_GAUSSIEN,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_components", (2, 3, 4, 5, 6, 8)), ("covariance_type", ("full", "diag"))),
        cout=Cout.LEGER,
        ref_livre=ref(SANS_CIBLE_MELANGE_GAUSSIEN),
        texte_id=SANS_CIBLE_MELANGE_GAUSSIEN,
    ),
)
