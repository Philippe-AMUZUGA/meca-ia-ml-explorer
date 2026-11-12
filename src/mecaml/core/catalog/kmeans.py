"""K moyennes : regroupement en k groupes compacts."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_KMEANS = "sans_cible.kmeans"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.cluster import KMeans

    return KMeans(**preparer(SANS_CIBLE_KMEANS, {"n_clusters": 3, "n_init": 10}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_KMEANS,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_clusters", (2, 3, 4, 5, 6, 8)),),
        cout=Cout.LEGER,
        ref_livre=ref(SANS_CIBLE_KMEANS),
        texte_id=SANS_CIBLE_KMEANS,
    ),
)
