"""Classification hiérarchique ascendante : fusions successives des groupes les plus proches."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_CLASSIFICATION_HIERARCHIQUE = "sans_cible.classification_hierarchique"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(
        **preparer(SANS_CIBLE_CLASSIFICATION_HIERARCHIQUE, {"n_clusters": 3, "linkage": "ward"}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_CLASSIFICATION_HIERARCHIQUE,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_clusters", (2, 3, 4, 5, 6, 8)), ("linkage", ("ward", "average", "complete"))),
        cout=Cout.LOURD,
        ref_livre=ref(SANS_CIBLE_CLASSIFICATION_HIERARCHIQUE),
        texte_id=SANS_CIBLE_CLASSIFICATION_HIERARCHIQUE,
    ),
)
