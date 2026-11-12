"""DBSCAN : regroupement par densité, les points isolés forment le bruit."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_DBSCAN = "sans_cible.dbscan"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.cluster import DBSCAN

    return DBSCAN(**preparer(SANS_CIBLE_DBSCAN, {"eps": 0.5, "min_samples": 5}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_DBSCAN,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("eps", (0.3, 0.5, 0.8, 1.2, 2.0)), ("min_samples", (3, 5, 10))),
        cout=Cout.LEGER,
        ref_livre=ref(SANS_CIBLE_DBSCAN),
        texte_id=SANS_CIBLE_DBSCAN,
    ),
)
