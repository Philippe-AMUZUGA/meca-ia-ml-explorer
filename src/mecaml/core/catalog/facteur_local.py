"""Facteur d'anomalie local : détection d'anomalies par densité relative aux voisins."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_FACTEUR_LOCAL = "sans_cible.facteur_local"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.neighbors import LocalOutlierFactor

    return LocalOutlierFactor(
        **preparer(SANS_CIBLE_FACTEUR_LOCAL, {"n_neighbors": 20, "contamination": "auto"}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_FACTEUR_LOCAL,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("n_neighbors", (10, 20, 35)), ("contamination", (0.01, 0.05, 0.1))),
        cout=Cout.LEGER,
        ref_livre=ref(SANS_CIBLE_FACTEUR_LOCAL),
        texte_id=SANS_CIBLE_FACTEUR_LOCAL,
    ),
)
