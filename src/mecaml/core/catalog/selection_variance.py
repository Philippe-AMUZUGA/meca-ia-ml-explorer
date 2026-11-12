"""Sélection par seuil de variance : écarte les variables presque constantes."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer, retirer_probleme
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SELECTION_VARIANCE = "selection.variance"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.feature_selection import VarianceThreshold

    _, reste = retirer_probleme(parametres, SELECTION_VARIANCE)
    return VarianceThreshold(**preparer(SELECTION_VARIANCE, {"threshold": 0.0}, reste))


ENTREES = (
    EntreeCatalogue(
        id=SELECTION_VARIANCE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("threshold", (0.0, 0.01, 0.1, 1.0)),),
        cout=Cout.LEGER,
        ref_livre=ref(SELECTION_VARIANCE),
        texte_id=SELECTION_VARIANCE,
    ),
)
