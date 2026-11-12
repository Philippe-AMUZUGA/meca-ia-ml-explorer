"""Élimination récursive de variables avec validation croisée interne (RFECV)."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer, retirer_probleme
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SELECTION_RFECV = "selection.rfecv"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression, Ridge

    probleme, reste = retirer_probleme(parametres, SELECTION_RFECV)
    base = LogisticRegression(max_iter=1000) if probleme == "classification" else Ridge()
    arguments = preparer(SELECTION_RFECV, {"step": 1, "cv": 3, "min_features_to_select": 1}, reste)
    return RFECV(base, **arguments)


ENTREES = (
    EntreeCatalogue(
        id=SELECTION_RFECV,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("min_features_to_select", (1, 3, 5)), ("step", (1, 2))),
        cout=Cout.MOYEN,
        ref_livre=ref(SELECTION_RFECV),
        texte_id=SELECTION_RFECV,
    ),
)
