"""Importance des variables par permutation : baisse du score quand une colonne est mélangée."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

EXPLICATION_PERMUTATION = "explication.permutation"


def _fabrique(graine: int, **parametres: Any) -> Callable[..., pd.DataFrame]:
    """Rend `calculer(pipeline, x, y, scoring=...)`, tableau trié par importance décroissante.

    Les colonnes de `x` sont les colonnes d'entrée du pipeline, avant tout prétraitement : le mélange
    d'une colonne se répercute donc à travers l'encodage, la mise à l'échelle et le modèle.
    """
    reglages = preparer(EXPLICATION_PERMUTATION, {"n_repeats": 10}, parametres)

    def calculer(pipeline: Any, x: pd.DataFrame, y: Any, *, scoring: Any = None) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance

        resultat = permutation_importance(
            pipeline,
            x,
            y,
            scoring=scoring,
            n_repeats=int(reglages["n_repeats"]),
            random_state=graine,
            n_jobs=1,
        )
        tableau = pd.DataFrame(
            {
                "variable": list(x.columns),
                "moyenne": resultat.importances_mean,
                "ecart_type": resultat.importances_std,
            }
        )
        return tableau.sort_values("moyenne", ascending=False, ignore_index=True)

    return calculer


ENTREES = (
    EntreeCatalogue(
        id=EXPLICATION_PERMUTATION,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("n_repeats", (5, 10, 20)),),
        cout=Cout.MOYEN,
        ref_livre=ref(EXPLICATION_PERMUTATION),
        texte_id=EXPLICATION_PERMUTATION,
    ),
)
