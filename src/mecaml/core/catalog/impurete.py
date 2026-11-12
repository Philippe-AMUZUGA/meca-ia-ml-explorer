"""Importance des variables par impureté : la part de la réduction d'impureté que chaque variable apporte.

Disponible pour les arbres, les forêts, les arbres extrêmes et le gradient boosting classique. Calculée sur
l'apprentissage, elle favorise les variables à nombreuses valeurs : la confirmer par la permutation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from mecaml.core.catalog._outils import noms_des_variables
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

EXPLICATION_IMPURETE = "explication.impurete"


def _fabrique(graine: int, **parametres: Any) -> Callable[..., pd.DataFrame]:
    """Rend `calculer(modele_ajuste)`, tableau (variable, importance) trié par importance décroissante."""
    if parametres:
        raise ValueError(f"{EXPLICATION_IMPURETE} : aucun paramètre.")

    def calculer(modele: Any) -> pd.DataFrame:
        final = modele[-1] if hasattr(modele, "steps") else modele
        importances = getattr(final, "feature_importances_", None)
        if importances is None:
            raise ValueError(
                f"{EXPLICATION_IMPURETE} : le modèle final n'a pas d'importance par impureté "
                "(arbre, forêt, arbres extrêmes, gradient boosting) ; utiliser la permutation."
            )
        noms = noms_des_variables(modele) if hasattr(modele, "steps") else None
        noms = noms or [f"x{k}" for k in range(len(importances))]
        tableau = pd.DataFrame({"variable": noms, "importance": importances})
        return tableau.sort_values("importance", ascending=False, ignore_index=True)

    return calculer


ENTREES = (
    EntreeCatalogue(
        id=EXPLICATION_IMPURETE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(EXPLICATION_IMPURETE),
        texte_id=EXPLICATION_IMPURETE,
    ),
)
