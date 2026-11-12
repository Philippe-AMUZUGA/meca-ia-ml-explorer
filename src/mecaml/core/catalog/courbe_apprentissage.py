"""Courbe d'apprentissage : score d'apprentissage et de validation selon la taille de l'échantillon."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_COURBE_APPRENTISSAGE = "validation.courbe_apprentissage"


def _fabrique(graine: int, **parametres: Any) -> Callable[..., dict[str, Any]]:
    """Rend `calculer(modele, x, y, cv=..., scoring=..., groups=...)`, qui rend tailles, app et val."""
    reglages = preparer(VALIDATION_COURBE_APPRENTISSAGE, {"n_points": 8, "fraction_min": 0.1}, parametres)

    def calculer(
        modele: Any, x: Any, y: Any, *, cv: Any, scoring: Any = None, groups: Any = None
    ) -> dict[str, Any]:
        import numpy as np
        from sklearn.model_selection import learning_curve

        tailles, app, val = learning_curve(
            modele,
            x,
            y,
            train_sizes=np.linspace(reglages["fraction_min"], 1.0, int(reglages["n_points"])),
            cv=cv,
            scoring=scoring,
            groups=groups,
            n_jobs=1,
            error_score=np.nan,
        )
        return {"tailles": tailles, "app": app, "val": val}

    return calculer


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_COURBE_APPRENTISSAGE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("n_points", (5, 8, 12)),),
        cout=Cout.MOYEN,
        ref_livre=ref(VALIDATION_COURBE_APPRENTISSAGE),
        texte_id=VALIDATION_COURBE_APPRENTISSAGE,
    ),
)
