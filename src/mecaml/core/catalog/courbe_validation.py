"""Courbe de validation : score d'apprentissage et de validation selon la valeur d'un hyperparamètre."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_COURBE_VALIDATION = "validation.courbe_validation"
VALEURS_MAX = 20


def _fabrique(graine: int, **parametres: Any) -> Callable[..., dict[str, Any]]:
    """Rend `calculer(modele, x, y, nom=..., valeurs=..., cv=...)`, qui rend valeurs, app et val."""
    if parametres:
        raise ValueError(f"{VALIDATION_COURBE_VALIDATION} : aucun paramètre à la construction.")

    def calculer(
        modele: Any,
        x: Any,
        y: Any,
        *,
        nom: str,
        valeurs: Any,
        cv: Any,
        scoring: Any = None,
        groups: Any = None,
    ) -> dict[str, Any]:
        import numpy as np
        from sklearn.model_selection import validation_curve

        valeurs = list(valeurs)
        if not 2 <= len(valeurs) <= VALEURS_MAX:
            raise ValueError(
                f"{VALIDATION_COURBE_VALIDATION} : de 2 à {VALEURS_MAX} valeurs attendues, "
                f"{len(valeurs)} reçues."
            )
        app, val = validation_curve(
            modele,
            x,
            y,
            param_name=nom,
            param_range=valeurs,
            cv=cv,
            scoring=scoring,
            groups=groups,
            n_jobs=1,
            error_score=np.nan,
        )
        return {"valeurs": valeurs, "app": app, "val": val}

    return calculer


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_COURBE_VALIDATION,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.MOYEN,
        ref_livre=ref(VALIDATION_COURBE_VALIDATION),
        texte_id=VALIDATION_COURBE_VALIDATION,
    ),
)
