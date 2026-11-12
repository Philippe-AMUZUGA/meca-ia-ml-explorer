"""Recherche exhaustive d'hyperparamètres sur une grille bornée (GridSearchCV)."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import exiger, preparer, verifier_grille
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_RECHERCHE_GRILLE = "validation.recherche_grille"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import GridSearchCV

    exiger(VALIDATION_RECHERCHE_GRILLE, parametres, ("estimator", "param_grid"))
    verifier_grille(VALIDATION_RECHERCHE_GRILLE, parametres["param_grid"])
    return GridSearchCV(**preparer(VALIDATION_RECHERCHE_GRILLE, {"cv": 5, "n_jobs": 1}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_RECHERCHE_GRILLE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("cv", (3, 5, 10)),),
        cout=Cout.MOYEN,
        ref_livre=ref(VALIDATION_RECHERCHE_GRILLE),
        texte_id=VALIDATION_RECHERCHE_GRILLE,
    ),
)
