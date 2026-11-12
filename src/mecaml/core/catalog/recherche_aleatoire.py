"""Recherche aléatoire d'hyperparamètres, nombre de tirages borné (RandomizedSearchCV)."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import exiger, preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_RECHERCHE_ALEATOIRE = "validation.recherche_aleatoire"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import RandomizedSearchCV

    exiger(VALIDATION_RECHERCHE_ALEATOIRE, parametres, ("estimator", "param_distributions"))
    defauts = {"cv": 5, "n_iter": 20, "n_jobs": 1}
    return RandomizedSearchCV(**preparer(VALIDATION_RECHERCHE_ALEATOIRE, defauts, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_RECHERCHE_ALEATOIRE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("n_iter", (10, 20, 30)), ("cv", (3, 5))),
        cout=Cout.MOYEN,
        ref_livre=ref(VALIDATION_RECHERCHE_ALEATOIRE),
        texte_id=VALIDATION_RECHERCHE_ALEATOIRE,
    ),
)
