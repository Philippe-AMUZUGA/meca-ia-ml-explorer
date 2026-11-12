"""Validation croisée en k plis, simple et stratifiée."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_KFOLD = "validation.kfold"
VALIDATION_KFOLD_STRATIFIE = "validation.kfold_stratifie"


def _fabrique_validation_kfold(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import KFold

    return KFold(**preparer(VALIDATION_KFOLD, {"n_splits": 5, "shuffle": True}, parametres, graine))


def _fabrique_validation_kfold_stratifie(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(
        **preparer(VALIDATION_KFOLD_STRATIFIE, {"n_splits": 5, "shuffle": True}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_KFOLD,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique_validation_kfold,
        grille_bornee=(("n_splits", (3, 5, 10)),),
        cout=Cout.LEGER,
        ref_livre=ref(VALIDATION_KFOLD),
        texte_id=VALIDATION_KFOLD,
    ),
    EntreeCatalogue(
        id=VALIDATION_KFOLD_STRATIFIE,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_validation_kfold_stratifie,
        grille_bornee=(("n_splits", (3, 5, 10)),),
        cout=Cout.LEGER,
        ref_livre=ref(VALIDATION_KFOLD_STRATIFIE),
        texte_id=VALIDATION_KFOLD_STRATIFIE,
    ),
)
