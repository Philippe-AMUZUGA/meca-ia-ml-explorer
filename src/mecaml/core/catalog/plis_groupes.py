"""Validation par groupes : une machine ou un lot n'est jamais à cheval sur deux plis."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

VALIDATION_PLIS_GROUPES = "validation.plis_groupes"
VALIDATION_PLIS_GROUPES_STRATIFIES = "validation.plis_groupes_stratifies"
VALIDATION_GROUPES_ALEATOIRES = "validation.groupes_aleatoires"


def _fabrique_validation_plis_groupes(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import GroupKFold

    return GroupKFold(
        **preparer(VALIDATION_PLIS_GROUPES, {"n_splits": 5, "shuffle": True}, parametres, graine)
    )


def _fabrique_validation_plis_groupes_stratifies(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(
        **preparer(VALIDATION_PLIS_GROUPES_STRATIFIES, {"n_splits": 5, "shuffle": True}, parametres, graine)
    )


def _fabrique_validation_groupes_aleatoires(graine: int, **parametres: Any) -> Any:
    from sklearn.model_selection import GroupShuffleSplit

    return GroupShuffleSplit(
        **preparer(VALIDATION_GROUPES_ALEATOIRES, {"n_splits": 5, "test_size": 0.2}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=VALIDATION_PLIS_GROUPES,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique_validation_plis_groupes,
        grille_bornee=(("n_splits", (3, 5, 10)),),
        cout=Cout.LEGER,
        ref_livre=ref(VALIDATION_PLIS_GROUPES),
        texte_id=VALIDATION_PLIS_GROUPES,
    ),
    EntreeCatalogue(
        id=VALIDATION_PLIS_GROUPES_STRATIFIES,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_validation_plis_groupes_stratifies,
        grille_bornee=(("n_splits", (3, 5, 10)),),
        cout=Cout.LEGER,
        ref_livre=ref(VALIDATION_PLIS_GROUPES_STRATIFIES),
        texte_id=VALIDATION_PLIS_GROUPES_STRATIFIES,
    ),
    EntreeCatalogue(
        id=VALIDATION_GROUPES_ALEATOIRES,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique_validation_groupes_aleatoires,
        grille_bornee=(("n_splits", (3, 5, 10)), ("test_size", (0.2, 0.3))),
        cout=Cout.LEGER,
        ref_livre=ref(VALIDATION_GROUPES_ALEATOIRES),
        texte_id=VALIDATION_GROUPES_ALEATOIRES,
    ),
)
