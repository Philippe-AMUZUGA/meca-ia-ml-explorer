"""SVM à une classe : frontière autour des données normales."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_SVM_UNE_CLASSE = "sans_cible.svm_une_classe"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.svm import OneClassSVM

    return OneClassSVM(
        **preparer(SANS_CIBLE_SVM_UNE_CLASSE, {"nu": 0.05, "kernel": "rbf", "gamma": "scale"}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_SVM_UNE_CLASSE,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("nu", (0.01, 0.05, 0.1)), ("gamma", ("scale", "auto"))),
        cout=Cout.LOURD,
        ref_livre=ref(SANS_CIBLE_SVM_UNE_CLASSE),
        texte_id=SANS_CIBLE_SVM_UNE_CLASSE,
    ),
)
