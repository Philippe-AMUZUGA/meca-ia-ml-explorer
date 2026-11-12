"""Machine à vecteurs de support, en régression (SVR) et en classification (SVC)."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REGRESSION_SVR = "regression.svr"
CLASSIFICATION_SVC = "classification.svc"


def _fabrique_regression_svr(graine: int, **parametres: Any) -> Any:
    from sklearn.svm import SVR

    return SVR(**preparer(REGRESSION_SVR, {"C": 1.0, "kernel": "rbf"}, parametres))


def _fabrique_classification_svc(graine: int, **parametres: Any) -> Any:
    from sklearn.svm import SVC

    return SVC(**preparer(CLASSIFICATION_SVC, {"C": 1.0, "kernel": "rbf"}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_SVR,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_svr,
        grille_bornee=(("C", (0.1, 1.0, 10.0)), ("epsilon", (0.01, 0.1, 1.0))),
        cout=Cout.LOURD,
        ref_livre=ref(REGRESSION_SVR),
        texte_id=REGRESSION_SVR,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_SVC,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_svc,
        grille_bornee=(("C", (0.1, 1.0, 10.0, 100.0)), ("class_weight", (None, "balanced"))),
        cout=Cout.LOURD,
        ref_livre=ref(CLASSIFICATION_SVC),
        texte_id=CLASSIFICATION_SVC,
    ),
)
