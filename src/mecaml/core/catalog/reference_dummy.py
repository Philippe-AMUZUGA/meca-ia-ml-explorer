"""Référence naïve : prédit la moyenne ou la classe majoritaire, sans lire les variables."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

REFERENCE_DUMMY_REGRESSION = "reference.dummy_regression"
REFERENCE_DUMMY_CLASSIFICATION = "reference.dummy_classification"


def _fabrique_reference_dummy_regression(graine: int, **parametres: Any) -> Any:
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(**preparer(REFERENCE_DUMMY_REGRESSION, {"strategy": "mean"}, parametres))


def _fabrique_reference_dummy_classification(graine: int, **parametres: Any) -> Any:
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(
        **preparer(REFERENCE_DUMMY_CLASSIFICATION, {"strategy": "prior"}, parametres, graine)
    )


ENTREES = (
    EntreeCatalogue(
        id=REFERENCE_DUMMY_REGRESSION,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_reference_dummy_regression,
        grille_bornee=(("strategy", ("mean", "median")),),
        cout=Cout.LEGER,
        ref_livre=ref(REFERENCE_DUMMY_REGRESSION),
        texte_id=REFERENCE_DUMMY_REGRESSION,
    ),
    EntreeCatalogue(
        id=REFERENCE_DUMMY_CLASSIFICATION,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_reference_dummy_classification,
        grille_bornee=(("strategy", ("prior", "most_frequent", "stratified")),),
        cout=Cout.LEGER,
        ref_livre=ref(REFERENCE_DUMMY_CLASSIFICATION),
        texte_id=REFERENCE_DUMMY_CLASSIFICATION,
    ),
)
