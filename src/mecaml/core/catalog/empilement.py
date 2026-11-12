"""Empilement (stacking) : un modèle final apprend à combiner les prédictions de modèles de base."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.catalog.vote import BASE_PAR_DEFAUT, modele_de_base, sous_estimateurs
from mecaml.core.types import Probleme

CLASSIFICATION_EMPILEMENT = "classification.empilement"
REGRESSION_EMPILEMENT = "regression.empilement"


def _fabrique_classification_empilement(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import StackingClassifier

    defauts = {
        "base": BASE_PAR_DEFAUT[Probleme.CLASSIFICATION],
        "final": "classification.logistique",
        "cv": 3,
    }
    arguments = preparer(CLASSIFICATION_EMPILEMENT, defauts, parametres)
    base = sous_estimateurs(CLASSIFICATION_EMPILEMENT, arguments.pop("base"), Probleme.CLASSIFICATION, graine)
    final = modele_de_base(CLASSIFICATION_EMPILEMENT, arguments.pop("final"), Probleme.CLASSIFICATION, graine)
    return StackingClassifier(estimators=base, final_estimator=final, n_jobs=1, **arguments)


def _fabrique_regression_empilement(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import StackingRegressor

    defauts = {"base": BASE_PAR_DEFAUT[Probleme.REGRESSION], "final": "regression.ridge", "cv": 3}
    arguments = preparer(REGRESSION_EMPILEMENT, defauts, parametres)
    base = sous_estimateurs(REGRESSION_EMPILEMENT, arguments.pop("base"), Probleme.REGRESSION, graine)
    final = modele_de_base(REGRESSION_EMPILEMENT, arguments.pop("final"), Probleme.REGRESSION, graine)
    return StackingRegressor(estimators=base, final_estimator=final, n_jobs=1, **arguments)


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_EMPILEMENT,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_empilement,
        grille_bornee=(("passthrough", (False, True)),),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_EMPILEMENT),
        texte_id=REGRESSION_EMPILEMENT,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_EMPILEMENT,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_empilement,
        grille_bornee=(("passthrough", (False, True)),),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_EMPILEMENT),
        texte_id=CLASSIFICATION_EMPILEMENT,
    ),
)
