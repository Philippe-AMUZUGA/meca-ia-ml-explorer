"""Régression logistique, avec pondération possible des classes."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

CLASSIFICATION_LOGISTIQUE = "classification.logistique"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**preparer(CLASSIFICATION_LOGISTIQUE, {"max_iter": 1000}, parametres, graine))


ENTREES = (
    EntreeCatalogue(
        id=CLASSIFICATION_LOGISTIQUE,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("C", (0.01, 0.1, 1.0, 10.0, 100.0)), ("class_weight", (None, "balanced"))),
        cout=Cout.LEGER,
        ref_livre=ref(CLASSIFICATION_LOGISTIQUE),
        texte_id=CLASSIFICATION_LOGISTIQUE,
    ),
)
