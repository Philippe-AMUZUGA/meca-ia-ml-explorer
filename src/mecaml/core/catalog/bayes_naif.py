"""Classifieur bayésien naïf gaussien : une loi normale par variable et par classe."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

CLASSIFICATION_BAYES_NAIF = "classification.bayes_naif"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB(**preparer(CLASSIFICATION_BAYES_NAIF, {}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=CLASSIFICATION_BAYES_NAIF,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("var_smoothing", (1e-9, 1e-8, 1e-7, 1e-6)),),
        cout=Cout.LEGER,
        ref_livre=ref(CLASSIFICATION_BAYES_NAIF),
        texte_id=CLASSIFICATION_BAYES_NAIF,
    ),
)
