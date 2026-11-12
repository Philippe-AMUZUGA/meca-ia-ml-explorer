"""Analyse discriminante linéaire : frontières linéaires entre classes."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

CLASSIFICATION_ANALYSE_DISCRIMINANTE = "classification.analyse_discriminante"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return LinearDiscriminantAnalysis(
        **preparer(CLASSIFICATION_ANALYSE_DISCRIMINANTE, {"solver": "svd"}, parametres)
    )


ENTREES = (
    EntreeCatalogue(
        id=CLASSIFICATION_ANALYSE_DISCRIMINANTE,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("solver", ("svd", "lsqr")),),
        cout=Cout.LEGER,
        ref_livre=ref(CLASSIFICATION_ANALYSE_DISCRIMINANTE),
        texte_id=CLASSIFICATION_ANALYSE_DISCRIMINANTE,
    ),
)
