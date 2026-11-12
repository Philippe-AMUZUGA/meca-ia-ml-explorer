"""Imputation simple des valeurs manquantes (médiane, moyenne ou valeur la plus fréquente)."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

MANQUANTS_SIMPLE = "manquants.simple"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.impute import SimpleImputer

    return SimpleImputer(**preparer(MANQUANTS_SIMPLE, {"strategy": "median"}, parametres))


ENTREES = (
    EntreeCatalogue(
        id=MANQUANTS_SIMPLE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("strategy", ("mean", "median", "most_frequent")),),
        cout=Cout.LEGER,
        ref_livre=ref(MANQUANTS_SIMPLE),
        texte_id=MANQUANTS_SIMPLE,
    ),
)
