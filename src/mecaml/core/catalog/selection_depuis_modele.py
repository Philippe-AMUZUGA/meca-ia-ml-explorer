"""Sélection par l'importance des variables d'une forêt aléatoire."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer, retirer_probleme
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SELECTION_DEPUIS_MODELE = "selection.depuis_modele"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    probleme, reste = retirer_probleme(parametres, SELECTION_DEPUIS_MODELE)
    foret = RandomForestClassifier if probleme == "classification" else RandomForestRegressor
    base = foret(n_estimators=50, max_depth=8, n_jobs=1, random_state=graine)
    arguments = preparer(SELECTION_DEPUIS_MODELE, {"threshold": "median"}, reste)
    return SelectFromModel(base, **arguments)


ENTREES = (
    EntreeCatalogue(
        id=SELECTION_DEPUIS_MODELE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("threshold", ("median", "mean", "1.25*mean")),),
        cout=Cout.MOYEN,
        ref_livre=ref(SELECTION_DEPUIS_MODELE),
        texte_id=SELECTION_DEPUIS_MODELE,
    ),
)
