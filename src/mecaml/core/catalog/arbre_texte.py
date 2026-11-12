"""Arbre de décision écrit en texte : les règles se lisent de haut en bas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mecaml.core.catalog._outils import noms_des_variables, preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

EXPLICATION_ARBRE_TEXTE = "explication.arbre_texte"


def _fabrique(graine: int, **parametres: Any) -> Callable[..., str]:
    """Rend `calculer(modele_ajuste)`, texte des règles ; le modèle final doit être un arbre de décision."""
    reglages = preparer(EXPLICATION_ARBRE_TEXTE, {"max_depth": 4}, parametres)

    def calculer(modele: Any) -> str:
        from sklearn.tree import export_text

        est_pipeline = hasattr(modele, "steps")
        arbre = modele[-1] if est_pipeline else modele
        if not hasattr(arbre, "tree_"):
            raise ValueError(
                f"{EXPLICATION_ARBRE_TEXTE} : le modèle final doit être un arbre de décision ajusté."
            )
        noms = noms_des_variables(modele) if est_pipeline else None
        return export_text(arbre, feature_names=noms, max_depth=int(reglages["max_depth"]))

    return calculer


ENTREES = (
    EntreeCatalogue(
        id=EXPLICATION_ARBRE_TEXTE,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("max_depth", (2, 3, 4, 6)),),
        cout=Cout.LEGER,
        ref_livre=ref(EXPLICATION_ARBRE_TEXTE),
        texte_id=EXPLICATION_ARBRE_TEXTE,
    ),
)
