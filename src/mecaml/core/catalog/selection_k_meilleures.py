"""Sélection des k meilleures variables selon un test univarié ou l'information mutuelle."""

from __future__ import annotations

from functools import partial
from typing import Any

from mecaml.core.catalog._outils import preparer, retirer_probleme
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SELECTION_K_MEILLEURES = "selection.k_meilleures"
SCORES = ("f", "information_mutuelle")


def _fonction_de_score(score: str, probleme: str, graine: int) -> Any:
    from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

    if score not in SCORES:
        raise ValueError(f"{SELECTION_K_MEILLEURES} : score {score!r} inconnu, choisir parmi {SCORES}.")
    classification = probleme == "classification"
    if score == "f":
        return f_classif if classification else f_regression
    information = mutual_info_classif if classification else mutual_info_regression
    return partial(information, random_state=graine)


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.feature_selection import SelectKBest

    probleme, reste = retirer_probleme(parametres, SELECTION_K_MEILLEURES)
    score = reste.pop("score", "f")
    arguments = preparer(SELECTION_K_MEILLEURES, {"k": 5}, reste)
    return SelectKBest(score_func=_fonction_de_score(score, probleme, graine), **arguments)


ENTREES = (
    EntreeCatalogue(
        id=SELECTION_K_MEILLEURES,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION}),
        fabrique=_fabrique,
        grille_bornee=(("k", (3, 5, 10, "all")), ("score", SCORES)),
        cout=Cout.LEGER,
        ref_livre=ref(SELECTION_K_MEILLEURES),
        texte_id=SELECTION_K_MEILLEURES,
    ),
)
