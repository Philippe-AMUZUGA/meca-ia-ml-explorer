"""Vote de plusieurs modèles du catalogue : majorité ou probabilités, ou moyenne en régression."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

CLASSIFICATION_VOTE = "classification.vote"
REGRESSION_VOTE = "regression.vote"
BASE_PAR_DEFAUT = {
    Probleme.CLASSIFICATION: (
        "classification.logistique",
        "classification.arbre",
        "classification.plus_proches_voisins",
    ),
    Probleme.REGRESSION: (
        "regression.ridge",
        "regression.arbre",
        "regression.plus_proches_voisins",
    ),
}
BASE_MAX = 6
COMPOSITIONS = frozenset(
    {"classification.vote", "classification.empilement", "regression.vote", "regression.empilement"}
)


def modele_de_base(technique: str, id_technique: str, probleme: Probleme, graine: int) -> Any:
    """Modèle du registre servant de brique à une composition ; refuse une composition imbriquée."""
    from mecaml.core.catalog import entree as entree_du_registre

    entree = entree_du_registre(id_technique)
    if (
        probleme not in entree.problemes
        or id_technique in COMPOSITIONS
        or entree.famille
        not in (
            "regression",
            "classification",
        )
    ):
        raise ValueError(f"{technique} : `{id_technique}` ne peut pas servir de modèle de base.")
    return entree.construire(graine)


def sous_estimateurs(technique: str, ids: Any, probleme: Probleme, graine: int) -> list[tuple[str, Any]]:
    """Modèles de base (nom, objet) construits depuis le registre ; refuse les choix incohérents."""
    ids = (ids,) if isinstance(ids, str) else tuple(ids)
    if not 2 <= len(ids) <= BASE_MAX or len(set(ids)) != len(ids):
        raise ValueError(f"{technique} : `base` attend de 2 à {BASE_MAX} techniques distinctes.")
    return [(i.split(".", 1)[1], modele_de_base(technique, i, probleme, graine)) for i in ids]


def _fabrique_classification_vote(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import VotingClassifier

    defauts = {"base": BASE_PAR_DEFAUT[Probleme.CLASSIFICATION], "voting": "soft"}
    arguments = preparer(CLASSIFICATION_VOTE, defauts, parametres)
    base = sous_estimateurs(CLASSIFICATION_VOTE, arguments.pop("base"), Probleme.CLASSIFICATION, graine)
    return VotingClassifier(estimators=base, **arguments)


def _fabrique_regression_vote(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import VotingRegressor

    arguments = preparer(REGRESSION_VOTE, {"base": BASE_PAR_DEFAUT[Probleme.REGRESSION]}, parametres)
    base = sous_estimateurs(REGRESSION_VOTE, arguments.pop("base"), Probleme.REGRESSION, graine)
    return VotingRegressor(estimators=base, **arguments)


ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_VOTE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression_vote,
        grille_bornee=(),
        cout=Cout.MOYEN,
        ref_livre=ref(REGRESSION_VOTE),
        texte_id=REGRESSION_VOTE,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_VOTE,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification_vote,
        grille_bornee=(("voting", ("soft", "hard")),),
        cout=Cout.MOYEN,
        ref_livre=ref(CLASSIFICATION_VOTE),
        texte_id=CLASSIFICATION_VOTE,
    ),
)
