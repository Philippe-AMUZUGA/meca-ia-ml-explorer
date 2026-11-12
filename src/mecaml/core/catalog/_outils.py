"""
MécaML : aides communes aux modules de techniques.

Trois services, sans dépendance à scikit-learn (import paresseux dans les fabriques) :

- la graine unique du projet est imposée à l'objet construit (reproductibilité, I8) ;
- les bornes de calcul (I9) refusent un paramètre démesuré avec un message actionnable ;
- les défauts de la technique sont fusionnés avec les paramètres choisis par l'utilisateur.

Convention de la famille `selection` : ses fabriques reçoivent en plus `probleme`
(« regression » ou « classification »), car le score ou le modèle de base en dépend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import Any

from mecaml.core.catalog.entree import GRILLE_MAX_COMBINAISONS
from mecaml.core.hyperparametres import COUCHES_MAX, UNITES_MAX, controler

__all__ = [
    "COUCHES_MAX",
    "LIMITES",
    "UNITES_MAX",
    "avec_graine",
    "exiger",
    "noms_des_variables",
    "preparer",
    "retirer_probleme",
    "verifier_grille",
]

# Bornes de calcul (I9) : au delà, la technique est refusée plutôt qu'exécutée sans fin.
LIMITES: dict[str, int] = {
    "n_estimators": 500,  # arbres d'une forêt, d'un bagging ou d'un adaboost
    "max_iter": 10_000,  # passes d'un solveur ou d'un réseau
    "max_depth": 50,  # profondeur d'un arbre
    "n_neighbors": 500,  # voisins
    "degree": 4,  # degré d'une expansion polynomiale
    "n_bins": 50,  # classes d'une discrétisation
    "n_clusters": 50,  # groupes d'un regroupement
    "n_iter": 50,  # tirages d'une recherche aléatoire
    "n_splits": 20,  # plis d'une validation croisée
    "n_repeats": 50,  # mélanges d'une importance par permutation
}

# Taille maximale d'un réseau de neurones : COUCHES_MAX couches de UNITES_MAX unités (`hyperparametres`).


def noms_des_variables(pipeline: Any) -> list[str] | None:
    """Noms des colonnes qui entrent dans le dernier modèle, si le pipeline sait les donner."""
    try:
        return [str(nom) for nom in pipeline[:-1].get_feature_names_out()]
    except (AttributeError, ValueError, TypeError):
        return None


def avec_graine(parametres: Mapping[str, Any], graine: int) -> dict[str, Any]:
    """Paramètres munis de la graine unique du projet (elle l'emporte sur toute autre)."""
    return {**parametres, "random_state": graine}


def retirer_probleme(parametres: Mapping[str, Any], technique: str) -> tuple[str, dict[str, Any]]:
    """Sépare `probleme` des autres paramètres (famille `selection`) ; refuse une valeur inconnue."""
    reste = dict(parametres)
    probleme = reste.pop("probleme", "regression")
    if probleme not in ("regression", "classification", "sans_cible"):
        raise ValueError(
            f"{technique} : `probleme` vaut {probleme!r} ; attendu « regression », « classification » "
            "ou « sans_cible »."
        )
    return str(probleme), reste


def exiger(technique: str, parametres: Mapping[str, Any], noms: Sequence[str]) -> None:
    """Refuse la construction si un paramètre obligatoire manque."""
    absents = [nom for nom in noms if nom not in parametres]
    if absents:
        raise ValueError(f"{technique} : paramètre(s) obligatoire(s) absent(s) : {', '.join(absents)}.")


def verifier_grille(technique: str, grille: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> int:
    """Nombre de combinaisons d'une grille de recherche ; refus au delà de la borne (I9)."""
    blocs = [grille] if isinstance(grille, Mapping) else list(grille)
    total = sum(prod(len(valeurs) for valeurs in bloc.values()) for bloc in blocs)
    if total > GRILLE_MAX_COMBINAISONS:
        raise ValueError(
            f"{technique} : grille de {total} combinaisons, borne {GRILLE_MAX_COMBINAISONS} ; "
            "réduire les valeurs candidates ou préférer la recherche aléatoire."
        )
    return total


def _verifier_borne(technique: str, nom: str, valeur: Any) -> None:
    borne = LIMITES[nom]
    if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
        return
    if valeur > borne:
        raise ValueError(
            f"{technique} : `{nom}` = {valeur} dépasse la borne {borne} (garde-fou de calcul) ; "
            "choisir une valeur plus faible."
        )


def _verifier_couches(technique: str, valeur: Any) -> None:
    couches = (valeur,) if isinstance(valeur, int) else tuple(valeur)
    if len(couches) > COUCHES_MAX or any(not isinstance(u, int) or not 1 <= u <= UNITES_MAX for u in couches):
        raise ValueError(
            f"{technique} : `hidden_layer_sizes` = {valeur!r} hors bornes "
            f"({COUCHES_MAX} couches au plus, de 1 à {UNITES_MAX} unités chacune)."
        )


def preparer(
    technique: str,
    defauts: Mapping[str, Any],
    parametres: Mapping[str, Any],
    graine: int | None = None,
) -> dict[str, Any]:
    """Arguments de construction : défauts, choix de l'utilisateur, graine, bornes vérifiées."""
    final = {**defauts, **parametres}
    if graine is not None:
        final = avec_graine(final, graine)
    for nom in LIMITES:
        if nom in final:
            _verifier_borne(technique, nom, final[nom])
    if "hidden_layer_sizes" in final:
        _verifier_couches(technique, final["hidden_layer_sizes"])
    controler(technique, final)
    return final
