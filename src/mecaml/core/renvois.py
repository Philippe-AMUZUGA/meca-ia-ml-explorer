"""
MécaML : table unique des renvois au livre.

Tout renvoi de l'application vers le livre s'écrit ici, une seule fois, sous la forme d'un identifiant
`T<cc>-<s>-<k>` de la liste du livre (`mecaml.core.livre`) : technique du catalogue, métrique, thème d'un
constat (audit, cible, découpage). Les autres modules lisent cette table (`catalog._refs.ref`,
`guards.renvoi`, `metrics.METRIQUES`, `split`) et n'écrivent aucun identifiant du livre : un test le vérifie.

Une technique de l'application que le livre ne nomme pas (extension) renvoie à la section du livre la plus
proche ; `EXTENSIONS` les recense. La table est contrôlée à l'import : un renvoi hors de la liste du livre, ou
vers une technique marquée hors application, empêche de démarrer.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from mecaml.core.livre import TECHNIQUES_LIVRE

__all__ = [
    "EXTENSIONS",
    "METRIQUES",
    "TECHNIQUES",
    "TECHNIQUES_OPTIONNELLES",
    "THEMES",
    "metrique",
    "technique",
    "theme",
]

# Techniques du catalogue : identifiant du registre vers technique du livre.
TECHNIQUES: Mapping[str, str] = MappingProxyType(
    {
        # Valeurs manquantes et encodage
        "manquants.simple": "T04-1-1",
        "manquants.knn": "T04-1-1",
        "manquants.indicateur": "T04-1-1",
        "encodage.one_hot": "T04-1-2",
        "encodage.ordinal": "T04-1-2",
        # Transformations
        "transformation.standard": "T04-2-1",
        "transformation.minmax": "T04-2-1",
        "transformation.robuste": "T04-2-1",
        "transformation.puissance": "T04-2-2",
        "transformation.polynomiale": "T05-1-2",
        "transformation.discretisation": "T04-2-2",
        "transformation.ecretage_tukey": "T03-1-2",
        # Variables temporelles
        "chronologique.variables_temporelles": "T12-1-1",
        "chronologique.cible_decalee": "T12-2-1",
        # Sélection de variables
        "selection.variance": "T11-2-1",
        "selection.k_meilleures": "T11-2-1",
        "selection.depuis_modele": "T11-2-1",
        "selection.rfecv": "T11-2-1",
        # Régression
        "regression.lineaire": "T05-1-1",
        "regression.ridge": "T05-2-1",
        "regression.lasso": "T05-2-1",
        "regression.elastic_net": "T05-2-1",
        "regression.quantile": "T05-2-1",
        "regression.plus_proches_voisins": "T06-2-1",
        "regression.arbre": "T08-1-1",
        "regression.foret": "T08-2-1",
        "regression.arbres_extremes": "T08-2-1",
        "regression.bagging": "T08-2-1",
        "regression.adaboost": "T08-3-1",
        "regression.gradient_boosting": "T08-3-1",
        "regression.gradient_histogramme": "T08-3-1",
        "regression.vote": "T08-3-2",
        "regression.empilement": "T08-3-2",
        "regression.svr": "T07-2-1",
        "regression.reseau_neurones": "T13-2-1",
        # Classification
        "classification.logistique": "T06-1-1",
        "classification.plus_proches_voisins": "T06-2-1",
        "classification.bayes_naif": "T06-2-2",
        "classification.analyse_discriminante": "T06-2-3",
        "classification.arbre": "T08-1-1",
        "classification.foret": "T08-2-1",
        "classification.arbres_extremes": "T08-2-1",
        "classification.bagging": "T08-2-1",
        "classification.adaboost": "T08-3-1",
        "classification.gradient_boosting": "T08-3-1",
        "classification.gradient_histogramme": "T08-3-1",
        "classification.vote": "T08-3-2",
        "classification.empilement": "T08-3-2",
        "classification.svc": "T07-1-1",
        "classification.reseau_neurones": "T13-2-1",
        # Référence naïve
        "reference.dummy_regression": "T02-1-2",
        "reference.dummy_classification": "T02-1-2",
        "reference.persistance": "T12-1-2",
        # Validation
        "validation.kfold": "T02-2-1",
        "validation.kfold_stratifie": "T02-2-1",
        "validation.plis_groupes": "T02-2-1",
        "validation.plis_groupes_stratifies": "T02-2-1",
        "validation.groupes_aleatoires": "T02-2-1",
        "validation.serie_temporelle": "T12-1-2",
        "validation.exclusion_unique": "T02-2-1",
        "validation.recherche_grille": "T02-2-1",
        "validation.recherche_aleatoire": "T02-2-1",
        "validation.courbe_apprentissage": "T02-2-2",
        "validation.courbe_validation": "T02-2-2",
        # Sans cible
        "sans_cible.acp": "T11-1-1",
        "sans_cible.kmeans": "T10-1-1",
        "sans_cible.melange_gaussien": "T10-1-1",
        "sans_cible.classification_hierarchique": "T10-1-2",
        "sans_cible.dbscan": "T10-1-3",
        "sans_cible.foret_isolement": "T10-2-1",
        "sans_cible.facteur_local": "T10-2-2",
        "sans_cible.svm_une_classe": "T10-3-1",
        # Explication
        "explication.permutation": "T08-3-3",
        "explication.impurete": "T08-3-3",
        "explication.arbre_texte": "T08-1-1",
    }
)

# Techniques de l'application que le livre ne nomme pas : elles renvoient à la section la plus proche.
EXTENSIONS: frozenset[str] = frozenset(
    {
        "selection.depuis_modele",
        "regression.quantile",
        "regression.arbres_extremes",
        "regression.adaboost",
        "regression.gradient_boosting",
        "regression.vote",
        "regression.empilement",
        "classification.arbres_extremes",
        "classification.gradient_boosting",
        "validation.plis_groupes_stratifies",
        "validation.groupes_aleatoires",
        "validation.exclusion_unique",
        "validation.recherche_grille",
        "validation.recherche_aleatoire",
        "sans_cible.melange_gaussien",
    }
)

# Métriques : identifiant de la métrique vers technique du livre.
METRIQUES: Mapping[str, str] = MappingProxyType(
    {
        "r2": "T02-3-1",
        "mae": "T02-3-1",
        "rmse": "T02-3-1",
        "mape": "T02-3-1",
        "exactitude": "T02-3-2",
        "exactitude_equilibree": "T02-3-2",
        "precision": "T02-3-2",
        "rappel": "T02-3-2",
        "f1": "T02-3-2",
        "roc_auc": "T09-1-1",
        "precision_moyenne": "T09-1-2",
        "matrice_confusion": "T02-3-2",
        "cout": "T09-2-1",
        "silhouette": "T10-1-1",
        "rand_ajuste": "T10-1-1",
        "variance_expliquee": "T11-1-1",
        "taux_anomalies": "T10-2-1",
    }
)

# Thèmes des constats (audit, contrôle de la cible, découpage). None : le livre n'en parle pas.
THEMES: Mapping[str, str | None] = MappingProxyType(
    {
        "profil": "T03-1-3",  # profil d'un jeu : types, uniques, constantes
        "doublons": "T03-1-3",
        "manquants": "T04-1-1",
        "cardinalite": "T04-1-2",  # encodage des catégories
        "fuite": "T02-1-1",  # fuite de données et morcellement du jeu
        "relation_monotone": "T03-2-1",  # corrélations de Pearson et de Spearman
        "aberrantes": "T03-1-2",  # valeurs aberrantes, règle de Tukey
        "temps_repere": "T12-1-1",  # index temporel
        "temps_validation": "T12-1-2",  # validation glissante
        "cible_passee": "T12-2-1",  # autorégression : la cible passée doit être connue à la prévision
        "cible": "T02-1-1",
        "cible_regression": "T02-3-1",  # métriques de régression
        "cible_classification": "T02-3-2",  # métriques de classification
        "plis": "T02-2-1",  # validation croisée
        "classe_rare": "T09-1-2",  # courbe précision-rappel
        "echantillon": "T02-2-1",
        "personnel": None,
        "decoupage.morcellement": "T02-1-1",  # fuite de données et morcellement du jeu
        "decoupage.plis": "T02-2-1",  # validation croisée, plis stratifiés et groupés
        "decoupage.dates": "T12-1-2",  # validation glissante d'une série
    }
)


# Techniques facultatives (réseaux Keras) : le livre les explique au chapitre 14 ; leur fiche est marquée
# hors application dans la liste publiée, car l'application ne les offre que si Keras est installé.
TECHNIQUES_OPTIONNELLES: Mapping[str, str] = MappingProxyType(
    {"regression.reseau_dense": "T14-1-1", "classification.reseau_dense": "T14-2-1"}
)


def technique(id_technique: str) -> str:
    """Identifiant du livre pour une technique du catalogue ; `KeyError` si elle n'est pas recensée."""
    try:
        return TECHNIQUES_OPTIONNELLES.get(id_technique) or TECHNIQUES[id_technique]
    except KeyError:
        raise KeyError(f"Renvoi au livre manquant pour la technique {id_technique!r}.") from None


def metrique(id_metrique: str) -> str:
    """Identifiant du livre pour une métrique ; `KeyError` explicite si elle n'est pas recensée."""
    try:
        return METRIQUES[id_metrique]
    except KeyError:
        raise KeyError(f"Renvoi au livre manquant pour la métrique {id_metrique!r}.") from None


def theme(nom: str) -> str | None:
    """Identifiant du livre pour un thème de constat (None si le livre n'en parle pas)."""
    try:
        return THEMES[nom]
    except KeyError:
        raise KeyError(f"Renvoi au livre manquant pour le thème {nom!r}.") from None


def _verifier() -> None:
    """Chaque renvoi désigne une technique de la liste du livre qui n'est pas hors application."""
    for quoi, table in (("technique", TECHNIQUES), ("métrique", METRIQUES), ("thème", THEMES)):
        for cle, identifiant in table.items():
            if identifiant is None:
                continue
            fiche = TECHNIQUES_LIVRE.get(identifiant)
            if fiche is None or fiche.hors_application:
                raise ValueError(
                    f"Renvoi {identifiant!r} de {quoi} {cle!r} absent de la liste du livre, "
                    "ou hors application."
                )


_verifier()
