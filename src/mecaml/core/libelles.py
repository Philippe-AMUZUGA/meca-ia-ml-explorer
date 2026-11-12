"""
MécaML : libellés français des identifiants que le rapport et les figures montrent à l'utilisateur.

Le cœur manipule des identifiants stables (`regression.ridge`, `validation.kfold`, `f1`) ; l'utilisateur lit
des mots. Les titres des techniques sont ceux de `TITRES`, les mêmes que dans les listes de l'interface ; les
métriques portent leur libellé dans le registre. Un identifiant inconnu se rend tel quel : un libellé
manquant ne casse jamais un rapport.
"""

from __future__ import annotations

import re

from mecaml.core import metrics

__all__ = ["TITRES", "famille", "metrique", "metrique_courte", "technique"]

TITRES: dict[str, str] = {
    "manquants.simple": "Imputation simple (médiane, moyenne, valeur fréquente)",
    "manquants.knn": "Imputation par plus proches voisins",
    "manquants.indicateur": "Indicateur de valeur manquante",
    "encodage.one_hot": "Encodage one-hot",
    "encodage.ordinal": "Encodage ordinal",
    "transformation.standard": "Standardisation",
    "transformation.minmax": "Mise à l'échelle min-max",
    "transformation.robuste": "Mise à l'échelle robuste (médiane et quartiles)",
    "transformation.puissance": "Transformation de puissance",
    "transformation.polynomiale": "Variables polynomiales",
    "transformation.discretisation": "Discrétisation en classes",
    "transformation.ecretage_tukey": "Écrêtage des valeurs aberrantes (Tukey)",
    "chronologique.variables_temporelles": "Variables temporelles (retards, fenêtres, différences)",
    "chronologique.cible_decalee": "Cible du pas précédent",
    "selection.variance": "Sélection par variance minimale",
    "selection.k_meilleures": "Sélection des k meilleures variables",
    "selection.depuis_modele": "Sélection par un modèle",
    "selection.rfecv": "Élimination récursive avec validation croisée (RFECV)",
    "regression.lineaire": "Régression linéaire",
    "regression.ridge": "Régression de crête (Ridge)",
    "regression.lasso": "Régression Lasso",
    "regression.elastic_net": "Régression Elastic Net",
    "regression.quantile": "Régression quantile",
    "regression.plus_proches_voisins": "Plus proches voisins (régression)",
    "regression.arbre": "Arbre de régression",
    "regression.foret": "Forêt aléatoire (régression)",
    "regression.arbres_extremes": "Arbres extrêmement aléatoires (régression)",
    "regression.bagging": "Bagging (régression)",
    "regression.gradient_histogramme": "Gradient boosting sur histogrammes (régression)",
    "regression.gradient_boosting": "Gradient boosting (régression)",
    "regression.svr": "Machine à vecteurs de support (SVR)",
    "regression.reseau_neurones": "Réseau de neurones (régression)",
    "regression.adaboost": "AdaBoost (régression)",
    "regression.vote": "Vote de modèles (régression)",
    "regression.empilement": "Empilement de modèles (régression)",
    "classification.plus_proches_voisins": "Plus proches voisins (classification)",
    "classification.arbre": "Arbre de classification",
    "classification.foret": "Forêt aléatoire (classification)",
    "classification.arbres_extremes": "Arbres extrêmement aléatoires (classification)",
    "classification.bagging": "Bagging (classification)",
    "classification.gradient_histogramme": "Gradient boosting sur histogrammes (classification)",
    "classification.gradient_boosting": "Gradient boosting (classification)",
    "classification.svc": "Machine à vecteurs de support (SVC)",
    "classification.reseau_neurones": "Réseau de neurones (classification)",
    "classification.logistique": "Régression logistique",
    "classification.bayes_naif": "Bayésien naïf",
    "classification.analyse_discriminante": "Analyse discriminante linéaire",
    "classification.adaboost": "AdaBoost (classification)",
    "classification.vote": "Vote de modèles (classification)",
    "classification.empilement": "Empilement de modèles (classification)",
    "reference.dummy_regression": "Référence naïve : la moyenne",
    "reference.dummy_classification": "Référence naïve : la classe majoritaire",
    "reference.persistance": "Référence : persistance (cible du pas précédent)",
    "validation.kfold": "Plis aléatoires",
    "validation.kfold_stratifie": "Plis aléatoires gardant les proportions des classes",
    "validation.plis_groupes": "Plis d'unités entières",
    "validation.plis_groupes_stratifies": "Plis d'unités entières gardant les proportions des classes",
    "validation.groupes_aleatoires": "Tirages d'unités entières",
    "validation.serie_temporelle": "Fenêtre croissante sur les dates",
    "validation.exclusion_unique": "Exclusion d'une ligne à la fois",
    "validation.recherche_grille": "Recherche par grille",
    "validation.recherche_aleatoire": "Recherche aléatoire",
    "validation.courbe_apprentissage": "Courbe d'apprentissage",
    "validation.courbe_validation": "Courbe de validation",
    "sans_cible.acp": "Analyse en composantes principales",
    "sans_cible.kmeans": "K-moyennes",
    "sans_cible.melange_gaussien": "Mélange gaussien",
    "sans_cible.classification_hierarchique": "Classification hiérarchique",
    "sans_cible.dbscan": "DBSCAN",
    "sans_cible.foret_isolement": "Forêt d'isolement",
    "sans_cible.facteur_local": "Facteur d'anomalie local",
    "sans_cible.svm_une_classe": "Machine à vecteurs de support à une classe",
    "explication.permutation": "Importance par permutation",
    "explication.impurete": "Importance par impureté",
    "explication.arbre_texte": "Règles de l'arbre en texte",
}
"""Titre français de chaque technique du catalogue ; les listes de l'interface les reprennent."""

_AVANT_PONCTUATION = re.compile(r" ([:;!?»])")
_INSECABLE = " "
TITRES = {cle: _AVANT_PONCTUATION.sub(_INSECABLE + r"\1", titre) for cle, titre in TITRES.items()}


# Techniques facultatives (Keras) : leurs titres sont à part, `TITRES` étant celui du registre de base.
TITRES_OPTIONNELS = {
    "regression.reseau_dense": "Réseau dense Keras (régression)",
    "classification.reseau_dense": "Réseau dense Keras (classification)",
}


def technique(identifiant: str) -> str:
    """Titre français d'une technique (modèle, prétraitement, validation) ; l'identifiant à défaut."""
    return TITRES.get(identifiant) or TITRES_OPTIONNELS.get(identifiant, identifiant)


def metrique(texte: str) -> str:
    """Libellé d'une métrique, avec ses paramètres s'il y en a (« cout:fp=1,fn=10 ») ; le texte à défaut."""
    try:
        m, parametres = metrics.analyser(texte)
    except ValueError:
        return texte
    if not parametres:
        return m.libelle
    reglages = ", ".join(f"{nom} = {valeur:g}" for nom, valeur in parametres.items())
    return f"{m.libelle.split(' (')[0]} ({reglages})"


_RE_SIGLE = re.compile(r"\(([A-Z0-9²]{2,})\)$")


def metrique_courte(texte: str) -> str:
    """Nom court d'une métrique pour un axe ou un en-tête : son sigle (RMSE, MAE) ou son libellé raccourci."""
    try:
        m, _ = metrics.analyser(texte)
    except ValueError:
        return texte
    sigle = _RE_SIGLE.search(m.libelle)
    return sigle.group(1) if sigle else m.libelle.split(" (")[0]


_FAMILLES = {
    "manquants": "Valeurs manquantes",
    "encodage": "Encodage",
    "transformation": "Transformation",
    "selection": "Sélection de variables",
    "chronologique": "Variables temporelles",
    "regression": "Régression",
    "classification": "Classification",
    "reference": "Référence naïve",
    "validation": "Validation",
    "sans_cible": "Sans cible",
    "explication": "Explication",
}


def famille(identifiant: str) -> str:
    """Famille d'une technique en français ; l'identifiant à défaut."""
    return _FAMILLES.get(identifiant, identifiant)
