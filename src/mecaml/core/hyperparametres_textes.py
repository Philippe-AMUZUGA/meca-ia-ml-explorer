"""
MécaML : libellés et aides des réglages des modèles, en français.

Un libellé et une aide d'une ligne par nom de paramètre ; une clé « identifiant.nom » l'emporte quand le sens
change d'un modèle à l'autre. Les mots de la section `MOTS` composent les bornes et les messages de refus.
"""

from __future__ import annotations

MOTS: dict[str, str] = {
    "a": "à",
    "oui": "oui",
    "non": "non",
    "aucun": "aucun",
    "illimite": "aucune limite",
    "auto": "auto",
    "toutes": "toutes",
    "defaut": "défaut",
    "couches": "{mini} à {maxi} unités par couche, {n} couches au plus",
    "liste": "{mini} à {maxi} modèles distincts",
    "part": "entier de {mini} à {maxi}, ou part de variance entre 0 et 1",
    "pas_nombre": "n'est pas un nombre.",
    "pas_fini": "n'est pas un nombre fini.",
    "pas_entier": "n'est pas un entier.",
    "pas_booleen": "doit être oui ou non.",
    "hors_bornes": "est hors bornes ({bornes}).",
    "hors_liste": "n'est pas permis ({choix}).",
}

# Clé : nom du paramètre, ou « identifiant.nom » ; valeur : (libellé, aide d'une ligne).
TEXTES: dict[str, tuple[str, str]] = {
    "fit_intercept": ("Ordonnée à l'origine", "Ajoute un terme constant au modèle."),
    "positive": ("Coefficients positifs", "Impose des coefficients supérieurs ou égaux à zéro."),
    "tol": ("Tolérance d'arrêt", "Le calcul s'arrête quand le gain passe sous ce seuil."),
    "alpha": ("Régularisation (alpha)", "Plus alpha est grand, plus les coefficients sont freinés."),
    "regression.quantile.alpha": (
        "Pénalité L1 (alpha)",
        "Freine les coefficients ; zéro désactive la pénalité.",
    ),
    "regression.reseau_neurones.alpha": (
        "Pénalité L2 (alpha)",
        "Freine les poids du réseau contre le surapprentissage.",
    ),
    "classification.reseau_neurones.alpha": (
        "Pénalité L2 (alpha)",
        "Freine les poids du réseau contre le surapprentissage.",
    ),
    "ccp_alpha": ("Élagage (ccp_alpha)", "Coupe les branches peu utiles ; zéro garde tout l'arbre."),
    "solver": (
        "Méthode de résolution",
        "Algorithme de calcul ; « auto » ou le défaut convient presque toujours.",
    ),
    "max_iter": ("Itérations maximales", "Plafond de passes de calcul avant l'arrêt."),
    "regression.gradient_histogramme.max_iter": (
        "Nombre d'arbres",
        "Nombre d'étapes de boosting, un arbre par étape.",
    ),
    "classification.gradient_histogramme.max_iter": (
        "Nombre d'arbres",
        "Nombre d'étapes de boosting, un arbre par étape.",
    ),
    "selection": (
        "Ordre de mise à jour",
        "Cyclique : variable par variable ; aléatoire : parfois plus rapide.",
    ),
    "l1_ratio": ("Part de Lasso (l1_ratio)", "0 donne Ridge, 1 donne Lasso."),
    "quantile": ("Quantile visé", "0,5 prédit la médiane ; 0,9 le seuil dépassé une fois sur dix."),
    "n_neighbors": ("Nombre de voisins", "Moins de voisins : modèle souple ; plus : modèle lissé."),
    "weights": ("Poids des voisins", "uniform : même voix ; distance : les plus proches pèsent plus."),
    "p": ("Distance (puissance p)", "1 : distance de Manhattan ; 2 : distance euclidienne."),
    "algorithm": ("Structure de recherche", "Accélère la recherche de voisins ; le résultat ne change pas."),
    "leaf_size": ("Taille des feuilles", "Réglage de vitesse de la structure de recherche."),
    "max_depth": (
        "Profondeur maximale",
        "Limite la complexité ; une profondeur faible évite le surapprentissage.",
    ),
    "min_samples_leaf": ("Lignes minimales par feuille", "Une feuille plus fournie lisse le modèle."),
    "min_samples_split": ("Lignes minimales avant coupure", "Un nœud plus petit n'est plus coupé."),
    "criterion": ("Critère de coupure", "Mesure qui choisit la meilleure coupure à chaque nœud."),
    "class_weight": ("Poids des classes", "« balanced » donne plus de poids aux classes rares."),
    "max_leaf_nodes": ("Feuilles maximales", "Plafond du nombre de feuilles de l'arbre."),
    "classification.gradient_histogramme.max_leaf_nodes": (
        "Feuilles maximales par arbre",
        "Plafond du nombre de feuilles de chaque arbre.",
    ),
    "regression.gradient_histogramme.max_leaf_nodes": (
        "Feuilles maximales par arbre",
        "Plafond du nombre de feuilles de chaque arbre.",
    ),
    "n_estimators": ("Nombre d'arbres", "Plus d'arbres stabilise le résultat, au prix du temps de calcul."),
    "regression.adaboost.n_estimators": ("Nombre de modèles", "Nombre de petits modèles enchaînés."),
    "classification.adaboost.n_estimators": ("Nombre de modèles", "Nombre de petits modèles enchaînés."),
    "regression.bagging.n_estimators": (
        "Nombre de modèles",
        "Nombre de modèles ajustés sur des tirages différents.",
    ),
    "classification.bagging.n_estimators": (
        "Nombre de modèles",
        "Nombre de modèles ajustés sur des tirages différents.",
    ),
    "bootstrap": ("Tirage avec remise", "Chaque arbre apprend sur un tirage différent des lignes."),
    "max_samples": ("Part des lignes par modèle", "Fraction des lignes tirée pour chaque modèle."),
    "max_features": ("Part des variables par modèle", "Fraction des variables tirée pour chaque modèle."),
    "bootstrap_features": (
        "Tirage des variables avec remise",
        "Les variables aussi sont tirées avec remise.",
    ),
    "learning_rate": ("Pas d'apprentissage", "Poids de chaque étape ; petit : lent mais précis."),
    "max_bins": ("Classes de discrétisation", "Nombre de classes par variable pour accélérer le calcul."),
    "l2_regularization": ("Pénalité L2", "Freine les feuilles pour limiter le surapprentissage."),
    "subsample": ("Part des lignes par arbre", "Fraction des lignes tirée pour chaque arbre."),
    "loss": ("Fonction de perte", "Erreur que l'apprentissage cherche à réduire."),
    "C": ("Pénalité d'erreur (C)", "Grand C : suit mieux les données, au risque du surapprentissage."),
    "classification.logistique.C": (
        "Inverse de la régularisation (C)",
        "Petit C : modèle plus freiné ; grand C : plus libre.",
    ),
    "epsilon": ("Marge sans pénalité (epsilon)", "Écart toléré entre prédiction et mesure."),
    "kernel": ("Noyau", "Forme de la frontière : rbf souple, linear droite."),
    "gamma": ("Portée du noyau (gamma)", "« scale » adapte la portée à l'étendue des données."),
    "degree": ("Degré du noyau poly", "Sert seulement avec le noyau poly."),
    "coef0": ("Terme libre du noyau", "Sert avec les noyaux poly et sigmoid."),
    "hidden_layer_sizes": ("Couches cachées", "Unités par couche, séparées par des virgules : 50, 50."),
    "activation": ("Fonction d'activation", "Non-linéarité appliquée à chaque couche cachée."),
    "learning_rate_init": ("Pas d'apprentissage", "Taille des pas de l'optimiseur au départ."),
    "early_stopping": ("Arrêt précoce", "Arrête quand le score cesse de progresser sur une part réservée."),
    "n_iter_no_change": ("Patience (itérations)", "Itérations sans progrès avant l'arrêt."),
    "var_smoothing": (
        "Lissage des variances",
        "Fraction de la variance ajoutée pour éviter les divisions par zéro.",
    ),
    "shrinkage": ("Rétrécissement", "Stabilise la covariance ; sert avec lsqr ou eigen."),
    "base": ("Modèles de base", "Modèles combinés : de 2 à 6, choisis dans le catalogue."),
    "voting": ("Type de vote", "soft : moyenne des probabilités ; hard : vote majoritaire."),
    "passthrough": ("Passer aussi les variables", "Le modèle final reçoit aussi les variables d'origine."),
    "cv": ("Plis internes", "Plis servant à produire les prédictions des modèles de base."),
    "strategy": ("Règle de la référence", "Ce que la référence prédit à chaque ligne."),
    "n_components": ("Nombre de composantes", "Dimensions conservées par l'analyse."),
    "sans_cible.melange_gaussien.n_components": (
        "Nombre de groupes",
        "Nombre de lois gaussiennes du mélange.",
    ),
    "whiten": ("Variances égalisées", "Ramène chaque composante à la même variance."),
    "svd_solver": ("Méthode de décomposition", "Algorithme de l'analyse ; « auto » choisit selon la taille."),
    "n_clusters": ("Nombre de groupes", "Nombre de groupes à former."),
    "n_init": ("Départs aléatoires", "Nombre de départs ; le meilleur résultat est gardé."),
    "init": ("Initialisation", "Choix des centres de départ."),
    "covariance_type": ("Forme des covariances", "full : ellipses libres ; spherical : sphères."),
    "reg_covar": (
        "Régularisation des covariances",
        "Valeur ajoutée à la diagonale pour garder les covariances inversibles.",
    ),
    "init_params": ("Initialisation", "Méthode de départ des composantes."),
    "linkage": ("Critère de fusion", "ward : variance minimale ; single : plus proche voisin."),
    "metric": ("Distance", "Mesure de ressemblance entre deux lignes."),
    "eps": ("Rayon de voisinage (eps)", "Deux points plus proches que eps sont voisins."),
    "min_samples": ("Voisins minimaux", "Voisins requis pour former un groupe dense."),
    "contamination": (
        "Part d'anomalies attendue",
        "Fraction des lignes jugées anormales ; « auto » laisse le modèle décider.",
    ),
    "nu": ("Part d'anomalies tolérée (nu)", "Borne la part de points hors de la frontière."),
    "couches": ("Couches cachées", "Unités par couche, séparées par des virgules : 32, 32."),
    "epochs": ("Époques", "Passes complètes sur l'apprentissage."),
    "taux_apprentissage": ("Pas d'apprentissage", "Taille des pas de l'optimiseur Adam."),
    "taille_lot": ("Taille de lot", "Lignes vues avant chaque mise à jour des poids."),
    "abandon": ("Taux d'abandon", "Part des unités éteintes au hasard pendant l'apprentissage."),
    "arret_precoce": ("Arrêt précoce", "Arrête quand le score cesse de progresser sur 10 % des lignes."),
}
