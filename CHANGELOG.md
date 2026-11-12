# Journal des versions

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et les numéros de version suivent le [versionnage sémantique](https://semver.org/lang/fr/).

## [Non publié]

### Corrigé

- Intégration continue : deux tests de lecture des formules d'un classeur supposaient que openpyxl écrit une valeur vide sous la forme `<v></v>`, ce qui n'est vrai qu'avec lxml ; ils acceptent aussi `<v />`, forme écrite sans lxml, et la suite est verte sur une installation limitée à `requirements-dev.txt`.
- Intégration continue : l'étape des tests écrit sa sortie dans un fichier et recopie les tests en échec dans l'annotation de l'exécution, lisible sans ouvrir de session GitHub.
- Le test qui écarte du dépôt les termes à ne pas publier compare désormais des empreintes SHA-256 : le fichier ne contient plus aucun de ces termes.

## [1.1.0] : 2026-11-10

Interface allégée et réglage des modèles du catalogue.

### Ajouté

- Un schéma de réglages par modèle (`core/hyperparametres.py`) pour les 42 modèles du catalogue (régressions, classifieurs, ensembles, regroupements, détection d'anomalies, références naïves) : 3 à 6 réglages principaux quand scikit-learn en offre autant, des réglages avancés, un type, des bornes et une valeur par défaut. Les libellés français et l'aide d'une ligne de chaque réglage sont dans `core/hyperparametres_textes.py`.
- Une fenêtre de réglage refaite (`Régler`, ou double clic sur la ligne) : libellé, champ adapté au type, plage permise, valeur par défaut, aide d'une ligne, repli **Avancé**, bouton **Défauts**, choix des modèles combinés d'un vote ou d'un empilement.
- Le refus d'une valeur hors bornes, avec un message qui nomme le réglage, la valeur et la plage, par toutes les voies d'entrée : fenêtre, projet rejoué, ligne de commande, recherche de réglages.
- Deux modèles facultatifs, le réseau dense Keras en régression et en classification (`core/catalog/reseau_dense.py`), proposés seulement si Keras est installé, jamais requis (`requirements-keras.txt`, extra `keras` du paquet). Ils se comparent comme les autres ; l'export du script étant impossible pour eux, il est refusé avec un message clair tant qu'ils figurent dans la comparaison.
- Des tests : schémas, valeurs par défaut égales à celles des modèles, bornes, refus, textes, fenêtre de réglage, réseaux Keras (ignorés sans Keras).

### Modifié

- Les textes d'interface passent de 50 648 à 29 241 caractères, soit 42 % de moins (hors libellés et aides des réglages, nouveaux et rangés à part) : plus de bandeau à trois repères sous chaque titre (le titre est le nom de l'étape), écran d'accueil sans liste d'étapes, libellés et messages raccourcis, deux phrases au plus par technique dans la liste des modèles (ce qu'elle fait, ce qu'il faut regarder).
- La colonne « Réglages » de la liste des modèles affiche les libellés français et les valeurs.
- Le garde-fou de taille des réseaux scikit-learn reprend les constantes du schéma de réglages.

### Corrigé

- Le bouton « Fermer » du bandeau de message n'est plus visible tant qu'aucun message ne s'affiche.
- Le README décrit le mode d'essai et la confidentialité des données ; le fichier `.gitignore` écarte les roues et les journaux.

### Retiré

- Les textes pédagogiques « pourquoi » des techniques et les titres répétés dans `etapes_fr.toml` (les titres viennent du cœur).

## [1.0.0] : 2026-10-04

Première version publique de MécaML Explorer, application de bureau qui guide un projet d'apprentissage automatique sur données tabulaires et accompagne le livre Méca-IA.

### Ajouté

- Un parcours guidé en neuf étapes, de l'accueil à l'export : données, audit, objectif, découpage, prétraitement, modèles, diagnostic et décision, test, export. Un état de session par étape, des transitions vérifiées par le cœur.
- Les garde-fous du livre, appliqués sans pouvoir être contournés : jeu de test scellé jusqu'à l'étape 8 et ouvert une seule fois ; découpage selon la structure (lignes indépendantes, unités entières, série chronologique) ; ajustement confiné aux plis de validation ; référence naïve dans toute comparaison ; refus de la cible passée non déclarée et du compte à rebours ; identifiants exclus des variables ; bornes de taille et de calcul ; aucun accès réseau, écriture dans le seul dossier d'export.
- Un audit des données : rôle proposé pour chaque colonne avec sa raison, constats bloquants, alertes et informations, avec acquittement des alertes.
- Un catalogue de 75 techniques : imputation, encodage, mises à l'échelle et transformations, variables temporelles, sélection de variables, régressions et classifieurs classiques, ensembles d'arbres, gradient boosting, machines à vecteurs de support, réseaux de neurones scikit-learn, vote et empilement, validations (plis, unités, séries, recherche d'hyperparamètres, courbes), regroupements, détection d'anomalies, analyse en composantes principales, explications.
- 17 métriques : R², MAE, RMSE, MAPE, exactitude, exactitude équilibrée, précision, rappel, F1, aire ROC, précision moyenne, matrice de confusion, coût de décision, silhouette, indice de Rand ajusté, variance expliquée, part d'anomalies.
- Un diagnostic hors pli : résidus, matrice de confusion, courbes ROC et précision rappel, seuil et coût, lecture du modèle, importances, courbe d'apprentissage, mises en garde ; pour l'analyse sans cible, composantes, groupes et anomalies.
- Des textes pédagogiques en français pour chaque technique et chaque métrique (ce qu'elle fait, pourquoi, ce qu'il faut regarder), et trois repères par écran : ce que vous voyez, ce que cela signifie, ce que vous pouvez faire.
- Un renvoi au livre Méca-IA pour chaque conseil : chapitre, section et identifiant stable de la forme `T08-2-1`.
- Un export en cinq formats : rapport HTML sans adresse externe, classeur des chiffres, modèle ajusté, script Python équivalent et projet rejouable, avec empreintes SHA-256.
- Une ligne de commande, `python -m mecaml.cli`, pour rejouer un projet sans interface (`rejouer`), auditer un fichier (`auditer`) et proposer un projet (`proposer`). Le rejeu d'un projet exporté retrouve les scores de l'interface.
- Des cibles codées en entiers (0 et 1 pour une panne ou un défaut, chiffres pour des classes) qui se diagnostiquent, se testent et s'exportent comme des étiquettes textuelles ; un nombre de plis proposé qui ne dépasse jamais le nombre d'unités, de dates ou de lignes de la classe rare de l'apprentissage ; une erreur imprévue d'une bibliothèque de calcul affichée en français, avec le nom de l'étape ; des noms français, et non des identifiants, dans les figures, le rapport et les lignes d'état.
- Quatre jeux d'exemple dans `examples/`, un guide utilisateur par écran (`USER_GUIDE.md`) et un README.
- Un lanceur Windows (`Lancer_MecaML.bat`), la commande `python -m mecaml` (et `mecaml` après `pip install .`) avec un essai automatique (`-t`), un projet exporté d'exemple (`examples/projet_exemple/`), les empreintes des jeux d'exemple (`examples/empreintes.json`) et la spécification PyInstaller d'un exécutable autonome (`build/`).

### Sécurité et confidentialité

- L'application n'ouvre aucune connexion réseau et n'écrit que dans le dossier choisi à l'export.
- Aucun modèle sérialisé n'est jamais chargé par l'application ; les formules d'un classeur sont lues comme des valeurs, jamais évaluées.

### Licence

- Licence MIT, copyright 2026 Philippe Amuzuga.
