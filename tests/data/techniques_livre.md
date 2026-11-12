# Techniques du livre Méca-IA

Liste des techniques du livre, fichier de référence des renvois. Fichier généré depuis les cartes `\fiche` du livre
construit ; ne pas le modifier à la main. Identifiant `T<cc>-<s>-<k>` : chapitre, section, rang dans la section ;
jamais réutilisé. Une fiche porte l'identifiant de sa carte `\fiche` ; la section est celle de la fiche dans le livre
construit. Marque `[hors application]` : technique hors du périmètre de l'application (images, séquences, réseaux
Keras), exclusion motivée : données non tabulaires ou dépendance TensorFlow ; le livre le dit au chapitre 14.

Une technique de l'application renvoie à une fiche de ce premier tableau ; à défaut, au chapitre ou à la section qui la
couvre, donnés par le deuxième tableau.

## Fiches du livre (38)

| id | chapitre | section | libellé |
|:-|:-|:-|:-|
| T05-1-1 | 5 | 5.1 | Régression linéaire multiple (`sklearn.linear_model.LinearRegression`) |
| T05-1-2 | 5 | 5.1 | Régression polynomiale (`sklearn.preprocessing.PolynomialFeatures`) |
| T05-2-1 | 5 | 5.2 | Régressions régularisées (`sklearn.linear_model.Ridge`, `Lasso`, `ElasticNet`) |
| T06-1-1 | 6 | 6.1 | Régression logistique (`sklearn.linear_model.LogisticRegression`) |
| T06-2-1 | 6 | 6.2 | k plus proches voisins (`sklearn.neighbors.KNeighborsClassifier`) |
| T06-2-2 | 6 | 6.2 | Bayes naïf (`sklearn.naive_bayes.GaussianNB`) |
| T06-2-3 | 6 | 6.2 | Analyse discriminante (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`) |
| T07-1-1 | 7 | 7.1 | Séparateur à vaste marge (`sklearn.svm.SVC`) |
| T07-2-1 | 7 | 7.2 | Régression à vecteurs de support (`sklearn.svm.SVR`) |
| T08-1-1 | 8 | 8.1 | Arbre de décision (`sklearn.tree.DecisionTreeClassifier`) |
| T08-2-1 | 8 | 8.2 | Bagging et forêt aléatoire (`sklearn.ensemble.RandomForestClassifier`) |
| T08-3-1 | 8 | 8.3 | AdaBoost et gradient boosting (`sklearn.ensemble.HistGradientBoostingClassifier`) |
| T08-3-2 | 8 | 8.3 | Vote et empilement (`sklearn.ensemble.StackingClassifier`) |
| T08-3-3 | 8 | 8.3 | Importance des variables (`sklearn.inspection.permutation_importance`) |
| T09-1-1 | 9 | 9.1 | Courbe ROC (`sklearn.metrics.roc_curve`) |
| T09-1-2 | 9 | 9.1 | Courbe précision-rappel (`sklearn.metrics.precision_recall_curve`) |
| T09-2-1 | 9 | 9.2 | Matrice de coûts et seuil recommandé (`sklearn.metrics.confusion_matrix`) |
| T10-1-1 | 10 | 10.1 | K-means (`sklearn.cluster.KMeans`) |
| T10-1-2 | 10 | 10.1 | Classification hiérarchique (`sklearn.cluster.AgglomerativeClustering`) |
| T10-1-3 | 10 | 10.1 | DBSCAN (`sklearn.cluster.DBSCAN`) |
| T10-2-1 | 10 | 10.2 | Isolation Forest (`sklearn.ensemble.IsolationForest`) |
| T10-2-2 | 10 | 10.2 | Local Outlier Factor (`sklearn.neighbors.LocalOutlierFactor`) |
| T10-3-1 | 10 | 10.3 | Machine à vecteurs de support à une classe (`sklearn.svm.OneClassSVM`) |
| T11-1-1 | 11 | 11.1 | Analyse en composantes principales (`sklearn.decomposition.PCA`) |
| T11-2-1 | 11 | 11.2 | Sélection de variables (`sklearn.feature_selection.RFECV`) |
| T12-2-1 | 12 | 12.2 | Autorégression sur décalages (`sklearn.linear_model.Ridge`) |
| T12-3-1 | 12 | 12.3 | Lissage exponentiel (`pandas.Series.ewm`) |
| T13-1-1 | 13 | 13.1 | Perceptron et rétropropagation (`numpy`) |
| T13-2-1 | 13 | 13.2 | Perceptron multicouche (`sklearn.neural_network.MLPRegressor`) |
| T14-1-1 | 14 | 14.1 | Réseau dense en régression (`keras.layers.Dense`) [hors application] |
| T14-2-1 | 14 | 14.2 | Réseau dense en classification (`keras.layers.Dense`) [hors application] |
| T15-1-1 | 15 | 15.1 | Réseau convolutif (`keras.layers.Conv2D`) [hors application] |
| T16-1-1 | 16 | 16.1 | Réseau récurrent à mémoire (`keras.layers.LSTM`) [hors application] |
| T16-2-1 | 16 | 16.2 | Mécanisme d'attention (`keras.layers.MultiHeadAttention`) [hors application] |
| T17-1-1 | 17 | 17.1 | Substitut d'un calcul par éléments finis (`sklearn.preprocessing.PolynomialFeatures`) |
| T17-2-1 | 17 | 17.2 | Loi de puissance par régression sur les logarithmes (`sklearn.compose.TransformedTargetRegressor`) |
| T19-1-1 | 19 | 19.1 | Sérialisation d'un modèle et de son contexte (`joblib.dump`) |
| T19-2-1 | 19 | 19.2 | Surveillance de dérive : indice de stabilité de population et carte de contrôle (`numpy.histogram`) |

## Techniques sans fiche (couvertes dans une autre fiche ou exclues) (25)

Aucune carte `\fiche` ne porte ces identifiants : chaque ligne donne la section du livre qui les couvre (colonne
section) et, dans le libellé après « sans fiche », la sous-section, la page et la fiche voisine. Aucune n'est exclue.

| id | chapitre | section | libellé |
|:-|:-|:-|:-|
| T01-1-1 | 1 | 1.1 | Régression linéaire simple, première approche (`LinearRegression`), sans fiche : sous-section 1.1.2, page 5 |
| T01-1-2 | 1 | 1.2 | Arbre de régression, première approche (`DecisionTreeRegressor`), sans fiche : sous-section 1.2.2, page 10 ; fiche T08-1-1 |
| T01-2-1 | 1 | 1.2 | Base de test mise de côté (`train_test_split`), sans fiche : sous-section 1.2.1, page 9 |
| T01-2-2 | 1 | 1.2 | Surapprentissage et domaine de validité (interpoler contre extrapoler), sans fiche : sous-sections 1.2.2 et 1.2.3, pages 10 et 11 |
| T02-1-1 | 2 | 2.1 | Fuite de données et morcellement du jeu (apprentissage, validation, test), sans fiche : sous-sections 2.1.3 et 2.1.4, pages 16 à 19 |
| T02-1-2 | 2 | 2.1 | Référence naïve (`DummyClassifier`, `DummyRegressor`), sans fiche : sous-section 2.1.3, page 16 (classification) ; sous-section 5.1.4, page 86 (régression) |
| T02-2-1 | 2 | 2.2 | Validation croisée (`KFold`, `StratifiedKFold`, `GroupKFold`, `cross_val_score`), sans fiche : sous-section 2.2.2, page 22 |
| T02-2-2 | 2 | 2.2 | Courbes de validation et d'apprentissage (`validation_curve`, `learning_curve`), sans fiche : sous-section 2.2.3, page 24 |
| T02-3-1 | 5 | 5.1 | Métriques de régression (R², MAE, RMSE, MAPE), sans fiche : sous-section 5.1.1, page 75 (MAPE : sous-section 12.3.2, page 299) |
| T02-3-2 | 2 | 2.3 | Métriques de classification (matrice de confusion, exactitude, rappel, précision, F-score, moyenne macro), sans fiche : sous-sections 2.3.1 à 2.3.4, pages 28 à 33 |
| T03-1-1 | 3 | 3.1 | Statistiques descriptives et distributions, sans fiche : sous-section 3.1.3, page 39 |
| T03-1-2 | 3 | 3.1 | Valeurs aberrantes (règle de Tukey) et effet de levier, sans fiche : sous-sections 3.1.3 et 3.1.4, pages 39 et 40 |
| T03-1-3 | 3 | 3.1 | Profil d'un jeu : dimensions, types, manquants, valeurs uniques, constantes, doublons, sans fiche : sous-sections 3.1.1 et 3.1.2, pages 36 à 38 |
| T03-2-1 | 3 | 3.2 | Corrélations de Pearson et de Spearman, graphiques de paires (`pairplot`), sans fiche : sous-sections 3.2.1 et 3.2.2, pages 41 à 44 |
| T03-2-2 | 3 | 3.2 | Tableau croisé entre cible et catégorie (`crosstab`), sans fiche : sous-section 3.2.3, page 46 |
| T03-2-3 | 3 | 3.2 | Fiche de décisions ENISE : variable, constat, opération prévue, sans fiche : sous-section 3.2.4, page 49 |
| T04-1-1 | 4 | 4.2 | Valeurs manquantes, I de ENISE (`SimpleImputer`, `KNNImputer`, `MissingIndicator`), sans fiche : sous-section 4.2.1, page 59 |
| T04-1-2 | 4 | 4.1 | Encodage des catégories, E de ENISE (`OneHotEncoder`, `OrdinalEncoder`), sans fiche : sous-section 4.1.2, page 52 |
| T04-2-1 | 4 | 4.1 | Normalisation, N de ENISE, choisie selon la distribution (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), sans fiche : sous-section 4.1.3, page 55 |
| T04-2-2 | 4 | 4.2 | Extraction, second E de ENISE (`PolynomialFeatures`, `PowerTransformer`, `KBinsDiscretizer`), sans fiche : sous-section 4.2.4, page 65 ; fiche T05-1-2 pour le polynôme |
| T04-2-3 | 4 | 4.2 | Chaîne de traitement sans fuite (`Pipeline`, `ColumnTransformer`), sans fiche : sous-sections 4.1.4 et 4.2.5, pages 56 et 67 |
| T04-2-4 | 4 | 4.2 | Sélection, S de ENISE, première approche (`VarianceThreshold`, `SelectKBest`), sans fiche : sous-section 4.2.2, page 60 ; fiche T11-2-1 |
| T12-1-1 | 12 | 12.1 | Index temporel, rééchantillonnage et moyennes glissantes (`resample`, `rolling`, `ewm`), sans fiche : sous-sections 12.1.1 et 12.1.2, pages 277 et 278 |
| T12-1-2 | 12 | 12.1 | Validation glissante (`TimeSeriesSplit`) et référence par persistance, sans fiche : sous-section 12.1.3, page 279 ; fiche T12-2-1 |
| T12-4-1 | 12 | 12.4 | Cible à horizon de panne et classes déséquilibrées (`class_weight`), sans fiche : sous-sections 12.4.1 et 12.4.2, page 302 |

## Détail des fiches (famille, sous-section, page)

| fiche | famille | sous-section | page |
|:-|:-|:-|:-|
| `T05-1-1` | régression | 5.1.2 | 76 |
| `T05-1-2` | régression | 5.1.3 | 81 |
| `T05-2-1` | régression | 5.2.1 | 90 |
| `T06-1-1` | classification | 6.1.1 | 102 |
| `T06-2-1` | classification et régression | 6.2.1 | 110 |
| `T06-2-2` | classification | 6.2.2 | 114 |
| `T06-2-3` | classification | 6.2.3 | 118 |
| `T07-1-1` | classification | 7.1.1 | 127 |
| `T07-2-1` | régression | 7.2.1 | 137 |
| `T08-1-1` | classification et régression | 8.1.1 | 147 |
| `T08-2-1` | ensembles | 8.2.1 | 157 |
| `T08-3-1` | ensembles | 8.3.1 | 166 |
| `T08-3-2` | ensembles | 8.3.2 | 173 |
| `T08-3-3` | interprétation | 8.3.3 | 179 |
| `T09-1-1` | évaluation | 9.1.1 | 189 |
| `T09-1-2` | évaluation | 9.1.2 | 194 |
| `T09-2-1` | décision | 9.2.1 | 204 |
| `T10-1-1` | partitionnement | 10.1.1 | 218 |
| `T10-1-2` | partitionnement | 10.1.2 | 223 |
| `T10-1-3` | partitionnement | 10.1.3 | 227 |
| `T10-2-1` | détection d'anomalies | 10.2.1 | 234 |
| `T10-2-2` | détection d'anomalies | 10.2.2 | 239 |
| `T10-3-1` | détection d'anomalies | 10.3.1 | 247 |
| `T11-1-1` | réduction de dimension | 11.1.1 | 257 |
| `T11-2-1` | sélection de variables | 11.2.1 | 266 |
| `T12-2-1` | séries temporelles | 12.2.1 | 284 |
| `T12-3-1` | séries temporelles | 12.3.1 | 293 |
| `T13-1-1` | réseaux de neurones | 13.1.1 | 313 |
| `T13-2-1` | réseaux de neurones | 13.2.1 | 321 |
| `T14-1-1` | réseaux de neurones | 14.1.1 | 332 |
| `T14-2-1` | réseaux de neurones | 14.2.1 | 343 |
| `T15-1-1` | réseaux de neurones | 15.1.1 | 355 |
| `T16-1-1` | réseaux de neurones | 16.1.1 | 374 |
| `T16-2-1` | réseaux de neurones | 16.2.1 | 384 |
| `T17-1-1` | substitution de calcul | 17.1.1 | 398 |
| `T17-2-1` | substitution de calcul | 17.2.1 | 406 |
| `T19-1-1` | exploitation d'un modèle | 19.1.2 | 444 |
| `T19-2-1` | exploitation d'un modèle | 19.2.2 | 453 |
