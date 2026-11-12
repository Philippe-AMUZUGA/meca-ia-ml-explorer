"""
MécaML : techniques du livre Méca-IA, avec leur identifiant stable `T<cc>-<s>-<k>`.

Chaque renvoi au livre de l'application (techniques du catalogue, métriques, constats) est un
identifiant de cette liste : chapitre, section, rang dans la section. Module généré par
`tests/outils/generer_livre.py` depuis `tests/data/techniques_livre.md` ;
ne pas le modifier à la main.
Les techniques marquées hors application (images, séquences, réseaux Keras) sont exclues de
l'application : données non tabulaires ou dépendance à TensorFlow.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["TECHNIQUES_LIVRE", "TechniqueLivre", "est_identifiant", "technique_du_livre"]

RE_IDENTIFIANT = re.compile(r"^T\d{2}-\d+-\d+$")


@dataclass(frozen=True)
class TechniqueLivre:
    """Une technique du livre : où le lecteur la trouve."""

    id: str
    chapitre: int
    section: str
    libelle: str
    hors_application: bool


TECHNIQUES_LIVRE: Mapping[str, TechniqueLivre] = MappingProxyType(
    {
        "T05-1-1": TechniqueLivre(
            "T05-1-1",
            5,
            "5.1",
            "Régression linéaire multiple (`sklearn.linear_model.LinearRegression`)",
            False,
        ),
        "T05-1-2": TechniqueLivre(
            "T05-1-2",
            5,
            "5.1",
            "Régression polynomiale (`sklearn.preprocessing.PolynomialFeatures`)",
            False,
        ),
        "T05-2-1": TechniqueLivre(
            "T05-2-1",
            5,
            "5.2",
            "Régressions régularisées (`sklearn.linear_model.Ridge`, `Lasso`, `ElasticNet`)",
            False,
        ),
        "T06-1-1": TechniqueLivre(
            "T06-1-1",
            6,
            "6.1",
            "Régression logistique (`sklearn.linear_model.LogisticRegression`)",
            False,
        ),
        "T06-2-1": TechniqueLivre(
            "T06-2-1",
            6,
            "6.2",
            "k plus proches voisins (`sklearn.neighbors.KNeighborsClassifier`)",
            False,
        ),
        "T06-2-2": TechniqueLivre(
            "T06-2-2",
            6,
            "6.2",
            "Bayes naïf (`sklearn.naive_bayes.GaussianNB`)",
            False,
        ),
        "T06-2-3": TechniqueLivre(
            "T06-2-3",
            6,
            "6.2",
            "Analyse discriminante (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)",
            False,
        ),
        "T07-1-1": TechniqueLivre(
            "T07-1-1",
            7,
            "7.1",
            "Séparateur à vaste marge (`sklearn.svm.SVC`)",
            False,
        ),
        "T07-2-1": TechniqueLivre(
            "T07-2-1",
            7,
            "7.2",
            "Régression à vecteurs de support (`sklearn.svm.SVR`)",
            False,
        ),
        "T08-1-1": TechniqueLivre(
            "T08-1-1",
            8,
            "8.1",
            "Arbre de décision (`sklearn.tree.DecisionTreeClassifier`)",
            False,
        ),
        "T08-2-1": TechniqueLivre(
            "T08-2-1",
            8,
            "8.2",
            "Bagging et forêt aléatoire (`sklearn.ensemble.RandomForestClassifier`)",
            False,
        ),
        "T08-3-1": TechniqueLivre(
            "T08-3-1",
            8,
            "8.3",
            "AdaBoost et gradient boosting (`sklearn.ensemble.HistGradientBoostingClassifier`)",
            False,
        ),
        "T08-3-2": TechniqueLivre(
            "T08-3-2",
            8,
            "8.3",
            "Vote et empilement (`sklearn.ensemble.StackingClassifier`)",
            False,
        ),
        "T08-3-3": TechniqueLivre(
            "T08-3-3",
            8,
            "8.3",
            "Importance des variables (`sklearn.inspection.permutation_importance`)",
            False,
        ),
        "T09-1-1": TechniqueLivre(
            "T09-1-1",
            9,
            "9.1",
            "Courbe ROC (`sklearn.metrics.roc_curve`)",
            False,
        ),
        "T09-1-2": TechniqueLivre(
            "T09-1-2",
            9,
            "9.1",
            "Courbe précision-rappel (`sklearn.metrics.precision_recall_curve`)",
            False,
        ),
        "T09-2-1": TechniqueLivre(
            "T09-2-1",
            9,
            "9.2",
            "Matrice de coûts et seuil recommandé (`sklearn.metrics.confusion_matrix`)",
            False,
        ),
        "T10-1-1": TechniqueLivre(
            "T10-1-1",
            10,
            "10.1",
            "K-means (`sklearn.cluster.KMeans`)",
            False,
        ),
        "T10-1-2": TechniqueLivre(
            "T10-1-2",
            10,
            "10.1",
            "Classification hiérarchique (`sklearn.cluster.AgglomerativeClustering`)",
            False,
        ),
        "T10-1-3": TechniqueLivre(
            "T10-1-3",
            10,
            "10.1",
            "DBSCAN (`sklearn.cluster.DBSCAN`)",
            False,
        ),
        "T10-2-1": TechniqueLivre(
            "T10-2-1",
            10,
            "10.2",
            "Isolation Forest (`sklearn.ensemble.IsolationForest`)",
            False,
        ),
        "T10-2-2": TechniqueLivre(
            "T10-2-2",
            10,
            "10.2",
            "Local Outlier Factor (`sklearn.neighbors.LocalOutlierFactor`)",
            False,
        ),
        "T10-3-1": TechniqueLivre(
            "T10-3-1",
            10,
            "10.3",
            "Machine à vecteurs de support à une classe (`sklearn.svm.OneClassSVM`)",
            False,
        ),
        "T11-1-1": TechniqueLivre(
            "T11-1-1",
            11,
            "11.1",
            "Analyse en composantes principales (`sklearn.decomposition.PCA`)",
            False,
        ),
        "T11-2-1": TechniqueLivre(
            "T11-2-1",
            11,
            "11.2",
            "Sélection de variables (`sklearn.feature_selection.RFECV`)",
            False,
        ),
        "T12-2-1": TechniqueLivre(
            "T12-2-1",
            12,
            "12.2",
            "Autorégression sur décalages (`sklearn.linear_model.Ridge`)",
            False,
        ),
        "T12-3-1": TechniqueLivre(
            "T12-3-1",
            12,
            "12.3",
            "Lissage exponentiel (`pandas.Series.ewm`)",
            False,
        ),
        "T13-1-1": TechniqueLivre(
            "T13-1-1",
            13,
            "13.1",
            "Perceptron et rétropropagation (`numpy`)",
            False,
        ),
        "T13-2-1": TechniqueLivre(
            "T13-2-1",
            13,
            "13.2",
            "Perceptron multicouche (`sklearn.neural_network.MLPRegressor`)",
            False,
        ),
        "T14-1-1": TechniqueLivre(
            "T14-1-1",
            14,
            "14.1",
            "Réseau dense en régression (`keras.layers.Dense`)",
            True,
        ),
        "T14-2-1": TechniqueLivre(
            "T14-2-1",
            14,
            "14.2",
            "Réseau dense en classification (`keras.layers.Dense`)",
            True,
        ),
        "T15-1-1": TechniqueLivre(
            "T15-1-1",
            15,
            "15.1",
            "Réseau convolutif (`keras.layers.Conv2D`)",
            True,
        ),
        "T16-1-1": TechniqueLivre(
            "T16-1-1",
            16,
            "16.1",
            "Réseau récurrent à mémoire (`keras.layers.LSTM`)",
            True,
        ),
        "T16-2-1": TechniqueLivre(
            "T16-2-1",
            16,
            "16.2",
            "Mécanisme d'attention (`keras.layers.MultiHeadAttention`)",
            True,
        ),
        "T17-1-1": TechniqueLivre(
            "T17-1-1",
            17,
            "17.1",
            "Substitut d'un calcul par éléments finis (`sklearn.preprocessing.PolynomialFeatures`)",
            False,
        ),
        "T17-2-1": TechniqueLivre(
            "T17-2-1",
            17,
            "17.2",
            "Loi de puissance par régression sur les logarithmes "
            "(`sklearn.compose.TransformedTargetRegressor`)",
            False,
        ),
        "T19-1-1": TechniqueLivre(
            "T19-1-1",
            19,
            "19.1",
            "Sérialisation d'un modèle et de son contexte (`joblib.dump`)",
            False,
        ),
        "T19-2-1": TechniqueLivre(
            "T19-2-1",
            19,
            "19.2",
            "Surveillance de dérive : indice de stabilité de population et carte de"
            " contrôle (`numpy.histogram`)",
            False,
        ),
        "T01-1-1": TechniqueLivre(
            "T01-1-1",
            1,
            "1.1",
            "Régression linéaire simple, première approche (`LinearRegression`), "
            "sans fiche : sous-section 1.1.2, page 5",
            False,
        ),
        "T01-1-2": TechniqueLivre(
            "T01-1-2",
            1,
            "1.2",
            "Arbre de régression, première approche (`DecisionTreeRegressor`), sans"
            " fiche : sous-section 1.2.2, page 10 ; fiche T08-1-1",
            False,
        ),
        "T01-2-1": TechniqueLivre(
            "T01-2-1",
            1,
            "1.2",
            "Base de test mise de côté (`train_test_split`), sans fiche : sous-section 1.2.1, page 9",
            False,
        ),
        "T01-2-2": TechniqueLivre(
            "T01-2-2",
            1,
            "1.2",
            "Surapprentissage et domaine de validité (interpoler contre "
            "extrapoler), sans fiche : sous-sections 1.2.2 et 1.2.3, pages 10 et 11",
            False,
        ),
        "T02-1-1": TechniqueLivre(
            "T02-1-1",
            2,
            "2.1",
            "Fuite de données et morcellement du jeu (apprentissage, validation, "
            "test), sans fiche : sous-sections 2.1.3 et 2.1.4, pages 16 à 19",
            False,
        ),
        "T02-1-2": TechniqueLivre(
            "T02-1-2",
            2,
            "2.1",
            "Référence naïve (`DummyClassifier`, `DummyRegressor`), sans fiche : "
            "sous-section 2.1.3, page 16 (classification) ; sous-section 5.1.4, "
            "page 86 (régression)",
            False,
        ),
        "T02-2-1": TechniqueLivre(
            "T02-2-1",
            2,
            "2.2",
            "Validation croisée (`KFold`, `StratifiedKFold`, `GroupKFold`, "
            "`cross_val_score`), sans fiche : sous-section 2.2.2, page 22",
            False,
        ),
        "T02-2-2": TechniqueLivre(
            "T02-2-2",
            2,
            "2.2",
            "Courbes de validation et d'apprentissage (`validation_curve`, "
            "`learning_curve`), sans fiche : sous-section 2.2.3, page 24",
            False,
        ),
        "T02-3-1": TechniqueLivre(
            "T02-3-1",
            5,
            "5.1",
            "Métriques de régression (R², MAE, RMSE, MAPE), sans fiche : sous-"
            "section 5.1.1, page 75 (MAPE : sous-section 12.3.2, page 299)",
            False,
        ),
        "T02-3-2": TechniqueLivre(
            "T02-3-2",
            2,
            "2.3",
            "Métriques de classification (matrice de confusion, exactitude, rappel,"
            " précision, F-score, moyenne macro), sans fiche : sous-sections 2.3.1 "
            "à 2.3.4, pages 28 à 33",
            False,
        ),
        "T03-1-1": TechniqueLivre(
            "T03-1-1",
            3,
            "3.1",
            "Statistiques descriptives et distributions, sans fiche : sous-section 3.1.3, page 39",
            False,
        ),
        "T03-1-2": TechniqueLivre(
            "T03-1-2",
            3,
            "3.1",
            "Valeurs aberrantes (règle de Tukey) et effet de levier, sans fiche : "
            "sous-sections 3.1.3 et 3.1.4, pages 39 et 40",
            False,
        ),
        "T03-1-3": TechniqueLivre(
            "T03-1-3",
            3,
            "3.1",
            "Profil d'un jeu : dimensions, types, manquants, valeurs uniques, "
            "constantes, doublons, sans fiche : sous-sections 3.1.1 et 3.1.2, pages"
            " 36 à 38",
            False,
        ),
        "T03-2-1": TechniqueLivre(
            "T03-2-1",
            3,
            "3.2",
            "Corrélations de Pearson et de Spearman, graphiques de paires "
            "(`pairplot`), sans fiche : sous-sections 3.2.1 et 3.2.2, pages 41 à 44",
            False,
        ),
        "T03-2-2": TechniqueLivre(
            "T03-2-2",
            3,
            "3.2",
            "Tableau croisé entre cible et catégorie (`crosstab`), sans fiche : sous-section 3.2.3, page 46",
            False,
        ),
        "T03-2-3": TechniqueLivre(
            "T03-2-3",
            3,
            "3.2",
            "Fiche de décisions ENISE : variable, constat, opération prévue, sans "
            "fiche : sous-section 3.2.4, page 49",
            False,
        ),
        "T04-1-1": TechniqueLivre(
            "T04-1-1",
            4,
            "4.2",
            "Valeurs manquantes, I de ENISE (`SimpleImputer`, `KNNImputer`, "
            "`MissingIndicator`), sans fiche : sous-section 4.2.1, page 59",
            False,
        ),
        "T04-1-2": TechniqueLivre(
            "T04-1-2",
            4,
            "4.1",
            "Encodage des catégories, E de ENISE (`OneHotEncoder`, "
            "`OrdinalEncoder`), sans fiche : sous-section 4.1.2, page 52",
            False,
        ),
        "T04-2-1": TechniqueLivre(
            "T04-2-1",
            4,
            "4.1",
            "Normalisation, N de ENISE, choisie selon la distribution "
            "(`StandardScaler`, `MinMaxScaler`, `RobustScaler`), sans fiche : sous-"
            "section 4.1.3, page 55",
            False,
        ),
        "T04-2-2": TechniqueLivre(
            "T04-2-2",
            4,
            "4.2",
            "Extraction, second E de ENISE (`PolynomialFeatures`, "
            "`PowerTransformer`, `KBinsDiscretizer`), sans fiche : sous-section "
            "4.2.4, page 65 ; fiche T05-1-2 pour le polynôme",
            False,
        ),
        "T04-2-3": TechniqueLivre(
            "T04-2-3",
            4,
            "4.2",
            "Chaîne de traitement sans fuite (`Pipeline`, `ColumnTransformer`), "
            "sans fiche : sous-sections 4.1.4 et 4.2.5, pages 56 et 67",
            False,
        ),
        "T04-2-4": TechniqueLivre(
            "T04-2-4",
            4,
            "4.2",
            "Sélection, S de ENISE, première approche (`VarianceThreshold`, "
            "`SelectKBest`), sans fiche : sous-section 4.2.2, page 60 ; fiche "
            "T11-2-1",
            False,
        ),
        "T12-1-1": TechniqueLivre(
            "T12-1-1",
            12,
            "12.1",
            "Index temporel, rééchantillonnage et moyennes glissantes (`resample`, "
            "`rolling`, `ewm`), sans fiche : sous-sections 12.1.1 et 12.1.2, pages "
            "277 et 278",
            False,
        ),
        "T12-1-2": TechniqueLivre(
            "T12-1-2",
            12,
            "12.1",
            "Validation glissante (`TimeSeriesSplit`) et référence par persistance,"
            " sans fiche : sous-section 12.1.3, page 279 ; fiche T12-2-1",
            False,
        ),
        "T12-4-1": TechniqueLivre(
            "T12-4-1",
            12,
            "12.4",
            "Cible à horizon de panne et classes déséquilibrées (`class_weight`), "
            "sans fiche : sous-sections 12.4.1 et 12.4.2, page 302",
            False,
        ),
    }
)


def est_identifiant(texte: object) -> bool:
    """Vrai si le texte est un identifiant du livre qui existe dans la liste."""
    return isinstance(texte, str) and texte in TECHNIQUES_LIVRE


def technique_du_livre(identifiant: str) -> TechniqueLivre:
    """Technique d'identifiant donné ; `KeyError` explicite sinon."""
    try:
        return TECHNIQUES_LIVRE[identifiant]
    except KeyError:
        raise KeyError(f"Renvoi au livre inconnu : {identifiant!r}.") from None
