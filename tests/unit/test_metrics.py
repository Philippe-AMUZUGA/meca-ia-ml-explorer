"""Métriques : valeurs calculées à la main, classe positive, cas indéfinis, coût, sans cible, registre."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mecaml.core import metrics
from mecaml.core.metrics import ErreurMetrique
from mecaml.core.types import Probleme

# ============================================================
# RÉGRESSION : valeurs vérifiables de tête
# ============================================================
Y = np.array([3.0, 5.0, 7.0, 9.0])
PRED = np.array([2.0, 5.0, 8.0, 9.0])  # erreurs : -1, 0, +1, 0


def test_regression_valeurs_a_la_main() -> None:
    # MAE = (1 + 0 + 1 + 0) / 4 ; RMSE = racine((1 + 0 + 1 + 0) / 4) ; R² = 1 - 2 / 20
    assert metrics.calculer("mae", y_vrai=Y, y_pred=PRED) == pytest.approx(0.5)
    assert metrics.calculer("rmse", y_vrai=Y, y_pred=PRED) == pytest.approx(math.sqrt(0.5))
    assert metrics.calculer("r2", y_vrai=Y, y_pred=PRED) == pytest.approx(1 - 2 / 20)


def test_r2_indefini_avec_une_ligne() -> None:
    with pytest.raises(ErreurMetrique, match="deux lignes"):
        metrics.calculer("r2", y_vrai=np.array([1.0]), y_pred=np.array([1.0]))


# ============================================================
# CLASSIFICATION BINAIRE
# ============================================================
# 10 lignes : 3 positives (1), 7 négatives (0). Prédictions : TP=2, FN=1, FP=2, TN=5.
Y_BIN = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P_BIN = np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0])


def test_classification_binaire_valeurs_a_la_main() -> None:
    donnees = {"y_vrai": Y_BIN, "y_pred": P_BIN, "classe_positive": 1}
    assert metrics.calculer("exactitude", **donnees) == pytest.approx(0.7)
    assert metrics.calculer("precision", **donnees) == pytest.approx(2 / 4)
    assert metrics.calculer("rappel", **donnees) == pytest.approx(2 / 3)
    assert metrics.calculer("f1", **donnees) == pytest.approx(2 * (0.5 * 2 / 3) / (0.5 + 2 / 3))
    # exactitude équilibrée = moyenne du rappel des deux classes = (2/3 + 5/7) / 2
    assert metrics.calculer("exactitude_equilibree", **donnees) == pytest.approx((2 / 3 + 5 / 7) / 2)


def test_l_exactitude_seule_trompe_sur_une_classe_rare() -> None:
    """Un modèle qui ne prédit jamais la classe rare a une bonne exactitude et un rappel nul."""
    y = np.array([1] * 5 + [0] * 95)
    pred = np.zeros(100, dtype=int)
    assert metrics.calculer("exactitude", y_vrai=y, y_pred=pred) == pytest.approx(0.95)
    assert metrics.calculer("rappel", y_vrai=y, y_pred=pred, classe_positive=1) == 0.0
    assert metrics.calculer("f1", y_vrai=y, y_pred=pred, classe_positive=1) == 0.0
    assert metrics.calculer("exactitude_equilibree", y_vrai=y, y_pred=pred) == pytest.approx(0.5)


def test_roc_auc_et_precision_moyenne_binaires() -> None:
    y = np.array([0, 0, 1, 1])
    parfait = np.array([0.1, 0.2, 0.8, 0.9])
    inverse = 1 - parfait
    constant = np.full(4, 0.5)
    assert metrics.calculer("roc_auc", y_vrai=y, scores=parfait, classe_positive=1) == pytest.approx(1.0)
    assert metrics.calculer("roc_auc", y_vrai=y, scores=inverse, classe_positive=1) == pytest.approx(0.0)
    assert metrics.calculer("roc_auc", y_vrai=y, scores=constant, classe_positive=1) == pytest.approx(0.5)
    assert metrics.calculer(
        "precision_moyenne", y_vrai=y, scores=parfait, classe_positive=1
    ) == pytest.approx(1.0)
    # la précision moyenne d'un score constant est la proportion de positifs
    assert metrics.calculer(
        "precision_moyenne", y_vrai=y, scores=constant, classe_positive=1
    ) == pytest.approx(0.5)


def test_scores_a_deux_colonnes_suivent_les_classes() -> None:
    y = np.array(["ok", "ok", "ko", "ko"])
    proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.8, 0.2]])  # colonnes déclarées : ko, ok
    juste = metrics.calculer("roc_auc", y_vrai=y, scores=proba, classes=["ko", "ok"], classe_positive="ko")
    assert juste == pytest.approx(1.0)
    # si l'on déclare l'ordre inverse des colonnes, la colonne lue pour « ko » change : le score s'effondre
    inverse = metrics.calculer("roc_auc", y_vrai=y, scores=proba, classes=["ok", "ko"], classe_positive="ko")
    assert inverse == pytest.approx(0.0)


def test_une_seule_classe_dans_le_pli_est_indefini() -> None:
    y = np.array([1, 1, 1])
    with pytest.raises(ErreurMetrique, match="seule classe"):
        metrics.calculer(
            "roc_auc", y_vrai=y, scores=np.array([0.2, 0.5, 0.9]), classes=[0, 1], classe_positive=1
        )
    with pytest.raises(ErreurMetrique, match="positive"):
        metrics.calculer(
            "precision_moyenne",
            y_vrai=np.array([0, 0, 0]),
            scores=np.array([0.1, 0.2, 0.3]),
            classes=[0, 1],
            classe_positive=1,
        )


# ============================================================
# CLASSE POSITIVE : PROPOSÉE, JAMAIS DÉDUITE EN SILENCE
# ============================================================
@pytest.mark.parametrize(
    ("y", "attendue", "origine"),
    [
        (np.array([0, 1, 0, 0]), 1, "zero_un"),
        (np.array([True, False, False]), True, "zero_un"),
        (np.array([1.0, 0.0, 0.0]), 1.0, "zero_un"),
        (np.array(["conforme"] * 8 + ["defaut"] * 2), "defaut", "lexique"),
        (
            np.array(["conforme"] * 2 + ["defaut"] * 8),
            "defaut",
            "lexique",
        ),  # le lexique l'emporte sur la rareté
        (np.array(["a"] * 6 + ["b"] * 2), "b", "rarete"),
        (np.array(["a", "a", "b", "b"]), "b", "rarete"),  # égalité : la dernière dans l'ordre alphabétique
    ],
)
def test_proposition_de_classe_positive(y: np.ndarray, attendue: object, origine: str) -> None:
    assert metrics.proposer_classe_positive(y) == (attendue, origine)


@pytest.mark.parametrize(
    ("classes", "attendue"),
    [
        (("conforme", "non-conforme"), "non-conforme"),  # Qualite_soudure : la classe saine est la plus rare
        (("critique", "Stable"), "critique"),  # Criticite_Fissures
        (("Defaut_present", "Pas_de_defaut"), "Defaut_present"),  # Identification_Conformite
        (("Pas_de_défaut", "Défaut présent"), "Défaut présent"),  # accents et majuscules
        (("conforme", "rebut"), "rebut"),  # Controle_Conformite
        (("ok", "panne"), "panne"),
        (("non", "oui"), "oui"),
        (("faux", "vrai"), "vrai"),
        (("sain", "fissure"), "fissure"),
    ],
)
def test_le_lexique_cherche_le_defaut_pas_la_piece_saine(classes: tuple[str, str], attendue: str) -> None:
    """Même quand la classe saine est la plus rare, la proposition reste celle du défaut."""
    rare_saine = np.array([classes[0]] * 3 + [classes[1]] * 7)
    rare_defaut = np.array([classes[0]] * 7 + [classes[1]] * 3)
    for y in (rare_saine, rare_defaut):
        assert metrics.proposer_classe_positive(y) == (attendue, "lexique")


def test_proposition_exige_deux_classes() -> None:
    with pytest.raises(ErreurMetrique, match="deux classes"):
        metrics.proposer_classe_positive(np.array([1, 2, 3]))


@pytest.mark.parametrize("identifiant", ["precision", "rappel", "f1", "precision_moyenne", "roc_auc", "cout"])
def test_sans_classe_positive_une_classification_binaire_est_refusee(identifiant: str) -> None:
    """Plus de classe la plus rare implicite : sans classe positive fixée, la métrique ne se calcule pas."""
    donnees = {"y_vrai": Y_BIN, "y_pred": P_BIN, "scores": np.linspace(0, 1, 10), "classes": [0, 1]}
    with pytest.raises(ErreurMetrique, match="Classe positive non fixée"):
        metrics.calculer(identifiant, **donnees)


def test_le_choix_de_la_classe_positive_change_les_mesures() -> None:
    donnees = {"y_vrai": Y_BIN, "y_pred": P_BIN, "classes": [0, 1]}
    sur_un = metrics.calculer("rappel", classe_positive=1, **donnees)
    sur_zero = metrics.calculer("rappel", classe_positive=0, **donnees)
    assert sur_un == pytest.approx(2 / 3) and sur_zero == pytest.approx(5 / 7)


def test_precision_rappel_f1_indefinis_sans_ligne_positive_dans_le_pli() -> None:
    """Un pli sans ligne positive donne NaN et une note, non un zéro qui ferait baisser la moyenne."""
    y = np.zeros(8, dtype=int)
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    for identifiant in ("precision", "rappel", "f1"):
        with pytest.raises(ErreurMetrique, match="Aucune ligne de la classe positive"):
            metrics.calculer(identifiant, y_vrai=y, y_pred=pred, classes=[0, 1], classe_positive=1)
    notes: list[str] = []
    valeurs = metrics.calculer_standard(
        Probleme.CLASSIFICATION, notes, y_vrai=y, y_pred=pred, classes=[0, 1], classe_positive=1
    )
    assert all(
        math.isnan(valeurs[nom]) for nom in ("precision", "rappel", "f1", "roc_auc", "precision_moyenne")
    )
    assert valeurs["exactitude"] == pytest.approx(7 / 8)
    assert sum("Aucune ligne de la classe positive" in n for n in notes) == 3


# ============================================================
# MULTICLASSE
# ============================================================
Y_MULTI = np.array(["a", "a", "b", "b", "c", "c"])
P_MULTI = np.array(["a", "b", "b", "b", "c", "a"])


def test_multiclasse_moyenne_macro() -> None:
    donnees = {"y_vrai": Y_MULTI, "y_pred": P_MULTI}
    # précision par classe : a = 1/2, b = 2/3, c = 1/1 ; rappel : a = 1/2, b = 1, c = 1/2
    assert metrics.calculer("precision", **donnees) == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert metrics.calculer("rappel", **donnees) == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert metrics.calculer("exactitude", **donnees) == pytest.approx(4 / 6)


def test_roc_auc_multiclasse_un_contre_reste() -> None:
    classes = ["a", "b", "c"]
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.1, 0.2, 0.7]]
    )
    assert metrics.calculer("roc_auc", y_vrai=Y_MULTI, scores=proba, classes=classes) == pytest.approx(1.0)
    assert metrics.calculer(
        "precision_moyenne", y_vrai=Y_MULTI, scores=proba, classes=classes
    ) == pytest.approx(1.0)


def test_matrice_de_confusion() -> None:
    attendue = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert (metrics.matrice_confusion(Y_MULTI, P_MULTI, ["a", "b", "c"]) == attendue).all()
    assert metrics.matrice_confusion(Y_MULTI, P_MULTI).shape == (3, 3)


# ============================================================
# COÛT DE DÉCISION
# ============================================================
def test_cout_binaire_a_la_main() -> None:
    # FP = 2, FN = 1 sur 10 lignes : (2 * 1 + 1 * 10) / 10
    assert metrics.cout_binaire(Y_BIN, P_BIN, 1, cout_fp=1.0, cout_fn=10.0) == pytest.approx(1.2)
    assert metrics.calculer(
        "cout:fp=1,fn=10", y_vrai=Y_BIN, y_pred=P_BIN, classe_positive=1
    ) == pytest.approx(1.2)
    assert metrics.calculer("cout", y_vrai=Y_BIN, y_pred=P_BIN, classe_positive=1) == pytest.approx(
        0.3
    )  # coûts 1 et 1 : taux d'erreur


def test_cout_au_seuil() -> None:
    scores = np.array([0.9, 0.8, 0.4, 0.6, 0.55, 0.2, 0.1, 0.1, 0.05, 0.0])
    # seuil 0.5 : positifs prédits = lignes 0, 1, 3, 4 -> TP=2 (0,1), FN=1 (2), FP=2 (3,4)
    assert metrics.cout_au_seuil(Y_BIN, scores, 1, 0.5, 1.0, 10.0) == pytest.approx(1.2)
    # seuil 0.3 : ligne 2 devient positive -> FN=0, FP=2
    assert metrics.cout_au_seuil(Y_BIN, scores, 1, 0.3, 1.0, 10.0) == pytest.approx(0.2)
    cout = metrics.calculer(
        "cout:fp=1,fn=10", y_vrai=Y_BIN, scores=scores, seuil=0.3, classes=[0, 1], classe_positive=1
    )
    assert cout == pytest.approx(0.2)


def test_cout_refus() -> None:
    with pytest.raises(ErreurMetrique, match="deux classes"):
        metrics.calculer("cout", y_vrai=Y_MULTI, y_pred=P_MULTI)
    with pytest.raises(ErreurMetrique, match="positifs ou nuls"):
        metrics.cout_binaire(Y_BIN, P_BIN, 1, cout_fp=-1.0)


# ============================================================
# ANALYSE DU TEXTE DE LA MÉTRIQUE
# ============================================================
def test_analyser() -> None:
    m, parametres = metrics.analyser("cout:fp=2.5,fn=10")
    assert m.id == "cout"
    assert parametres == {"fp": 2.5, "fn": 10.0}
    assert metrics.analyser("f1")[1] == {}


@pytest.mark.parametrize(
    ("texte", "motif"),
    [
        ("", "vide"),
        ("inconnue", "Métrique inconnue"),
        ("f1:fp=1", "invalide"),
        ("cout:xx=1", "invalide"),
        ("cout:fp=abc", "invalide"),
    ],
)
def test_analyser_refus(texte: str, motif: str) -> None:
    with pytest.raises(ErreurMetrique, match=motif):
        metrics.analyser(texte)


def test_valider_selon_le_probleme() -> None:
    assert metrics.valider("rmse", Probleme.REGRESSION).id == "rmse"
    assert metrics.valider("silhouette", Probleme.SANS_CIBLE).id == "silhouette"
    with pytest.raises(ErreurMetrique, match="ne s'applique pas"):
        metrics.valider("f1", Probleme.REGRESSION)
    with pytest.raises(ErreurMetrique, match="tableau"):
        metrics.valider("matrice_confusion", Probleme.CLASSIFICATION)
    with pytest.raises(ErreurMetrique, match="matrice_confusion"):
        metrics.calculer("matrice_confusion", y_vrai=Y_BIN, y_pred=P_BIN)


def test_donnee_absente_est_signalee() -> None:
    with pytest.raises(ErreurMetrique, match="y_pred"):
        metrics.calculer("f1", y_vrai=Y_BIN)


# ============================================================
# MÉTRIQUES STANDARD
# ============================================================
def test_calculer_standard_classification_avec_et_sans_scores() -> None:
    notes: list[str] = []
    valeurs = metrics.calculer_standard(
        Probleme.CLASSIFICATION, notes, y_vrai=Y_BIN, y_pred=P_BIN, classes=[0, 1], classe_positive=1
    )
    assert set(valeurs) == set(metrics.STANDARD[Probleme.CLASSIFICATION])
    assert math.isnan(valeurs["roc_auc"]) and math.isnan(valeurs["precision_moyenne"])
    assert any("roc_auc" in n for n in notes)
    assert valeurs["rappel"] == pytest.approx(2 / 3)


def test_calculer_standard_regression() -> None:
    assert set(metrics.calculer_standard(Probleme.REGRESSION, None, y_vrai=Y, y_pred=PRED)) == {
        "r2",
        "mae",
        "rmse",
        "mape",
    }
    assert metrics.calculer_standard(Probleme.SANS_CIBLE, None) == {}


# ============================================================
# SANS CIBLE
# ============================================================
def _amas() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    points = np.r_[rng.normal(0, 0.2, (30, 2)), rng.normal(6, 0.2, (30, 2))]
    return points, np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]


def test_silhouette_amas_separes_et_bruit_ecarte() -> None:
    points, etiquettes = _amas()
    assert metrics.calculer("silhouette", x=points, etiquettes=etiquettes) > 0.9
    avec_bruit = etiquettes.copy()
    avec_bruit[:3] = -1  # le bruit de DBSCAN n'est pas un groupe
    assert metrics.calculer("silhouette", x=points, etiquettes=avec_bruit) > 0.9
    with pytest.raises(ErreurMetrique, match="Silhouette indéfinie"):
        metrics.calculer("silhouette", x=points, etiquettes=np.zeros(60, dtype=int))


def test_rand_ajuste_invariant_aux_noms_de_groupes() -> None:
    _, etiquettes = _amas()
    permutees = 1 - etiquettes
    assert metrics.calculer("rand_ajuste", etiquettes=permutees, etiquettes_ref=etiquettes) == pytest.approx(
        1.0
    )
    hasard = np.random.default_rng(1).integers(0, 2, 60)
    assert abs(metrics.calculer("rand_ajuste", etiquettes=hasard, etiquettes_ref=etiquettes)) < 0.2


def test_variance_expliquee_et_taux_d_anomalies() -> None:
    from sklearn.decomposition import PCA

    points, _ = _amas()
    acp = PCA(n_components=1).fit(points)
    assert metrics.calculer("variance_expliquee", modele=acp) == pytest.approx(
        acp.explained_variance_ratio_.sum()
    )
    with pytest.raises(ErreurMetrique, match="composantes principales"):
        metrics.calculer("variance_expliquee", modele=object())
    assert metrics.calculer("taux_anomalies", etiquettes=np.array([1, 1, -1, 1])) == pytest.approx(0.25)


# ============================================================
# MODÈLE AJUSTÉ : SCORES CONTINUS, ÉVALUATEUR
# ============================================================
def _jeu_binaire() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 2))
    y = np.where(x[:, 0] + 0.3 * rng.normal(size=120) > 0, "defaut", "conforme")
    return x, y


def test_scores_du_modele_binaire_positive_choisie() -> None:
    from sklearn.linear_model import LogisticRegression

    x, y = _jeu_binaire()
    modele = LogisticRegression().fit(x, y)  # classes_ = ['conforme', 'defaut']
    pour_defaut = metrics.scores_du_modele(modele, x, ["conforme", "defaut"], "defaut")
    pour_conforme = metrics.scores_du_modele(modele, x, ["conforme", "defaut"], "conforme")
    assert pour_defaut.shape == (120,)
    assert np.allclose(pour_defaut + pour_conforme, 1.0)
    auc = metrics.evaluer_modele(
        modele, x, y, "roc_auc", classes=["conforme", "defaut"], classe_positive="defaut"
    )
    assert auc > 0.9


def test_scores_du_modele_decision_binaire_change_de_signe() -> None:
    from sklearn.svm import SVC

    x, y = _jeu_binaire()
    modele = SVC().fit(x, y)  # pas de predict_proba : fonction de décision
    pos = metrics.scores_du_modele(modele, x, ["conforme", "defaut"], "defaut")
    neg = metrics.scores_du_modele(modele, x, ["conforme", "defaut"], "conforme")
    assert np.allclose(pos, -neg)
    auc_pos = metrics.evaluer_modele(
        modele, x, y, "roc_auc", classes=["conforme", "defaut"], classe_positive="defaut"
    )
    auc_neg = metrics.evaluer_modele(
        modele, x, y, "roc_auc", classes=["conforme", "defaut"], classe_positive="conforme"
    )
    assert auc_pos > 0.9 and auc_neg > 0.9  # chaque classe positive est bien classée par son propre score


def test_scores_d_un_pli_ou_une_classe_manque() -> None:
    """Un modèle qui n'a vu que deux des trois classes rend quand même une matrice à trois colonnes."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    y = np.array(["a"] * 30 + ["b"] * 30)
    modele = LogisticRegression().fit(x, y)
    scores = metrics.scores_du_modele(modele, x, ["a", "b", "c"], None)
    assert scores.shape == (60, 3)
    assert (scores[:, 2] < scores[:, :2].min()).all()  # la classe jamais vue est classée en dernier


def test_scorer_au_format_scikit_learn_change_le_signe_des_erreurs() -> None:
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 2))
    y = x[:, 0] + rng.normal(0, 0.5, 50)
    modele = LinearRegression().fit(x, y)
    assert metrics.scorer("rmse")(modele, x, y) < 0  # plus grand est meilleur : l'erreur passe en négatif
    assert metrics.scorer("r2")(modele, x, y) > 0
    brut = metrics.evaluer_modele(modele, x, y, "rmse")
    assert metrics.scorer("rmse")(modele, x, y) == pytest.approx(-brut)


# ============================================================
# REGISTRE
# ============================================================
def test_registre_des_metriques_complet_et_coherent() -> None:
    ids = [m.id for m in metrics.METRIQUES]
    assert len(ids) == len(set(ids))
    attendues = {"r2", "mae", "rmse", "exactitude", "exactitude_equilibree", "precision", "rappel", "f1"}
    attendues |= {"roc_auc", "precision_moyenne", "matrice_confusion", "cout", "silhouette", "rand_ajuste"}
    assert attendues <= set(ids)
    for m in metrics.METRIQUES:
        assert m.libelle and m.ref_livre and m.texte_id == f"metrique.{m.id}"
        assert m.meilleur in (metrics.HAUT, metrics.BAS, metrics.NEUTRE)
        assert m.problemes
    assert not metrics.metrique("matrice_confusion").scalaire
    assert metrics.metrique("rmse").meilleur == metrics.BAS
    assert metrics.metrique("cout").meilleur == metrics.BAS
    assert metrics.metrique("f1").meilleur == metrics.HAUT


# ============================================================
# MAPE, EXACTITUDE ÉQUILIBRÉE DIRECTE, STABILITÉ DE POPULATION
# ============================================================
def test_mape_a_la_main_et_indefinie_si_une_valeur_vraie_est_nulle() -> None:
    # erreurs relatives : 1/3, 0, 1/7, 0 -> moyenne (1/3 + 1/7) / 4
    assert metrics.calculer("mape", y_vrai=Y, y_pred=PRED) == pytest.approx((1 / 3 + 1 / 7) / 4)
    with pytest.raises(ErreurMetrique, match="MAPE indéfinie"):
        metrics.calculer("mape", y_vrai=np.array([0.0, 1.0]), y_pred=np.array([0.5, 1.0]))
    assert metrics.metrique("mape").meilleur == metrics.BAS


def test_exactitude_equilibree_egale_celle_de_scikit_learn_sans_avertissement() -> None:
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b", "c"], 60, p=[0.6, 0.3, 0.1])
    pred = rng.choice(["a", "b", "c", "d"], 60)  # « d » est absente de y : sklearn s'en plaindrait
    attendu = balanced_accuracy_score(y, pred) if "d" not in pred else None
    obtenu = metrics.calculer(
        "exactitude_equilibree", y_vrai=y, y_pred=pred
    )  # filterwarnings=error : aucune alerte
    if attendu is not None:
        assert obtenu == pytest.approx(attendu)
    manuel = np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)])
    assert obtenu == pytest.approx(manuel)


def test_indice_de_stabilite_de_population() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 2000)
    assert metrics.indice_stabilite_population(reference, rng.normal(0, 1, 2000)) < 0.1  # stable
    assert metrics.indice_stabilite_population(reference, rng.normal(0.5, 1, 2000)) > 0.1  # bouge
    assert metrics.indice_stabilite_population(reference, rng.normal(2, 1, 2000)) > 0.25  # a changé
    with pytest.raises(ErreurMetrique, match="constante"):
        metrics.indice_stabilite_population(np.ones(50), np.ones(50))
    with pytest.raises(ErreurMetrique, match="trop courts"):
        metrics.indice_stabilite_population([1.0], [1.0])


def test_renvois_au_livre_des_metriques() -> None:
    from mecaml.core.livre import est_identifiant

    assert all(est_identifiant(m.ref_livre) for m in metrics.METRIQUES), [m.id for m in metrics.METRIQUES]
    assert (
        metrics.metrique("cout").ref_livre == "T09-2-1" and metrics.metrique("roc_auc").ref_livre == "T09-1-1"
    )
