"""Lecture hors pli et sur le test : diagnostics, seuil, évaluateur du coffre (I1), courbe d'apprentissage."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml.core import evaluate, metrics, train
from mecaml.core.preprocess import colonnes_d_entree
from mecaml.core.session import (
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Etat,
    FixerObjectif,
    Pretraiter,
    Session,
    Tester,
)
from mecaml.core.types import ChoixTechnique, Decision, Probleme

CT = ChoixTechnique.creer
RIDGE = CT("regression.ridge")
ARBRE = CT("regression.arbre", max_depth=3)
LOGISTIQUE = CT("classification.logistique")
ARBRE_CLS = CT("classification.arbre", max_depth=3)


# ============================================================
# PROJETS PARTAGÉS
# ============================================================
def _mener(dossier: Path, jeu: pd.DataFrame, fabrique, **options) -> aides.Projet:
    return aides.mener_le_projet(aides.ecrire_csv(jeu, dossier / "jeu.csv"), fabrique, **options)


def _spec_ridge(_table):
    return jeux.spec_regression(modeles=(RIDGE,))


@pytest.fixture(scope="module")
def regression(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("evaluate_regression")
    return _mener(dossier, jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE, ARBRE)))


@pytest.fixture(scope="module")
def classification(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("evaluate_classification")
    jeu = jeux.jeu_classification(n=300)
    return _mener(dossier, jeu, lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS)))


@pytest.fixture(scope="module")
def multiclasse(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("evaluate_multiclasse")
    jeu = jeux.jeu_classification(n=300, classes=3)
    return _mener(dossier, jeu, lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,), binaire=False))


@pytest.fixture(scope="module")
def cible_binaire_entiere(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    """Classes 0 et 1 (pannes codées en entiers, cas courant des jeux de défauts)."""
    dossier = tmp_path_factory.mktemp("evaluate_cible_entiere")
    jeu = jeux.jeu_classification(n=300)
    jeu["y"] = (jeu["y"] == "defaut").astype(int)
    return _mener(
        dossier,
        jeu,
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS), classe_positive=1),
    )


@pytest.fixture(scope="module")
def cible_multiclasse_entiere(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    """Quatre classes codées 0 à 3."""
    dossier = tmp_path_factory.mktemp("evaluate_cible_chiffres")
    jeu = jeux.jeu_classification(n=300, classes=4)
    jeu["y"] = jeu["y"].str.removeprefix("classe_").astype(int)
    return _mener(dossier, jeu, lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,), binaire=False))


@pytest.fixture(scope="module")
def cible_entiere_chronologique(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    """Série ordonnée, cible 0 et 1 : la fenêtre croissante ne valide pas les premières dates."""
    dossier = tmp_path_factory.mktemp("evaluate_cible_entiere_serie")
    jeu = aides.jeu_chronologique()
    jeu["y"] = (jeu["y"] > jeu["y"].median()).astype(int)

    def spec(table):
        base = aides.spec_chronologique(table)
        return dataclasses.replace(
            base,
            probleme=Probleme.CLASSIFICATION,
            metrique="f1",
            classe_positive=1,
            modeles=(LOGISTIQUE, ARBRE_CLS),
        )

    return _mener(dossier, jeu, spec)


@pytest.fixture(scope="module")
def chronologique(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("evaluate_chronologique")
    return _mener(dossier, aides.jeu_chronologique(), aides.spec_chronologique)


# ============================================================
# FONCTIONS PURES
# ============================================================
def test_un_score_egal_au_seuil_est_declare_positif() -> None:
    predites = evaluate.predire_au_seuil([0.2, 0.5, 0.8], ("conforme", "defaut"), "defaut", 0.5)
    assert predites.tolist() == ["conforme", "defaut", "defaut"]


def test_le_seuil_s_applique_a_des_classes_numeriques_dans_n_importe_quel_ordre() -> None:
    assert evaluate.predire_au_seuil([0.1, 0.9], (1, 0), 0, 0.5).tolist() == [1, 0]
    assert evaluate.predire_au_seuil(np.array([0.4]), (0, 1), 1, 0.3).tolist() == [1]


def test_un_seuil_se_regle_pour_deux_classes() -> None:
    with pytest.raises(ValueError, match="deux classes"):
        evaluate.predire_au_seuil([0.1], ("a", "b", "c"), "a", 0.5)


def test_l_empreinte_des_predictions_ne_change_qu_avec_une_prediction() -> None:
    a = np.array([1.0, 2.0, 3.5])
    assert evaluate.empreinte_predictions(a) == evaluate.empreinte_predictions(a.copy())
    assert evaluate.empreinte_predictions(a) != evaluate.empreinte_predictions(a + np.array([0, 0, 1e-12]))
    assert evaluate.empreinte_predictions(np.array([1, 2])) == evaluate.empreinte_predictions(
        np.array([1.0, 2.0])
    )
    etiquettes = np.array(["conforme", "defaut"], dtype=object)
    assert evaluate.empreinte_predictions(etiquettes) == evaluate.empreinte_predictions(
        np.array(["conforme", "defaut"])
    )
    assert evaluate.empreinte_predictions(etiquettes) != evaluate.empreinte_predictions(etiquettes[::-1])


# ============================================================
# LIGNES D'APPRENTISSAGE DU PROJET
# ============================================================
def test_les_lignes_d_apprentissage_sont_celles_du_coffre_et_jamais_celles_du_test(
    regression: aides.Projet,
) -> None:
    app = evaluate.apprentissage_du_projet(regression.spec, regression.table, regression.coffre)
    assert list(app.X.columns) == list(colonnes_d_entree(regression.spec))
    assert len(app.X) == len(app.y) == len(app.donnees) == len(app.positions)
    test = np.setdiff1d(np.arange(regression.table.n_lignes), regression.coffre.positions_apprentissage())
    assert not set(app.positions) & set(test)
    assert app.contexte is None and app.n_sans_cible == 0
    assert app.y.dtype == float


def test_les_lignes_sans_cible_forment_le_contexte_et_sortent_de_la_validation(tmp_path: Path) -> None:
    projet = _mener(tmp_path, aides.jeu_serie_lacunaire(), aides.spec_lacunaire)
    app = evaluate.apprentissage_du_projet(projet.spec, projet.table, projet.coffre)
    assert app.n_sans_cible == len(app.contexte) > 0
    assert app.donnees[projet.spec.cible].notna().all()
    assert len(app.X) + app.n_sans_cible == projet.coffre.n_apprentissage
    assert list(app.contexte.columns) == list(app.X.columns)
    assert projet.resultat.n_sans_cible == app.n_sans_cible


def test_les_lignes_d_apprentissage_exigent_un_coffre_de_cette_table(
    regression: aides.Projet, tmp_path: Path
) -> None:
    spec, table = regression.spec, regression.table
    with pytest.raises(evaluate.ErreurEvaluation, match="sous scellé"):
        evaluate.apprentissage_du_projet(spec, table, None)
    autre = _mener(tmp_path, jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE,)))
    with pytest.raises(evaluate.ErreurEvaluation, match="pas été construit pour cette table"):
        evaluate.apprentissage_du_projet(spec, table, autre.coffre)
    with pytest.raises(evaluate.ErreurEvaluation, match="pas été construit pour cette table"):
        evaluate.apprentissage_du_projet(spec.avec(part_test=0.3), table, regression.coffre)
    sans_cible = jeux.spec_sans_cible()
    with pytest.raises(evaluate.ErreurEvaluation, match="sans cible"):
        evaluate.apprentissage_du_projet(sans_cible, table, regression.coffre)


# ============================================================
# DIAGNOSTIC HORS PLI : RÉGRESSION
# ============================================================
def test_les_residus_hors_pli_sont_le_vrai_moins_le_predit(regression: aides.Projet) -> None:
    res = regression.resultat
    diag = evaluate.diagnostiquer_validation(res, "regression.ridge")
    modele = res.modele("regression.ridge")
    assert diag.n == len(res.y_vrai) and diag.origine == "validation"
    assert np.array_equal(diag.residus, np.asarray(res.y_vrai) - np.asarray(modele.predictions))
    resume = diag.resume()
    assert resume["rmse"] == pytest.approx(math.sqrt(np.mean(diag.residus**2)))
    assert resume["mae"] == pytest.approx(np.mean(np.abs(diag.residus)))
    assert resume["biais"] == pytest.approx(np.mean(diag.residus))
    assert resume["ecart_maximal"] == pytest.approx(np.max(np.abs(diag.residus)))
    assert set(diag.plis) == set(range(len(res.plis)))
    assert np.array_equal(diag.positions, res.positions)


def test_une_ligne_jamais_validee_n_est_pas_lue(chronologique: aides.Projet) -> None:
    res = chronologique.resultat
    diag = evaluate.diagnostiquer_validation(res, "regression.ridge")
    assert diag.n < len(res.y_vrai)  # la fenêtre croissante ne valide pas les premières dates
    assert not np.isnan(diag.y_pred.astype(float)).any()
    validees = np.concatenate([val for _, val in res.plis])
    assert set(
        np.flatnonzero(~np.isnan(np.asarray(res.modele("regression.ridge").predictions, float)))
    ) == set(validees)


def test_les_residus_se_lisent_en_regression_seulement(classification: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    with pytest.raises(evaluate.ErreurEvaluation, match="régression"):
        _ = diag.residus
    with pytest.raises(evaluate.ErreurEvaluation, match="Les résidus"):
        evaluate.figure_regression(diag)


def test_un_modele_en_echec_n_a_pas_de_diagnostic(regression: aides.Projet) -> None:
    with pytest.raises(KeyError):
        evaluate.diagnostiquer_validation(regression.resultat, "regression.inconnu")
    modele_en_echec = train.ResultatModele(CT("regression.ridge"), {}, 0.0, erreur="ValueError : essai")
    resultat = type(regression.resultat)(**{**regression.resultat.__dict__, "modeles": (modele_en_echec,)})
    with pytest.raises(evaluate.ErreurEvaluation, match="pas de prédictions"):
        evaluate.diagnostiquer_validation(resultat, "regression.ridge")


# ============================================================
# DIAGNOSTIC HORS PLI : CLASSIFICATION
# ============================================================
def test_la_matrice_de_confusion_hors_pli_est_celle_de_scikit_learn(classification: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    attendue = confusion_matrix(diag.y_vrai, diag.y_pred, labels=list(diag.classes))
    assert np.array_equal(diag.matrice_confusion(), attendue)
    assert diag.matrice_confusion().sum() == diag.n
    assert diag.resume()["erreurs"] == diag.n - np.trace(attendue)


def test_une_cible_entiere_garde_son_type_hors_pli_et_se_diagnostique(
    cible_binaire_entiere: aides.Projet, cible_multiclasse_entiere: aides.Projet
) -> None:
    """Les prédictions hors pli ont le type des classes, jamais un tableau d'objets."""
    for projet, identifiant, forme in (
        (cible_binaire_entiere, "classification.logistique", (2, 2)),
        (cible_binaire_entiere, "classification.arbre", (2, 2)),
        (cible_multiclasse_entiere, "classification.logistique", (4, 4)),
    ):
        resultat = projet.resultat
        assert np.asarray(resultat.modele(identifiant).predictions).dtype.kind == "i"
        diag = evaluate.diagnostiquer_validation(resultat, identifiant)
        assert diag.y_pred.dtype.kind == "i" and diag.y_vrai.dtype.kind == "i"
        matrice = diag.matrice_confusion()
        assert matrice.shape == forme and matrice.sum() == diag.n == len(resultat.y_vrai)
        assert np.array_equal(matrice, confusion_matrix(diag.y_vrai, diag.y_pred, labels=list(diag.classes)))
        evaluate.figure_confusion(diag)


def test_une_cible_entiere_binaire_donne_roc_et_seuil(cible_binaire_entiere: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(
        cible_binaire_entiere.resultat, "classification.logistique", seuil=0.3
    )
    assert diag.binaire and diag.classe_positive == 1 and set(np.unique(diag.y_pred)) <= {0, 1}
    assert 0.5 < diag.courbe_roc()[2] <= 1.0
    evaluate.figure_roc_pr(diag)


def test_une_cible_entiere_chronologique_ecarte_les_lignes_jamais_validees(
    cible_entiere_chronologique: aides.Projet,
) -> None:
    resultat = cible_entiere_chronologique.resultat
    diag = evaluate.diagnostiquer_validation(resultat, "classification.logistique")
    validees = np.concatenate([val for _, val in resultat.plis])
    assert diag.n == len(set(validees)) < len(resultat.y_vrai)
    assert diag.y_pred.dtype.kind == "i" and set(np.unique(diag.y_pred)) <= {0, 1}
    assert diag.matrice_confusion().sum() == diag.n


def test_les_courbes_roc_et_precision_rappel_hors_pli(classification: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    reel = diag.y_vrai == diag.classe_positive
    fpr, tpr, aire = diag.courbe_roc()
    assert aire == pytest.approx(roc_auc_score(reel, diag.scores))
    assert fpr[0] == 0 and tpr[-1] == 1
    precision, rappel, moyenne = diag.courbe_pr()
    assert moyenne == pytest.approx(average_precision_score(reel, diag.scores))
    assert rappel[0] == 1 and precision[-1] == 1


def test_la_courbe_selon_le_seuil_recoupe_le_cout_de_la_metrique(classification: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    courbe = diag.courbe_seuil(cout_fp=1.0, cout_fn=7.0, points=21)
    assert list(courbe.columns) == [
        "seuil",
        "precision",
        "rappel",
        "f1",
        "exactitude",
        "cout",
        "positifs_predits",
    ]
    reel = diag.y_vrai == diag.classe_positive
    for ligne in courbe.itertuples():
        attendu = metrics.cout_au_seuil(reel, diag.scores, True, ligne.seuil, 1.0, 7.0)
        assert ligne.cout == pytest.approx(attendu)
        predit = diag.scores >= ligne.seuil
        tp, fp, fn = (reel & predit).sum(), (~reel & predit).sum(), (reel & ~predit).sum()
        assert ligne.positifs_predits == tp + fp
        if tp + fn:
            assert ligne.rappel == pytest.approx(tp / (tp + fn))
        if 2 * tp + fp + fn:
            assert ligne.f1 == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert courbe["rappel"].dropna().is_monotonic_decreasing  # plus le seuil monte, moins on détecte
    assert courbe["seuil"].is_monotonic_increasing and len(courbe) <= 21


def test_un_seuil_change_les_etiquettes_hors_pli_et_pas_les_scores(classification: aides.Projet) -> None:
    sans = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    avec = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique", seuil=0.2)
    assert avec.seuil == 0.2 and np.array_equal(avec.scores, sans.scores)
    attendues = evaluate.predire_au_seuil(sans.scores, sans.classes, sans.classe_positive, 0.2)
    assert np.array_equal(avec.y_pred, attendues)
    assert (avec.y_pred == avec.classe_positive).sum() > (sans.y_pred == sans.classe_positive).sum()


def test_un_seuil_ne_se_regle_que_pour_deux_classes_a_scores(
    multiclasse: aides.Projet, regression: aides.Projet
) -> None:
    with pytest.raises(evaluate.ErreurEvaluation, match="deux classes dont le modèle donne les scores"):
        evaluate.diagnostiquer_validation(multiclasse.resultat, "classification.logistique", seuil=0.5)
    diag = evaluate.diagnostiquer_validation(multiclasse.resultat, "classification.logistique")
    assert not diag.binaire and diag.matrice_confusion().shape == (3, 3)
    with pytest.raises(evaluate.ErreurEvaluation, match="deux classes avec scores"):
        diag.courbe_roc()
    with pytest.raises(evaluate.ErreurEvaluation, match="deux classes avec scores"):
        diag.courbe_seuil()
    with pytest.raises(evaluate.ErreurEvaluation):
        evaluate.diagnostiquer_validation(regression.resultat, "regression.ridge", seuil=0.5)


def test_la_courbe_selon_le_seuil_a_un_nombre_de_points_borne(classification: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    for points in (1, 5000):
        with pytest.raises(evaluate.ErreurEvaluation, match="de 2 à 1001 points"):
            diag.courbe_seuil(points=points)


# ============================================================
# SUR LES LIGNES DU TEST
# ============================================================
def _modele_final(projet: aides.Projet) -> train.ModeleAjuste:
    choix = next(c for c in projet.spec.modeles if c.id == projet.decision.modele_id)
    return train.ajuster_final(projet.spec, choix, projet.table, projet.coffre)


def test_les_scores_sur_des_lignes_sont_ceux_de_train(classification: aides.Projet) -> None:
    modele = _modele_final(classification)
    lignes = classification.table.donnees.iloc[:60]
    scores, diag, sans_cible = evaluate.diagnostiquer_lignes(modele, classification.spec, lignes)
    attendus = train.evaluer_lignes(modele, classification.spec, lignes)
    assert scores == dict(attendus) and sans_cible == attendus.n_sans_cible == 0
    assert diag.origine == "test" and diag.n == attendus.n_lignes == 60


def test_les_lignes_sans_cible_sont_predites_puis_comptees(tmp_path: Path) -> None:
    projet = _mener(tmp_path, aides.jeu_serie_lacunaire(), aides.spec_lacunaire)
    modele = _modele_final(projet)
    lignes = projet.table.donnees.iloc[:50]
    scores, diag, sans_cible = evaluate.diagnostiquer_lignes(modele, projet.spec, lignes)
    assert sans_cible == int(lignes[projet.spec.cible].isna().sum()) > 0
    assert diag.n == len(lignes) - sans_cible and not np.isnan(diag.y_pred.astype(float)).any()
    attendus = train.evaluer_lignes(modele, projet.spec, lignes)
    assert scores == dict(attendus) and sans_cible == attendus.n_sans_cible


def test_le_seuil_de_la_decision_gouverne_les_etiquettes_du_test(classification: aides.Projet) -> None:
    modele = _modele_final(classification)
    lignes = classification.table.donnees.iloc[:80]
    sans_seuil, diag_sans, _ = evaluate.diagnostiquer_lignes(modele, classification.spec, lignes)
    avec_seuil, diag_avec, _ = evaluate.diagnostiquer_lignes(modele, classification.spec, lignes, seuil=0.15)
    assert diag_avec.seuil == 0.15
    assert avec_seuil["rappel"] >= sans_seuil["rappel"]  # un seuil bas déclare plus de positifs
    assert avec_seuil["roc_auc"] == sans_seuil["roc_auc"]  # il ne touche pas aux scores
    assert (diag_avec.y_pred == diag_avec.classe_positive).sum() >= (
        diag_sans.y_pred == diag_sans.classe_positive
    ).sum()


def test_l_evaluation_suppose_une_cible() -> None:
    modele = train.ModeleAjuste(CT("regression.ridge"), None, None, None)
    with pytest.raises(evaluate.ErreurEvaluation, match="suppose une cible"):
        evaluate.diagnostiquer_lignes(modele, jeux.spec_sans_cible(), pd.DataFrame())


# ============================================================
# I1 : L'ÉVALUATEUR AJUSTE AVANT L'OUVERTURE ET NE LIT RIEN D'AUTRE ENSUITE
# ============================================================
def test_le_modele_est_ajuste_avant_l_ouverture_du_coffre(classification: aides.Projet) -> None:
    coffre = classification.coffre
    avant = coffre.ouvertures
    evaluer = evaluate.preparer_test(
        classification.spec, classification.table, coffre, classification.decision, importances=False
    )
    assert coffre.ouvertures == avant  # la préparation n'ouvre rien
    lignes = coffre.ouvrir(classification.decision)
    resultat = evaluer(lignes)
    assert coffre.ouvertures == avant + 1
    assert isinstance(resultat, evaluate.ResultatTest)
    assert resultat.modele is not None and resultat.modele_id == classification.decision.modele_id


def test_un_ajustement_qui_echoue_laisse_le_test_scelle(
    regression: aides.Projet, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projet = _mener(tmp_path, jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE,)))

    def echec(*_, **__):
        raise train.ErreurEntrainement("ajustement impossible")

    monkeypatch.setattr(train, "ajuster_final", echec)
    with pytest.raises(train.ErreurEntrainement, match="ajustement impossible"):
        evaluate.preparer_test(projet.spec, projet.table, projet.coffre, projet.decision)
    assert projet.coffre.ouvertures == 0


def test_l_evaluateur_ne_refait_aucun_ajustement_et_ne_touche_pas_aux_lignes(
    regression: aides.Projet,
) -> None:
    evaluer = evaluate.preparer_test(
        regression.spec, regression.table, regression.coffre, regression.decision, importances=False
    )
    lignes = regression.coffre.ouvrir(regression.decision)
    copie = lignes.copy()
    # Le pipeline ajusté est celui du résultat : un second appel à `fit` le changerait.
    resultat = evaluer(lignes)
    pipeline = resultat.modele.pipeline
    coefficients = pipeline[-1].coef_.copy() if hasattr(pipeline[-1], "coef_") else None
    evaluer(lignes)
    pd.testing.assert_frame_equal(lignes, copie)
    if coefficients is not None:
        assert np.array_equal(pipeline[-1].coef_, coefficients)


def test_le_modele_evalue_ne_depend_pas_des_valeurs_du_test(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=120)
    (tmp_path / "n").mkdir()
    (tmp_path / "s").mkdir()
    normal = _mener(tmp_path / "n", jeu, _spec_ridge)
    saccage = jeu.copy()
    test = np.setdiff1d(np.arange(len(jeu)), normal.coffre.positions_apprentissage())
    saccage.loc[test, ["a", "b", "d"]] = 1e6
    saccage.loc[test, "y"] = -1e6
    autre = _mener(tmp_path / "s", saccage, _spec_ridge)
    assert np.array_equal(normal.coffre.positions_apprentissage(), autre.coffre.positions_apprentissage())
    m1 = evaluate.preparer_test(normal.spec, normal.table, normal.coffre, normal.decision, importances=False)
    m2 = evaluate.preparer_test(autre.spec, autre.table, autre.coffre, autre.decision, importances=False)
    r1 = m1(normal.coffre.ouvrir(normal.decision))
    r2 = m2(autre.coffre.ouvrir(autre.decision))
    assert np.array_equal(r1.modele.pipeline[-1].coef_, r2.modele.pipeline[-1].coef_)  # même ajustement
    assert r1.scores != r2.scores  # seul le jugement change avec les valeurs du test


def test_le_resultat_du_test_ne_garde_aucune_ligne_du_test(classification: aides.Projet) -> None:
    evaluer = evaluate.preparer_test(
        classification.spec,
        classification.table,
        classification.coffre,
        classification.decision,
        importances=True,
        repetitions=3,
    )
    resultat = evaluer(classification.coffre.ouvrir(classification.decision))
    assert resultat.diagnostic.n == resultat.n_lignes
    assert resultat.valeur() == resultat.scores[classification.spec.metrique]
    assert list(resultat.tableau().columns) == ["metrique", "valeur"]
    assert resultat.importances is not None and set(resultat.importances["variable"]) == set(
        classification.spec.variables()
    )
    assert resultat.empreinte_predictions == evaluate.empreinte_predictions(resultat.diagnostic.y_pred)
    for champ in ("diagnostic", "importances"):
        assert not isinstance(getattr(resultat, champ), pd.DataFrame) or list(
            getattr(resultat, champ).columns
        ) == ["variable", "moyenne", "ecart_type", "plis_valides"]


def test_l_evaluateur_dans_la_session_ouvre_le_test_une_seule_fois(
    regression: aides.Projet, tmp_path: Path
) -> None:
    projet = _mener(tmp_path, jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE,)))
    session = Session()
    for action in (
        Charger(projet.table),
        Auditer(),
        FixerObjectif(projet.spec),
        Decouper(projet.coffre, spec=projet.spec),
        Pretraiter(projet.spec),
        Entrainer(projet.resultat, spec=projet.resultat.spec),
        Decider(projet.decision),
    ):
        session = session.appliquer(action)
    assert session.etat is Etat.DIAGNOSTIC and projet.coffre.ouvertures == 0
    evaluer = evaluate.preparer_test(
        projet.spec, projet.table, projet.coffre, projet.decision, importances=False
    )
    session = session.appliquer(Tester(evaluer))
    assert session.etat is Etat.TESTE and projet.coffre.ouvertures == 1 and not session.test_optimiste
    assert isinstance(session.resultat_test, evaluate.ResultatTest)


def test_l_evaluateur_applique_le_seuil_de_la_decision(classification: aides.Projet) -> None:
    avec_seuil = dataclasses.replace(classification.decision, seuil=0.15)
    coffre = classification.coffre
    spec, table = classification.spec, classification.table
    sans = evaluate.preparer_test(spec, table, coffre, classification.decision, importances=False)(
        coffre.ouvrir(classification.decision)
    )
    avec = evaluate.preparer_test(spec, table, coffre, avec_seuil, importances=False)(
        coffre.ouvrir(avec_seuil)
    )
    assert sans.seuil is None and avec.seuil == 0.15 and avec.diagnostic.seuil == 0.15
    positives = [(r.diagnostic.y_pred == r.classe_positive).sum() for r in (sans, avec)]
    assert positives[1] > positives[0]  # un seuil bas déclare plus de positifs que le seuil par défaut
    assert avec.scores["roc_auc"] == sans.scores["roc_auc"]  # il ne touche pas aux scores
    assert avec.empreinte_predictions != sans.empreinte_predictions


def test_l_evaluateur_refuse_ce_qui_ne_se_teste_pas(regression: aides.Projet) -> None:
    sans_cible = jeux.spec_sans_cible()
    with pytest.raises(evaluate.ErreurEvaluation, match=r"sans cible|pas de jeu de test"):
        evaluate.preparer_test(sans_cible, regression.table, regression.coffre, regression.decision)
    inconnu = Decision("regression.foret", regression.spec.metrique, 1.0)
    with pytest.raises(evaluate.ErreurEvaluation, match="ne figure pas parmi les modèles comparés"):
        evaluate.preparer_test(regression.spec, regression.table, regression.coffre, inconnu)


# ============================================================
# COURBE D'APPRENTISSAGE
# ============================================================
def test_la_courbe_d_apprentissage_suit_les_plis_et_garde_le_sens_de_la_metrique(
    regression: aides.Projet,
) -> None:
    courbe = evaluate.courbe_d_apprentissage(
        regression.resultat, "regression.ridge", regression.table, regression.coffre, points=4
    )
    assert courbe.apprentissage.shape == courbe.validation.shape == (4, len(regression.resultat.plis))
    assert list(courbe.tailles) == sorted(courbe.tailles)
    assert courbe.sens == metrics.BAS and (courbe.validation > 0).all()  # une erreur reste positive
    moyennes = courbe.moyennes()
    assert list(moyennes.columns)[:2] == ["taille", "apprentissage"]
    assert moyennes["validation"].iloc[-1] == pytest.approx(
        regression.resultat.valeur(regression.resultat.modele("regression.ridge")), rel=0.15
    )
    assert courbe.ecart_final() >= 0


def test_la_courbe_d_apprentissage_d_une_classification_va_dans_le_sens_haut(
    classification: aides.Projet,
) -> None:
    courbe = evaluate.courbe_d_apprentissage(
        classification.resultat,
        "classification.logistique",
        classification.table,
        classification.coffre,
        points=3,
    )
    assert courbe.sens == metrics.HAUT and ((courbe.validation >= 0) | np.isnan(courbe.validation)).all()


def test_la_courbe_d_apprentissage_ne_lit_pas_le_test(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=120)
    (tmp_path / "n").mkdir()
    (tmp_path / "s").mkdir()
    normal = _mener(tmp_path / "n", jeu, _spec_ridge)
    saccage = jeu.copy()
    test = np.setdiff1d(np.arange(len(jeu)), normal.coffre.positions_apprentissage())
    saccage.loc[test, ["a", "b", "d", "y"]] = 1e6
    autre = _mener(tmp_path / "s", saccage, _spec_ridge)
    c1 = evaluate.courbe_d_apprentissage(
        normal.resultat, "regression.ridge", normal.table, normal.coffre, points=3
    )
    c2 = evaluate.courbe_d_apprentissage(
        autre.resultat, "regression.ridge", autre.table, autre.coffre, points=3
    )
    assert np.array_equal(c1.validation, c2.validation) and np.array_equal(c1.apprentissage, c2.apprentissage)


def test_la_courbe_d_apprentissage_refuse_un_modele_hors_comparaison(regression: aides.Projet) -> None:
    with pytest.raises(evaluate.ErreurEvaluation, match="ne figure pas"):
        evaluate.courbe_d_apprentissage(
            regression.resultat, "regression.foret", regression.table, regression.coffre
        )


# ============================================================
# FIGURES
# ============================================================
def _textes(fig) -> str:
    morceaux = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    for axe in fig.axes:
        morceaux += [axe.get_title(), axe.get_xlabel(), axe.get_ylabel()]
    return " ".join(morceaux)


def test_les_figures_de_regression_ont_trois_panneaux_en_francais(regression: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(regression.resultat, "regression.ridge")
    fig = evaluate.figure_regression(diag)
    assert len(fig.axes) == 3
    texte = _textes(fig)
    assert (
        "valeur prédite" in texte and "résidu" in texte and "Régression de crête (Ridge) (hors pli" in texte
    )


def test_les_figures_de_classification(classification: aides.Projet, multiclasse: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(classification.resultat, "classification.logistique")
    assert "classe réelle" in _textes(evaluate.figure_confusion(diag))
    assert len(evaluate.figure_roc_pr(diag).axes) == 2
    courbe = diag.courbe_seuil()
    fig = evaluate.figure_seuil(courbe, seuil=0.3, seuil_cout=0.4)
    assert len(fig.axes) == 2 and any(len(a.lines) >= 2 for a in fig.axes)
    multi = evaluate.diagnostiquer_validation(multiclasse.resultat, "classification.logistique")
    assert len(evaluate.figure_confusion(multi).axes) >= 1
    with pytest.raises(evaluate.ErreurEvaluation):
        evaluate.figure_roc_pr(multi)


def test_la_figure_de_la_courbe_d_apprentissage(regression: aides.Projet) -> None:
    courbe = evaluate.courbe_d_apprentissage(
        regression.resultat, "regression.ridge", regression.table, regression.coffre, points=3
    )
    fig = evaluate.figure_courbe_apprentissage(courbe)
    assert "lignes d'apprentissage" in _textes(fig) and len(fig.axes[0].lines) == 2


def test_un_nuage_de_points_est_borne_et_reproductible() -> None:
    assert len(evaluate._echantillon(100)) == 100
    k = evaluate._echantillon(10_000)
    assert len(k) == evaluate.POINTS_MAX and np.array_equal(k, evaluate._echantillon(10_000))
    assert np.array_equal(k, np.sort(k))


def test_la_matrice_de_confusion_a_un_axe_par_classe(multiclasse: aides.Projet) -> None:
    diag = evaluate.diagnostiquer_validation(multiclasse.resultat, "classification.logistique")
    fig = evaluate.figure_confusion(diag)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [str(c) for c in diag.classes]
    assert diag.probleme is Probleme.CLASSIFICATION
