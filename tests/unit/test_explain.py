"""Explications : importance par permutation (plis, test), impureté, arbre en texte, coefficients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml.core import evaluate, explain, train
from mecaml.core.tasks import Annulation, ErreurAnnulation
from mecaml.core.types import ChoixTechnique

CT = ChoixTechnique.creer
RIDGE = CT("regression.ridge")
ARBRE = CT("regression.arbre", max_depth=3)
LOGISTIQUE = CT("classification.logistique")


def _mener(dossier: Path, jeu: pd.DataFrame, fabrique, **options) -> aides.Projet:
    return aides.mener_le_projet(aides.ecrire_csv(jeu, dossier / "jeu.csv"), fabrique, **options)


def _jeu_avec_bruit(n: int = 200, graine: int = 1) -> pd.DataFrame:
    """Une variable qui explique tout, une qui explique un peu, une de pur bruit, une modalité sans effet."""
    rng = np.random.default_rng(graine)
    df = pd.DataFrame(
        {
            "forte": rng.normal(size=n),
            "faible": rng.normal(size=n),
            "bruit": rng.normal(size=n),
            "c": rng.choice(["x", "y"], n),
        }
    )
    df["y"] = 5.0 * df["forte"] + 0.5 * df["faible"] + rng.normal(0, 0.2, n)
    return df


def _spec_bruit(table):
    from mecaml.core.types import Probleme, Role, SpecProjet

    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="y",
        roles=(
            ("forte", Role.NUMERIQUE),
            ("faible", Role.NUMERIQUE),
            ("bruit", Role.NUMERIQUE),
            ("c", Role.CATEGORIELLE),
            ("y", Role.CIBLE),
        ),
        metrique="rmse",
        pretraitement=(CT("encodage.one_hot"), CT("transformation.standard")),
        modeles=(RIDGE, ARBRE),
    )


@pytest.fixture(scope="module")
def bruit(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    return _mener(tmp_path_factory.mktemp("explain_bruit"), _jeu_avec_bruit(), _spec_bruit)


@pytest.fixture(scope="module")
def classification(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("explain_cls")
    return _mener(
        dossier,
        jeux.jeu_classification(n=300),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, CT("classification.foret", n_estimators=20))),
    )


# ============================================================
# PERMUTATION, PLI PAR PLI
# ============================================================
def test_la_variable_qui_explique_tout_a_la_plus_grande_importance(bruit: aides.Projet) -> None:
    tableau = explain.importances_validation(bruit.resultat, "regression.ridge", bruit.table, bruit.coffre)
    assert list(tableau.columns) == list(explain.COLONNES_IMPORTANCES)
    assert tableau["variable"].tolist()[:2] == ["forte", "faible"]
    assert set(tableau["variable"]) == {
        "forte",
        "faible",
        "bruit",
        "c",
    }  # les variables du modèle, non les colonnes
    par_nom = tableau.set_index("variable")
    assert par_nom.loc["forte", "moyenne"] > 10 * abs(par_nom.loc["bruit", "moyenne"]) > 0
    assert (
        par_nom.loc["forte", "moyenne"] > par_nom.loc["faible", "moyenne"] > 0
    )  # une erreur qui monte est une baisse
    assert (tableau["plis_valides"] == len(bruit.resultat.plis)).all()
    assert (tableau["ecart_type"] >= 0).all() and tableau["moyenne"].is_monotonic_decreasing


def test_l_importance_se_mesure_sur_les_lignes_de_validation_de_chaque_pli(bruit: aides.Projet) -> None:
    """Contre-calcul : ajustement sur le pli d'apprentissage, permutation sur celui de validation."""
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    spec = bruit.resultat.spec
    donnees = evaluate.apprentissage_du_projet(spec, bruit.table, bruit.coffre)
    pipeline = train.construire_pipeline(spec, RIDGE, None, donnees.contexte)
    variables = list(spec.variables())
    par_pli = []
    for app, val in bruit.resultat.plis:
        ajuste = clone(pipeline).fit(donnees.X.iloc[app], donnees.y[app])
        mesure = permutation_importance(
            ajuste,
            donnees.X.iloc[val][variables],
            donnees.y[val],
            scoring="neg_root_mean_squared_error",
            n_repeats=3,
            random_state=spec.graine,
            n_jobs=1,
        )
        par_pli.append(pd.Series(mesure.importances_mean, index=variables))
    attendu = pd.concat(par_pli, axis=1).mean(axis=1)
    tableau = explain.importances_validation(
        bruit.resultat, "regression.ridge", bruit.table, bruit.coffre, repetitions=3
    )
    obtenu = tableau.set_index("variable")["moyenne"]
    assert np.allclose(obtenu.loc[variables].to_numpy(), attendu.loc[variables].to_numpy(), rtol=1e-9)


def test_les_importances_sont_reproductibles(bruit: aides.Projet) -> None:
    a = explain.importances_validation(
        bruit.resultat, "regression.ridge", bruit.table, bruit.coffre, repetitions=3
    )
    b = explain.importances_validation(
        bruit.resultat, "regression.ridge", bruit.table, bruit.coffre, repetitions=3
    )
    pd.testing.assert_frame_equal(a, b)


def test_les_importances_hors_pli_ne_lisent_pas_le_test(tmp_path: Path) -> None:
    jeu = _jeu_avec_bruit()
    (tmp_path / "n").mkdir()
    (tmp_path / "s").mkdir()
    normal = _mener(tmp_path / "n", jeu, _spec_bruit)
    saccage = jeu.copy()
    test = np.setdiff1d(np.arange(len(jeu)), normal.coffre.positions_apprentissage())
    saccage.loc[test, ["forte", "faible", "bruit", "y"]] = -1e6
    autre = _mener(tmp_path / "s", saccage, _spec_bruit)
    a = explain.importances_validation(normal.resultat, "regression.ridge", normal.table, normal.coffre)
    b = explain.importances_validation(autre.resultat, "regression.ridge", autre.table, autre.coffre)
    pd.testing.assert_frame_equal(a, b)


def test_une_classification_se_lit_dans_le_sens_du_score(classification: aides.Projet) -> None:
    tableau = explain.importances_validation(
        classification.resultat, "classification.logistique", classification.table, classification.coffre
    )
    assert tableau["variable"].iloc[0] in {"a", "b"}  # les deux mesures portent la cible, la modalité non
    assert tableau.set_index("variable").loc["c", "moyenne"] < tableau["moyenne"].iloc[0]
    assert (tableau["plis_valides"] >= 1).all()


def test_une_serie_se_lit_sur_ses_variables_sans_toucher_a_la_date_ni_a_l_unite(tmp_path: Path) -> None:
    projet = _mener(tmp_path, aides.jeu_serie_lacunaire(), aides.spec_lacunaire)
    tableau = explain.importances_validation(
        projet.resultat, "regression.ridge", projet.table, projet.coffre, repetitions=2
    )
    assert set(tableau["variable"]) == {"x1", "x2"}  # ni machine, ni cycle, ni la cible
    assert tableau["moyenne"].notna().all()


def test_l_annulation_arrete_le_calcul_entre_deux_plis(bruit: aides.Projet) -> None:
    jeton = Annulation()
    jeton.demander()
    with pytest.raises(ErreurAnnulation):
        explain.importances_validation(
            bruit.resultat, "regression.ridge", bruit.table, bruit.coffre, annulation=jeton
        )


def test_les_refus_de_l_importance_par_plis(bruit: aides.Projet, tmp_path: Path) -> None:
    with pytest.raises(explain.ErreurExplication, match="ne figure pas parmi les modèles comparés"):
        explain.importances_validation(bruit.resultat, "regression.foret", bruit.table, bruit.coffre)
    sans_cible = _mener(
        tmp_path,
        jeux.jeu_sans_cible(n=80),
        lambda t: jeux.spec_sans_cible(modeles=(CT("sans_cible.kmeans", n_clusters=2),)),
    )
    with pytest.raises(explain.ErreurExplication, match="suppose une cible"):
        explain.importances_validation(sans_cible.resultat, "sans_cible.kmeans", sans_cible.table, None)


def test_l_agregation_ignore_les_plis_sans_valeur_et_classe_les_valeurs_manquantes_en_dernier() -> None:
    p1 = pd.DataFrame({"variable": ["a", "b", "c"], "moyenne": [1.0, 0.2, np.nan]})
    p2 = pd.DataFrame({"variable": ["a", "b", "c"], "moyenne": [3.0, np.nan, np.nan]})
    tableau = explain._agreger([p1, p2], ["a", "b", "c"])
    assert tableau["variable"].tolist() == ["a", "b", "c"]
    assert tableau["moyenne"].tolist()[:2] == [2.0, 0.2] and np.isnan(tableau["moyenne"].iloc[2])
    assert tableau["plis_valides"].tolist() == [2, 1, 0]
    assert tableau["ecart_type"].tolist() == [pytest.approx(np.sqrt(2.0)), 0.0, 0.0]


# ============================================================
# PERMUTATION, SUR DES LIGNES QUE LE MODÈLE N'A PAS VUES
# ============================================================
def _modele_final(projet: aides.Projet) -> train.ModeleAjuste:
    choix = next(c for c in projet.spec.modeles if c.id == projet.decision.modele_id)
    return train.ajuster_final(projet.spec, choix, projet.table, projet.coffre)


def test_l_importance_sur_des_lignes_ignore_celles_sans_cible(bruit: aides.Projet) -> None:
    modele = _modele_final(bruit)
    lignes = bruit.table.donnees.iloc[:80].copy()
    complet = explain.importances_sur_lignes(modele, bruit.spec, lignes, repetitions=3)
    lignes.loc[lignes.index[::7], "y"] = np.nan
    lacunaire = explain.importances_sur_lignes(modele, bruit.spec, lignes, repetitions=3)
    assert complet["variable"].iloc[0] == lacunaire["variable"].iloc[0] == "forte"
    assert lacunaire["plis_valides"].tolist() == [1, 1, 1, 1] and list(lacunaire.columns) == list(
        explain.COLONNES_IMPORTANCES
    )
    assert not np.allclose(
        complet["moyenne"].to_numpy(), lacunaire["moyenne"].to_numpy()
    )  # d'autres lignes, d'autres chiffres


def test_le_modele_vu_a_travers_ses_variables_transmet_ce_que_le_pipeline_sait_faire(
    bruit: aides.Projet, classification: aides.Projet
) -> None:
    ridge = _modele_final(bruit).pipeline
    x = bruit.table.donnees[list(ridge.feature_names_in_)]
    vue = explain._SurLesVariables(ridge, x[["bruit"]], list(x.columns))
    variables = x[["forte", "faible", "c"]]
    assert np.array_equal(vue.predict(variables), ridge.predict(x))
    assert not hasattr(vue, "predict_proba")  # une régression n'en a pas
    with pytest.raises(AttributeError):
        _ = vue.inexistant
    logistique = _modele_final(classification).pipeline
    xc = classification.table.donnees[list(logistique.feature_names_in_)]
    vue_c = explain._SurLesVariables(logistique, xc[["c"]], list(xc.columns))
    assert np.allclose(vue_c.predict_proba(xc[["a", "b"]]), logistique.predict_proba(xc))
    assert list(vue_c.classes_) == list(logistique.classes_)


# ============================================================
# LECTURE DU MODÈLE AJUSTÉ
# ============================================================
def test_l_importance_par_impurete_somme_a_un(bruit: aides.Projet) -> None:
    choix = ARBRE
    modele = train.ajuster_final(bruit.spec, choix, bruit.table, bruit.coffre)
    tableau = explain.importance_par_impurete(modele)
    assert list(tableau.columns) == ["variable", "importance"]
    assert tableau["importance"].sum() == pytest.approx(1.0) and tableau["importance"].is_monotonic_decreasing
    assert tableau["variable"].iloc[0] == "forte"


def test_l_importance_par_impurete_et_l_arbre_refusent_un_modele_qui_n_en_a_pas(bruit: aides.Projet) -> None:
    ridge = train.ajuster_final(bruit.spec, RIDGE, bruit.table, bruit.coffre)
    with pytest.raises(explain.ErreurExplication, match="n'a pas d'importance par impureté"):
        explain.importance_par_impurete(ridge)
    with pytest.raises(explain.ErreurExplication, match="doit être un arbre de décision"):
        explain.arbre_texte(ridge)


def test_l_arbre_en_texte_donne_les_regles_de_haut_en_bas(bruit: aides.Projet) -> None:
    arbre = train.ajuster_final(bruit.spec, ARBRE, bruit.table, bruit.coffre)
    texte = explain.arbre_texte(arbre, profondeur=2)
    assert "forte" in texte and "value:" in texte
    assert (
        texte.count("<=") >= 1
        and len(texte.splitlines()) < len(explain.arbre_texte(arbre, profondeur=4).splitlines()) + 1
    )


def test_les_coefficients_d_un_modele_lineaire(bruit: aides.Projet, classification: aides.Projet) -> None:
    ridge = train.ajuster_final(bruit.spec, RIDGE, bruit.table, bruit.coffre)
    coefficients = explain.coefficients(ridge)
    assert (
        list(coefficients.columns) == ["variable", "coefficient"]
        and len(coefficients) == ridge.pipeline[-1].coef_.shape[0]
    )
    assert (
        abs(coefficients.set_index("variable").loc["forte", "coefficient"])
        == coefficients["coefficient"].abs().max()
    )
    arbre = train.ajuster_final(bruit.spec, ARBRE, bruit.table, bruit.coffre)
    assert explain.coefficients(arbre) is None
    binaire = explain.coefficients(_modele_final(classification))
    assert list(binaire.columns) == ["variable", "coefficient"]


def test_les_coefficients_d_une_classification_a_trois_classes_ont_une_colonne_par_classe(
    tmp_path: Path,
) -> None:
    projet = _mener(
        tmp_path,
        jeux.jeu_classification(n=300, classes=3),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,), binaire=False),
    )
    coefficients = explain.coefficients(_modele_final(projet))
    assert next(iter(coefficients.columns)) == "variable" and coefficients.shape[1] == 1 + 3


# ============================================================
# FIGURE
# ============================================================
def test_la_figure_des_importances_garde_les_plus_fortes_et_saute_les_valeurs_manquantes(
    bruit: aides.Projet,
) -> None:
    tableau = explain.importances_validation(
        bruit.resultat, "regression.ridge", bruit.table, bruit.coffre, repetitions=2
    )
    tableau = pd.concat(
        [
            tableau,
            pd.DataFrame({"variable": ["x"], "moyenne": [np.nan], "ecart_type": [0.0], "plis_valides": [0]}),
        ]
    )
    fig = explain.figure_importances(tableau, "Importance des variables", maximum=3)
    axe = fig.axes[0]
    assert (
        len(axe.patches) == 3
        and "baisse du score" in axe.get_xlabel()
        and axe.get_title() == "Importance des variables"
    )
    assert [t.get_text() for t in axe.get_yticklabels()][-1] == "forte"  # la plus forte en haut
