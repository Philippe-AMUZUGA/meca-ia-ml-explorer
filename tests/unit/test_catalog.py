"""Catalogue des techniques : forme du registre, ajustement de chaque entrée, bornes, graines, couverture.

Chaque entrée est construite puis ajustée sur un jeu synthétique minuscule en moins de 5 secondes ;
chaque combinaison de chaque grille doit se construire ; la matrice de techniques du projet est
couverte, une technique sans entrée ayant une exclusion motivée.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecaml.core import cible_passee, metrics, renvois
from mecaml.core.catalog import FAMILLES, GRILLE_MAX_COMBINAISONS, REGISTRE, Cout, entree, entrees
from mecaml.core.catalog._outils import LIMITES, preparer, verifier_grille
from mecaml.core.catalog._refs import ref
from mecaml.core.livre import TECHNIQUES_LIVRE, est_identifiant
from mecaml.core.types import Probleme

EXTENSIONS = renvois.EXTENSIONS

RACINE = Path(__file__).resolve().parents[2]
SRC = RACINE / "src"
BUDGET_S = 5.0
IDS = [e.id for e in REGISTRE]


@pytest.fixture(scope="module", autouse=True)
def _prechauffage() -> None:
    """Le premier gradient boosting d'un processus coûte plusieurs secondes à froid : hors des mesures."""
    from mecaml.core.train import prechauffer

    prechauffer()


# ============================================================
# JEUX SYNTHÉTIQUES
# ============================================================
@cache
def _numerique() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(150, 6)), columns=list("abcdef"))
    y_reg = (2 * x["a"] + x["b"] + rng.normal(0, 0.2, 150)).to_numpy()
    y_cls = np.where(x["a"] + 0.3 * x["b"] > 0, "defaut", "conforme")
    return x, y_reg, y_cls


@cache
def _avec_manquants() -> pd.DataFrame:
    x, _, _ = _numerique()
    x = x.copy()
    x.iloc[::7, 0] = np.nan
    x.iloc[::11, 3] = np.nan
    return x


@cache
def _categoriel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"u": rng.choice(["x", "y", "z"], 90), "v": rng.choice(["p", "q"], 90)})
    df.iloc[::13, 0] = None
    return df


@cache
def _amas() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(np.r_[rng.normal(0, 0.3, (60, 3)), rng.normal(4, 0.3, (60, 3))], columns=list("uvw"))


# ============================================================
# AJUSTEMENT D'UNE ENTRÉE, PAR FAMILLE
# ============================================================
def _verifier_transformation(objet: object, x: pd.DataFrame, attendu_sans_nan: bool = True) -> None:
    sortie = np.asarray(objet.fit_transform(x))  # type: ignore[attr-defined]
    assert sortie.shape[0] == len(x) and sortie.ndim == 2
    if attendu_sans_nan:
        assert not np.isnan(sortie.astype(float)).any()


def _essai_preparation(e) -> None:
    objet = e.construire(0)
    if e.famille == "manquants":
        _verifier_transformation(objet, _avec_manquants(), attendu_sans_nan=e.id != "manquants.indicateur")
    elif e.famille == "encodage":
        _verifier_transformation(objet, _categoriel())
    else:
        _verifier_transformation(objet, _numerique()[0])


def _essai_selection(e) -> None:
    x, y_reg, y_cls = _numerique()
    for probleme, y in (("regression", y_reg), ("classification", y_cls)):
        if Probleme(probleme) not in e.problemes:
            continue
        objet = e.construire(0, probleme=probleme)
        sortie = objet.fit_transform(x, y)
        assert 1 <= sortie.shape[1] <= x.shape[1]
    if Probleme.SANS_CIBLE in e.problemes:
        assert e.construire(0, probleme="sans_cible").fit_transform(x).shape[0] == len(x)


def _declarees(e) -> dict[str, bool]:
    """Une technique qui lit la cible passée exige d'être déclarée connue à la prévision."""
    return {"cible_connue": True} if e.id in cible_passee.TECHNIQUES else {}


def _essai_modele(e) -> None:
    x, y_reg, y_cls = _numerique()
    y = y_reg if Probleme.REGRESSION in e.problemes else y_cls
    modele = e.construire(0, **_declarees(e)).fit(x, y)
    prediction = modele.predict(x)
    assert len(prediction) == len(x)
    if Probleme.CLASSIFICATION in e.problemes:
        assert set(prediction) <= {"defaut", "conforme"}
    else:
        assert np.isfinite(prediction).all()


def _essai_sans_cible(e) -> None:
    x = _amas()
    objet = e.construire(0)
    if hasattr(objet, "fit_predict"):
        assert len(objet.fit_predict(x)) == len(x)
    else:
        assert objet.fit(x).transform(x).shape[0] == len(x)


def _essai_validation(e) -> None:
    x, y_reg, y_cls = _numerique()
    groupes = np.repeat(np.arange(15), 10)
    if e.id in ("validation.recherche_grille", "validation.recherche_aleatoire"):
        _essai_recherche(e, x, y_reg)
    elif e.id == "validation.courbe_apprentissage":
        _essai_courbe_apprentissage(e, x, y_reg)
    elif e.id == "validation.courbe_validation":
        _essai_courbe_validation(e, x, y_reg)
    else:
        reglages = {"n_splits": 3} if "n_splits" in dict(e.grille_bornee) else {}
        decoupeur = e.construire(0, **reglages)
        y = (
            y_cls
            if Probleme.CLASSIFICATION in e.problemes and Probleme.REGRESSION not in e.problemes
            else y_reg
        )
        petit = slice(0, 30) if e.id == "validation.exclusion_unique" else slice(None)
        a_groupes = (
            e.id in ("validation.plis_groupes", "validation.plis_groupes_stratifies") or "aleatoires" in e.id
        )
        plis = list(decoupeur.split(x[petit], y[petit], groupes[petit] if a_groupes else None))
        assert plis and all(len(a) and len(v) for a, v in plis)


def _essai_recherche(e, x, y) -> None:
    from sklearn.linear_model import Ridge

    grille = {"alpha": [0.1, 1.0, 10.0]}
    parametres = {"estimator": Ridge(), "cv": 3}
    if e.id == "validation.recherche_grille":
        recherche = e.construire(0, param_grid=grille, **parametres)
    else:
        recherche = e.construire(0, param_distributions=grille, n_iter=3, **parametres)
    assert recherche.fit(x, y).best_params_["alpha"] in grille["alpha"]


def _essai_courbe_apprentissage(e, x, y) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    courbe = e.construire(0, n_points=4)(Ridge(), x, y, cv=KFold(3))
    assert courbe["tailles"].shape == (4,) and courbe["val"].shape == (4, 3)


def _essai_courbe_validation(e, x, y) -> None:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    courbe = e.construire(0)(Ridge(), x, y, nom="alpha", valeurs=[0.1, 1.0, 10.0], cv=KFold(3))
    assert courbe["val"].shape == (3, 3)


def _essai_explication(e) -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    x, y_reg, _ = _numerique()
    if e.id == "explication.permutation":
        from sklearn.linear_model import Ridge

        modele = make_pipeline(StandardScaler(), Ridge()).fit(x, y_reg)
        tableau = e.construire(0, n_repeats=3)(modele, x, y_reg)
        assert list(tableau.columns) == ["variable", "moyenne", "ecart_type"]
        assert tableau["variable"].iloc[0] in ("a", "b")  # les deux variables qui portent la cible
    elif e.id == "explication.impurete":
        modele = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=3, random_state=0)).fit(
            x, y_reg
        )
        tableau = e.construire(0)(modele)
        assert list(tableau.columns) == ["variable", "importance"]
        assert tableau["variable"].iloc[0] in ("a", "b") and np.isclose(tableau["importance"].sum(), 1.0)
        with pytest.raises(ValueError, match="pas d'importance par impureté"):
            from sklearn.linear_model import Ridge

            e.construire(0)(make_pipeline(StandardScaler(), Ridge()).fit(x, y_reg))
    else:
        modele = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=2, random_state=0)).fit(
            x, y_reg
        )
        texte = e.construire(0)(modele)
        assert "feature" not in texte and "<=" in texte  # noms de colonnes lus dans le pipeline


def _essai_chronologique(e) -> None:
    rng = np.random.default_rng(3)
    lignes = pd.DataFrame(
        {"t": np.tile(np.arange(10), 3), "g": np.repeat(["m1", "m2", "m3"], 10), "a": rng.normal(size=30)}
    )
    commun = {"colonne_date": "t", "colonne_groupe": "g", "colonnes": ("a",)}
    with pytest.raises(ValueError, match="colonne_date"):
        e.construire(0, **_declarees(e))
    if e.id == "chronologique.cible_decalee":
        declaree = _declarees(e)
        objet = e.construire(0, retards=(1, 2), **commun, **declaree)
        sortie = objet.fit_transform(lignes)
        assert sortie.shape == (30, 2) and np.isnan(sortie[0, 0])
        assert list(objet.get_feature_names_out()) == ["a_retard_1", "a_retard_2"]
        for interdit in ("fenetres", "differences", "demi_vies"):  # inclurait la valeur courante de la cible
            with pytest.raises(ValueError, match="seuls les retards"):
                e.construire(
                    0, **commun, **declaree, **{interdit: (3,) if interdit != "demi_vies" else (2.0,)}
                )
        return
    objet = e.construire(0, retards=(1, 2), fenetres=(3,), **commun)
    sortie = objet.fit_transform(lignes)
    assert sortie.shape == (30, 4) and np.isnan(sortie[0, 0])  # le premier retard d'une série est inconnu
    assert list(objet.get_feature_names_out()) == [
        "a_retard_1",
        "a_retard_2",
        "a_moyenne_3",
        "a_ecart_type_3",
    ]


def _essayer(e) -> None:
    if e.famille == "chronologique":
        _essai_chronologique(e)
    elif e.famille in ("manquants", "encodage", "transformation"):
        _essai_preparation(e)
    elif e.famille == "selection":
        _essai_selection(e)
    elif e.famille in ("regression", "classification", "reference"):
        _essai_modele(e)
    elif e.famille == "sans_cible":
        _essai_sans_cible(e)
    elif e.famille == "validation":
        _essai_validation(e)
    else:
        _essai_explication(e)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("identifiant", IDS)
def test_chaque_entree_s_ajuste_en_moins_de_cinq_secondes(identifiant: str) -> None:
    e = entree(identifiant)
    debut = time.perf_counter()
    _essayer(e)
    assert time.perf_counter() - debut < BUDGET_S, f"{identifiant} dépasse {BUDGET_S} s"


# ============================================================
# FORME DU REGISTRE
# ============================================================
def test_registre_identifiants_uniques_et_ordonnes_par_famille() -> None:
    assert len(IDS) == len(set(IDS)) >= 50
    assert {e.famille for e in REGISTRE} == FAMILLES


def test_renvois_au_livre_et_textes_pedagogiques() -> None:
    assert set(renvois.TECHNIQUES) == set(IDS), "chaque technique du registre a un renvoi, et réciproquement"
    for e in REGISTRE:
        assert e.ref_livre == ref(e.id) and est_identifiant(e.ref_livre), e.id
        assert e.texte_id == e.id
    with pytest.raises(KeyError, match="Renvoi au livre manquant"):
        ref("regression.absente")


def test_perimetre_des_problemes_par_famille() -> None:
    for e in REGISTRE:
        if e.famille in ("regression", "classification", "sans_cible"):
            assert e.problemes == {Probleme(e.famille)}, e.id
        if e.id == "chronologique.variables_temporelles":
            assert Probleme.SANS_CIBLE in e.problemes, e.id
        if e.id == "chronologique.cible_decalee":
            assert e.problemes == {Probleme.REGRESSION}  # une cible décalée numérique
        if e.id == "reference.persistance":
            assert e.problemes == {Probleme.REGRESSION}
        if e.famille == "reference":
            assert len(e.problemes) == 1
        if e.famille in ("manquants", "encodage", "transformation"):
            assert e.problemes == set(Probleme), e.id
    assert entrees(probleme=Probleme.SANS_CIBLE, famille="sans_cible")
    assert {e.id for e in entrees(famille="reference")} == {
        "reference.dummy_regression",
        "reference.dummy_classification",
        "reference.persistance",
    }


def test_grilles_bornees() -> None:
    for e in REGISTRE:
        assert e.combinaisons <= GRILLE_MAX_COMBINAISONS, e.id
        assert all(len(valeurs) >= 2 for _, valeurs in e.grille_bornee), e.id
    assert max(e.combinaisons for e in REGISTRE) <= 30


def test_couts_les_techniques_coupees_sont_celles_de_i9() -> None:
    lourdes = {e.id for e in REGISTRE if e.cout is Cout.LOURD}
    assert lourdes == {
        "regression.svr",
        "classification.svc",
        "sans_cible.svm_une_classe",
        "sans_cible.classification_hierarchique",
        "validation.exclusion_unique",
    }


def test_import_du_catalogue_sans_scikit_learn() -> None:
    programme = "import sys, mecaml.core.catalog as c\nprint(len(c.REGISTRE), 'sklearn' in sys.modules)\n"
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}
    sortie = subprocess.run(
        [sys.executable, "-c", programme], capture_output=True, text=True, env=env, timeout=120
    )
    assert sortie.returncode == 0, sortie.stderr
    n, charge = sortie.stdout.split()
    assert int(n) == len(REGISTRE) and charge == "False"


# ============================================================
# GRILLES, GRAINE, BORNES
# ============================================================
SANS_GRILLE_CONSTRUCTIBLE = {"validation.recherche_grille", "validation.recherche_aleatoire"}
# Familles dont la fabrique reçoit un contexte du projet (problème, date) ou rend une fonction.
EXIGENT_UN_CONTEXTE = ("selection", "explication", "chronologique")


@pytest.mark.parametrize("identifiant", [i for i in IDS if i not in SANS_GRILLE_CONSTRUCTIBLE])
def test_chaque_combinaison_de_la_grille_se_construit(identifiant: str) -> None:
    from itertools import product

    e = entree(identifiant)
    noms = [nom for nom, _ in e.grille_bornee]
    supplement = {"probleme": "regression"} if e.famille == "selection" else {}
    supplement |= {"colonne_date": "t", "colonnes": ("x",)} if e.famille == "chronologique" else {}
    supplement |= _declarees(e)
    for valeurs in product(*(v for _, v in e.grille_bornee)):
        assert e.construire(0, **supplement, **dict(zip(noms, valeurs, strict=True))) is not None


def test_la_graine_du_projet_l_emporte_sur_toute_autre() -> None:
    avec_graine = 0
    for e in REGISTRE:
        if e.famille in EXIGENT_UN_CONTEXTE or e.id in SANS_GRILLE_CONSTRUCTIBLE:
            continue
        objet = e.construire(7, **_declarees(e))
        parametres = objet.get_params() if hasattr(objet, "get_params") else {}
        if "random_state" in parametres:
            avec_graine += 1
            assert parametres["random_state"] == 7, e.id
            assert e.construire(7, random_state=99).get_params()["random_state"] == 7, e.id
    assert avec_graine >= 15  # arbres, forêts, boosting, réseaux, regroupements, découpeurs...


def test_deux_constructions_identiques_donnent_les_memes_parametres() -> None:
    for e in REGISTRE:
        if e.famille in EXIGENT_UN_CONTEXTE or e.id in SANS_GRILLE_CONSTRUCTIBLE:
            continue
        a, b = e.construire(3, **_declarees(e)), e.construire(3, **_declarees(e))
        if hasattr(a, "get_params"):
            assert repr(a.get_params()) == repr(b.get_params()), e.id


@pytest.mark.parametrize(
    ("identifiant", "parametres", "motif"),
    [
        ("regression.foret", {"n_estimators": 501}, "n_estimators.*500"),
        ("classification.foret", {"n_estimators": 10_000}, "n_estimators"),
        ("regression.reseau_neurones", {"max_iter": 10_001}, "max_iter"),
        ("regression.arbre", {"max_depth": 51}, "max_depth"),
        ("regression.plus_proches_voisins", {"n_neighbors": 501}, "n_neighbors"),
        ("transformation.polynomiale", {"degree": 5}, "degree"),
        ("transformation.discretisation", {"n_bins": 51}, "n_bins"),
        ("sans_cible.kmeans", {"n_clusters": 51}, "n_clusters"),
        ("validation.kfold", {"n_splits": 21}, "n_splits"),
        ("regression.reseau_neurones", {"hidden_layer_sizes": (10, 10, 10, 10)}, "hors bornes"),
        ("regression.reseau_neurones", {"hidden_layer_sizes": (1000,)}, "hors bornes"),
        ("explication.permutation", {"n_repeats": 51}, "n_repeats"),
    ],
)
def test_parametres_demesures_refuses(identifiant: str, parametres: dict, motif: str) -> None:
    with pytest.raises(ValueError, match=motif):
        entree(identifiant).construire(0, **parametres)


def test_borne_exacte_acceptee() -> None:
    assert entree("regression.foret").construire(0, n_estimators=LIMITES["n_estimators"]) is not None


def test_verifier_grille_et_parametres_obligatoires() -> None:
    assert verifier_grille("essai", {"a": [1, 2], "b": [1, 2, 3]}) == 6
    assert verifier_grille("essai", [{"a": [1, 2]}, {"b": [1, 2, 3]}]) == 5
    with pytest.raises(ValueError, match="borne 50"):
        verifier_grille("essai", {"a": list(range(8)), "b": list(range(8))})
    with pytest.raises(ValueError, match="obligatoire"):
        entree("validation.recherche_grille").construire(0)
    from sklearn.linear_model import Ridge

    grande = {"alpha": list(range(60))}
    with pytest.raises(ValueError, match="grille de 60"):
        entree("validation.recherche_grille").construire(0, estimator=Ridge(), param_grid=grande)
    with pytest.raises(ValueError, match="n_iter"):
        entree("validation.recherche_aleatoire").construire(
            0, estimator=Ridge(), param_distributions={"alpha": [1, 2]}, n_iter=51
        )


def test_preparer_fusionne_defauts_choix_et_graine() -> None:
    assert preparer("t", {"a": 1, "b": 2}, {"b": 3}, graine=5) == {"a": 1, "b": 3, "random_state": 5}
    assert preparer("t", {"a": 1}, {}) == {"a": 1}


# ============================================================
# COMPOSITIONS : VOTE ET EMPILEMENT
# ============================================================
def test_vote_et_empilement_utilisent_le_catalogue() -> None:
    x, _, y_cls = _numerique()
    vote = entree("classification.vote").construire(
        0, base=("classification.logistique", "classification.arbre"), voting="hard"
    )
    assert [nom for nom, _ in vote.estimators] == ["logistique", "arbre"]
    assert len(vote.fit(x, y_cls).predict(x)) == len(x)
    pile = entree("classification.empilement").construire(0, final="classification.arbre")
    assert type(pile.final_estimator).__name__ == "DecisionTreeClassifier"


@pytest.mark.parametrize(
    ("parametres", "motif"),
    [
        ({"base": ("classification.logistique",)}, "de 2 à 6"),
        ({"base": ("classification.logistique", "classification.logistique")}, "distinctes"),
        ({"base": ("classification.logistique", "classification.vote")}, "modèle de base"),
        ({"base": ("classification.logistique", "regression.ridge")}, "modèle de base"),
        ({"base": ("classification.logistique", "classification.inconnue")}, "Technique inconnue"),
    ],
)
def test_vote_refus(parametres: dict, motif: str) -> None:
    with pytest.raises((ValueError, KeyError), match=motif):
        entree("classification.vote").construire(0, **parametres)


def test_selection_rejette_un_probleme_inconnu() -> None:
    with pytest.raises(ValueError, match="probleme"):
        entree("selection.variance").construire(0, probleme="autre")


# ============================================================
# COUVERTURE DE LA LISTE DES TECHNIQUES DU LIVRE
# ============================================================
# La liste est celle du livre (`tests/data/techniques_livre.md`, identifiants T<cc>-<s>-<k>). Chaque technique
# hors de la marque [hors application] a une ou plusieurs entrées du catalogue ("entree"), une ou plusieurs
# métriques ("metrique"), un paramètre ("parametre") ou une exclusion motivée ("exclusion").
LIVRE = Path(__file__).resolve().parents[1] / "data" / "techniques_livre.md"
E, M, P, X = "entree", "metrique", "parametre", "exclusion"
COUVERTURE: dict[str, tuple[str, str | tuple[str, ...]]] = {
    "T01-1-1": (E, "regression.lineaire"),
    "T01-1-2": (E, "regression.arbre"),
    "T01-2-1": (X, "base de test mise de côté par le découpage scellé de split.py, avant tout prétraitement"),
    "T01-2-2": (E, "validation.courbe_apprentissage"),
    "T02-1-1": (
        X,
        "fuite et morcellement traités par l'audit des données et le coffre scellé, contrôles du cœur",
    ),
    "T02-1-2": (E, ("reference.dummy_regression", "reference.dummy_classification")),
    "T02-2-1": (E, ("validation.kfold", "validation.kfold_stratifie", "validation.plis_groupes")),
    "T02-2-2": (E, ("validation.courbe_validation", "validation.courbe_apprentissage")),
    "T02-3-1": (M, ("r2", "mae", "rmse", "mape")),
    "T02-3-2": (M, ("matrice_confusion", "exactitude", "exactitude_equilibree", "precision", "rappel", "f1")),
    "T03-1-1": (
        X,
        "statistiques descriptives et distributions : aperçu et profil de l'audit, rien à fabriquer",
    ),
    "T03-1-2": (E, "transformation.ecretage_tukey"),
    "T03-1-3": (
        X,
        "profil du jeu (types, manquants, constantes, doublons) : audit de lecture, hors catalogue",
    ),
    "T03-2-1": (
        X,
        "corrélations et paires : audit et figures du rapport, seaborn exclu au profit de matplotlib",
    ),
    "T03-2-2": (X, "tableau croisé entre cible et catégorie : constat de l'audit, hors catalogue"),
    "T03-2-3": (
        X,
        "fiche de décisions : journal des choix du projet dans le rapport exporté, hors catalogue",
    ),
    "T04-1-1": (E, ("manquants.simple", "manquants.knn", "manquants.indicateur")),
    "T04-1-2": (E, ("encodage.one_hot", "encodage.ordinal")),
    "T04-2-1": (E, ("transformation.standard", "transformation.minmax", "transformation.robuste")),
    "T04-2-2": (
        E,
        ("transformation.polynomiale", "transformation.puissance", "transformation.discretisation"),
    ),
    "T04-2-3": (X, "Pipeline et ColumnTransformer sont la colonne vertébrale de l'application, non un choix"),
    "T04-2-4": (E, ("selection.variance", "selection.k_meilleures")),
    "T05-1-1": (E, "regression.lineaire"),
    "T05-1-2": (E, ("transformation.polynomiale", "regression.lineaire")),
    "T05-2-1": (E, ("regression.ridge", "regression.lasso", "regression.elastic_net")),
    "T06-1-1": (E, "classification.logistique"),
    "T06-2-1": (E, ("classification.plus_proches_voisins", "regression.plus_proches_voisins")),
    "T06-2-2": (E, "classification.bayes_naif"),
    "T06-2-3": (E, "classification.analyse_discriminante"),
    "T07-1-1": (E, "classification.svc"),
    "T07-2-1": (E, "regression.svr"),
    "T08-1-1": (E, ("classification.arbre", "regression.arbre", "explication.arbre_texte")),
    "T08-2-1": (
        E,
        ("classification.bagging", "regression.bagging", "classification.foret", "regression.foret"),
    ),
    "T08-3-1": (
        E,
        ("classification.adaboost", "classification.gradient_histogramme", "regression.gradient_histogramme"),
    ),
    "T08-3-2": (E, ("classification.vote", "classification.empilement")),
    "T08-3-3": (E, ("explication.impurete", "explication.permutation")),
    "T09-1-1": (M, "roc_auc"),
    "T09-1-2": (M, "precision_moyenne"),
    "T09-2-1": (M, "cout"),
    "T10-1-1": (E, "sans_cible.kmeans"),
    "T10-1-2": (E, "sans_cible.classification_hierarchique"),
    "T10-1-3": (E, "sans_cible.dbscan"),
    "T10-2-1": (E, "sans_cible.foret_isolement"),
    "T10-2-2": (E, "sans_cible.facteur_local"),
    "T10-3-1": (E, "sans_cible.svm_une_classe"),
    "T11-1-1": (E, "sans_cible.acp"),
    "T11-2-1": (E, ("selection.variance", "selection.k_meilleures", "selection.rfecv")),
    "T12-1-1": (E, "chronologique.variables_temporelles"),
    "T12-1-2": (E, ("validation.serie_temporelle", "reference.persistance")),
    "T12-2-1": (E, "chronologique.cible_decalee"),
    "T12-3-1": (
        E,
        "chronologique.variables_temporelles",
    ),  # lissage exponentiel ; horizons : EXCLUSIONS_PARTIELLES
    "T12-4-1": (P, "class_weight"),
    "T13-1-1": (
        X,
        "calcul pas à pas du perceptron : illustration du livre, l'application emploie le MLP de sklearn",
    ),
    "T13-2-1": (E, ("regression.reseau_neurones", "classification.reseau_neurones")),
    "T17-1-1": (
        X,
        "substitut d'un calcul par éléments finis : usage de la régression, domaine de validité lu en S7",
    ),
    "T17-2-1": (
        X,
        "loi de puissance par régression sur les logarithmes : usage de la régression, hors catalogue",
    ),
    "T19-1-1": (
        X,
        "sérialisation : export joblib à l'étape S9 (écriture seule, chargement interdit par I10)",
    ),
    "T19-2-1": (
        X,
        "surveillance de dérive en production hors périmètre ; indice de stabilité fourni au rapport",
    ),
}


def _techniques_du_fichier() -> dict[str, bool]:
    """Identifiants du fichier de la liste du livre, avec la marque « hors application »."""
    trouvees = {}
    for ligne in LIVRE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(T\d{2}-\d+-\d+)\s*\|.*\|\s*(.+?)\s*\|\s*$", ligne)
        if m:
            trouvees[m.group(1)] = m.group(2).endswith("[hors application]")
    return trouvees


TECHNIQUES = _techniques_du_fichier()
A_COUVRIR = sorted(i for i, hors in TECHNIQUES.items() if not hors)


def test_la_liste_du_livre_est_lue() -> None:
    assert len(TECHNIQUES) == 63 and len(A_COUVRIR) == 58
    assert set(TECHNIQUES) == set(TECHNIQUES_LIVRE), (
        "regénérer livre.py : python tests/outils/generer_livre.py"
    )
    hors = {i for i, h in TECHNIQUES.items() if h}
    assert hors == {"T14-1-1", "T14-2-1", "T15-1-1", "T16-1-1", "T16-2-1"}  # réseaux Keras : motif du livre


@pytest.mark.parametrize("technique", A_COUVRIR)
def test_chaque_technique_du_livre_a_une_entree_ou_une_exclusion_motivee(technique: str) -> None:
    assert technique in COUVERTURE, f"{technique} : ni entrée ni exclusion motivée dans la couverture"
    nature, cibles = COUVERTURE[technique]
    cibles = (cibles,) if isinstance(cibles, str) else cibles
    if nature == E:
        assert all(c in IDS for c in cibles), cibles
    elif nature == M:
        assert all(c in {m.id for m in metrics.METRIQUES} for c in cibles), cibles
    elif nature == P:
        avec = [e.id for e in entrees(probleme=Probleme.CLASSIFICATION) if cibles[0] in dict(e.grille_bornee)]
        assert len(avec) >= 4, "class_weight est un paramètre de plusieurs classifieurs"
    else:
        assert len(cibles[0].split()) >= 8, "une exclusion doit être motivée"


def test_la_couverture_ne_cite_que_des_techniques_du_livre_non_exclues() -> None:
    assert set(COUVERTURE) <= set(A_COUVRIR), sorted(set(COUVERTURE) - set(A_COUVRIR))
    assert set(COUVERTURE) == set(A_COUVRIR)


def test_chaque_entree_renvoie_a_une_technique_du_livre_ou_est_une_extension_declaree() -> None:
    for e in REGISTRE:
        assert e.ref_livre in TECHNIQUES and not TECHNIQUES[e.ref_livre], (e.id, e.ref_livre)
    assert set(IDS) >= EXTENSIONS
    # une entrée citée par la couverture n'est jamais une extension, et inversement
    citees = {
        c
        for n, cibles in COUVERTURE.values()
        if n == E
        for c in ([cibles] if isinstance(cibles, str) else cibles)
    }
    assert not EXTENSIONS & citees, sorted(EXTENSIONS & citees)
    assert set(IDS) - citees - EXTENSIONS == set(), sorted(set(IDS) - citees - EXTENSIONS)


def _citations() -> dict[tuple[str, str], set[str]]:
    """Pour chaque entrée ou métrique que la couverture cite : les techniques du livre qui la citent."""
    citations: dict[tuple[str, str], set[str]] = {}
    for technique, (nature, cibles) in COUVERTURE.items():
        if nature in (E, M):
            for cible in [cibles] if isinstance(cibles, str) else cibles:
                citations.setdefault((nature, cible), set()).add(technique)
    return citations


def test_le_renvoi_d_une_entree_ou_d_une_metrique_citee_est_une_technique_qui_la_cite() -> None:
    """Accord de la couverture et des renvois : un renvoi faux mais valide (la crête renvoyée à la régression
    linéaire multiple, par exemple) ne passe pas."""
    citations = _citations()
    assert len(citations) > 60
    for (nature, identifiant), techniques in citations.items():
        renvoi = entree(identifiant).ref_livre if nature == E else metrics.metrique(identifiant).ref_livre
        assert renvoi in techniques, (identifiant, renvoi, sorted(techniques))


def test_la_regression_lineaire_renvoie_a_la_regression_multiple_et_non_a_la_premiere_approche() -> None:
    assert entree("regression.lineaire").ref_livre == "T05-1-1"
    assert TECHNIQUES_LIVRE["T05-1-1"].libelle.startswith("Régression linéaire multiple")


def test_les_renvois_des_metriques_sont_ceux_de_la_table_unique() -> None:
    assert set(renvois.METRIQUES) == {m.id for m in metrics.METRIQUES}
    for m in metrics.METRIQUES:
        assert m.ref_livre == renvois.metrique(m.id) and est_identifiant(m.ref_livre), m.id
    with pytest.raises(KeyError, match="Renvoi au livre manquant"):
        renvois.metrique("inconnue")


CORE = SRC / "mecaml" / "core"
TABLES_DE_RENVOIS = ("renvois.py", "livre.py")  # la table des renvois et la liste du livre


def _identifiants_ecrits(racine: Path) -> list[str]:
    """Emplacements (fichier:ligne) d'un identifiant `T<cc>-<s>-<k>` écrit hors des deux modules permis."""
    motif = re.compile(r"\bT\d{2}-\d+-\d+\b")
    return [
        f"{chemin.relative_to(racine).as_posix()}:{numero}"
        for chemin in sorted(racine.rglob("*.py"))
        if chemin.relative_to(racine).as_posix() not in TABLES_DE_RENVOIS
        for numero, ligne in enumerate(chemin.read_text(encoding="utf-8").splitlines(), 1)
        if motif.search(ligne)
    ]


def test_aucun_identifiant_du_livre_n_est_ecrit_hors_de_la_table_unique_des_renvois(tmp_path: Path) -> None:
    assert _identifiants_ecrits(CORE) == []
    # le contrôle voit un identifiant écrit ailleurs : un second tableau de renvois ne passerait pas
    (tmp_path / "livre.py").write_text('IDENTIFIANTS = ("T01-1-1",)\n', encoding="utf-8")
    (tmp_path / "split.py").write_text('REF = "T02-1-1"\n', encoding="utf-8")
    assert _identifiants_ecrits(tmp_path) == ["split.py:1"]


def test_les_cles_de_texte_du_registre_et_des_metriques() -> None:
    """Chaque entrée et chaque métrique porte la clé de son texte : celles que les écrans doivent fournir."""
    cles = [e.texte_id for e in REGISTRE] + [m.texte_id for m in metrics.METRIQUES]
    assert len(cles) == len(set(cles)) == len(REGISTRE) + len(metrics.METRIQUES)
    assert all(re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)+", cle) for cle in cles)
    assert {e.texte_id for e in REGISTRE} == set(IDS)  # clé d'une technique : son identifiant
    assert {m.texte_id for m in metrics.METRIQUES} == {f"metrique.{m.id}" for m in metrics.METRIQUES}
    assert not any(cle.startswith("decoupage.") for cle in cles)  # clés de renvoi, non de texte
    assert {
        "transformation.ecretage_tukey",
        "reference.persistance",
        "chronologique.cible_decalee",
        "sans_cible.classification_hierarchique",
        "validation.plis_groupes_stratifies",
        "metrique.mape",
    } <= set(cles)


def test_les_techniques_ajoutees_pour_couvrir_le_livre_existent_ou_sont_exclues_avec_un_motif() -> None:
    """Les techniques ajoutées pour couvrir le livre existent, ou sont exclues avec un motif."""
    assert "mape" in {m.id for m in metrics.METRIQUES}
    assert {"sans_cible.classification_hierarchique", "reference.persistance"} <= set(IDS)
    assert {"chronologique.cible_decalee", "transformation.ecretage_tukey"} <= set(IDS)
    assert COUVERTURE["T19-2-1"][0] == X and "indice de stabilité" in COUVERTURE["T19-2-1"][1]


# Techniques que le livre réunit sous un seul identifiant et que l'application couvre en partie : la partie
# sans entrée est exclue avec un motif.
EXCLUSIONS_PARTIELLES = {
    "T12-3-1": (
        "prévision à plusieurs horizons : elle suppose une cible décalée vers l'avant, un modèle par "
        "horizon et des plis qui respectent l'horizon, alors que l'application valide un seul horizon, "
        "la ligne suivante"
    ),
}


def test_la_prevision_a_plusieurs_horizons_est_exclue_avec_un_motif() -> None:
    """T12-3-1 réunit le lissage exponentiel (couvert) et la prévision à plusieurs horizons (exclue)."""
    # la fiche porte le lissage ; la section 12.3 du livre traite les plusieurs horizons
    assert TECHNIQUES_LIVRE["T12-3-1"].libelle.startswith("Lissage exponentiel")
    assert TECHNIQUES_LIVRE["T12-3-1"].section == "12.3"
    assert COUVERTURE["T12-3-1"][0] == E and "horizons" in EXCLUSIONS_PARTIELLES["T12-3-1"]


@pytest.mark.parametrize("technique", sorted(EXCLUSIONS_PARTIELLES))
def test_la_partie_sans_entree_d_une_technique_est_exclue_avec_un_motif(technique: str) -> None:
    nature, cibles = COUVERTURE[technique]
    assert nature == E, "la partie couverte a une entrée"
    assert len(EXCLUSIONS_PARTIELLES[technique].split()) >= 8, "une exclusion doit être motivée"
    # T12-3-1 : le lissage exponentiel est la moyenne exponentielle (`demi_vies`) des variables temporelles
    objet = entree(cibles).construire(0, colonne_date="t", colonnes=("a",), demi_vies=(3.0,))
    assert objet.demi_vies == (3.0,)


@pytest.mark.parametrize("identifiant", sorted(cible_passee.TECHNIQUES))
def test_la_cible_passee_exige_une_declaration_explicite(identifiant: str) -> None:
    e = entree(identifiant)
    commun = {"colonne_date": "t", "colonnes": ("a",)} if e.famille == "chronologique" else {}
    for refus in ({}, {"cible_connue": False}, {"cible_connue": "oui"}, {"cible_connue": 1}):
        with pytest.raises(ValueError, match=r"cible_connue=True"):
            e.construire(0, **commun, **refus)
    assert e.construire(0, **commun, cible_connue=True) is not None
