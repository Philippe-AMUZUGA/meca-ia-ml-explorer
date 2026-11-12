"""Réglages des modèles : schémas, valeurs par défaut, bornes, refus avec message, textes français."""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from mecaml.core import catalog, hyperparametres
from mecaml.core.catalog import REGISTRE, entree
from mecaml.core.catalog._outils import LIMITES
from mecaml.core.catalog.reseau_dense import ENTREES as ENTREES_DENSES
from mecaml.core.catalog.vote import BASE_PAR_DEFAUT
from mecaml.core.hyperparametres import (
    SCHEMAS,
    ErreurHyperparametre,
    Genre,
    aide,
    avances,
    controler,
    convertir,
    libelle,
    principaux,
    texte_bornes,
    texte_valeur,
    valider,
)
from mecaml.core.hyperparametres_textes import MOTS, TEXTES

SRC = Path(__file__).resolve().parents[2] / "src"
IDS_DENSES = {e.id for e in ENTREES_DENSES}
MODELES = [e.id for e in REGISTRE if e.famille in ("regression", "classification", "sans_cible", "reference")]
MODELES = [i for i in MODELES if i != "reference.persistance"]
# Modèles dont scikit-learn n'offre pas trois réglages utiles : tous leurs réglages sont déclarés.
MOINS_DE_TROIS = {
    "classification.bayes_naif",
    "classification.adaboost",
    "regression.vote",
    "classification.vote",
    "reference.dummy_regression",
    "reference.dummy_classification",
}
TIRETS = re.compile(f"[{chr(0x2013)}{chr(0x2014)}]|-{{2}}")
COUPLES = [(i, h.nom) for i, schema in SCHEMAS.items() for h in schema]
NOMBRES = [(i, h) for i, schema in SCHEMAS.items() for h in schema if h.genre in (Genre.ENTIER, Genre.REEL)]


def _reglage(identifiant: str, nom: str) -> hyperparametres.Hyperparametre:
    return next(h for h in SCHEMAS[identifiant] if h.nom == nom)


# ============================================================
# COUVERTURE DU CATALOGUE
# ============================================================
def test_chaque_modele_du_catalogue_declare_ses_reglages() -> None:
    assert sorted(set(MODELES) - set(SCHEMAS)) == []
    assert sorted(set(SCHEMAS) - set(MODELES) - IDS_DENSES) == []


def test_de_trois_a_six_reglages_principaux_sauf_les_modeles_pauvres() -> None:
    for identifiant in SCHEMAS:
        n = len(principaux(identifiant))
        assert 1 <= n <= 6, identifiant
        assert n >= 3 or identifiant in MOINS_DE_TROIS, f"{identifiant} : {n} réglage(s) principal(aux)"
    assert all(len(principaux(i)) == len(SCHEMAS[i]) - len(avances(i)) for i in SCHEMAS)


def test_les_modeles_pauvres_n_ont_pas_trois_reglages_utiles() -> None:
    assert all(len(SCHEMAS[i]) <= 3 for i in MOINS_DE_TROIS)


def test_les_noms_de_reglage_sont_uniques_par_modele() -> None:
    for identifiant, schema in SCHEMAS.items():
        noms = [h.nom for h in schema]
        assert len(noms) == len(set(noms)), identifiant


# ============================================================
# VALEURS PAR DÉFAUT ET BORNES
# ============================================================
@pytest.mark.parametrize(("identifiant", "nom"), [c for c in COUPLES if c[0] not in IDS_DENSES])
def test_la_valeur_par_defaut_est_celle_du_modele(identifiant: str, nom: str) -> None:
    h = _reglage(identifiant, nom)
    if h.genre is Genre.LISTE:
        assert tuple(h.defaut) == BASE_PAR_DEFAUT[next(iter(entree(identifiant).problemes))]
        return
    effective = entree(identifiant).construire(42).get_params(deep=False)[nom]
    equivalent = nom == "max_samples" and effective is None  # None : toutes les lignes
    assert effective == h.defaut or equivalent, f"{identifiant}.{nom} : {h.defaut!r} contre {effective!r}"


@pytest.mark.parametrize(("identifiant", "h"), NOMBRES)
def test_les_bornes_du_reglage_ne_depassent_pas_le_garde_fou_de_calcul(identifiant: str, h) -> None:
    if h.nom in LIMITES:
        assert h.maximum <= LIMITES[h.nom], f"{identifiant}.{h.nom}"
    assert h.minimum < h.maximum
    assert h.minimum <= (h.defaut if h.defaut is not None and not isinstance(h.defaut, str) else h.minimum)


@pytest.mark.parametrize(("identifiant", "nom"), COUPLES)
def test_le_defaut_fait_l_aller_retour_par_son_texte(identifiant: str, nom: str) -> None:
    h = _reglage(identifiant, nom)
    if h.genre is Genre.LISTE:
        assert convertir(identifiant, nom, texte_valeur(h, h.defaut)) == tuple(h.defaut)
    else:
        assert convertir(identifiant, nom, texte_valeur(h, h.defaut)) == h.defaut


@pytest.mark.parametrize("identifiant", [i for i in SCHEMAS if i not in IDS_DENSES])
def test_les_bornes_et_les_choix_permis_construisent_le_modele(identifiant: str) -> None:
    e = entree(identifiant)
    for h in SCHEMAS[identifiant]:
        if h.genre in (Genre.ENTIER, Genre.REEL):
            valeurs = [h.minimum, h.maximum]
        elif h.genre is Genre.CHOIX:
            valeurs = [*h.choix, *(v for v, _ in h.speciaux)]
        elif h.genre is Genre.BOOLEEN:
            valeurs = [True, False]
        else:
            continue
        for brut in valeurs:
            e.construire(42, **{h.nom: convertir(identifiant, h.nom, brut)})


# ============================================================
# REFUS HORS BORNES, AVEC UN MESSAGE CLAIR
# ============================================================
@pytest.mark.parametrize(("identifiant", "h"), NOMBRES)
def test_une_valeur_hors_bornes_est_refusee_avec_le_libelle_et_la_plage(identifiant: str, h) -> None:
    for hors in (h.minimum - 1, h.maximum + 1):
        with pytest.raises(ErreurHyperparametre) as erreur:
            convertir(identifiant, h.nom, hors)
        message = str(erreur.value)
        assert libelle(identifiant, h.nom) in message and "hors bornes" in message
        assert texte_bornes(h) in message


def test_le_coeur_refuse_un_reglage_hors_bornes_a_la_construction() -> None:
    with pytest.raises(ErreurHyperparametre, match=r"Nombre d'arbres : « 0 » est hors bornes \(1 à 500"):
        entree("regression.foret").construire(42, n_estimators=0)
    with pytest.raises(ErreurHyperparametre, match="Quantile visé"):
        entree("regression.quantile").construire(42, quantile=1.5)
    with pytest.raises(ErreurHyperparametre, match="Part de Lasso"):
        entree("regression.elastic_net").construire(42, l1_ratio=-0.1)


def test_un_reglage_hors_bornes_n_atteint_pas_la_bibliotheque() -> None:
    """Le refus vient du cœur : aucune erreur de scikit-learn n'est lue par l'utilisateur."""
    with pytest.raises(ErreurHyperparametre):
        entree("classification.svc").construire(42, C=-1)
    with pytest.raises(ErreurHyperparametre):
        entree("sans_cible.dbscan").construire(42, eps=0)


@pytest.mark.parametrize(
    ("brut", "motif"),
    [
        ("abc", "n'est pas un nombre"),
        (2.5, "n'est pas un entier"),
        (float("nan"), "n'est pas un nombre fini"),
        (math.inf, "n'est pas un nombre fini"),
        (True, "n'est pas un nombre"),
        ([3], "n'est pas un nombre"),
    ],
)
def test_une_valeur_qui_n_est_pas_un_entier_est_refusee(brut: object, motif: str) -> None:
    with pytest.raises(ErreurHyperparametre, match=motif):
        convertir("regression.foret", "n_estimators", brut)


def test_choix_booleens_couches_et_listes_sont_controles() -> None:
    with pytest.raises(ErreurHyperparametre, match="n'est pas permis"):
        convertir("regression.ridge", "solver", "inconnu")
    with pytest.raises(ErreurHyperparametre, match="oui ou non"):
        convertir("regression.ridge", "positive", "peut-être")
    with pytest.raises(ErreurHyperparametre, match="hors bornes"):
        convertir("regression.reseau_neurones", "hidden_layer_sizes", "10, 10, 10, 10")
    with pytest.raises(ErreurHyperparametre, match="hors bornes"):
        convertir("regression.reseau_neurones", "hidden_layer_sizes", "0")
    with pytest.raises(ErreurHyperparametre, match="hors bornes"):
        convertir("classification.vote", "base", "classification.arbre")
    assert convertir("regression.reseau_neurones", "hidden_layer_sizes", "50, 20") == (50, 20)
    assert convertir("regression.reseau_neurones", "hidden_layer_sizes", "(30,)") == (30,)
    assert convertir("regression.reseau_neurones", "hidden_layer_sizes", [8, 8]) == (8, 8)


def test_un_reglage_inconnu_est_refuse_par_la_saisie_mais_pas_par_le_controle_du_coeur() -> None:
    with pytest.raises(KeyError, match="n'a pas de réglage"):
        valider("regression.ridge", {"inexistant": "1"})
    controler("regression.ridge", {"inexistant": 1, "n_jobs": 1})  # passe sans contrôle


# ============================================================
# SAISIE
# ============================================================
def test_valider_ne_retient_que_ce_qui_differe_du_defaut_et_lit_la_virgule() -> None:
    saisies = {"alpha": "0,5", "fit_intercept": "oui", "positive": "", "solver": "auto", "tol": "1e-4"}
    assert valider("regression.ridge", saisies) == {"alpha": 0.5}


def test_valider_comprend_les_mots_des_valeurs_speciales() -> None:
    retenu = valider("regression.foret", {"max_depth": "8", "n_estimators": " 200 "})
    assert retenu == {"max_depth": 8, "n_estimators": 200}
    assert valider("regression.foret", {"max_depth": "aucune limite"}) == {}  # le défaut est déjà sans limite
    assert valider("regression.arbre", {"max_depth": "aucune limite"}) == {"max_depth": None}
    assert valider("sans_cible.foret_isolement", {"contamination": "0,1"}) == {"contamination": 0.1}
    assert valider("sans_cible.foret_isolement", {"contamination": "auto"}) == {}
    assert valider("sans_cible.acp", {"n_components": "0,9"}) == {"n_components": 0.9}
    assert valider("sans_cible.acp", {"n_components": "2"}) == {"n_components": 2}
    assert valider("classification.foret", {"class_weight": "balanced"}) == {"class_weight": "balanced"}


def test_les_valeurs_valides_sont_ajustables() -> None:
    """Des réglages changés se construisent et entraînent un petit jeu sans erreur de la bibliothèque."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] * 2 + rng.normal(scale=0.1, size=60)
    classes = (y > 0).astype(int)
    essais = {
        "regression.foret": {"n_estimators": 5, "min_samples_leaf": 2, "max_depth": 4},
        "regression.gradient_boosting": {"n_estimators": 10, "subsample": 0.8, "max_depth": 2},
        "regression.reseau_neurones": {"hidden_layer_sizes": (8, 4), "max_iter": 30, "activation": "tanh"},
        "regression.svr": {"C": 5.0, "kernel": "linear", "epsilon": 0.2},
        "regression.lasso": {"alpha": 0.05, "positive": True, "max_iter": 500},
        "regression.plus_proches_voisins": {"n_neighbors": 3, "weights": "distance", "p": 1},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # 30 itérations de réseau : voulu pour aller vite
        for identifiant, params in essais.items():
            modele = entree(identifiant).construire(42, **params)
            modele.fit(x, y)
            assert modele.predict(x).shape == (60,), identifiant
    essais_classes = {
        "classification.arbre": {"max_depth": 3, "class_weight": "balanced", "criterion": "entropy"},
        "classification.logistique": {"C": 0.5, "max_iter": 200},
        "classification.foret": {"n_estimators": 5, "class_weight": "balanced_subsample"},
        "classification.vote": {"voting": "hard"},
        "classification.empilement": {"cv": 2, "passthrough": True},
    }
    for identifiant, params in essais_classes.items():
        modele = entree(identifiant).construire(42, **params)
        modele.fit(x, classes)
        assert set(modele.predict(x)) <= {0, 1}, identifiant


# ============================================================
# TEXTES
# ============================================================
@pytest.mark.parametrize(("identifiant", "nom"), COUPLES)
def test_chaque_reglage_a_un_libelle_francais_et_une_aide_d_une_ligne(identifiant: str, nom: str) -> None:
    lib, ligne = libelle(identifiant, nom), aide(identifiant, nom)
    assert lib != nom and (lib[0].isupper() or lib[0].isdigit()), f"{identifiant}.{nom} : libellé"
    assert 3 <= len(lib) <= 40
    assert ligne and "\n" not in ligne and ligne.endswith(".") and len(ligne) <= 80, (
        f"{identifiant}.{nom} : aide"
    )
    assert not TIRETS.search(lib + ligne)


def test_chaque_texte_de_reglage_sert_a_un_modele() -> None:
    noms = {h.nom for schema in SCHEMAS.values() for h in schema}
    complets = {f"{i}.{h.nom}" for i, schema in SCHEMAS.items() for h in schema}
    assert sorted(c for c in TEXTES if c not in noms and c not in complets) == []


def test_les_mots_des_bornes_et_des_messages_sont_complets() -> None:
    attendus = {
        "a",
        "oui",
        "non",
        "aucun",
        "illimite",
        "auto",
        "toutes",
        "defaut",
        "couches",
        "liste",
        "part",
    }
    attendus |= {"pas_nombre", "pas_fini", "pas_entier", "pas_booleen", "hors_bornes", "hors_liste"}
    assert attendus <= set(MOTS)
    assert not any(TIRETS.search(m) for m in MOTS.values())


def test_la_plage_et_la_valeur_se_lisent_a_la_francaise() -> None:
    assert texte_bornes(_reglage("regression.foret", "n_estimators")) == "1 à 500"
    assert texte_bornes(_reglage("regression.foret", "max_depth")) == "1 à 50, aucune limite"
    assert texte_bornes(_reglage("regression.ridge", "solver")).startswith("auto, svd")
    assert texte_valeur(_reglage("regression.ridge", "alpha"), 0.5) == "0,5"
    assert texte_valeur(_reglage("regression.ridge", "alpha"), 1000000.0) == "1 000 000"
    assert texte_valeur(_reglage("classification.bayes_naif", "var_smoothing"), 1e-9) == "1e-9"
    assert texte_valeur(_reglage("regression.foret", "max_depth"), None) == "aucune limite"
    assert texte_valeur(_reglage("regression.ridge", "positive"), False) == "non"


# ============================================================
# SERVICES DU CHOIX
# ============================================================
def test_un_choix_enregistre_se_controle_par_le_meme_cœur() -> None:
    from mecaml.core.types import ChoixTechnique

    choix = ChoixTechnique.creer("regression.foret", n_estimators=120, max_depth=7)
    controler(choix.id, choix.parametres_dict())
    hors = ChoixTechnique.creer("regression.foret", n_estimators=501)
    with pytest.raises(ValueError, match="borne"):
        entree(hors.id).construire(42, **hors.parametres_dict())


def test_les_reglages_sont_dans_le_registre_sans_importer_scikit_learn() -> None:
    import subprocess
    import sys

    programme = (
        "import sys, mecaml.core.hyperparametres as h\nprint(len(h.SCHEMAS), 'sklearn' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    sortie = subprocess.run(
        [sys.executable, "-c", programme], capture_output=True, text=True, env=env, check=True
    )
    n, charge = sortie.stdout.split()
    assert int(n) == len(SCHEMAS) and charge == "False"


def test_le_catalogue_et_les_optionnelles_ne_se_melangent_pas() -> None:
    assert not (IDS_DENSES & {e.id for e in REGISTRE})
    assert catalog.entrees()[: len(REGISTRE)] == REGISTRE
