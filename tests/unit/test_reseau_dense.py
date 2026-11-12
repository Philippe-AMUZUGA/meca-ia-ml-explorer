"""Réseau dense Keras : technique facultative, absente du registre de base, jamais requise."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aides_apprentissage as aides
from mecaml.core import catalog, hyperparametres, libelles, train
from mecaml.core.catalog import OPTIONNELLES, REGISTRE, optionnelles_disponibles, reseau_dense
from mecaml.core.codegen import ErreurExport, verifier_ceinture
from mecaml.core.types import ChoixTechnique, Probleme, Table

SRC = Path(__file__).resolve().parents[2] / "src"
CT = ChoixTechnique.creer
AVEC_KERAS = reseau_dense.keras_present()
IDS = {"regression.reseau_dense", "classification.reseau_dense"}


# ============================================================
# SANS KERAS : le registre de base ne change pas
# ============================================================
def test_les_reseaux_denses_sont_hors_du_registre_de_base() -> None:
    assert {e.id for e in OPTIONNELLES} == IDS
    assert not IDS & {e.id for e in REGISTRE}
    assert len(REGISTRE) == 75


def test_sans_keras_les_reseaux_sont_inconnus_et_le_message_le_dit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reseau_dense, "keras_present", lambda: False)
    assert optionnelles_disponibles() == ()
    assert catalog.entrees() == REGISTRE
    assert IDS.isdisjoint(e.id for e in catalog.entrees(probleme=Probleme.CLASSIFICATION))
    with pytest.raises(KeyError, match="demande Keras"):
        catalog.entree("regression.reseau_dense")
    with pytest.raises(KeyError, match="Technique inconnue"):
        catalog.entree("regression.inexistante")


def test_avec_keras_les_reseaux_s_ajoutent_apres_le_registre(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reseau_dense, "keras_present", lambda: True)
    assert catalog.entrees()[: len(REGISTRE)] == REGISTRE
    assert {e.id for e in catalog.entrees()[len(REGISTRE) :]} == IDS
    assert catalog.entree("classification.reseau_dense").cout.value == "lourd"
    assert [e.id for e in catalog.entrees(probleme=Probleme.REGRESSION)][-1] == "regression.reseau_dense"


def test_la_detection_de_keras_n_importe_ni_keras_ni_scikit_learn() -> None:
    programme = (
        "import sys, mecaml.core.catalog as c\n"
        "c.optionnelles_disponibles()\n"
        "print('keras' in sys.modules, 'tensorflow' in sys.modules, 'sklearn' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    sortie = subprocess.run(
        [sys.executable, "-c", programme], capture_output=True, text=True, env=env, check=True
    )
    assert sortie.stdout.split() == ["False", "False", "False"]


def test_sans_keras_sur_le_chemin_le_catalogue_se_liste_sans_erreur() -> None:
    """Environnement sans Keras simulé : les dossiers de paquets tiers sont retirés du chemin de recherche."""
    programme = (
        "import sys\n"
        "import mecaml.core.catalog as c\n"
        "sys.path[:] = [p for p in sys.path if 'site-packages' not in p]\n"
        "assert not c.optionnelles_disponibles()\n"
        "print(len(c.entrees()) == len(c.REGISTRE))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    sortie = subprocess.run([sys.executable, "-c", programme], capture_output=True, text=True, env=env)
    assert sortie.returncode == 0, sortie.stderr
    assert sortie.stdout.split() == ["True"]


def test_les_titres_les_textes_et_les_reglages_des_reseaux_existent() -> None:
    for identifiant in IDS:
        assert libelles.technique(identifiant) != identifiant
        assert set(reseau_dense.TEXTES[identifiant]) == {"quoi", "regarder"}
        assert 3 <= len(hyperparametres.principaux(identifiant)) <= 6
        assert hyperparametres.avances(identifiant)
        for h in hyperparametres.schema(identifiant):
            assert hyperparametres.libelle(identifiant, h.nom) != h.nom, h.nom
            assert hyperparametres.aide(identifiant, h.nom), h.nom


def test_les_reglages_des_reseaux_ont_des_bornes_de_calcul() -> None:
    schema = {h.nom: h for h in hyperparametres.schema("regression.reseau_dense")}
    assert schema["epochs"].maximum == 200 and schema["taille_lot"].maximum == 4096
    assert schema["abandon"].maximum < 1
    with pytest.raises(hyperparametres.ErreurHyperparametre, match="Époques"):
        hyperparametres.convertir("regression.reseau_dense", "epochs", 201)
    with pytest.raises(hyperparametres.ErreurHyperparametre, match="Couches cachées"):
        hyperparametres.convertir("classification.reseau_dense", "couches", "8, 8, 8, 8")


def test_l_export_refuse_un_reseau_keras_avec_un_message_clair() -> None:
    df = aides.jeu_regression()
    table = Table(df, "regression")
    spec = aides.spec_regression(modeles=(CT("reference.dummy_regression"), CT("regression.reseau_dense")))
    coffre = aides.coffre_par_pas(table, spec)
    with pytest.raises(ErreurExport, match=r"Réseau dense Keras \(régression\) n'a pas de script"):
        verifier_ceinture(spec, table, coffre)


# ============================================================
# AVEC KERAS : ajustement, reproductibilité, comportement scikit-learn
# ============================================================
pytestmark_keras = pytest.mark.skipif(not AVEC_KERAS, reason="Keras absent de cet environnement (facultatif)")


def _jeu(n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 3))
    return x, x[:, 0] * 2 - x[:, 1] + rng.normal(scale=0.05, size=n)


@pytestmark_keras
def test_la_regression_apprend_et_se_reproduit_a_graine_egale() -> None:
    x, y = _jeu()
    petits = {"couches": (16,), "epochs": 120, "taux_apprentissage": 0.02, "arret_precoce": False}
    a = catalog.entree("regression.reseau_dense").construire(42, **petits).fit(x, y)
    b = catalog.entree("regression.reseau_dense").construire(42, **petits).fit(x, y)
    np.testing.assert_allclose(a.predict(x), b.predict(x), rtol=0, atol=1e-6)
    assert a.predict(x).shape == (80,) and a.n_features_in_ == 3
    reste = np.mean((a.predict(x) - y) ** 2) / np.var(y)
    assert reste < 0.5, "le réseau doit mieux faire que la moyenne"


@pytestmark_keras
def test_la_classification_rend_des_probabilites_et_les_classes_d_origine() -> None:
    x, y = _jeu()
    classes = np.where(y > 0, "defaut", "conforme")
    modele = catalog.entree("classification.reseau_dense").construire(
        42, couches=(16,), epochs=120, taux_apprentissage=0.02, arret_precoce=False
    )
    modele.fit(x, classes)
    proba = modele.predict_proba(x)
    assert proba.shape == (80, 2) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert set(modele.predict(x)) <= {"defaut", "conforme"} and list(modele.classes_) == [
        "conforme",
        "defaut",
    ]
    assert (modele.predict(x) == classes).mean() > 0.8


@pytestmark_keras
def test_le_reseau_se_clone_et_refuse_les_valeurs_manquantes() -> None:
    from sklearn.base import clone

    x, y = _jeu()
    modele = catalog.entree("regression.reseau_dense").construire(42, couches=(4,), epochs=2)
    copie = clone(modele)
    assert copie.get_params() == modele.get_params()
    x_trou = x.copy()
    x_trou[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        copie.fit(x_trou, y)


@pytestmark_keras
def test_le_reseau_se_compare_dans_un_pipeline_et_la_validation_croisee() -> None:
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = _jeu()
    modele = make_pipeline(
        StandardScaler(),
        catalog.entree("regression.reseau_dense").construire(
            42, couches=(8,), epochs=15, arret_precoce=False
        ),
    )
    scores = cross_val_score(modele, x, y, cv=3, scoring="r2")
    assert scores.shape == (3,) and np.all(np.isfinite(scores))


@pytestmark_keras
def test_l_entrainement_de_l_application_accepte_un_reseau_dense() -> None:
    """Le réseau passe par `train.entrainer` comme les autres modèles, face à la référence naïve."""
    df = aides.jeu_regression()
    table = Table(df, "regression")
    modeles = (
        CT("regression.reseau_dense", couches=(4,), epochs=3, arret_precoce=False),
        CT("regression.ridge"),
    )
    spec = aides.spec_regression(modeles=modeles)
    resultat = train.entrainer(spec, table, aides.coffre_par_pas(table, spec))
    assert resultat.modele("regression.reseau_dense").ok and resultat.reference().ok


def test_les_listes_de_modeles_de_l_application_proposent_les_reseaux_seulement_avec_keras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reseau_dense, "keras_present", lambda: False)
    assert IDS.isdisjoint(e.id for p in Probleme for e in train.modeles_disponibles(p))
    monkeypatch.setattr(reseau_dense, "keras_present", lambda: True)
    assert "regression.reseau_dense" in {e.id for e in train.modeles_disponibles(Probleme.REGRESSION)}
    assert "classification.reseau_dense" in {e.id for e in train.modeles_disponibles(Probleme.CLASSIFICATION)}
    assert not IDS & {e.id for e in train.modeles_disponibles(Probleme.SANS_CIBLE)}
