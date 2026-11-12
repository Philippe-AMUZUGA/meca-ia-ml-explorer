"""I2, ajustement confiné : tout ajustement vit dans le Pipeline, donc dans les plis.

Deux preuves. Statique : aucun module du cœur n'appelle `fit_transform` ou `partial_fit`, et seuls les
modules d'apprentissage appellent `fit` ou `fit_predict`, sur des pipelines. Dynamique : un transformeur
espion, placé en tête du prétraitement, enregistre les lignes qu'il voit ; il n'ajuste jamais que sur les
lignes d'apprentissage d'un pli, ne transforme jamais une ligne de test, et l'ajustement final porte sur
toute la base d'apprentissage et rien d'autre.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

import aides_apprentissage as aides
from mecaml.core import preprocess, train
from mecaml.core.catalog import REGISTRE, Cout, EntreeCatalogue
from mecaml.core.types import ChoixTechnique, Probleme, Table

CT = ChoixTechnique.creer
CORE = Path(__file__).resolve().parents[2] / "src" / "mecaml" / "core"
# Modules autorisés à ajuster, toujours sur un Pipeline : l'entraînement, et les modules qui en dérivent.
# `reseau_dense.py` écrit l'ajustement d'un estimateur : son `fit` est celui que le pipeline appelle.
AJUSTEURS_PERMIS = {
    "train.py",
    "evaluate.py",
    "decision.py",
    "explain.py",
    "unsupervised.py",
    "reseau_dense.py",
}
RECEPTEURS_PERMIS = {"modele", "pipeline"}


# ============================================================
# PREUVE STATIQUE
# ============================================================
def _appels_de_methodes(chemin: Path) -> list[tuple[str, str, int]]:
    """(méthode, nom du récepteur ou « ? », ligne) de chaque appel `objet.methode(...)` du module."""
    arbre = ast.parse(chemin.read_text(encoding="utf-8"), filename=str(chemin))
    appels = []
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Call) and isinstance(noeud.func, ast.Attribute):
            recepteur = noeud.func.value.id if isinstance(noeud.func.value, ast.Name) else "?"
            appels.append((noeud.func.attr, recepteur, noeud.lineno))
    return appels


@pytest.mark.parametrize("chemin", sorted(CORE.rglob("*.py")), ids=lambda p: p.relative_to(CORE).as_posix())
def test_aucun_ajustement_hors_du_pipeline(chemin: Path) -> None:
    for methode, recepteur, ligne in _appels_de_methodes(chemin):
        lieu = f"{chemin.name}:{ligne}"
        assert methode not in ("fit_transform", "partial_fit"), f"{lieu} : {methode} hors d'un Pipeline"
        if methode in ("fit", "fit_predict"):
            assert chemin.name in AJUSTEURS_PERMIS, (
                f"{lieu} : ajustement dans un module qui ne doit pas ajuster"
            )
            assert recepteur in RECEPTEURS_PERMIS, (
                f"{lieu} : {methode} sur {recepteur!r}, pas sur un pipeline"
            )


def test_l_analyse_statique_detecte_les_fautes(tmp_path: Path) -> None:
    faute = tmp_path / "faute.py"
    faute.write_text("def f(x, y):\n    return scaler.fit_transform(x)\n", encoding="utf-8")
    assert [(m, r) for m, r, _ in _appels_de_methodes(faute)] == [("fit_transform", "scaler")]
    ok = tmp_path / "ok.py"
    ok.write_text("def f(pipeline, x, y):\n    pipeline.fit(x, y)\n", encoding="utf-8")
    assert [(m, r) for m, r, _ in _appels_de_methodes(ok)] == [("fit", "pipeline")]


def test_la_construction_du_pipeline_n_ajuste_rien() -> None:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    pipeline = train.construire_pipeline(aides.spec_regression(), CT("regression.ridge"))
    for _, etape in pipeline.steps:
        with pytest.raises(NotFittedError):
            check_is_fitted(etape)


# ============================================================
# PREUVE DYNAMIQUE : TRANSFORMEUR ESPION
# ============================================================
class Espion(TransformerMixin, BaseEstimator):
    """Transformeur d'identité qui note les lignes (par étiquette d'index) qu'il ajuste et transforme."""

    ajustees: ClassVar[list[frozenset[int]]] = []
    transformees: ClassVar[list[frozenset[int]]] = []

    def fit(self, X, y=None):  # noqa: N803
        Espion.ajustees.append(frozenset(X.index))
        return self

    def transform(self, X):  # noqa: N803
        Espion.transformees.append(frozenset(X.index))
        return X.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


ENTREE_ESPION = EntreeCatalogue(
    id="transformation.espion",
    problemes=frozenset(Probleme),
    fabrique=lambda graine, **parametres: Espion(),
    grille_bornee=(),
    cout=Cout.LEGER,
    ref_livre="C04",
    texte_id="transformation.espion",
)


@pytest.fixture
def espion(monkeypatch: pytest.MonkeyPatch):
    """Espion enregistré en tête du prétraitement numérique (avant l'imputation, pour voir l'index)."""
    monkeypatch.setitem(
        preprocess.STADES, "transformation.espion", preprocess.Stade(5, "numerique", frozenset({"numerique"}))
    )
    Espion.ajustees, Espion.transformees = [], []
    return (*REGISTRE, ENTREE_ESPION)


def _projet():
    df = aides.jeu_regression()
    table = Table(df, "regression")
    pretraitement = (CT("transformation.espion"), *aides.PRETRAITEMENT_STANDARD)
    spec = aides.spec_regression(
        pretraitement=pretraitement, modeles=(CT("regression.ridge"), CT("regression.arbre"))
    )
    return table, spec, aides.coffre_par_pas(table, spec)


def test_l_espion_n_ajuste_que_sur_les_lignes_d_apprentissage_de_chaque_pli(espion) -> None:
    table, spec, coffre = _projet()
    res = train.entrainer(spec, table, coffre, registre=espion)
    n_modeles, n_plis = len(res.modeles), len(res.plis)
    assert len(Espion.ajustees) == n_modeles * n_plis, "un ajustement par modèle et par pli, aucun autre"
    apprentissage_des_plis = {frozenset(res.positions[a].tolist()) for a, _ in res.plis}
    assert len(apprentissage_des_plis) == n_plis
    for ajustees in Espion.ajustees:
        assert ajustees in apprentissage_des_plis, "ajustement sur autre chose que l'apprentissage d'un pli"
    assert all(Espion.ajustees.count(a) == n_modeles for a in apprentissage_des_plis)
    toute_la_base = frozenset(res.positions.tolist())
    assert toute_la_base not in Espion.ajustees, (
        "aucun ajustement sur toute la base pendant la validation croisée"
    )


def test_aucune_ligne_de_test_n_est_vue_ni_ajustee_ni_transformee(espion) -> None:
    table, spec, coffre = _projet()
    train.entrainer(spec, table, coffre, registre=espion)
    test = set(range(table.n_lignes)) - set(coffre.positions_apprentissage().tolist())
    assert test, "le jeu doit avoir des lignes de test"
    vues = set().union(*Espion.ajustees, *Espion.transformees)
    assert not vues & test


def test_chaque_pli_est_transforme_sur_sa_validation_apres_l_ajustement(espion) -> None:
    table, spec, coffre = _projet()
    res = train.entrainer(spec, table, coffre, registre=espion)
    validations = {frozenset(res.positions[v].tolist()) for _, v in res.plis}
    transformees = set(Espion.transformees)
    assert validations <= transformees
    # à l'ajustement, le pipeline transforme l'apprentissage du pli (sortie de l'étape), jamais plus
    apprentissages = {frozenset(res.positions[a].tolist()) for a, _ in res.plis}
    assert transformees <= validations | apprentissages


def test_l_ajustement_final_porte_sur_toute_la_base_d_apprentissage_et_rien_d_autre(espion) -> None:
    table, spec, coffre = _projet()
    res = train.entrainer(spec, table, coffre, registre=espion)
    avant = len(Espion.ajustees)
    train.ajuster_final(res.spec, CT("regression.ridge"), table, coffre, registre=espion)
    nouveaux = Espion.ajustees[avant:]
    assert nouveaux == [frozenset(coffre.positions_apprentissage().tolist())]


def test_la_graine_unique_donne_des_plis_identiques_donc_des_ajustements_identiques(espion) -> None:
    table, spec, coffre = _projet()
    train.entrainer(spec, table, coffre, registre=espion)
    premiers = list(Espion.ajustees)
    Espion.ajustees, Espion.transformees = [], []
    train.entrainer(spec, table, coffre, registre=espion)
    assert Espion.ajustees == premiers
