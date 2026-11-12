"""Variables temporelles : valeurs exactes, garde-fous, transformeur compatible scikit-learn."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from mecaml.core import features_ts
from mecaml.core.features_ts import (
    PROFONDEUR_MAX,
    VARIABLES_MAX,
    ConfigTemporelle,
    construire_variables,
    fabrique_variables_temporelles,
)
from mecaml.core.split import separer_sans_cible

SRC = Path(__file__).resolve().parents[2] / "src"
VariablesTemporelles = features_ts.VariablesTemporelles


def _serie() -> pd.DataFrame:
    """Deux unités aux lignes mêlées : A (1, 2, 4, 7, 11) et B (10, 20, 30)."""
    return pd.DataFrame(
        {
            "u": ["B", "A", "A", "B", "A", "A", "B", "A"],
            "t": [3, 2, 1, 1, 3, 5, 2, 4],
            "x": [30.0, 2.0, 1.0, 10.0, 4.0, 11.0, 20.0, 7.0],
        },
        index=[f"r{i}" for i in range(8)],
    )


def _config(**champs: object) -> ConfigTemporelle:
    return ConfigTemporelle(**{"colonne_date": "t", "colonnes": ("x",), "colonne_groupe": "u", **champs})


def _par_unite(resultat: pd.DataFrame, df: pd.DataFrame, colonne: str) -> dict[str, list]:
    rangee = resultat.assign(u=df["u"], t=df["t"]).sort_values(["u", "t"])
    return {u: rangee.loc[rangee["u"] == u, colonne].round(10).tolist() for u in ("A", "B")}


# ============================================================
# CONFIGURATION
# ============================================================
def test_config_normalisee_et_noms() -> None:
    c = _config(retards=(3, 1, 3), fenetres=(4,), differences=(1,), demi_vies=(2,), fenetres_frequence=(8,))
    assert c.retards == (1, 3) and c.demi_vies == (2.0,)
    assert c.noms() == (
        "x_retard_1",
        "x_retard_3",
        "x_difference_1",
        "x_moyenne_4",
        "x_ecart_type_4",
        "x_moyenne_exp_2",
        "x_frequence_dominante_8",
        "x_centroide_spectral_8",
    )
    assert c.profondeur == 20  # dix demi-vies de 2 lignes


@pytest.mark.parametrize(
    ("champs", "erreur", "motif"),
    [
        ({"retards": (0,)}, ValueError, "retards"),
        ({"retards": (PROFONDEUR_MAX + 1,)}, ValueError, "retards"),
        ({"retards": (1.5,)}, TypeError, "entiers"),
        ({"retards": (True,)}, TypeError, "entiers"),
        ({"retards": 2}, TypeError, "liste"),
        ({"fenetres": (1,)}, ValueError, "fenetres"),
        ({"fenetres_frequence": (3,)}, ValueError, "fenetres_frequence"),
        ({"demi_vies": (0,)}, ValueError, "demi-vie"),
        ({"demi_vies": ("2",)}, TypeError, "non numérique"),
        ({"demi_vies": "2"}, TypeError, "nombres"),
        ({"retards": ()}, ValueError, "aucune variable"),
        ({"colonne_date": ""}, ValueError, "colonne_date"),
        ({"colonne_groupe": ""}, ValueError, "colonne_groupe"),
        ({"colonnes": "x"}, TypeError, "noms de colonnes"),
        ({"colonnes": ()}, ValueError, "vide"),
        ({"colonnes": ("x", "x")}, ValueError, "doublons"),
        ({"colonnes": ("x", "t")}, ValueError, "ne se transforment pas"),
    ],
)
def test_config_refus(champs: dict, erreur: type, motif: str) -> None:
    with pytest.raises(erreur, match=motif):
        _config(**champs)


def test_config_borne_du_nombre_de_variables() -> None:
    colonnes = tuple(f"c{i}" for i in range(VARIABLES_MAX // 2 + 1))
    with pytest.raises(ValueError, match="au plus 500"):
        ConfigTemporelle("t", colonnes, retards=(1, 2))


# ============================================================
# VALEURS
# ============================================================
def test_retards_et_differences_par_unite() -> None:
    df = _serie()
    r = construire_variables(df, _config(retards=(1, 2), differences=(1,)))
    assert list(r.index) == list(df.index)
    a = _par_unite(r, df, "x_retard_1")
    assert a["B"][1:] == [10.0, 20.0] and np.isnan(a["B"][0])
    assert _par_unite(r, df, "x_retard_2")["A"][2:] == [1.0, 2.0, 4.0]
    d = _par_unite(r, df, "x_difference_1")
    assert d["A"][1:] == [1.0, 2.0, 3.0, 4.0] and d["B"][1:] == [10.0, 10.0]


def test_fenetres_glissantes_et_moyenne_exponentielle() -> None:
    df = _serie()
    r = construire_variables(df, _config(retards=(), fenetres=(3,), demi_vies=(1.0,)))
    moy = _par_unite(r, df, "x_moyenne_3")
    assert moy["A"][2:] == [round(7 / 3, 10), round(13 / 3, 10), round(22 / 3, 10)]
    assert all(np.isnan(v) for v in moy["A"][:2])
    assert _par_unite(r, df, "x_ecart_type_3")["B"][2] == round(10.0, 10)
    attendu = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0]).ewm(halflife=1.0).mean().round(10).tolist()
    assert _par_unite(r, df, "x_moyenne_exp_1")["A"] == attendu


def test_ecart_type_nul_sur_une_fenetre_constante() -> None:
    df = pd.DataFrame({"t": np.arange(12), "x": [87.901235] * 6 + [88.518519] * 6})
    r = construire_variables(df, ConfigTemporelle("t", ("x",), retards=(), fenetres=(4,)))
    assert (r["x_ecart_type_4"].iloc[[3, 4, 5, 9, 10, 11]] == 0.0).all()
    assert (r["x_ecart_type_4"].iloc[6:9] > 0).all()


def test_descripteurs_de_frequence() -> None:
    n, w = 64, 16
    t = np.arange(n)
    df = pd.DataFrame({"t": t, "x": np.sin(2 * np.pi * t / 8) + 3.0, "c": np.full(n, 2.0)})
    config = ConfigTemporelle("t", ("x", "c"), retards=(), fenetres_frequence=(w,))
    r = construire_variables(df, config)
    assert r["x_frequence_dominante_16"].iloc[: w - 1].isna().all()
    assert np.allclose(r["x_frequence_dominante_16"].iloc[w - 1 :], 1 / 8)
    assert np.allclose(r["x_centroide_spectral_16"].iloc[w - 1 :], 1 / 8, atol=1e-9)
    assert r["c_frequence_dominante_16"].isna().all()  # fenêtre constante : sans fréquence


def test_valeurs_manquantes_propagees_sans_erreur() -> None:
    df = pd.DataFrame({"t": np.arange(30), "x": np.arange(30.0)})
    df.loc[10, "x"] = np.nan
    config = ConfigTemporelle("t", ("x",), retards=(1,), fenetres=(3,), fenetres_frequence=(4,))
    r = construire_variables(df, config)
    assert np.isnan(r.loc[11, "x_retard_1"]) and r.loc[12, "x_retard_1"] == 11.0
    assert r.loc[10:12, "x_moyenne_3"].isna().all() and r.loc[13, "x_moyenne_3"] == 12.0
    assert r.loc[10:13, "x_frequence_dominante_4"].isna().all()


def test_sans_unite_une_seule_serie() -> None:
    df = _serie().drop(columns="u").assign(t=np.arange(8)[::-1])
    r = construire_variables(df, ConfigTemporelle("t", ("x",), retards=(1,)))
    assert r.loc["r0", "x_retard_1"] == 2.0  # t=7 précédé de r1 (t=6)
    assert np.isnan(r.loc["r7", "x_retard_1"])


def test_booleens_acceptes() -> None:
    df = pd.DataFrame({"t": [1, 2, 3], "x": [True, False, True]})
    assert construire_variables(df, ConfigTemporelle("t", ("x",)))["x_retard_1"].tolist()[1:] == [1.0, 0.0]


def test_dates_horodatees() -> None:
    df = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=5, freq="4h")[::-1], "x": np.arange(5.0)})
    r = construire_variables(df, ConfigTemporelle("t", ("x",)))
    assert r["x_retard_1"].tolist()[:4] == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("modifier", "erreur", "motif"),
    [
        (lambda df: df.assign(t=[3, 2, 1, 1, 3, 5, 3, 4]), ValueError, "en double"),
        (lambda df: df.assign(t=[3, 2, None, 1, 3, 5, 2, 4]), ValueError, "la date"),
        (lambda df: df.assign(u=["B", None, "A", "B", "A", "A", "B", "A"]), ValueError, "l'unité"),
        (lambda df: df.assign(x=list("abcdefgh")), ValueError, "n'est pas numérique"),
        (lambda df: df.drop(columns="x"), ValueError, "absentes"),
        (lambda df: df.to_numpy(), TypeError, "DataFrame"),
    ],
)
def test_refus(modifier, erreur: type, motif: str) -> None:
    with pytest.raises(erreur, match=motif):
        construire_variables(modifier(_serie()), _config())


# ============================================================
# TRANSFORMEUR
# ============================================================
def _transformeur(**parametres: object):
    return VariablesTemporelles(colonne_date="t", colonne_groupe="u", **parametres)


def test_transformeur_interface_scikit_learn() -> None:
    tr = _transformeur(retards=(1, 2), fenetres=(3,))
    copie = clone(tr)
    assert copie.get_params() == tr.get_params() and copie is not tr
    df = _serie()
    sortie = tr.fit_transform(df)
    assert sortie.shape == (8, 4)
    assert list(tr.get_feature_names_out()) == ["x_retard_1", "x_retard_2", "x_moyenne_3", "x_ecart_type_3"]
    assert tr.n_features_in_ == 3 and list(tr.feature_names_in_) == ["u", "t", "x"]
    assert (
        type(tr).__module__ == "mecaml.core.features_ts" and type(tr).__qualname__ == "VariablesTemporelles"
    )


def test_transformeur_non_ajuste_et_entree_invalide() -> None:
    tr = _transformeur()
    with pytest.raises(NotFittedError):
        tr.transform(_serie())
    with pytest.raises(NotFittedError):
        tr.get_feature_names_out()
    with pytest.raises(TypeError, match="DataFrame"):
        tr.fit(_serie().to_numpy())
    tr.fit(_serie())
    with pytest.raises(TypeError, match="DataFrame"):
        tr.transform(_serie().to_numpy())


def test_transformeur_dans_un_pipeline_et_sortie_pandas() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"u": np.repeat(["a", "b", "c"], 20), "t": np.tile(np.arange(20), 3), "x": rng.normal(size=60)}
    )
    df["z"] = rng.normal(size=60)
    y = df["x"].shift(-1).fillna(0.0)
    colonnes = ColumnTransformer(
        [("temps", _transformeur(colonnes=("x", "z"), retards=(1, 2)), ["u", "t", "x", "z"])]
    )
    pipeline = make_pipeline(colonnes, SimpleImputer(), Ridge()).fit(df, y)
    assert pipeline.predict(df).shape == (60,)
    colonnes.set_output(transform="pandas")
    sortie = colonnes.fit_transform(df)
    assert isinstance(sortie, pd.DataFrame) and "temps__z_retard_2" in sortie.columns


def test_transformeur_colonnes_par_defaut_et_memoire() -> None:
    df = _serie().assign(y=1.0)
    tr = _transformeur(retards=(1,)).fit(df)
    assert tr.config_.colonnes == ("x", "y")
    assert tr.historique_ is not None and len(tr.historique_) == 2  # une ligne par unité
    sans = _transformeur(memoire=False).fit(_serie())
    assert sans.historique_ is None


def test_transformeur_memoire_prolonge_les_unites() -> None:
    df = _serie()
    passe, futur = df[df["t"] <= 2], df[df["t"] > 2]
    avec = _transformeur(retards=(1,)).fit(passe).transform(futur)
    sans = _transformeur(retards=(1,), memoire=False).fit(passe).transform(futur)
    complet = construire_variables(df, _config()).loc[futur.index, "x_retard_1"].to_numpy()
    assert np.array_equal(avec[:, 0], complet)
    assert np.isnan(sans[:, 0]).sum() == 2  # première ligne future de chaque unité


def test_transformeur_pickle_meme_processus() -> None:
    tr = _transformeur(retards=(1,), fenetres=(2,)).fit(_serie())
    copie = pickle.loads(pickle.dumps(tr))
    assert type(copie) is type(tr)
    assert np.array_equal(copie.transform(_serie()), tr.transform(_serie()), equal_nan=True)


def test_transformeur_pickle_relu_dans_un_processus_neuf(tmp_path: Path) -> None:
    chemin = tmp_path / "transformeur.pkl"
    chemin.write_bytes(pickle.dumps(_transformeur(retards=(2,)).fit(_serie())))
    programme = (
        "import pickle, sys\n"
        "tr = pickle.loads(open(sys.argv[1], 'rb').read())\n"
        "print(type(tr).__name__, tr.retards)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}
    sortie = subprocess.run(
        [sys.executable, "-c", programme, str(chemin)], capture_output=True, text=True, env=env, timeout=120
    )
    assert sortie.returncode == 0, sortie.stderr
    assert sortie.stdout.strip() == "VariablesTemporelles (2,)"


def test_import_du_module_sans_scikit_learn() -> None:
    programme = (
        "import sys\nimport mecaml.core.features_ts as f\n"
        "avant = 'sklearn' in sys.modules\nf.VariablesTemporelles\n"
        "print(avant, 'sklearn' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1"}
    sortie = subprocess.run(
        [sys.executable, "-c", programme], capture_output=True, text=True, env=env, timeout=120
    )
    assert sortie.stdout.strip() == "False True", sortie.stderr


def test_attribut_inconnu_du_module() -> None:
    with pytest.raises(AttributeError, match="Inexistant"):
        _ = features_ts.Inexistant


def test_fabrique_du_registre() -> None:
    tr = fabrique_variables_temporelles(42, colonne_date="t", retards=(1, 3))
    assert isinstance(tr, VariablesTemporelles) and tr.retards == (1, 3)


# ============================================================
# LIGNES SANS CIBLE : CONTEXTE DES SÉRIES
# ============================================================
TOUTES = {
    "retards": (1, 3),
    "differences": (1,),
    "fenetres": (4,),
    "demi_vies": (2.0,),
    "fenetres_frequence": (8,),
}


def _serie_trouee() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Série t = 0..99, x = t ; cible manquante aux instants 50 à 52 et 80, 81."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"t": np.arange(100), "x": np.arange(100.0) + rng.normal(0, 0.1, 100)})
    df["y"] = df["x"] * 2
    df.loc[[50, 51, 52, 80, 81], "y"] = np.nan
    avec, sans = separer_sans_cible(df, "y")
    return df, avec[["t", "x"]], sans[["t", "x"]]


def _complet(df: pd.DataFrame, lignes: pd.Index, **familles: object) -> np.ndarray:
    return construire_variables(df, ConfigTemporelle("t", ("x",), **familles)).loc[lignes].to_numpy()


def test_retard_calcule_sur_la_serie_complete() -> None:
    """Le retard d'une ligne est la valeur de l'instant précédent, que cet instant ait une cible ou non."""
    df, avec, sans = _serie_trouee()
    tr = VariablesTemporelles(colonne_date="t", colonnes=("x",), contexte=sans, **TOUTES)
    sortie = pd.DataFrame(tr.fit_transform(avec), index=avec.index, columns=tr.get_feature_names_out())
    assert sortie.loc[53, "x_retard_1"] == df.loc[52, "x"]
    np.testing.assert_array_equal(sortie.to_numpy(), _complet(df, avec.index, **TOUTES))
    sans_contexte = VariablesTemporelles(colonne_date="t", colonnes=("x",)).fit_transform(avec)
    assert sans_contexte[avec.index.get_loc(53), 0] == df.loc[49, "x"]  # la ligne sans cible serait sautée


def test_contexte_dans_un_pli_chronologique() -> None:
    """Pli : apprentissage avant l'instant 70, validation après ; le contexte comble les deux côtés."""
    df, avec, sans = _serie_trouee()
    familles = {k: v for k, v in TOUTES.items() if k != "demi_vies"}  # mémoire exacte
    app, val = avec[avec["t"] < 70], avec[avec["t"] >= 70]
    tr = VariablesTemporelles(colonne_date="t", colonnes=("x",), contexte=sans, **familles).fit(app)
    np.testing.assert_allclose(tr.transform(val), _complet(df, val.index, **familles), rtol=1e-12, atol=1e-12)


def test_lignes_connues_posterieures_sans_effet() -> None:
    df, avec, sans = _serie_trouee()
    tr = VariablesTemporelles(colonne_date="t", colonnes=("x",), contexte=sans, retards=(1, 2)).fit(avec)
    debut = avec[avec["t"] < 40]
    np.testing.assert_array_equal(tr.transform(debut), _complet(df, debut.index, retards=(1, 2)))


def test_contexte_refus() -> None:
    _, avec, sans = _serie_trouee()
    with pytest.raises(TypeError, match="contexte"):
        VariablesTemporelles(colonne_date="t", contexte=sans.to_numpy()).fit(avec)
    with pytest.raises(ValueError, match="en double"):
        VariablesTemporelles(colonne_date="t", contexte=avec.head(3)).fit(avec)
    with pytest.raises(ValueError, match="absentes"):
        VariablesTemporelles(colonne_date="t", colonnes=("x",), contexte=sans[["t"]]).fit(avec)


def test_contexte_vide_ou_sans_memoire() -> None:
    _, avec, sans = _serie_trouee()
    vide = VariablesTemporelles(colonne_date="t", contexte=sans.iloc[:0]).fit(avec)
    assert np.array_equal(vide.transform(avec), _transformeur_seul(avec), equal_nan=True)
    sans_memoire = VariablesTemporelles(colonne_date="t", contexte=sans, memoire=False).fit(avec)
    assert len(sans_memoire.historique_) == len(sans) == 5


def _transformeur_seul(avec: pd.DataFrame) -> np.ndarray:
    return VariablesTemporelles(colonne_date="t").fit(avec).transform(avec)


def test_contexte_clone_et_pickle() -> None:
    _, avec, sans = _serie_trouee()
    tr = VariablesTemporelles(colonne_date="t", contexte=sans, fenetres=(3,))
    copie = clone(tr)
    assert copie.contexte is not sans and copie.contexte.equals(sans)
    tr.fit(avec)
    relu = pickle.loads(pickle.dumps(tr))
    assert np.array_equal(relu.transform(avec), tr.transform(avec), equal_nan=True)


def test_duree_sur_une_table_de_la_taille_du_suivi_de_machines() -> None:
    """100 unités, 18 855 lignes, 7 capteurs, 77 variables : moins de 2 s."""
    rng = np.random.default_rng(1)
    longueurs = rng.integers(130, 248, 100)
    longueurs[-1] += 18_855 - longueurs.sum()
    unites = np.repeat(np.arange(100), longueurs)
    cycles = np.concatenate([np.arange(1, k + 1) for k in longueurs])
    capteurs = {f"s{i}": rng.normal(size=unites.size) for i in range(7)}
    df = pd.DataFrame({"machine": unites, "cycle": cycles, **capteurs})
    config = ConfigTemporelle(
        "cycle",
        tuple(capteurs),
        "machine",
        retards=(1, 2, 5),
        fenetres=(5, 20),
        differences=(1,),
        demi_vies=(3.0,),
        fenetres_frequence=(16,),
    )
    debut = time.perf_counter()
    r = construire_variables(df, config)
    assert r.shape == (18_855, 77)
    assert time.perf_counter() - debut < 2.0
