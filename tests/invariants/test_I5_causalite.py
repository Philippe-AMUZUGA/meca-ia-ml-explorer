"""I5 Causalité : modifier les valeurs postérieures à t ne change aucune variable construite à t, par unité.

Perturbation sur le suivi de 100 machines (TimeSeries_RUL) et sur la maintenance
horodatée (AI4I), lus dans MECAML_JEUX s'il est défini ; répliques synthétiques de
même forme toujours jouées. Toutes les familles de variables sont exercées.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecaml.core import features_ts
from mecaml.core.features_ts import ConfigTemporelle, construire_variables
from mecaml.core.split import sceller
from mecaml.core.types import Probleme, Role, SpecProjet, Structure, Table

CAPTEURS = ("Temp_Carter_K", "Vib_RMS_AxeZ_mm_s", "Courant_Moteur_Amp")
MESURES = ("T_palier_C", "Couple_Nm", "Vibration_mm_s")
FAMILLES = {
    "retards": (1, 3),
    "differences": (1, 2),
    "fenetres": (4, 10),
    "demi_vies": (2.0,),
    "fenetres_frequence": (8,),
}


def _lire(fichier: str) -> pd.DataFrame:
    dossier = os.environ.get("MECAML_JEUX")
    if not dossier or not (Path(dossier) / fichier).is_file():
        pytest.skip(f"{fichier} absent de MECAML_JEUX")
    return pd.read_excel(Path(dossier) / fichier)


def _replique_rul() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    durees = rng.integers(25, 70, 100)
    cycle = np.concatenate([np.arange(1, d + 1) for d in durees])
    df = pd.DataFrame({"ID_Machine": np.repeat(np.arange(1, 101), durees), "Cycle_Operation_Heures": cycle})
    for c in CAPTEURS:
        df[c] = rng.normal(size=len(df)).cumsum()
    df["Duree_Vie_Restante_Cycles"] = np.repeat(durees, durees) - cycle
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # lignes mêlées


def _replique_ai4i() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 2500
    df = pd.DataFrame({"time": np.round(np.arange(n) / 6, 4)})
    for c in MESURES:
        df[c] = rng.normal(size=n)
    df.loc[rng.choice(n, 60, replace=False), "T_palier_C"] = np.nan  # manques, comme le jeu réel
    df["Panne"] = (rng.random(n) < 0.034).astype(int)
    return df


@functools.cache
def _jeu(nom: str, source: str) -> pd.DataFrame:
    if source == "replique":
        return _replique_rul() if nom == "rul" else _replique_ai4i()
    return _lire("MecaIA-TimeSeries_RUL.xlsx" if nom == "rul" else "MecaIA-AI4I_maintenance.xlsx")


@pytest.fixture(params=["reel", "replique"])
def source(request: pytest.FixtureRequest) -> str:
    return request.param


def _config_rul(**familles: object) -> ConfigTemporelle:
    return ConfigTemporelle("Cycle_Operation_Heures", CAPTEURS, "ID_Machine", **(familles or FAMILLES))


def _config_ai4i() -> ConfigTemporelle:
    return ConfigTemporelle("time", MESURES, **FAMILLES)


def _perturber(df: pd.DataFrame, masque: np.ndarray, colonnes: tuple[str, ...], graine: int) -> pd.DataFrame:
    """Remplace les valeurs des lignes du masque par du bruit, des manques et des valeurs extrêmes."""
    rng = np.random.default_rng(graine)
    autre = df.copy()
    for c in colonnes:
        bruit = rng.normal(1e3, 1e3, int(masque.sum()))
        bruit[rng.random(bruit.size) < 0.1] = np.nan
        autre.loc[masque, c] = bruit
    return autre


def _egales(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


# ============================================================
# PERTURBATION DU FUTUR
# ============================================================
@pytest.mark.parametrize("t", [5, 20, 40, 60])
def test_I5_cent_machines_futur_perturbe(source, t) -> None:
    df = _jeu("rul", source)
    config = _config_rul()
    reference = construire_variables(df, config)
    futur = (df["Cycle_Operation_Heures"] > t).to_numpy()
    perturbe = construire_variables(_perturber(df, futur, CAPTEURS, t), config)
    assert _egales(reference[~futur], perturbe[~futur])
    assert not _egales(reference[futur], perturbe[futur])  # le futur, lui, a bien changé


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.9])
def test_I5_maintenance_horodatee_futur_perturbe(source, fraction) -> None:
    df = _jeu("ai4i", source)
    config = _config_ai4i()
    reference = construire_variables(df, config)
    t = df["time"].quantile(fraction)
    futur = (df["time"] > t).to_numpy()
    perturbe = construire_variables(_perturber(df, futur, MESURES, int(100 * fraction)), config)
    assert _egales(reference[~futur], perturbe[~futur])


def test_I5_une_machine_ne_voit_pas_les_autres(source) -> None:
    df = _jeu("rul", source)
    config = _config_rul()
    reference = construire_variables(df, config)
    autres = (df["ID_Machine"] != 7).to_numpy()
    perturbe = construire_variables(_perturber(df, autres, CAPTEURS, 7), config)
    assert _egales(reference[~autres], perturbe[~autres])


@pytest.mark.parametrize("famille", sorted(FAMILLES))
def test_I5_chaque_famille_est_causale(famille) -> None:
    df = _jeu("rul", "replique")
    config = _config_rul(**{famille: FAMILLES[famille], **({"retards": ()} if famille != "retards" else {})})
    reference = construire_variables(df, config)
    futur = (df["Cycle_Operation_Heures"] > 30).to_numpy()
    perturbe = construire_variables(_perturber(df, futur, CAPTEURS, 1), config)
    assert _egales(reference[~futur], perturbe[~futur])


def test_I5_valeur_presente_seulement_pour_t() -> None:
    """Changer la valeur à t change les variables à t et après, jamais avant."""
    df = _jeu("rul", "replique")
    config = _config_rul()
    reference = construire_variables(df, config)
    a_t = ((df["ID_Machine"] == 3) & (df["Cycle_Operation_Heures"] == 12)).to_numpy()
    perturbe = construire_variables(_perturber(df, a_t, CAPTEURS, 2), config)
    avant = ((df["ID_Machine"] == 3) & (df["Cycle_Operation_Heures"] < 12)).to_numpy()
    assert _egales(reference[avant], perturbe[avant])
    assert not _egales(reference[a_t], perturbe[a_t])


def test_I5_ordre_des_lignes_sans_effet() -> None:
    df = _jeu("rul", "replique")
    config = _config_rul()
    reference = construire_variables(df, config)
    melange = df.sample(frac=1.0, random_state=11)
    assert _egales(construire_variables(melange, config).loc[df.index], reference)


# ============================================================
# TRANSFORMEUR AJUSTÉ SUR L'APPRENTISSAGE, APPLIQUÉ AU TEST
# ============================================================
def _spec_ai4i() -> SpecProjet:
    roles = (("time", Role.DATE), ("Panne", Role.CIBLE), *((c, Role.NUMERIQUE) for c in MESURES))
    return SpecProjet(Probleme.CLASSIFICATION, "Panne", roles, "f1_macro", Structure.CHRONOLOGIQUE, graine=42)


def test_I5_memoire_d_apprentissage_et_futur_du_test(source) -> None:
    """Le test hérite des dernières lignes de l'apprentissage ; son propre futur ne remonte pas."""
    df = _jeu("ai4i", source)
    table = Table(df, nom="ai4i")
    _, coffre = sceller(table, _spec_ai4i())
    app = coffre.positions_apprentissage()
    test = np.setdiff1d(np.arange(len(df)), app)
    retards = {
        k: v for k, v in FAMILLES.items() if k != "demi_vies"
    }  # mémoire exacte hors moyenne exponentielle
    tr = features_ts.VariablesTemporelles(colonne_date="time", colonnes=MESURES, **retards).fit(df.iloc[app])
    sortie = tr.transform(df.iloc[test])
    complet = construire_variables(df, ConfigTemporelle("time", MESURES, **retards)).iloc[test]
    np.testing.assert_allclose(sortie, complet.to_numpy(), rtol=1e-9, atol=1e-9)  # sommes glissantes
    t = df["time"].iloc[test].median()
    futur = (df["time"].iloc[test] > t).to_numpy()
    lignes_test = df.iloc[test]
    perturbe = tr.transform(_perturber(lignes_test, futur, MESURES, 5))
    assert np.array_equal(sortie[~futur], perturbe[~futur], equal_nan=True)


# ============================================================
# LIGNES SANS CIBLE : CONTEXTE CAUSAL DES SÉRIES
# ============================================================
def _sans_cible(df: pd.DataFrame, cible: str, part: float = 0.05) -> pd.Series:
    """Masque de lignes dont la cible est rendue manquante (tirage fixe)."""
    return pd.Series(np.random.default_rng(9).random(len(df)) < part, index=df.index)


def test_I5_contexte_sans_cible_complete_la_serie_sans_fuite(source) -> None:
    """Lignes sans cible en contexte : variables égales au calcul sur la série complète ;
    perturber le contexte postérieur à t ne change rien à t."""
    df = _jeu("rul", source)
    manque = _sans_cible(df, "Duree_Vie_Restante_Cycles")
    colonnes = ["ID_Machine", "Cycle_Operation_Heures", *CAPTEURS]
    avec, contexte = df.loc[~manque, colonnes], df.loc[manque, colonnes]
    parametres = {
        "colonne_date": "Cycle_Operation_Heures",
        "colonne_groupe": "ID_Machine",
        "colonnes": CAPTEURS,
    }
    tr = features_ts.VariablesTemporelles(contexte=contexte, **parametres, **FAMILLES)
    sortie = tr.fit_transform(avec)
    complet = construire_variables(df, _config_rul()).loc[avec.index].to_numpy()
    assert np.array_equal(sortie, complet, equal_nan=True)
    t = 30
    futur = (contexte["Cycle_Operation_Heures"] > t).to_numpy()
    autre = features_ts.VariablesTemporelles(
        contexte=_perturber(contexte, futur, CAPTEURS, 3), **parametres, **FAMILLES
    ).fit_transform(avec)
    avant = (avec["Cycle_Operation_Heures"] <= t).to_numpy()
    assert np.array_equal(sortie[avant], autre[avant], equal_nan=True)
