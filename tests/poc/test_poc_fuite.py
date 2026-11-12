"""Preuve de concept : détection bon marché des identifiants et des fuites par la cible.

Critère : moins de 2 s de détection par jeu, fuites connues détectées. Indicateurs,
calculés sur au plus 20 000 lignes tirées avec une graine fixe :

- identifiant : nom de type identifiant, ou colonne non réelle à valeurs toutes distinctes ;
- variable réelle et cible réelle : max(|Pearson|, |Spearman|) ;
- une variable catégorielle et une réelle : séparation, part des lignes dont la valeur
  réelle ne tombe dans l'intervalle que d'une seule modalité (1 = la modalité se déduit
  de la valeur, ou l'inverse : cible sœur) ;
- deux catégorielles : V de Cramér.

Les jeux sont lus dans le dossier désigné par la variable d'environnement MECAML_JEUX
(copies locales, jamais versionnées). Lancer : `python -m pytest -m lent -s tests/poc`.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.lent

SEUIL_FUITE = 0.99
BUDGET_S = 2.0
LIGNES_MAX = 20_000
MODALITES_MAX = 50
MOTS_ID = frozenset({"id", "identifiant", "numero", "num", "index", "ref"})
_RE_SANS_NOM = re.compile(r"^unnamed:\s*\d+$", re.I)
_RE_MOTS = re.compile(r"[_\s\-.]+|(?<=[a-z])(?=[A-Z])")


@dataclass(frozen=True)
class Indice:
    colonne: str
    nature: str  # « identifiant » ou nom de l'indicateur
    valeur: float


def _reelle(s: pd.Series) -> bool:
    """Réelle : nombres flottants, ou entiers à plus de MODALITES_MAX valeurs."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    return pd.api.types.is_float_dtype(s) or s.nunique() > MODALITES_MAX


def est_identifiant(nom: str, s: pd.Series) -> bool:
    """Nom de type identifiant, ou colonne entière ou textuelle à valeurs toutes distinctes."""
    if _RE_SANS_NOM.match(nom) or MOTS_ID & {m.lower() for m in _RE_MOTS.split(nom) if m}:
        return True
    if pd.api.types.is_float_dtype(s) or len(s) < 20:
        return False
    return bool(s.notna().all() and s.nunique() == len(s))


def _pearson_colonnes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Pearson| de chaque colonne de `x` (n, p) avec `y` (n,), sans valeur manquante."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    normes = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(xc.T @ yc) / normes
    return np.nan_to_num(r, nan=0.0)


def correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """max(|Pearson|, |Spearman|) de chaque colonne de `x` avec `y`, paires complètes."""
    from scipy.stats import rankdata

    ok_y = np.isfinite(y)
    x, y = x[ok_y], y[ok_y]
    resultat = np.zeros(x.shape[1])
    completes = np.isfinite(x).all(axis=0)
    if completes.any():
        xa = x[:, completes]
        spearman = _pearson_colonnes(rankdata(xa, axis=0), rankdata(y))
        resultat[completes] = np.maximum(_pearson_colonnes(xa, y), spearman)
    for j in np.flatnonzero(~completes):
        ok = np.isfinite(x[:, j])
        if ok.sum() >= 3:
            xj, yj = x[ok, j][:, None], y[ok]
            spearman = _pearson_colonnes(rankdata(xj, axis=0), rankdata(yj))[0]
            resultat[j] = max(_pearson_colonnes(xj, yj)[0], spearman)
    return resultat


def separation(valeurs: np.ndarray, modalites: pd.Series) -> float:
    """Part des lignes dont la valeur n'est couverte que par l'intervalle de sa modalité."""
    ok = np.isfinite(valeurs) & modalites.notna().to_numpy()
    v = valeurs[ok]
    g = modalites[ok]
    if v.size < 3:
        return 0.0
    bornes = pd.DataFrame({"v": v, "g": g.to_numpy()}).groupby("g", observed=True)["v"].agg(["min", "max"])
    if not 2 <= len(bornes) <= MODALITES_MAX:
        return 0.0
    couvertures = np.zeros(v.size, dtype=np.int32)
    for bas, haut in bornes.itertuples(index=False):
        couvertures += (v >= bas) & (v <= haut)
    return float(np.mean(couvertures == 1))


def cramer_v(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér, sans correction de biais."""
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    n = table.to_numpy().sum()
    attendu = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table.to_numpy() - attendu) ** 2 / attendu).sum())
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def detecter(df: pd.DataFrame, cible: str, graine: int = 0) -> list[Indice]:
    """Identifiants et indicateurs de fuite, triés par valeur décroissante."""
    indices = [Indice(c, "identifiant", 1.0) for c in df.columns if c != cible and est_identifiant(c, df[c])]
    exclues = {i.colonne for i in indices} | {cible}
    echantillon = df.sample(n=LIGNES_MAX, random_state=graine) if len(df) > LIGNES_MAX else df
    y = echantillon[cible]
    y_reelle = _reelle(y)
    y_num = y.to_numpy(dtype=float, na_value=np.nan) if y_reelle else None
    reelles = [c for c in echantillon.columns if c not in exclues and _reelle(echantillon[c])]
    if y_reelle and reelles:
        matrice = echantillon[reelles].to_numpy(dtype=float, na_value=np.nan)
        for c, valeur in zip(reelles, correlations(matrice, y_num), strict=True):
            indices.append(Indice(c, "corrélation", float(valeur)))
        exclues |= set(reelles)
    for c in echantillon.columns:
        if c in exclues:
            continue
        x = echantillon[c]
        x_reelle = c in reelles
        if x_reelle:
            nature, valeur = "séparation", separation(x.to_numpy(dtype=float, na_value=np.nan), y)
        elif y_reelle:
            nature, valeur = "séparation", separation(y_num, x)
        else:
            nature, valeur = "V de Cramér", cramer_v(x, y)
        indices.append(Indice(c, nature, valeur))
    return sorted(indices, key=lambda i: -i.valeur)


def _fuites(indices: list[Indice]) -> set[str]:
    return {i.colonne for i in indices if i.nature != "identifiant" and i.valeur >= SEUIL_FUITE}


def _rapport(nom: str, forme: tuple[int, int], lecture: float, duree: float, indices: list[Indice]) -> None:
    premiers = ", ".join(f"{i.colonne} ({i.nature} {i.valeur:.3f})" for i in indices[:5])
    ids = [i.colonne for i in indices if i.nature == "identifiant"]
    print(
        f"\n[POC fuite] {nom} {forme[0]} x {forme[1]} : lecture {lecture:.2f} s, "
        f"détection {duree:.3f} s ; identifiants {ids} ; "
        f"fuites >= {SEUIL_FUITE} {sorted(_fuites(indices))} ; premiers : {premiers}"
    )


# Jeux autorisés : (fichier, cible, identifiants attendus, fuites attendues)
JEUX = [
    ("MecaIA-Reduction_Rebuts.xlsx", "indice_performance_qualite", set(), {"taux_rebuts_pct"}),
    ("MecaIA-Qualite_soudure.xlsx", "qualite_soudure", {"id_soudure"}, {"controle_visuel_resultat"}),
    ("MecaIA-Suivi_Production_Machines.xlsx", "indice_sante_machine_0_100", set(), {"etat_sante_machine"}),
]


def _dossier_jeux() -> Path:
    dossier = os.environ.get("MECAML_JEUX")
    if not dossier or not Path(dossier).is_dir():
        pytest.skip("MECAML_JEUX ne désigne pas un dossier de jeux")
    return Path(dossier)


@pytest.mark.parametrize(("fichier", "cible", "ids_attendus", "fuites_attendues"), JEUX)
def test_poc_detection_sur_jeu_autorise(fichier, cible, ids_attendus, fuites_attendues) -> None:
    chemin = _dossier_jeux() / fichier
    if not chemin.is_file():
        pytest.skip(f"{fichier} absent de MECAML_JEUX")
    debut = time.perf_counter()
    df = pd.read_excel(chemin, sheet_name=0)
    lecture = time.perf_counter() - debut
    debut = time.perf_counter()
    indices = detecter(df, cible)
    duree = time.perf_counter() - debut
    _rapport(fichier, df.shape, lecture, duree, indices)
    assert ids_attendus <= {i.colonne for i in indices if i.nature == "identifiant"}
    assert fuites_attendues <= _fuites(indices)
    assert duree < BUDGET_S


def test_poc_detection_a_la_borne_200000x500() -> None:
    """Table synthétique aux bornes, une fuite et un identifiant plantés."""
    rng = np.random.default_rng(0)
    n, p = 200_000, 497
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i:03d}" for i in range(p)])
    df["cible"] = df["x000"] * 2.0 + rng.normal(scale=0.5, size=n)
    df["cible_x100"] = df["cible"] * 100.0  # cible sœur
    df["Piece_ID"] = np.arange(n)
    debut = time.perf_counter()
    indices = detecter(df, "cible")
    duree = time.perf_counter() - debut
    _rapport("synthétique", df.shape, 0.0, duree, indices)
    assert "Piece_ID" in {i.colonne for i in indices if i.nature == "identifiant"}
    assert _fuites(indices) == {"cible_x100"}
    assert duree < BUDGET_S
