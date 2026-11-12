"""
MécaML : lecture des modèles sans cible (S5 à S7) : réduction de dimension, regroupement, anomalies.

Un projet sans cible ajuste chaque modèle une fois, sur toutes les lignes (`train.entrainer`) ; il n'y a ni
jeu de test, ni validation croisée, et le modèle ne se juge que par ce qu'il montre. Ce module lit le pipeline
ajusté, les étiquettes et les détails que `train` a gardés dans chaque `ResultatModele` :

- analyse en composantes principales : variance de chaque composante, poids des variables, coordonnées ; -
  regroupement : effectif de chaque groupe, valeur moyenne de chaque variable par groupe, silhouette par
  groupe ; - anomalies : score de chaque ligne, lignes les plus anormales, part déclarée anormale.

Aucune de ces lectures n'est une preuve : un regroupement existe toujours, et la silhouette dit seulement si
les groupes sont bien séparés dans les variables choisies. Les figures sont des `matplotlib.figure.Figure`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mecaml.core import libelles, train
from mecaml.core.preprocess import colonnes_d_entree
from mecaml.core.types import Probleme, SpecProjet, Table

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "AnalyseAnomalies",
    "AnalyseComposantes",
    "AnalyseRegroupement",
    "ErreurAnalyse",
    "analyser",
    "analyser_anomalies",
    "analyser_composantes",
    "analyser_regroupement",
    "figure_anomalies",
    "figure_composantes",
    "figure_plan",
    "figure_profils",
]

ECHANTILLON_SILHOUETTE = 5000
REGROUPEMENTS = (
    "sans_cible.kmeans",
    "sans_cible.melange_gaussien",
    "sans_cible.dbscan",
    "sans_cible.classification_hierarchique",
)
COMPOSANTES = "sans_cible.acp"
COULEURS = ("#1f4e79", "#c55a11", "#548235", "#7030a0", "#bf9000", "#2e75b6", "#833c0b", "#7f7f7f")


class ErreurAnalyse(ValueError):
    """Analyse impossible pour ce modèle ou ces données ; le message dit quoi faire."""


@dataclass(frozen=True, eq=False)
class AnalyseComposantes:
    """Composantes principales : variance expliquée, poids des variables, coordonnées des lignes."""

    modele_id: str
    variance: np.ndarray  # part de variance de chaque composante
    poids: pd.DataFrame  # une ligne par variable produite, une colonne par composante
    coordonnees: np.ndarray  # (lignes, composantes)

    @property
    def cumulee(self) -> np.ndarray:
        """Variance cumulée."""
        return np.cumsum(self.variance)


@dataclass(frozen=True, eq=False)
class AnalyseRegroupement:
    """Groupes trouvés : effectifs, profil moyen des variables d'origine, silhouette de chaque groupe."""

    modele_id: str
    etiquettes: np.ndarray
    effectifs: pd.Series  # étiquette de groupe vers nombre de lignes (-1 : bruit)
    profils: pd.DataFrame  # moyenne de chaque variable numérique par groupe
    silhouette_par_groupe: pd.Series  # vide si moins de deux groupes
    plan: np.ndarray  # (lignes, 2) : coordonnées de lecture, première et deuxième composante

    @property
    def n_groupes(self) -> int:
        """Nombre de groupes, bruit exclu."""
        return int((self.effectifs.index != -1).sum())


@dataclass(frozen=True, eq=False)
class AnalyseAnomalies:
    """Anomalies : score de chaque ligne (plus grand : plus anormal), lignes déclarées anormales."""

    modele_id: str
    scores: np.ndarray
    etiquettes: np.ndarray  # -1 : anormale, 1 : normale
    plan: np.ndarray  # (lignes, 2)

    @property
    def taux(self) -> float:
        """Part des lignes déclarées anormales."""
        return float(np.mean(self.etiquettes == -1))

    def plus_anormales(self, donnees: pd.DataFrame, nombre: int = 10) -> pd.DataFrame:
        """Les `nombre` lignes les plus anormales, avec leur score (position = ligne de la table)."""
        ordre = np.argsort(-self.scores, kind="stable")[:nombre]
        lignes = donnees.iloc[ordre].copy()
        lignes.insert(0, "score_anomalie", self.scores[ordre])
        lignes.insert(0, "position", ordre)
        return lignes.reset_index(drop=True)


# ============================================================
# OUTILS
# ============================================================
def _modele(resultat: train.Resultat, id_modele: str) -> train.ResultatModele:
    if resultat.spec.probleme is not Probleme.SANS_CIBLE:
        raise ErreurAnalyse("Cette lecture est celle d'un projet sans cible.")
    try:
        modele = resultat.modele(id_modele)
    except KeyError as erreur:
        raise ErreurAnalyse(str(erreur.args[0])) from None
    if not modele.ok or modele.pipeline is None:
        raise ErreurAnalyse(
            f"Le modèle {id_modele} n'a pas été ajusté : {modele.erreur or 'aucun pipeline'}."
        )
    return modele


def _entrees(spec: SpecProjet, table: Table) -> pd.DataFrame:
    return table.donnees[list(colonnes_d_entree(spec))]


def _transformees(modele: train.ResultatModele, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(modele.pipeline[:-1].transform(x), dtype=float)


def _plan(transformees: np.ndarray, graine: int) -> np.ndarray:
    """Deux premières composantes principales des variables produites, pour tracer les lignes dans un plan."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    k = min(2, transformees.shape[1], transformees.shape[0])
    pipeline = Pipeline([("plan", PCA(n_components=k, svd_solver="full", random_state=graine))])
    coordonnees = pipeline.fit(transformees).transform(transformees)
    if k < 2:
        coordonnees = np.column_stack([coordonnees, np.zeros(len(coordonnees))])
    return np.asarray(coordonnees)


def _noms_produits(modele: train.ResultatModele, n: int) -> list[str]:
    try:
        return [str(nom) for nom in modele.pipeline[:-1].get_feature_names_out()]
    except (AttributeError, ValueError, TypeError):
        return [f"x{k}" for k in range(n)]


# ============================================================
# ANALYSE EN COMPOSANTES PRINCIPALES
# ============================================================
def analyser_composantes(resultat: train.Resultat, id_modele: str, table: Table) -> AnalyseComposantes:
    """Variance, poids des variables et coordonnées de l'analyse en composantes principales ajustée."""
    modele = _modele(resultat, id_modele)
    final = modele.pipeline[-1]
    if not hasattr(final, "explained_variance_ratio_"):
        raise ErreurAnalyse(f"{id_modele} n'est pas une analyse en composantes principales.")
    x = _transformees(modele, _entrees(resultat.spec, table))
    composantes = np.asarray(final.components_, dtype=float)
    noms = _noms_produits(modele, composantes.shape[1])
    colonnes = [f"CP{k + 1}" for k in range(composantes.shape[0])]
    poids = pd.DataFrame(composantes.T, index=noms, columns=colonnes)
    return AnalyseComposantes(
        id_modele,
        np.asarray(final.explained_variance_ratio_, dtype=float),
        poids,
        np.asarray(final.transform(x)),
    )


# ============================================================
# REGROUPEMENT
# ============================================================
def _silhouette_par_groupe(x: np.ndarray, etiquettes: np.ndarray, graine: int) -> pd.Series:
    from sklearn.metrics import silhouette_samples

    garde = etiquettes != -1
    n_groupes = len(np.unique(etiquettes[garde]))
    if n_groupes < 2 or garde.sum() <= n_groupes:
        return pd.Series(dtype=float)
    indices = np.flatnonzero(garde)
    if len(indices) > ECHANTILLON_SILHOUETTE:
        indices = np.sort(
            np.random.default_rng(graine).choice(indices, ECHANTILLON_SILHOUETTE, replace=False)
        )
    valeurs = silhouette_samples(x[indices], etiquettes[indices])
    par_groupe = pd.Series(valeurs).groupby(etiquettes[indices]).mean()
    par_groupe.name = "silhouette"
    return par_groupe


def analyser_regroupement(resultat: train.Resultat, id_modele: str, table: Table) -> AnalyseRegroupement:
    """Groupes d'un regroupement ajusté : effectifs, profil moyen par groupe, silhouette de chaque groupe."""
    modele = _modele(resultat, id_modele)
    if modele.etiquettes is None:
        raise ErreurAnalyse(f"{id_modele} ne donne pas d'étiquettes de groupe.")
    etiquettes = np.asarray(modele.etiquettes)
    x = _transformees(modele, _entrees(resultat.spec, table))
    effectifs = pd.Series(etiquettes).value_counts().sort_index()
    effectifs.name = "effectif"
    numeriques = [c for c in resultat.spec.variables() if pd.api.types.is_numeric_dtype(table.donnees[c])]
    profils = table.donnees[numeriques].groupby(etiquettes).mean()
    return AnalyseRegroupement(
        id_modele,
        etiquettes,
        effectifs,
        profils,
        _silhouette_par_groupe(x, etiquettes, resultat.spec.graine),
        _plan(x, resultat.spec.graine),
    )


# ============================================================
# ANOMALIES
# ============================================================
def _scores_d_anomalie(final: Any, x: np.ndarray) -> np.ndarray:
    """Score d'anomalie (plus grand : plus anormal) d'un détecteur ajusté."""
    if hasattr(final, "negative_outlier_factor_"):
        return -np.asarray(final.negative_outlier_factor_, dtype=float)
    if hasattr(final, "score_samples"):
        return -np.asarray(final.score_samples(x), dtype=float)
    raise ErreurAnalyse("Ce détecteur ne donne pas de score par ligne.")


def analyser_anomalies(resultat: train.Resultat, id_modele: str, table: Table) -> AnalyseAnomalies:
    """Score d'anomalie de chaque ligne d'un détecteur ajusté, et lignes qu'il déclare anormales."""
    modele = _modele(resultat, id_modele)
    if id_modele not in train.ANOMALIES or modele.etiquettes is None:
        raise ErreurAnalyse(f"{id_modele} n'est pas un détecteur d'anomalies.")
    x = _transformees(modele, _entrees(resultat.spec, table))
    scores = _scores_d_anomalie(modele.pipeline[-1], x)
    return AnalyseAnomalies(id_modele, scores, np.asarray(modele.etiquettes), _plan(x, resultat.spec.graine))


def analyser(resultat: train.Resultat, id_modele: str, table: Table) -> Any:
    """Lecture adaptée au modèle : composantes, regroupement ou anomalies."""
    _modele(resultat, id_modele)  # projet sans cible, modèle présent et ajusté
    if id_modele == COMPOSANTES:
        return analyser_composantes(resultat, id_modele, table)
    if id_modele in REGROUPEMENTS:
        return analyser_regroupement(resultat, id_modele, table)
    if id_modele in train.ANOMALIES:
        return analyser_anomalies(resultat, id_modele, table)
    raise ErreurAnalyse(f"{id_modele} n'a pas de lecture sans cible connue.")


# ============================================================
# FIGURES
# ============================================================
def _style(axe: Any) -> None:
    axe.spines[["top", "right"]].set_visible(False)
    axe.grid(True, color="#e5e5e5", linewidth=0.6)
    axe.set_axisbelow(True)


def _nouvelle(largeur: float, hauteur: float) -> Figure:
    from matplotlib.figure import Figure

    return Figure(figsize=(largeur, hauteur), layout="constrained", dpi=100)


def figure_composantes(analyse: AnalyseComposantes) -> Figure:
    """Part de variance de chaque composante et variance cumulée."""
    k = len(analyse.variance)
    fig = _nouvelle(5.6, 3.4)
    axe = fig.subplots()
    axe.bar(range(1, k + 1), 100 * analyse.variance, color="#1f4e79")
    axe.plot(range(1, k + 1), 100 * analyse.cumulee, color="#c55a11", marker="o", markersize=3)
    axe.set(xlabel="composante", ylabel="variance expliquée (%)", title="Variance par composante")
    axe.set_xticks(range(1, k + 1))
    _style(axe)
    return fig


def figure_plan(plan: np.ndarray, etiquettes: np.ndarray, titre: str, anomalies: bool = False) -> Figure:
    """Lignes dans le plan des deux premières composantes, colorées par groupe ou par normalité."""
    fig = _nouvelle(5.6, 4.0)
    axe = fig.subplots()
    k = np.sort(np.random.default_rng(0).choice(len(plan), min(len(plan), 3000), replace=False))
    if anomalies:
        normal = etiquettes[k] != -1
        axe.scatter(plan[k][normal, 0], plan[k][normal, 1], s=9, alpha=0.5, color="#1f4e79", label="normales")
        axe.scatter(
            plan[k][~normal, 0], plan[k][~normal, 1], s=22, marker="x", color="#c00000", label="anormales"
        )
    else:
        for rang, groupe in enumerate(np.unique(etiquettes)):
            masque = etiquettes[k] == groupe
            nom = "bruit" if groupe == -1 else f"groupe {groupe}"
            couleur = "#7f7f7f" if groupe == -1 else COULEURS[rang % len(COULEURS)]
            axe.scatter(plan[k][masque, 0], plan[k][masque, 1], s=10, alpha=0.6, color=couleur, label=nom)
    axe.set(xlabel="première composante", ylabel="deuxième composante", title=titre)
    axe.legend(frameon=False, fontsize=8)
    _style(axe)
    return fig


def figure_profils(analyse: AnalyseRegroupement) -> Figure:
    """Valeur moyenne de chaque variable par groupe, écart à la moyenne générale en écarts types."""
    profils = analyse.profils
    ecart = profils.sub(profils.mean()).div(profils.std(ddof=0).replace(0.0, math.nan)).fillna(0.0)
    fig = _nouvelle(max(4.8, 1.0 * len(profils.columns) + 2.0), max(2.8, 0.6 * len(profils) + 1.6))
    axe = fig.subplots()
    image = axe.imshow(ecart.to_numpy(), cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    axe.set_xticks(range(len(profils.columns)), [str(c) for c in profils.columns], rotation=30, ha="right")
    axe.set_yticks(range(len(profils)), ["bruit" if g == -1 else f"groupe {g}" for g in profils.index])
    for i in range(len(profils)):
        for j in range(len(profils.columns)):
            axe.text(j, i, f"{profils.iloc[i, j]:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axe, label="écart à la moyenne (écarts types)")
    axe.set_title("Profil des groupes")
    return fig


def figure_anomalies(analyse: AnalyseAnomalies) -> Figure:
    """Répartition des scores d'anomalie, lignes déclarées anormales en rouge."""
    fig = _nouvelle(5.6, 3.4)
    axe = fig.subplots()
    normales = analyse.scores[analyse.etiquettes != -1]
    anormales = analyse.scores[analyse.etiquettes == -1]
    bornes = np.histogram_bin_edges(analyse.scores, bins=40)
    axe.hist(normales, bins=bornes, color="#1f4e79", label="normales")
    axe.hist(anormales, bins=bornes, color="#c00000", label="anormales")
    axe.set(
        xlabel="score d'anomalie (plus grand : plus anormal)",
        ylabel="lignes",
        title=libelles.technique(analyse.modele_id),
    )
    axe.legend(frameon=False, fontsize=8)
    _style(axe)
    return fig
