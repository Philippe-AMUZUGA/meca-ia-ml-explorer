"""
MécaML : décision du modèle retenu et de son seuil, prise sur la validation croisée avant tout test.

La décision (`types.Decision`) est le jeton qui ouvre le coffre de test : elle nomme un modèle de la
comparaison, la métrique fixée en S3, son score de validation et, pour deux classes, un seuil sur le score.
Elle ne lit que les prédictions hors pli de `train.Resultat` : le test reste scellé.

Garde-fous :

- une référence naïve (la moyenne, la classe majoritaire) sert de base de lecture, elle ne se retient pas ;
  - un modèle sans score défini pour la métrique du projet ne se retient pas ; - un seuil se règle pour deux
  classes dont le modèle donne des probabilités (entre 0 et 1) ; le score de validation d'une décision à
  seuil est recalculé pli par pli au seuil retenu, jamais repris de la décision sans seuil ; -
  `avertissements` dit ce qu'un lecteur pressé manquerait : un modèle qui ne bat pas la référence, deux
  modèles que l'écart entre plis ne permet pas de départager, une métrique définie sur une partie des plis,
  une classe positive proposée mais non confirmée.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mecaml.core import evaluate, libelles, metrics, train
from mecaml.core.guards import decimale
from mecaml.core.types import REFERENCES_NAIVES, Decision, Probleme

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "ErreurDecision",
    "SeuilPropose",
    "avertissements",
    "choisir_le_seuil",
    "decider",
    "figure_comparaison",
    "meilleur_modele",
    "modeles_a_retenir",
    "proposer",
    "score_au_seuil",
    "seuil_minimisant_le_cout",
]


class ErreurDecision(ValueError):
    """Décision impossible ou incohérente ; le message dit quoi choisir."""


# ============================================================
# MODÈLES QUE L'ON PEUT RETENIR
# ============================================================
def modeles_a_retenir(resultat: train.Resultat) -> tuple[train.ResultatModele, ...]:
    """Modèles ajustés à métrique définie, du meilleur au moins bon, références naïves exclues."""
    return tuple(
        m
        for m in resultat.classement()
        if m.ok and not math.isnan(resultat.valeur(m)) and m.id not in REFERENCES_NAIVES.values()
    )


def meilleur_modele(resultat: train.Resultat) -> train.ResultatModele:
    """Modèle au meilleur score de validation (hors référence naïve) ; `ErreurDecision` sinon."""
    candidats = modeles_a_retenir(resultat)
    if not candidats:
        raise ErreurDecision(
            "Aucun modèle n'a de score défini pour la métrique du projet : vérifier les échecs du tableau "
            "de S6, ou choisir une métrique que les modèles savent calculer."
        )
    return candidats[0]


def _modele_retenable(resultat: train.Resultat, id_modele: str) -> train.ResultatModele:
    try:
        modele = resultat.modele(id_modele)
    except KeyError as erreur:
        raise ErreurDecision(str(erreur.args[0])) from None
    if id_modele in REFERENCES_NAIVES.values():
        raise ErreurDecision(
            f"{libelles.technique(id_modele)} est la référence naïve : elle sert à mesurer le gain des "
            "modèles, elle ne se retient pas. Choisir un des modèles comparés."
        )
    if not modele.ok:
        raise ErreurDecision(
            f"Le modèle {libelles.technique(id_modele)} a échoué à l'entraînement ({modele.erreur}) : "
            "en choisir un autre."
        )
    if math.isnan(resultat.valeur(modele)):
        raise ErreurDecision(
            f"Le modèle {libelles.technique(id_modele)} n'a aucun score défini pour "
            f"« {libelles.metrique_courte(resultat.metrique)} » : choisir un autre "
            "modèle, ou une métrique qu'il sait calculer."
        )
    return modele


# ============================================================
# SEUIL
# ============================================================
@dataclass(frozen=True, eq=False)
class SeuilPropose:
    """Seuil qui minimise le coût moyen sur les prédictions hors pli, et la courbe qui le justifie."""

    seuil: float
    cout: float
    cout_fp: float
    cout_fn: float
    courbe: pd.DataFrame


def _diagnostic_au_seuil(
    resultat: train.Resultat, id_modele: str, seuil: float | None
) -> evaluate.Diagnostic:
    if (
        resultat.spec.probleme is not Probleme.CLASSIFICATION
        or resultat.classes is None
        or len(resultat.classes) != 2
    ):
        raise ErreurDecision("Un seuil se règle pour une classification à deux classes.")
    diag = evaluate.diagnostiquer_validation(resultat, id_modele)
    scores = diag.scores
    if scores is None or np.asarray(scores).ndim != 1:
        raise ErreurDecision(
            f"Le modèle {libelles.technique(id_modele)} ne donne pas de score par ligne : "
            "aucun seuil ne se règle."
        )
    if np.min(scores) < 0.0 or np.max(scores) > 1.0:
        raise ErreurDecision(
            f"Le modèle {libelles.technique(id_modele)} ne donne pas de probabilités (scores hors de 0 à "
            "1) : un seuil entre 0 "
            "et 1 n'aurait pas de sens. Choisir un modèle à probabilités (régression logistique, forêt...)."
        )
    if seuil is None:
        return diag
    return evaluate.diagnostiquer_validation(resultat, id_modele, seuil=seuil)


def _couts(resultat: train.Resultat, cout_fp: float | None, cout_fn: float | None) -> tuple[float, float]:
    """Coûts de décision : ceux demandés, sinon ceux de la métrique « cout:fp=...,fn=... », sinon 1 et 1."""
    metrique, parametres = metrics.analyser(resultat.metrique)
    par_defaut = parametres if metrique.id == "cout" else {}
    return (
        float(par_defaut.get("fp", 1.0) if cout_fp is None else cout_fp),
        float(par_defaut.get("fn", 1.0) if cout_fn is None else cout_fn),
    )


def seuil_minimisant_le_cout(
    resultat: train.Resultat,
    id_modele: str,
    *,
    cout_fp: float | None = None,
    cout_fn: float | None = None,
    points: int = 101,
) -> SeuilPropose:
    """Seuil du plus petit coût moyen sur les prédictions hors pli (à égalité, le plus proche de 0,5)."""
    _modele_retenable(resultat, id_modele)
    fp, fn = _couts(resultat, cout_fp, cout_fn)
    courbe = _diagnostic_au_seuil(resultat, id_modele, None).courbe_seuil(fp, fn, points)
    ligne = choisir_le_seuil(courbe)
    return SeuilPropose(float(ligne["seuil"]), float(ligne["cout"]), fp, fn, courbe)


def choisir_le_seuil(courbe: pd.DataFrame) -> pd.Series:
    """Ligne de la courbe au plus petit coût ; à coût égal, celle dont le seuil est le plus proche de 0,5."""
    meilleur = courbe["cout"].min()
    ex_aequo = courbe[np.isclose(courbe["cout"], meilleur, rtol=0.0, atol=1e-12)]
    return ex_aequo.iloc[(ex_aequo["seuil"] - 0.5).abs().argmin()]


def score_au_seuil(resultat: train.Resultat, id_modele: str, seuil: float) -> float:
    """Score de validation au seuil : tout score au moins égal au seuil est déclaré positif.

    Calculé pli par pli sur les prédictions hors pli, puis moyenné sur les plis où il est défini (comme la
    comparaison de S6).
    """
    diag = _diagnostic_au_seuil(resultat, id_modele, seuil)
    valeurs = []
    for k in np.unique(diag.plis):
        if k < 0:
            continue
        masque = diag.plis == k
        try:
            valeurs.append(
                metrics.calculer(
                    resultat.metrique,
                    y_vrai=diag.y_vrai[masque],
                    y_pred=diag.y_pred[masque],
                    scores=diag.scores[masque],
                    classes=diag.classes,
                    classe_positive=diag.classe_positive,
                    seuil=seuil,
                )
            )
        except metrics.ErreurMetrique:
            continue
    return float(np.mean(valeurs)) if valeurs else math.nan


# ============================================================
# DÉCISION
# ============================================================
def _justification(resultat: train.Resultat, modele: train.ResultatModele, seuil: float | None) -> str:
    valeur = resultat.valeur(modele)
    texte = (
        f"Meilleur score de validation croisée parmi les modèles comparés : {resultat.metrique} = "
        f"{decimale(valeur, 4)} en moyenne sur {modele.plis_valides(resultat.metrique)} plis"
    )
    gain = resultat.gain(modele.id)
    reference = resultat.reference()
    if gain is not None and reference is not None:
        texte += f", contre {decimale(resultat.valeur(reference), 4)} pour la référence naïve"
    texte += "."
    if seuil is not None:
        texte += f" Seuil réglé à {decimale(seuil, 3)} sur les prédictions hors pli."
    return texte


def decider(
    resultat: train.Resultat, id_modele: str, *, seuil: float | None = None, justification: str = ""
) -> Decision:
    """Décision sur un modèle de la comparaison, pour la métrique de S3, avec un seuil pour deux classes.

    Le score de validation est celui du tableau de S6, ou, avec un seuil, celui recalculé au seuil. Une
    justification vide est remplacée par un texte qui cite le score et la référence naïve.
    """
    modele = _modele_retenable(resultat, id_modele)
    if seuil is None:
        score = resultat.valeur(modele)
    else:
        if isinstance(seuil, bool) or not 0.0 <= float(seuil) <= 1.0:
            raise ErreurDecision("Le seuil se donne entre 0 et 1.")
        score = score_au_seuil(resultat, id_modele, float(seuil))
        if math.isnan(score):
            raise ErreurDecision(
                f"« {resultat.metrique} » n'est défini sur aucun pli au seuil {seuil} : choisir un autre "
                "seuil."
            )
    return Decision(
        id_modele,
        resultat.metrique,
        float(score),
        None if seuil is None else float(seuil),
        justification.strip() or _justification(resultat, modele, seuil),
    )


def proposer(resultat: train.Resultat) -> Decision:
    """Décision proposée par défaut : le meilleur modèle de la comparaison, sans seuil."""
    return decider(resultat, meilleur_modele(resultat).id)


# ============================================================
# CE QU'UN LECTEUR PRESSÉ MANQUERAIT
# ============================================================
def avertissements(resultat: train.Resultat, id_modele: str) -> tuple[str, ...]:
    """Mises en garde sur le modèle que l'on s'apprête à retenir, en phrases de l'écran de décision."""
    modele = _modele_retenable(resultat, id_modele)
    notes: list[str] = []
    if resultat.reference() is not None:
        gain = resultat.gain(id_modele)
        if gain is None or gain <= 0:
            notes.append(
                "Ce modèle ne fait pas mieux que la référence naïve : rien n'a été appris que la moyenne "
                "ou la classe majoritaire ne donnait déjà. Revoir les variables ou le prétraitement."
            )
    valides, total = modele.plis_valides(resultat.metrique), len(resultat.plis)
    if valides < total:
        notes.append(
            f"« {libelles.metrique_courte(resultat.metrique)} » n'est défini que sur {valides} plis sur "
            f"{total} (une classe manque dans les autres) : la moyenne repose sur moins de lignes."
        )
    suivant = next((m for m in modeles_a_retenir(resultat) if m.id != id_modele), None)
    if suivant is not None:
        ecart = abs(resultat.valeur(modele) - resultat.valeur(suivant))
        dispersion = max(modele.ecart_type(resultat.metrique), suivant.ecart_type(resultat.metrique))
        if ecart < dispersion:
            notes.append(
                f"L'écart avec {libelles.technique(suivant.id)} ({decimale(ecart, 4)}) est plus petit que la "
                f"dispersion entre plis ({decimale(dispersion, 4)}) : les deux modèles ne se distinguent "
                "pas ; préférer le plus simple."
            )
    if resultat.classe_positive_source not in (None, "specification"):
        notes.append(
            f"La classe positive ({resultat.classe_positive}) est une proposition non confirmée en S3 : "
            "les métriques de précision, de rappel et de F1 en dépendent."
        )
    notes.extend(f"Avertissement de calcul : {a}" for a in modele.avertissements)
    return tuple(notes)


# ============================================================
# FIGURE
# ============================================================
def figure_comparaison(resultat: train.Resultat) -> Figure:
    """Score moyen de chaque modèle et dispersion entre plis ; la référence naïve est en pointillés."""
    from matplotlib.figure import Figure

    classes = [m for m in resultat.classement() if m.ok and not math.isnan(resultat.valeur(m))]
    ids = [m.id for m in classes][::-1]
    noms = [libelles.technique(i) for i in ids]
    valeurs = [resultat.valeur(m) for m in classes][::-1]
    ecarts = [m.ecart_type(resultat.metrique) for m in classes][::-1]
    fig = Figure(figsize=(7.0, 0.4 * max(len(noms), 3) + 1.3), layout="constrained", dpi=100)
    axe = fig.subplots()
    couleurs = ["#c55a11" if i in REFERENCES_NAIVES.values() else "#1f4e79" for i in ids]
    axe.barh(noms, valeurs, xerr=ecarts, color=couleurs, ecolor="#7f7f7f")
    reference = resultat.reference()
    if reference is not None and not math.isnan(resultat.valeur(reference)):
        axe.axvline(resultat.valeur(reference), color="#c55a11", linestyle=":", linewidth=1)
    sens = {metrics.HAUT: "plus grand est meilleur", metrics.BAS: "plus petit est meilleur"}.get(
        resultat.meilleur, ""
    )
    metrique = libelles.metrique_courte(resultat.metrique)
    axe.set(xlabel=f"{metrique} ({sens})" if sens else metrique, title="Comparaison des modèles")
    axe.spines[["top", "right"]].set_visible(False)
    axe.grid(True, axis="x", color="#e5e5e5", linewidth=0.6)
    axe.set_axisbelow(True)
    return fig
