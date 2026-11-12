"""
MécaML : la cible passée n'entre dans un modèle que si elle est connue au moment de la prévision.

La cible décalée (`chronologique.cible_decalee`) et la référence par persistance (`reference.persistance`)
prédisent une ligne avec la valeur que la cible avait aux pas précédents. C'est permis quand cette valeur
est connue à l'instant de la prévision (une température relevée à chaque pas) ; c'est la réponse même
quand elle ne l'est qu'après coup (la durée de vie restante d'une machine ne se connaît qu'après sa panne).

Le cœur ne le devine pas : le projet le déclare dans le choix lui-même, par le paramètre
`cible_connue=True`, que la spécification garde et que le script exporté reproduit ; sans déclaration, le
choix est refusé. Une déclaration ne suffit pas non plus quand la cible décompte du même pas dans chaque
unité : c'est une durée restante, et le refus tient même déclaré.

Trois usages : les fabriques des deux techniques exigent la déclaration (`exiger_declaration`), l'audit en
fait un constat BLOQUANT (`constats`) et l'entraînement refuse (`train`). Seules les lignes d'apprentissage
se lisent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from mecaml.core import renvois
from mecaml.core.guards import decimale
from mecaml.core.types import ChoixTechnique, Constat, Niveau, SpecProjet

__all__ = [
    "CIBLE_DECALEE",
    "PARAMETRE",
    "PART_DECOMPTE",
    "PERSISTANCE",
    "TECHNIQUES",
    "choix_concernes",
    "constats",
    "declaree",
    "decompte",
    "exiger_declaration",
    "lit_la_cible_passee",
    "message_non_declaree",
]

CIBLE_DECALEE = "chronologique.cible_decalee"
PERSISTANCE = "reference.persistance"
TECHNIQUES = (CIBLE_DECALEE, PERSISTANCE)  # techniques qui lisent la cible des pas précédents
PARAMETRE = "cible_connue"  # déclaration du projet : la cible passée est connue à la prévision

# Compte à rebours : le même pas négatif, à TOLERANCE_PAS de sa valeur près (en part du pas), entre au
# moins PART_DECOMPTE des lignes consécutives d'une unité ; sous PAS_MIN pas, rien ne se juge.
PART_DECOMPTE = 0.9
TOLERANCE_PAS = 0.5
PAS_MIN = 10


# ============================================================
# DÉCLARATION
# ============================================================
def declaree(parametres: Mapping[str, Any]) -> bool:
    """Vrai si le choix déclare la cible passée connue au moment de la prévision (`cible_connue=True`)."""
    return parametres.get(PARAMETRE) is True


def message_non_declaree(id_technique: str) -> tuple[str, str]:
    """(message, action) pour un choix qui lit la cible passée sans le déclarer."""
    message = (
        f"{id_technique} prédit une ligne avec la valeur de la cible aux pas précédents : c'est la réponse "
        "même si cette valeur n'est connue qu'après coup (la durée de vie restante d'une machine, le "
        "résultat d'un contrôle), et le cœur ne peut pas le deviner."
    )
    action = (
        "Si la cible du pas précédent est connue au moment de la prévision, le déclarer en donnant "
        f"`{PARAMETRE}=True` à ce choix ; sinon le retirer et prévoir à partir des mesures seules."
    )
    return message, action


def exiger_declaration(id_technique: str, parametres: Mapping[str, Any]) -> None:
    """Refuse (`ValueError`) la construction d'un choix qui lit la cible passée sans le déclarer."""
    if not declaree(parametres):
        message, action = message_non_declaree(id_technique)
        raise ValueError(f"{message} {action}")


def choix_concernes(spec: SpecProjet) -> tuple[ChoixTechnique, ...]:
    """Choix du prétraitement et des modèles qui lisent la cible passée."""
    return tuple(c for c in (*spec.pretraitement, *spec.modeles) if c.id in TECHNIQUES)


def lit_la_cible_passee(spec: SpecProjet) -> bool:
    """Vrai si un choix de la spécification lit la cible des pas précédents."""
    return bool(choix_concernes(spec))


# ============================================================
# COMPTE À REBOURS
# ============================================================
def decompte(donnees: pd.DataFrame, spec: SpecProjet) -> tuple[float, float] | None:
    """(pas, part) si la cible décompte, sinon None.

    Les lignes sont rangées par date dans chaque unité (une seule unité hors structure groupée) ; le pas est
    l'écart à la ligne précédente de la même unité. La cible décompte quand au moins `PART_DECOMPTE` des
    pas valent le pas médian, négatif, à `TOLERANCE_PAS` de sa valeur près : une durée de vie restante
    perd un cycle à chaque cycle. Trop peu de pas, ou des dates qu'on ne sait pas ordonner : rien ne se juge.
    """
    cible, date, groupe = spec.cible, spec.colonne_date, spec.colonne_groupe
    requises = [c for c in (cible, date, groupe) if c is not None]
    if cible is None or date is None or any(c not in donnees.columns for c in requises):
        return None
    unite = pd.factorize(donnees[groupe])[0] if groupe is not None else np.zeros(len(donnees), dtype=int)
    valeurs = pd.to_numeric(donnees[cible], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    travail = pd.DataFrame({"y": valeurs, "date": donnees[date].to_numpy(), "unite": unite})
    travail = travail[(travail["unite"] >= 0) & travail["date"].notna()]
    try:
        travail = travail.sort_values(["unite", "date"], kind="mergesort")
    except TypeError:
        return None
    pas = travail.groupby("unite", sort=False)["y"].diff().dropna().to_numpy()
    if pas.size < PAS_MIN:
        return None
    mediane = float(np.median(pas))
    if mediane >= 0:
        return None
    part = float(np.mean(np.abs(pas - mediane) <= TOLERANCE_PAS * abs(mediane)))
    return (mediane, part) if part >= PART_DECOMPTE else None


def _message_decompte(cible: str, id_technique: str, pas: float, part: float) -> tuple[str, str]:
    pas_lisible = f"{round(abs(pas), 2):g}".replace(".", ",")
    message = (
        f"La cible « {cible} » baisse de {pas_lisible} à chaque ligne d'une même unité dans "
        f"{decimale(100 * part, 0)} % des pas : c'est un compte à rebours, la durée de vie restante, que "
        "l'on ne connaît qu'après la panne. La valeur du pas précédent donne celle du pas courant à un pas "
        f"près : {id_technique} reviendrait à lire la réponse, et le refus tient même déclaré."
    )
    action = "Retirer ce choix et prévoir à partir des mesures seules (les capteurs), sans la cible passée."
    return message, action


# ============================================================
# CONSTATS
# ============================================================
def constats(spec: SpecProjet, donnees: pd.DataFrame) -> tuple[Constat, ...]:
    """Constats BLOQUANT des choix qui lisent la cible passée : sans déclaration, ou cible qui décompte.

    `donnees` : les lignes d'apprentissage seules (jamais celles du test). Rien ne se mesure tant
    qu'aucun choix ne lit la cible passée.
    """
    par_technique: dict[str, list[ChoixTechnique]] = {}
    for choix in choix_concernes(spec):
        par_technique.setdefault(choix.id, []).append(choix)
    if not par_technique:
        return ()
    renvoi = renvois.theme("cible_passee")
    refus: list[Constat] = []
    declarees: list[str] = []
    for id_technique, choix in par_technique.items():
        if all(declaree(c.parametres_dict()) for c in choix):
            declarees.append(id_technique)
            continue
        message, action = message_non_declaree(id_technique)
        code = f"I3.cible_passee:{id_technique.split('.', 1)[1]}"
        refus.append(Constat(Niveau.BLOQUANT, code, message, action, renvoi))
    mesure = decompte(donnees, spec) if declarees and spec.cible else None
    if mesure is not None and spec.cible is not None:
        pas, part = mesure
        for id_technique in declarees:
            message, action = _message_decompte(spec.cible, id_technique, pas, part)
            code = f"I3.cible_decompte:{id_technique.split('.', 1)[1]}"
            refus.append(Constat(Niveau.BLOQUANT, code, message, action, renvoi))
    return tuple(refus)
