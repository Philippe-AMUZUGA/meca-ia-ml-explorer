"""Jeux synthétiques minuscules et fabriques de projets pour les tests du catalogue et de l'apprentissage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mecaml.core.types import ChoixTechnique, CoffreTest, Probleme, Role, SpecProjet, Structure, Table

PRETRAITEMENT_STANDARD = (
    ChoixTechnique.creer("manquants.simple", strategy="median"),
    ChoixTechnique.creer("encodage.one_hot"),
    ChoixTechnique.creer("transformation.standard"),
)


def jeu_regression(n: int = 240, graine: int = 0, manquants: bool = True) -> pd.DataFrame:
    """Deux mesures (dont une avec valeurs manquantes), une modalité, une cible linéaire bruitée."""
    rng = np.random.default_rng(graine)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "d": rng.uniform(0, 10, n),
            "c": rng.choice(["x", "y", "z"], n),
        }
    )
    df["y"] = 2.0 * df["a"] + df["b"] - 0.2 * df["d"] + (df["c"] == "x") + rng.normal(0, 0.1, n)
    if manquants:
        df.loc[df.index[::17], "a"] = np.nan
    return df


def jeu_classification(n: int = 300, graine: int = 0, classes: int = 2, rare: float = 0.3) -> pd.DataFrame:
    """Deux mesures, une modalité ; la cible dépend des mesures (classes équilibrées ou une classe rare)."""
    rng = np.random.default_rng(graine)
    a, b = rng.normal(size=n), rng.normal(size=n)
    score = a + 0.5 * b + rng.normal(0, 0.3, n)
    if classes == 2:
        seuil = np.quantile(score, 1 - rare)
        cible = np.where(score > seuil, "defaut", "conforme")
    else:
        bornes = np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1])
        cible = np.array([f"classe_{k}" for k in np.searchsorted(bornes, score)])
    df = pd.DataFrame({"a": a, "b": b, "c": rng.choice(["x", "y", "z"], n), "y": cible})
    df.loc[df.index[::19], "b"] = np.nan
    return df


def jeu_groupe(n_groupes: int = 10, longueur: int = 14, graine: int = 0) -> pd.DataFrame:
    """Machines suivies sur des cycles : dérive linéaire propre à chaque machine, durée de vie résiduelle.

    La cible décompte d'un cycle à chaque cycle (au bruit près) : la cible passée y donne la réponse.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    for g in range(n_groupes):
        pente = rng.uniform(0.5, 1.5)
        for t in range(longueur):
            lignes.append(
                {
                    "machine": f"M{g:02d}",
                    "cycle": t,
                    "x1": pente * t + rng.normal(0, 0.2),
                    "x2": rng.normal(),
                    "y": float(longueur - t) + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(lignes)


def jeu_serie_mesuree(n_groupes: int = 8, longueur: int = 60, graine: int = 0) -> pd.DataFrame:
    """Machines suivies sur des cycles ; la cible est une température relevée à chaque cycle.

    Elle s'installe lentement (autorégression de coefficient 0,9) : sa valeur du cycle précédent est connue
    à la prévision et en dit presque tout. À l'inverse de `jeu_groupe`, aucune durée de vie ne décompte.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    for g in range(n_groupes):
        temperature = 20.0 + rng.normal(0, 2)
        for t in range(longueur):
            x1 = rng.normal()
            temperature = 0.9 * temperature + 0.1 * (20.0 + 5.0 * x1) + rng.normal(0, 0.3)
            lignes.append(
                {"machine": f"M{g:02d}", "cycle": t, "x1": x1, "x2": rng.normal(), "y": temperature}
            )
    return pd.DataFrame(lignes)


def jeu_sans_cible(n: int = 200, graine: int = 0) -> pd.DataFrame:
    """Deux amas bien séparés dans le plan, plus une mesure de bruit."""
    rng = np.random.default_rng(graine)
    demi = n // 2
    return pd.DataFrame(
        {
            "u": np.r_[rng.normal(0, 0.3, demi), rng.normal(5, 0.3, n - demi)],
            "v": np.r_[rng.normal(0, 0.3, demi), rng.normal(5, 0.3, n - demi)],
            "bruit": rng.normal(size=n),
        }
    )


ROLES_REGRESSION = (
    ("a", Role.NUMERIQUE),
    ("b", Role.NUMERIQUE),
    ("d", Role.NUMERIQUE),
    ("c", Role.CATEGORIELLE),
    ("y", Role.CIBLE),
)
ROLES_CLASSIFICATION = (
    ("a", Role.NUMERIQUE),
    ("b", Role.NUMERIQUE),
    ("c", Role.CATEGORIELLE),
    ("y", Role.CIBLE),
)
ROLES_GROUPE = (
    ("machine", Role.GROUPE),
    ("cycle", Role.DATE),
    ("x1", Role.NUMERIQUE),
    ("x2", Role.NUMERIQUE),
    ("y", Role.CIBLE),
)
ROLES_SANS_CIBLE = (("u", Role.NUMERIQUE), ("v", Role.NUMERIQUE), ("bruit", Role.NUMERIQUE))


def spec_regression(modeles: tuple[ChoixTechnique, ...] = (), **changements: Any) -> SpecProjet:
    """Régression de `y` sur `a`, `b`, `d`, `c`, métrique RMSE, prétraitement standard."""
    base = {
        "probleme": Probleme.REGRESSION,
        "cible": "y",
        "roles": ROLES_REGRESSION,
        "metrique": "rmse",
        "pretraitement": PRETRAITEMENT_STANDARD,
        "modeles": modeles,
        **changements,
    }
    return SpecProjet(**base)


def spec_classification(
    modeles: tuple[ChoixTechnique, ...] = (), binaire: bool = True, **changements: Any
) -> SpecProjet:
    """Classification de `y`, métrique F1, prétraitement standard ; classe positive « defaut » si binaire."""
    base = {
        "probleme": Probleme.CLASSIFICATION,
        "cible": "y",
        "roles": ROLES_CLASSIFICATION,
        "metrique": "f1",
        "pretraitement": PRETRAITEMENT_STANDARD,
        "modeles": modeles,
        "classe_positive": "defaut" if binaire else None,
        **changements,
    }
    return SpecProjet(**base)


def spec_groupe(modeles: tuple[ChoixTechnique, ...] = (), **changements: Any) -> SpecProjet:
    """Durée résiduelle par machine : structure groupée et chronologique, métrique RMSE."""
    base = {
        "probleme": Probleme.REGRESSION,
        "cible": "y",
        "roles": ROLES_GROUPE,
        "metrique": "rmse",
        "structure": Structure.GROUPEE_CHRONOLOGIQUE,
        "n_plis": 3,
        "pretraitement": (ChoixTechnique.creer("transformation.standard"),),
        "modeles": modeles,
        **changements,
    }
    return SpecProjet(**base)


def spec_sans_cible(modeles: tuple[ChoixTechnique, ...] = (), **changements: Any) -> SpecProjet:
    """Regroupement des amas sur `u`, `v` et `bruit`."""
    base = {
        "probleme": Probleme.SANS_CIBLE,
        "roles": ROLES_SANS_CIBLE,
        "metrique": "silhouette",
        "pretraitement": (ChoixTechnique.creer("transformation.standard"),),
        "modeles": modeles,
        **changements,
    }
    return SpecProjet(**base)


# ============================================================
# DOUBLES MINIMAUX DU DÉCOUPAGE (tests indépendants de l'état de `split.py`)
# ============================================================
def coffre_par_pas(table: Table, spec: SpecProjet, pas: int = 5) -> CoffreTest:
    """Coffre dont le test est une ligne sur `pas`."""
    return CoffreTest.par_positions(table, np.arange(0, table.n_lignes, pas), decoupage=spec.decoupage())


def coffre_par_groupes(table: Table, spec: SpecProjet, groupes_test: tuple[str, ...]) -> CoffreTest:
    """Coffre dont le test réunit des groupes entiers."""
    colonne = spec.colonne_groupe
    positions = np.flatnonzero(table.donnees[colonne].isin(groupes_test).to_numpy())
    return CoffreTest.par_positions(table, positions, decoupage=spec.decoupage())


def colonnes_modele() -> tuple[str, ...]:
    """Colonnes d'entrée de la régression type (mesures et modalité)."""
    return ("a", "b", "d", "c")
