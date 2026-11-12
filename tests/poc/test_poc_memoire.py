"""Preuve de concept : les transitions partagent la table, le pic mémoire reste borné.

Critère : pic mémoire au plus 1,5 fois la taille de la table (100 000 lignes × 21
colonnes), mesuré par `tracemalloc` sur le chargement en `Table` (empreinte comprise)
et sur un parcours de S1 à S7 avec retour arrière et rejeu de l'audit.
Lancer : `python -m pytest -m lent -s tests/poc`.
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from mecaml.core.session import (
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Etat,
    FixerObjectif,
    Pretraiter,
    Revenir,
    Session,
)
from mecaml.core.types import ChoixTechnique, CoffreTest, Decision, Probleme, Role, SpecProjet, Table

pytestmark = pytest.mark.lent

N_LIGNES = 100_000
RATIO_MAX = 1.5
MODELES = (
    ChoixTechnique.creer("reference.dummy_regression"),
    ChoixTechnique.creer("regression.ridge", alpha=1.0),
)


def table_synthetique(n: int = N_LIGNES, graine: int = 0) -> pd.DataFrame:
    """16 mesures réelles, 2 entiers, 3 colonnes textuelles : 21 colonnes."""
    rng = np.random.default_rng(graine)
    colonnes: dict[str, np.ndarray] = {f"mesure_{i:02d}": rng.normal(size=n) for i in range(16)}
    colonnes["cycle"] = np.arange(n) % 250
    colonnes["lot"] = rng.integers(0, 400, n)
    colonnes["machine"] = rng.choice([f"M{i:03d}" for i in range(100)], n)
    colonnes["materiau"] = rng.choice(["acier", "alu", "inox", "titane"], n)
    colonnes["classe"] = rng.choice(["conforme", "non_conforme"], n)
    return pd.DataFrame(colonnes)


def _spec(cible: str) -> SpecProjet:
    roles = [(f"mesure_{i:02d}", Role.NUMERIQUE) for i in range(16) if f"mesure_{i:02d}" != cible]
    roles += [("cycle", Role.NUMERIQUE), ("lot", Role.NUMERIQUE), ("machine", Role.CATEGORIELLE)]
    roles += [("materiau", Role.CATEGORIELLE), ("classe", Role.IGNOREE), (cible, Role.CIBLE)]
    return SpecProjet(Probleme.REGRESSION, cible=cible, roles=tuple(roles), metrique="rmse")


def parcours_complet(table: Table) -> Session:
    """S1 à S7, retour en S3, nouvelle cible (audit rejoué), puis S7 de nouveau."""
    positions = np.random.default_rng(1).choice(table.n_lignes, table.n_lignes // 5, replace=False)
    decision = Decision("reference.dummy_regression", "rmse", 1.0)
    s = Session().appliquer(Charger(table)).appliquer(Auditer())
    for cible in ("mesure_00", "mesure_01"):
        spec = _spec(cible)
        s = s.appliquer(FixerObjectif(spec))
        s = s.appliquer(Decouper(CoffreTest.par_positions(table, positions, decoupage=spec.decoupage())))
        spec = spec.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))
        s = s.appliquer(Pretraiter(spec))
        s = s.appliquer(Entrainer({"reference.dummy_regression": 1.0}, spec=spec.avec(modeles=MODELES)))
        s = s.appliquer(Decider(decision))
        if cible == "mesure_00":
            s = s.appliquer(Revenir(Etat.OBJECTIF))
    return s


def test_poc_pic_memoire_des_transitions() -> None:
    df = table_synthetique()
    taille = int(df.memory_usage(index=True, deep=True).sum())
    gc.collect()
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        debut = time.perf_counter()
        table = Table(df, nom="synthetique_100000x21.csv")
        session = parcours_complet(table)
        intacte = table.est_intacte()
        duree = time.perf_counter() - debut
        _, pic = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    surcout = pic - base
    ratio = (taille + surcout) / taille
    print(
        f"\n[POC mémoire] table {df.shape[0]} x {df.shape[1]} : {taille / 2**20:.1f} Mio ; "
        f"surcoût de pic {surcout / 2**20:.2f} Mio ; ratio {ratio:.3f} (borne {RATIO_MAX}) ; "
        f"{len(session.historique)} transitions en {duree:.2f} s"
    )
    assert session.etat == Etat.DIAGNOSTIC and session.table is table
    assert session.table.donnees is df and intacte
    assert ratio <= RATIO_MAX


def test_poc_extraction_apprentissage_pour_information() -> None:
    """Mesure indicative : `apprentissage()` matérialise 80 % des lignes (copie pandas)."""
    df = table_synthetique()
    table = Table(df, nom="synthetique.csv")
    taille = int(df.memory_usage(index=True, deep=True).sum())
    decoupage = _spec("mesure_00").decoupage()
    coffre = CoffreTest.par_positions(table, np.arange(0, table.n_lignes, 5), decoupage=decoupage)
    gc.collect()
    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        app = coffre.apprentissage()
        _, pic = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    print(
        f"\n[POC mémoire] apprentissage() : {len(app)} lignes, surcoût de pic "
        f"{(pic - base) / 2**20:.2f} Mio soit {(pic - base) / taille:.2f} fois la table"
    )
    assert len(app) == table.n_lignes - table.n_lignes // 5
