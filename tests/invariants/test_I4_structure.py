"""I4 Structure : unités disjointes entre apprentissage et test ; max(date apprentissage) < min(date test).

Cas dorés sur le suivi de 100 machines jusqu'à la défaillance (TimeSeries_RUL) et sur
la maintenance horodatée (AI4I), lus dans le dossier désigné par MECAML_JEUX (copies
locales, jamais versionnées) ; répliques synthétiques de même forme toujours jouées.
Accord exact avec l'algorithme des bases de test du dépôt d'exercices.
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from mecaml.core.session import (
    Auditer,
    Charger,
    Decouper,
    Etat,
    FixerObjectif,
    Pretraiter,
    Session,
    TransitionRefusee,
)
from mecaml.core.split import (
    ErreurDecoupage,
    Methode,
    controler_decoupage,
    controler_structure,
    empreinte_positions,
    plis_validation,
    positions_test,
    sceller,
    sceller_base_test,
)
from mecaml.core.types import ChoixTechnique, CoffreTest, Probleme, Role, SpecProjet, Structure, Table

RUL_CAPTEURS = (
    "Reglage_Charge_Nominale_Pourcent",
    "Reglage_Vitesse_RPM",
    "Temp_Carter_K",
    "Pression_Ligne_bar",
    "Vib_RMS_AxeZ_mm_s",
    "Courant_Moteur_Amp",
    "Contamination_Huile_PPM",
)
AI4I_MESURES = ("T_ambiant_C", "T_palier_C", "Vitesse_rot_tr_min", "Couple_Nm", "Usure_palier_hrs")


# ============================================================
# JEUX : RÉELS (MECAML_JEUX) ET RÉPLIQUES SYNTHÉTIQUES
# ============================================================
def _lire(fichier: str) -> pd.DataFrame:
    dossier = os.environ.get("MECAML_JEUX")
    if not dossier or not (Path(dossier) / fichier).is_file():
        pytest.skip(f"{fichier} absent de MECAML_JEUX")
    return pd.read_excel(Path(dossier) / fichier)


def replique_rul(graine: int = 0) -> pd.DataFrame:
    """100 machines suivies cycle par cycle jusqu'à la défaillance, durées inégales."""
    rng = np.random.default_rng(graine)
    durees = rng.integers(20, 60, 100)
    machine = np.repeat(np.arange(1, 101), durees)
    cycle = np.concatenate([np.arange(1, d + 1) for d in durees])
    df = pd.DataFrame({"ID_Machine": machine, "Cycle_Operation_Heures": cycle})
    for i, c in enumerate(RUL_CAPTEURS):
        df[c] = rng.normal(10.0 * (i + 1), 1.0, len(df)) + 0.05 * cycle
    df["Duree_Vie_Restante_Cycles"] = np.repeat(durees, durees) - cycle
    return df


def replique_ai4i(graine: int = 0, n: int = 3000) -> pd.DataFrame:
    """Une série horodatée au pas de 4 heures, pannes rares (environ 3,4 %)."""
    rng = np.random.default_rng(graine)
    df = pd.DataFrame({"time": np.round(np.arange(n) / 6, 4)})
    for i, c in enumerate(AI4I_MESURES):
        df[c] = rng.normal(20.0 * (i + 1), 2.0, n)
    df["Panne"] = (rng.random(n) < 0.034).astype(int)
    return df


def spec_rul(structure: Structure = Structure.GROUPEE_CHRONOLOGIQUE, **changements: object) -> SpecProjet:
    roles = (
        ("ID_Machine", Role.GROUPE),
        ("Cycle_Operation_Heures", Role.DATE),
        ("Duree_Vie_Restante_Cycles", Role.CIBLE),
        *((c, Role.NUMERIQUE) for c in RUL_CAPTEURS),
    )
    spec = SpecProjet(Probleme.REGRESSION, "Duree_Vie_Restante_Cycles", roles, "rmse", structure, graine=42)
    return spec.avec(**changements) if changements else spec


def spec_ai4i(structure: Structure = Structure.CHRONOLOGIQUE, **changements: object) -> SpecProjet:
    roles = (("time", Role.DATE), ("Panne", Role.CIBLE), *((c, Role.NUMERIQUE) for c in AI4I_MESURES))
    spec = SpecProjet(Probleme.CLASSIFICATION, "Panne", roles, "f1_macro", structure, graine=42)
    return spec.avec(**changements) if changements else spec


@functools.cache
def _jeu(nom: str, source: str) -> pd.DataFrame:
    """Jeu réel lu une fois par session de tests, ou sa réplique synthétique."""
    if source == "replique":
        return replique_rul() if nom == "rul" else replique_ai4i()
    return _lire("MecaIA-TimeSeries_RUL.xlsx" if nom == "rul" else "MecaIA-AI4I_maintenance.xlsx")


@pytest.fixture(params=["reel", "replique"])
def source(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def rul(source: str) -> pd.DataFrame:
    return _jeu("rul", source)


@pytest.fixture
def ai4i(source: str) -> pd.DataFrame:
    return _jeu("ai4i", source)


def _cotes(coffre: CoffreTest, n: int) -> tuple[np.ndarray, np.ndarray]:
    app = coffre.positions_apprentissage()
    return app, np.setdiff1d(np.arange(n), app)


# ============================================================
# UNITÉS DISJOINTES
# ============================================================
@pytest.mark.parametrize("structure", [Structure.GROUPEE, Structure.GROUPEE_CHRONOLOGIQUE], ids=str)
def test_I4_cent_machines_unites_entieres(rul, structure) -> None:
    table = Table(rul, nom="rul")
    spec = spec_rul(structure)
    decoupage, coffre = sceller(table, spec)
    app, test = _cotes(coffre, len(rul))
    m = rul["ID_Machine"].to_numpy()
    assert rul["ID_Machine"].nunique() == 100
    assert not set(m[app]) & set(m[test])
    assert len(set(m[test])) == math.ceil(0.2 * 100)  # 20 machines entières en test
    assert decoupage.methode is Methode.UNITES and decoupage.colonne_groupe == "ID_Machine"
    assert controler_structure(table, spec, coffre) == ()


def test_I4_plis_par_machines(rul) -> None:
    table = Table(rul, nom="rul")
    _, coffre = sceller(table, spec_rul())
    lignes = coffre.apprentissage()
    m = lignes["ID_Machine"].to_numpy()
    plis = plis_validation(lignes, spec_rul())
    assert len(plis) == 5
    for app, val in plis:
        assert not set(m[app]) & set(m[val])
    assert np.array_equal(np.sort(np.concatenate([v for _, v in plis])), np.arange(len(lignes)))


# ============================================================
# DATES ORDONNÉES
# ============================================================
def test_I4_maintenance_horodatee_coupure_chronologique(ai4i) -> None:
    table = Table(ai4i, nom="ai4i")
    spec = spec_ai4i()
    decoupage, coffre = sceller(table, spec)
    app, test = _cotes(coffre, len(ai4i))
    t = ai4i["time"].to_numpy()
    assert t[app].max() < t[test].min()
    assert test.size == math.ceil(0.2 * len(ai4i))  # dates toutes distinctes : part exacte
    assert decoupage.methode is Methode.CHRONOLOGIQUE and decoupage.colonne_groupe == "time"
    assert controler_structure(table, spec, coffre) == ()


def test_I4_plis_chronologiques(ai4i) -> None:
    _, coffre = sceller(Table(ai4i, nom="ai4i"), spec_ai4i())
    lignes = coffre.apprentissage()
    t = lignes["time"].to_numpy()
    for app, val in plis_validation(lignes, spec_ai4i()):
        assert t[app].max() < t[val].min()


def test_I4_cycles_partages_entre_machines_sans_date_a_cheval(rul) -> None:
    """Coupure chronologique sur des cycles communs à plusieurs machines : aucun cycle des deux côtés."""
    roles = tuple((c, r) for c, r in spec_rul().roles if r is not Role.GROUPE)
    spec = spec_rul(Structure.CHRONOLOGIQUE, roles=roles)
    _, coffre = sceller(Table(rul, nom="rul"), spec)
    app, test = _cotes(coffre, len(rul))
    c = rul["Cycle_Operation_Heures"].to_numpy()
    assert c[app].max() < c[test].min()
    assert test.size >= math.ceil(0.2 * len(rul))


# ============================================================
# DÉTECTION D'UNE STRUCTURE VIOLÉE ET GARDE DE LA SESSION
# ============================================================
def test_I4_controle_detecte_les_violations(rul) -> None:
    table = Table(rul, nom="rul")
    une_sur_cinq = np.arange(0, len(rul), 5)
    for spec, code in (
        (spec_rul(Structure.GROUPEE), "I4.unites_a_cheval"),
        (spec_rul(Structure.CHRONOLOGIQUE), "I4.dates_a_cheval"),
    ):
        coffre = CoffreTest.par_positions(table, une_sur_cinq, decoupage=spec.decoupage())
        constats = controler_structure(table, spec, coffre)
        assert [c.code for c in constats] == [code]
        assert constats[0].bloquant and constats[0].action and constats[0].ref_livre


def test_I4_controle_signale_une_colonne_incomplete() -> None:
    df = replique_rul().assign(ID_Machine=lambda d: d["ID_Machine"].astype(float))
    df.loc[3, "ID_Machine"] = np.nan
    table = Table(df, nom="trous")
    spec = spec_rul(Structure.GROUPEE)
    coffre = CoffreTest.par_positions(table, np.arange(0, len(df), 5), decoupage=spec.decoupage())
    assert [c.code for c in controler_structure(table, spec, coffre)] == ["I4.unites_incompletes"]
    with pytest.raises(ErreurDecoupage, match="incomplète"):
        sceller(table, spec)


def test_I4_session_bloque_un_coffre_a_cheval() -> None:
    """L'auditeur appelle le contrôle au rejeu de S4 : un coffre à cheval interdit d'avancer."""
    table = Table(replique_rul(), nom="rul")
    spec = spec_rul(Structure.GROUPEE)
    s = Session(auditeur=controler_structure)
    s = s.appliquer(Charger(table)).appliquer(Auditer()).appliquer(FixerObjectif(spec))
    a_cheval = CoffreTest.par_positions(table, np.arange(0, table.n_lignes, 5), decoupage=spec.decoupage())
    s4 = s.appliquer(Decouper(a_cheval))
    assert [c.code for c in s4.bloquants()] == ["I4.unites_a_cheval"]
    with pytest.raises(TransitionRefusee, match=r"I4\.unites_a_cheval"):
        s4.appliquer(Pretraiter(spec.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))))
    _, scelle = sceller(table, spec)
    s4 = s.appliquer(Decouper(scelle))
    assert s4.etat is Etat.DECOUPE and s4.bloquants() == ()


# ============================================================
# ACCORD AVEC LES BASES DE TEST DU DÉPÔT D'EXERCICES
# ============================================================
def _reference_depot(df: pd.DataFrame, methode: str, ts: float, rs: int, col: str | None) -> list[int]:
    """Algorithme du dépôt, réécrit à part : positions de la première feuille, avant tout filtrage."""
    pos = np.arange(len(df))
    if methode == "aleatoire":
        return sorted(int(i) for i in train_test_split(pos, test_size=ts, random_state=rs)[1])
    if methode == "stratifie":
        return sorted(
            int(i) for i in train_test_split(pos, test_size=ts, random_state=rs, stratify=df[col])[1]
        )
    if methode == "unites":
        _, test = next(
            GroupShuffleSplit(n_splits=1, test_size=ts, random_state=rs).split(pos, groups=df[col])
        )
        return sorted(int(i) for i in test)
    valeurs = df[col].to_numpy()
    coupure = np.sort(valeurs, kind="stable")[len(df) - math.ceil(ts * len(df))]
    return sorted(int(i) for i in pos[valeurs >= coupure])


@pytest.mark.parametrize(
    ("jeu", "methode", "ts", "colonne"),
    [
        ("rul", "unites", 0.2, "ID_Machine"),
        ("rul", "chronologique", 0.25, "Cycle_Operation_Heures"),
        ("rul", "aleatoire", 0.3, None),
        ("ai4i", "stratifie", 0.2, "Panne"),
        ("ai4i", "chronologique", 0.25, "time"),
    ],
)
def test_I4_positions_identiques_a_l_algorithme_du_depot(source, jeu, methode, ts, colonne) -> None:
    df = _jeu(jeu, source)
    obtenues = positions_test(df, methode, ts, 42, colonne)
    attendues = _reference_depot(df, methode, ts, 42, colonne)
    assert obtenues.tolist() == attendues
    assert empreinte_positions(obtenues) == empreinte_positions(attendues)


def test_I4_base_publiee_relue_a_l_identique() -> None:
    """Base de test publiée au manifeste du dépôt pour l'exercice E01-2 (Traitement_Acier)."""
    df = _lire("MecaIA-Traitement_Acier.xlsx")
    base_test = {
        "methode": "aleatoire",
        "test_size": 0.25,
        "random_state": 42,
        "colonne_groupe": None,
        "lignes_test": 38,
        "empreinte_sha256": "698dd81846207bb444bf2781bde22ce045a0112247435a668f23ce9d77e4d425",
    }
    roles = tuple((c, Role.CIBLE if c == "resistance_traction_MPa" else Role.NUMERIQUE) for c in df.columns)
    roles = tuple((c, Role.CATEGORIELLE if c == "type_alliage" else r) for c, r in roles)
    spec = SpecProjet(Probleme.REGRESSION, "resistance_traction_MPa", roles, "r2")
    spec_lue, lu, coffre = sceller_base_test(Table(df, nom="acier"), spec, base_test)
    assert lu.vers_base_test() == base_test and coffre.n_test == 38 and spec_lue.graine == 42
    with pytest.raises(ErreurDecoupage, match="diffère"):
        sceller_base_test(Table(df.iloc[1:].reset_index(drop=True), nom="filtree"), spec, base_test)


# ============================================================
# CLASSIFICATION GROUPÉE : CLASSES AU TEST ET DANS CHAQUE PLI (I7)
# ============================================================
def _pannes_rares(rul: pd.DataFrame, n_machines: int) -> pd.DataFrame:
    """Panne imminente (10 derniers cycles) sur les `n_machines` premières machines seulement."""
    rul = rul.copy()
    proche = rul["Duree_Vie_Restante_Cycles"] < 10
    rul["Panne"] = (proche & (rul["ID_Machine"] <= n_machines)).astype(int)
    return rul


def _spec_pannes(structure: Structure = Structure.GROUPEE, **changements: object) -> SpecProjet:
    roles = tuple(
        (c, Role.IGNOREE if c == "Duree_Vie_Restante_Cycles" else r) for c, r in spec_rul(structure).roles
    )
    roles = (*roles, ("Panne", Role.CIBLE))
    return SpecProjet(
        Probleme.CLASSIFICATION, "Panne", roles, "f1_macro", structure, graine=42, **changements
    )


@pytest.mark.parametrize("structure", [Structure.GROUPEE, Structure.GROUPEE_CHRONOLOGIQUE], ids=str)
def test_I4_classification_groupee_stratifiee_par_machines(rul, structure) -> None:
    """Pannes sur 20 machines sur 100 : test et plis par machines entières, chaque classe partout."""
    df = _pannes_rares(rul, 20)
    table, spec = Table(df, nom="pannes"), _spec_pannes(structure)
    decoupage, coffre = sceller(table, spec)
    app, test = _cotes(coffre, len(df))
    m, y = df["ID_Machine"].to_numpy(), df["Panne"].to_numpy()
    assert decoupage.methode is Methode.UNITES_STRATIFIEES
    assert not set(m[app]) & set(m[test]) and set(y[test]) == {0, 1}
    lignes = coffre.apprentissage()
    for a, v in plis_validation(lignes, spec):
        ma = lignes["ID_Machine"].to_numpy()
        assert not set(ma[a]) & set(ma[v]) and set(lignes["Panne"].to_numpy()[v]) == {0, 1}
    assert controler_decoupage(table, spec, coffre) == ()


def test_I4_classification_groupee_par_le_contrat_signale_le_test_sans_panne(rul) -> None:
    """Méthode `unites` du dépôt : les pannes sur des machines que son tirage laisse à l'apprentissage."""
    unites_test = set(
        rul["ID_Machine"].to_numpy()[positions_test(rul, Methode.UNITES, 0.2, 42, "ID_Machine")]
    )
    choisies = sorted(set(rul["ID_Machine"]) - unites_test)[:10]
    df = rul.assign(
        Panne=((rul["Duree_Vie_Restante_Cycles"] < 10) & rul["ID_Machine"].isin(choisies)).astype(int)
    )
    table, spec = Table(df, nom="pannes"), _spec_pannes()
    _, coffre = sceller(table, spec, methode="unites")
    assert "I7.classe_absente_test" in [c.code for c in controler_decoupage(table, spec, coffre)]
    _, stratifie = sceller(table, spec)
    assert "I7.classe_absente_test" not in [c.code for c in controler_decoupage(table, spec, stratifie)]


def test_I4_pli_groupe_sans_panne_signale(rul) -> None:
    """Plis par machines non stratifiés, pannes sur 3 machines : des plis sans panne, signalés."""
    df = _pannes_rares(rul, 3)
    spec = _spec_pannes(validation="validation.plis_groupes")
    table = Table(df, nom="pannes")
    test = np.flatnonzero(df["ID_Machine"].isin([3, 99, 100]).to_numpy())
    coffre = CoffreTest.par_positions(table, test, decoupage=spec.decoupage())
    assert [c.code for c in controler_decoupage(table, spec, coffre)] == ["I7.classe_absente_pli"]
