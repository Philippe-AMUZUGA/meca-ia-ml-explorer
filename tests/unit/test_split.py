"""Découpage déclaré, coffre scellé selon la structure, mode challenge, plis de validation."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mecaml.core import split
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
    CHAMPS_BASE_TEST,
    GRAINE_CONTRAT,
    Decoupage,
    ErreurDecoupage,
    Methode,
    classes_absentes,
    controler_classes,
    controler_decoupage,
    controler_part_test,
    controler_structure,
    empreinte_positions,
    methode_deduite,
    methodes_permises,
    part_obtenue,
    plis_validation,
    positions_test,
    sceller,
    sceller_base_test,
    sceller_fichier,
    separer_sans_cible,
    spec_pour_base_test,
    validation_appliquee,
    validation_deduite,
)
from mecaml.core.types import (
    LIGNES_MAX_LENTES,
    VALIDATIONS,
    ChoixTechnique,
    CoffreTest,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
    validations_permises,
)

EMPREINTE = "a" * 64
STRUCTURES = list(Structure)


def _spec_cls(spec_reg: SpecProjet, **changements: object) -> SpecProjet:
    """Classification de `y_cls` sur les rôles de la régression type."""
    roles = tuple((c, r) for c, r in spec_reg.roles if c not in ("y_reg", "y_cls"))
    spec = SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="y_cls",
        roles=(*roles, ("y_cls", Role.CIBLE), ("y_reg", Role.IGNOREE)),
        metrique="f1_macro",
        graine=GRAINE_CONTRAT,
    )
    return spec.avec(**changements) if changements else spec


def _test_de(coffre) -> np.ndarray:
    """Positions de test déduites de l'apprentissage (sans ouvrir le coffre)."""
    masque = np.ones(coffre.n_apprentissage + coffre.n_test, dtype=bool)
    masque[coffre.positions_apprentissage()] = False
    return np.flatnonzero(masque)


# ============================================================
# EMPREINTE ET DÉCOUPAGE DÉCLARÉ
# ============================================================
def test_empreinte_positions_forme_du_depot() -> None:
    attendu = hashlib.sha256(b"0,1,2,10").hexdigest()
    assert empreinte_positions([10, 2, 0, 1]) == attendu
    assert empreinte_positions(np.array([1, 0, 10, 2], dtype=np.int32)) == attendu


def test_decoupage_normalise() -> None:
    d = Decoupage("unites", 0.25, np.int64(42), "machine", np.int64(10), EMPREINTE)
    assert d.methode is Methode.UNITES
    assert (type(d.test_size), type(d.random_state), type(d.lignes_test)) == (float, int, int)
    assert d.conforme_contrat
    assert not Decoupage("aleatoire", 0.2, 0, None, 3, EMPREINTE).conforme_contrat


@pytest.mark.parametrize(
    ("champs", "motif"),
    [
        ({"test_size": 0.0}, "test_size"),
        ({"test_size": 1.0}, "test_size"),
        ({"test_size": True}, "test_size"),
        ({"test_size": "0.2"}, "test_size"),
        ({"random_state": -1}, "random_state"),
        ({"random_state": 1.5}, "random_state"),
        ({"random_state": True}, "random_state"),
        ({"colonne_groupe": " "}, "colonne_groupe"),
        ({"methode": "unites", "colonne_groupe": None}, "exige"),
        ({"methode": "chronologique", "colonne_groupe": None}, "exige"),
        ({"lignes_test": 0}, "lignes_test"),
        ({"empreinte_sha256": "A" * 64}, "empreinte"),
        ({"empreinte_sha256": "abc"}, "empreinte"),
        ({"methode": "kfold"}, "inconnue"),
    ],
)
def test_decoupage_refus(champs: dict, motif: str) -> None:
    base = {
        "methode": "aleatoire",
        "test_size": 0.2,
        "random_state": 42,
        "colonne_groupe": None,
        "lignes_test": 5,
        "empreinte_sha256": EMPREINTE,
    }
    with pytest.raises(ErreurDecoupage, match=motif):
        Decoupage(**{**base, **champs})


def test_base_test_aller_retour_json() -> None:
    d = Decoupage(Methode.CHRONOLOGIQUE, 0.25, 42, "time", 2500, EMPREINTE)
    bt = d.vers_base_test()
    assert tuple(bt) == CHAMPS_BASE_TEST
    assert bt["methode"] == "chronologique"
    assert Decoupage.depuis_base_test(json.loads(json.dumps(bt))) == d


@pytest.mark.parametrize(
    ("modifier", "motif"),
    [
        (lambda bt: bt.pop("lignes_test"), "manquants"),
        (lambda bt: bt.update(stratify="y"), "inconnus"),
        (lambda bt: bt.update(methode="fichier"), "n'appartient pas"),
    ],
)
def test_base_test_refus(modifier, motif: str) -> None:
    bt = Decoupage(Methode.ALEATOIRE, 0.2, 42, None, 3, EMPREINTE).vers_base_test()
    modifier(bt)
    with pytest.raises(ErreurDecoupage, match=motif):
        Decoupage.depuis_base_test(bt)
    with pytest.raises(ErreurDecoupage, match="dictionnaire"):
        Decoupage.depuis_base_test([("methode", "aleatoire")])


def test_fichier_ne_s_ecrit_pas_en_base_test() -> None:
    with pytest.raises(ErreurDecoupage, match="à part"):
        Decoupage(Methode.FICHIER, 0.2, 42, None, 3, EMPREINTE).vers_base_test()


# ============================================================
# MÉTHODE DÉDUITE DE LA STRUCTURE
# ============================================================
@pytest.mark.parametrize(
    ("structure", "classification", "attendue"),
    [
        (Structure.INDEPENDANTE, False, Methode.ALEATOIRE),
        (Structure.INDEPENDANTE, True, Methode.STRATIFIE),
        (Structure.GROUPEE, False, Methode.UNITES),
        (Structure.GROUPEE, True, Methode.UNITES_STRATIFIEES),
        (Structure.CHRONOLOGIQUE, True, Methode.CHRONOLOGIQUE),
        (Structure.GROUPEE_CHRONOLOGIQUE, False, Methode.UNITES),
        (Structure.GROUPEE_CHRONOLOGIQUE, True, Methode.UNITES_STRATIFIEES),
    ],
)
def test_methode_deduite(spec_reg, structure, classification, attendue) -> None:
    spec = _spec_cls(spec_reg, structure=structure) if classification else spec_reg.avec(structure=structure)
    assert methode_deduite(spec) is attendue
    assert attendue in methodes_permises(spec)


def test_methodes_permises(spec_reg) -> None:
    assert methodes_permises(spec_reg) == {Methode.ALEATOIRE}
    assert methodes_permises(_spec_cls(spec_reg)) == {Methode.ALEATOIRE, Methode.STRATIFIE}
    assert methodes_permises(spec_reg.avec(structure=Structure.GROUPEE_CHRONOLOGIQUE)) == {Methode.UNITES}
    groupee = _spec_cls(spec_reg, structure=Structure.GROUPEE)
    assert methodes_permises(groupee) == {
        Methode.UNITES_STRATIFIEES,
        Methode.UNITES,
    }  # unites : défis du dépôt


def test_validation_deduite_et_appliquee(spec_reg) -> None:
    attendues = {
        (Structure.INDEPENDANTE, False): "validation.kfold",
        (Structure.INDEPENDANTE, True): "validation.kfold_stratifie",
        (Structure.GROUPEE, False): "validation.plis_groupes",
        (Structure.GROUPEE, True): "validation.plis_groupes_stratifies",
        (Structure.GROUPEE_CHRONOLOGIQUE, True): "validation.plis_groupes_stratifies",
        (Structure.CHRONOLOGIQUE, True): "validation.serie_temporelle",
    }
    for (structure, classification), attendue in attendues.items():
        spec = (
            _spec_cls(spec_reg, structure=structure) if classification else spec_reg.avec(structure=structure)
        )
        assert validation_deduite(spec) == validation_appliquee(spec) == attendue
        assert attendue in validations_permises(spec.structure, spec.probleme)
    imposee = spec_reg.avec(validation="validation.exclusion_unique")
    assert validation_appliquee(imposee) == "validation.exclusion_unique"


def test_toute_validation_du_socle_a_ses_plis() -> None:
    assert set(split._PLIS) == set(VALIDATIONS)


# ============================================================
# SCELLÉ
# ============================================================
@pytest.mark.parametrize("structure", STRUCTURES, ids=str)
def test_sceller_selon_la_structure(table_petite, spec_reg, structure) -> None:
    spec = spec_reg.avec(structure=structure, graine=GRAINE_CONTRAT)
    decoupage, coffre = sceller(table_petite, spec)
    test = _test_de(coffre)
    assert decoupage.methode is methode_deduite(spec)
    assert coffre.decoupage == spec.decoupage()
    assert coffre.ouvertures == 0 and coffre.repose_sur(table_petite)
    assert decoupage.lignes_test == coffre.n_test == test.size
    assert decoupage.empreinte_sha256 == empreinte_positions(test)
    assert (decoupage.test_size, decoupage.random_state) == (spec.part_test, spec.graine)
    assert controler_structure(table_petite, spec, coffre) == ()


def test_sceller_colonne_declaree(table_petite, spec_reg) -> None:
    assert sceller(table_petite, spec_reg.avec(structure=Structure.GROUPEE))[0].colonne_groupe == "machine"
    assert (
        sceller(table_petite, spec_reg.avec(structure=Structure.CHRONOLOGIQUE))[0].colonne_groupe == "cycle"
    )
    assert sceller(table_petite, _spec_cls(spec_reg))[0].colonne_groupe is None  # strate : la cible


def test_sceller_reproductible_et_graine(table_petite, spec_reg) -> None:
    e1 = sceller(table_petite, spec_reg)[0].empreinte_sha256
    e2 = sceller(Table(table_petite.donnees.copy(), nom="copie.csv"), spec_reg)[0].empreinte_sha256
    e3 = sceller(table_petite, spec_reg.avec(graine=7))[0].empreinte_sha256
    assert e1 == e2 != e3


def test_sceller_methode_explicite(table_petite, spec_reg) -> None:
    spec = _spec_cls(spec_reg)
    assert sceller(table_petite, spec, methode="aleatoire")[0].methode is Methode.ALEATOIRE
    with pytest.raises(ErreurDecoupage, match="incompatible"):
        sceller(table_petite, spec, methode=Methode.UNITES)
    with pytest.raises(ErreurDecoupage, match="incompatible"):
        sceller(table_petite, spec_reg.avec(structure=Structure.GROUPEE), methode="aleatoire")


def test_sceller_refus(table_petite, spec_reg, spec_sc, df_petit) -> None:
    with pytest.raises(ErreurDecoupage, match="sans cible"):
        sceller(table_petite, spec_sc)
    with pytest.raises(TypeError, match="SpecProjet"):
        sceller(table_petite, "spec")
    with pytest.raises(ErreurDecoupage, match="absentes"):
        sceller(Table(df_petit.drop(columns="effort"), nom="t.csv"), spec_reg)
    trous = df_petit.copy()
    trous.loc[3, "machine"] = None
    with pytest.raises(ErreurDecoupage, match="incomplète"):
        sceller(Table(trous, nom="t.csv"), spec_reg.avec(structure=Structure.GROUPEE))
    une = df_petit.assign(machine="M1")
    with pytest.raises(ErreurDecoupage, match="Une seule unité"):
        sceller(Table(une, nom="t.csv"), spec_reg.avec(structure=Structure.GROUPEE))
    plate = df_petit.assign(cycle=1)
    with pytest.raises(ErreurDecoupage, match="première"):
        sceller(Table(plate, nom="t.csv"), spec_reg.avec(structure=Structure.CHRONOLOGIQUE))


def test_stratification_impossible_message(df_petit, spec_reg) -> None:
    rare = df_petit.copy()
    rare.loc[0, "y_cls"] = "unique"
    with pytest.raises(ErreurDecoupage, match="Regrouper les classes rares"):
        sceller(Table(rare, nom="t.csv"), _spec_cls(spec_reg))


def test_part_de_test_qui_vide_un_cote() -> None:
    df = pd.DataFrame({"u": ["a", "a", "b"], "t": [1, 2, 3]})
    with pytest.raises(ErreurDecoupage, match="ajuster la part de test"):
        positions_test(df, Methode.ALEATOIRE, 0.99, 42, None)
    with pytest.raises(ErreurDecoupage, match="2 unités ne se partagent pas"):
        positions_test(df, Methode.UNITES, 0.99, 42, "u")


def test_table_trop_petite() -> None:
    with pytest.raises(ErreurDecoupage, match="moins de deux"):
        positions_test(pd.DataFrame({"a": [1]}), Methode.ALEATOIRE, 0.5, 42, None)
    with pytest.raises(ErreurDecoupage, match="fichier"):
        positions_test(pd.DataFrame({"a": [1, 2]}), Methode.FICHIER, 0.5, 42, None)


def test_stratifie_garde_les_proportions() -> None:
    y = np.array(["ok"] * 90 + ["rebut"] * 10)
    test = positions_test(pd.DataFrame({"y": y}), Methode.STRATIFIE, 0.2, 42, "y")
    assert test.size == 20 and (y[test] == "rebut").sum() == 2


def test_stratification_cible_manquante_strate_a_part() -> None:
    y = pd.Series(["a"] * 40 + ["b"] * 40 + [None] * 20)
    test = positions_test(pd.DataFrame({"y": y}), Methode.STRATIFIE, 0.25, 42, "y")
    assert test.size == 25 and y.iloc[test].isna().sum() == 5


def test_chronologique_aucune_date_a_cheval_avec_ex_aequo() -> None:
    dates = np.repeat(np.arange(10), [1, 3, 5, 2, 8, 4, 6, 9, 7, 5])
    df = pd.DataFrame({"t": np.random.default_rng(0).permutation(dates)})
    test = positions_test(df, Methode.CHRONOLOGIQUE, 0.3, 42, "t")
    app = np.setdiff1d(np.arange(len(df)), test)
    assert df["t"].iloc[app].max() < df["t"].iloc[test].min()
    assert test.size >= np.ceil(0.3 * len(df))


@pytest.mark.parametrize(("n", "part", "attendu"), [(10, 0.25, 3), (7, 0.2, 2), (100, 0.2, 20)])
def test_chronologique_rang_de_coupure_arrondi_au_dessus(n, part, attendu) -> None:
    """Dates distinctes : le test compte ceil(part × n) lignes, comme la base du dépôt."""
    df = pd.DataFrame({"t": np.arange(n)[::-1]})
    test = positions_test(df, Methode.CHRONOLOGIQUE, part, 42, "t")
    assert test.tolist() == list(range(attendu))  # les dates les plus tardives, en tête du fichier


# ============================================================
# MODE CHALLENGE : BASE DE TEST DÉCLARÉE
# ============================================================
@pytest.mark.parametrize("structure", STRUCTURES, ids=str)
def test_sceller_base_test_relit_le_decoupage(table_petite, spec_reg, structure) -> None:
    spec = spec_reg.avec(structure=structure, graine=GRAINE_CONTRAT)
    decoupage, coffre = sceller(table_petite, spec)
    rechargee = Table(table_petite.donnees.copy(), nom="publie.xlsx")
    spec_lue, lu, coffre_lu = sceller_base_test(rechargee, spec, decoupage.vers_base_test())
    assert lu == decoupage and spec_lue.decoupage() == spec.decoupage()
    assert np.array_equal(coffre_lu.positions_apprentissage(), coffre.positions_apprentissage())
    assert coffre_lu.ouvertures == 0


def test_sceller_base_test_adapte_le_decoupage(table_petite, spec_reg) -> None:
    d = sceller(table_petite, spec_reg.avec(structure=Structure.GROUPEE, part_test=0.34, graine=42))[0]
    spec_lue, _, coffre = sceller_base_test(table_petite, spec_reg, d)
    assert (spec_lue.structure, spec_lue.part_test, spec_lue.graine) == (Structure.GROUPEE, 0.34, 42)
    assert spec_lue.objectif() == spec_reg.objectif() and coffre.decoupage == spec_lue.decoupage()


@pytest.mark.parametrize(
    ("changement", "motif"),
    [
        ({"lignes_test": 1}, "diffère"),
        ({"empreinte_sha256": EMPREINTE}, "diffère"),
        ({"random_state": 3}, "diffère"),
    ],
)
def test_sceller_base_test_refuse_une_autre_base(table_petite, spec_reg, changement, motif) -> None:
    bt = sceller(table_petite, spec_reg.avec(graine=42))[0].vers_base_test()
    with pytest.raises(ErreurDecoupage, match=motif):
        sceller_base_test(table_petite, spec_reg, {**bt, **changement})


def test_sceller_base_test_refuse_une_table_filtree(table_petite, spec_reg) -> None:
    bt = sceller(table_petite, spec_reg.avec(graine=42))[0].vers_base_test()
    filtree = Table(table_petite.donnees.iloc[1:].reset_index(drop=True), nom="filtree.csv")
    with pytest.raises(ErreurDecoupage, match="sans filtrer"):
        sceller_base_test(filtree, spec_reg, bt)


def test_spec_pour_base_test_roles(spec_reg, spec_sc) -> None:
    unites = Decoupage(Methode.UNITES, 0.2, 42, "materiau", 5, EMPREINTE)
    with pytest.raises(ErreurDecoupage, match="rôle GROUPE"):
        spec_pour_base_test(spec_reg, unites)
    dates = Decoupage(Methode.CHRONOLOGIQUE, 0.2, 42, "machine", 5, EMPREINTE)
    with pytest.raises(ErreurDecoupage, match="rôle DATE"):
        spec_pour_base_test(spec_reg, dates)
    chrono = spec_reg.avec(structure=Structure.CHRONOLOGIQUE)
    sur_machine = Decoupage(Methode.UNITES, 0.2, 42, "machine", 5, EMPREINTE)
    assert spec_pour_base_test(chrono, sur_machine).structure is Structure.GROUPEE_CHRONOLOGIQUE
    hasard = Decoupage(Methode.ALEATOIRE, 0.2, 42, None, 5, EMPREINTE)
    assert spec_pour_base_test(chrono, hasard).structure is Structure.INDEPENDANTE
    with pytest.raises(ErreurDecoupage, match="sceller_fichier"):
        spec_pour_base_test(spec_reg, Decoupage(Methode.FICHIER, 0.2, 42, None, 5, EMPREINTE))
    with pytest.raises(ErreurDecoupage, match="sans cible"):
        spec_pour_base_test(spec_sc, hasard)


def test_sceller_base_test_stratifie_sur_colonne_declaree(table_petite, spec_reg) -> None:
    pos = positions_test(table_petite.donnees, Methode.STRATIFIE, 0.25, 42, "materiau")
    bt = Decoupage(Methode.STRATIFIE, 0.25, 42, "materiau", pos.size, empreinte_positions(pos))
    spec_lue, _, coffre = sceller_base_test(table_petite, spec_reg, bt.vers_base_test())
    assert spec_lue.structure is Structure.INDEPENDANTE and coffre.n_test == pos.size


# ============================================================
# MODE CHALLENGE : TEST FOURNI À PART
# ============================================================
def _deux_tables(df: pd.DataFrame, masque: np.ndarray) -> tuple[Table, Table]:
    return (
        Table(df[~masque].reset_index(drop=True), nom="apprentissage.csv"),
        Table(df[masque].reset_index(drop=True), nom="test.csv"),
    )


def test_sceller_fichier(df_petit, spec_reg) -> None:
    app, test = _deux_tables(df_petit, df_petit["machine"].isin(["M5", "M6"]).to_numpy())
    spec = spec_reg.avec(structure=Structure.GROUPEE)
    decoupage, coffre = sceller_fichier(app, test, spec)
    assert decoupage.methode is Methode.FICHIER and coffre.mode_challenge
    assert decoupage.empreinte_sha256 == test.empreinte == coffre.empreinte_test
    assert decoupage.lignes_test == 20 and decoupage.test_size == pytest.approx(20 / 60)
    assert decoupage.colonne_groupe == "machine" and not decoupage.conforme_contrat
    assert controler_structure(app, spec, coffre) == ()  # contrôlé à la construction


def test_sceller_fichier_refuse_une_structure_violee(df_petit, spec_reg) -> None:
    melange = np.zeros(len(df_petit), dtype=bool)
    melange[::4] = True
    app, test = _deux_tables(df_petit, melange)
    with pytest.raises(ErreurDecoupage, match="unité"):
        sceller_fichier(app, test, spec_reg.avec(structure=Structure.GROUPEE))
    with pytest.raises(ErreurDecoupage, match="Dates à cheval"):
        sceller_fichier(app, test, spec_reg.avec(structure=Structure.CHRONOLOGIQUE))
    tot = df_petit["cycle"].to_numpy() <= 3  # le test précède l'apprentissage
    with pytest.raises(ErreurDecoupage, match="futur"):
        sceller_fichier(*_deux_tables(df_petit, tot), spec_reg.avec(structure=Structure.CHRONOLOGIQUE))
    tard = df_petit["cycle"].to_numpy() > 7
    assert sceller_fichier(*_deux_tables(df_petit, tard), spec_reg.avec(structure=Structure.CHRONOLOGIQUE))


def test_sceller_fichier_refus_de_types(table_petite, spec_reg, spec_sc) -> None:
    with pytest.raises(TypeError, match="Table"):
        sceller_fichier(table_petite, table_petite.donnees, spec_reg)
    with pytest.raises(ErreurDecoupage, match="sans cible"):
        sceller_fichier(table_petite, table_petite, spec_sc)


# ============================================================
# SESSION
# ============================================================
@pytest.mark.parametrize("structure", STRUCTURES, ids=str)
def test_session_accepte_le_coffre_scelle(fabrique_parcours, spec_reg, structure) -> None:
    spec = spec_reg.avec(structure=structure, graine=GRAINE_CONTRAT)
    s3 = fabrique_parcours(spec).jusqua(Etat.OBJECTIF)
    _, coffre = sceller(s3.table, s3.spec)
    assert s3.appliquer(Decouper(coffre)).etat is Etat.DECOUPE


def test_session_mode_challenge_change_le_decoupage(parcours, spec_reg) -> None:
    s3 = parcours.jusqua(Etat.OBJECTIF)
    declare = sceller(s3.table, spec_reg.avec(structure=Structure.GROUPEE, part_test=0.34, graine=42))[0]
    spec_lue, _, coffre = sceller_base_test(s3.table, s3.spec, declare.vers_base_test())
    s4 = s3.appliquer(Decouper(coffre, spec=spec_lue))
    assert s4.etat is Etat.DECOUPE and s4.spec.structure is Structure.GROUPEE


# ============================================================
# PLIS DE VALIDATION CROISÉE
# ============================================================
def _verifier_partition(plis, n: int) -> None:
    for app, val in plis:
        assert np.intersect1d(app, val).size == 0
        assert app.max() < n and val.max() < n


def test_plis_independants_couvrent_une_fois(df_petit, spec_reg) -> None:
    plis = plis_validation(df_petit, spec_reg)
    _verifier_partition(plis, len(df_petit))
    assert len(plis) == spec_reg.n_plis
    assert np.array_equal(np.sort(np.concatenate([v for _, v in plis])), np.arange(len(df_petit)))
    assert all(
        np.array_equal(a, b) for (a, _), (b, _) in zip(plis, plis_validation(df_petit, spec_reg), strict=True)
    )


def test_plis_stratifies(df_petit, spec_reg) -> None:
    plis = plis_validation(df_petit, _spec_cls(spec_reg))
    for _, val in plis:
        assert (df_petit["y_cls"].iloc[val] == "conforme").sum() == len(val) // 2


@pytest.mark.parametrize("structure", [Structure.GROUPEE, Structure.GROUPEE_CHRONOLOGIQUE], ids=str)
def test_plis_par_unites_entieres(df_petit, spec_reg, structure) -> None:
    plis = plis_validation(df_petit, spec_reg.avec(structure=structure, n_plis=3))
    _verifier_partition(plis, len(df_petit))
    m = df_petit["machine"].to_numpy()
    assert all(not set(m[a]) & set(m[v]) for a, v in plis)


def test_plis_chronologiques_valident_le_futur(df_petit, spec_reg) -> None:
    plis = plis_validation(df_petit, spec_reg.avec(structure=Structure.CHRONOLOGIQUE, n_plis=4))
    c = df_petit["cycle"].to_numpy()
    assert len(plis) == 4
    for app, val in plis:
        assert c[app].max() < c[val].min()
    assert [len(a) for a, _ in plis] == sorted(len(a) for a, _ in plis)  # fenêtre croissante


def test_plis_refus(df_petit, spec_reg) -> None:
    with pytest.raises(ErreurDecoupage, match="6 unités pour 7 plis"):
        plis_validation(df_petit, spec_reg.avec(structure=Structure.GROUPEE, n_plis=7))
    with pytest.raises(ErreurDecoupage, match="10 dates distinctes pour 10 plis"):
        plis_validation(df_petit, spec_reg.avec(structure=Structure.CHRONOLOGIQUE, n_plis=10))
    rare = df_petit.copy()
    rare.loc[:2, "y_cls"] = "rare"
    with pytest.raises(ErreurDecoupage, match="classe la plus rare compte 3"):
        plis_validation(rare, _spec_cls(spec_reg))
    with pytest.raises(ErreurDecoupage, match="4 lignes pour 5 plis"):
        plis_validation(df_petit.head(4), spec_reg)
    with pytest.raises(TypeError, match="SpecProjet"):
        plis_validation(df_petit, None)


def test_plis_sans_cible(df_petit, spec_sc) -> None:
    assert len(plis_validation(df_petit, spec_sc)) == spec_sc.n_plis


def test_controler_structure_sans_objet(table_petite, spec_reg, coffre_de) -> None:
    coffre = coffre_de(table_petite)
    assert controler_structure(table_petite, None, coffre) == ()
    assert controler_structure(table_petite, spec_reg, None) == ()
    sans_colonne = spec_reg.avec(
        roles=tuple((c, r) for c, r in spec_reg.roles if r not in (Role.GROUPE, Role.DATE))
    )
    assert controler_structure(table_petite, sans_colonne, coffre) == ()


# ============================================================
# VALIDATION IMPOSÉE PAR LA SPÉCIFICATION
# ============================================================
def test_validations_imposees(df_petit, spec_reg) -> None:
    exclusion = plis_validation(df_petit, spec_reg.avec(validation="validation.exclusion_unique"))
    assert len(exclusion) == len(df_petit) and all(len(v) == 1 for _, v in exclusion)
    aleatoires = plis_validation(
        df_petit, spec_reg.avec(structure=Structure.GROUPEE, validation="validation.groupes_aleatoires")
    )
    m = df_petit["machine"].to_numpy()
    assert len(aleatoires) == 5 and all(not set(m[a]) & set(m[v]) for a, v in aleatoires)
    groupee = spec_reg.avec(structure=Structure.GROUPEE)
    imposee = groupee.avec(validation="validation.plis_groupes")
    assert all(
        np.array_equal(a1, a2) and np.array_equal(v1, v2)
        for (a1, v1), (a2, v2) in zip(
            plis_validation(df_petit, groupee), plis_validation(df_petit, imposee), strict=True
        )
    )


def test_exclusion_unique_bornee() -> None:
    df = pd.DataFrame({"x": np.zeros(LIGNES_MAX_LENTES + 1), "y": np.zeros(LIGNES_MAX_LENTES + 1)})
    spec = SpecProjet(
        Probleme.REGRESSION,
        "y",
        (("x", Role.NUMERIQUE), ("y", Role.CIBLE)),
        "rmse",
        validation="validation.exclusion_unique",
    )
    with pytest.raises(ErreurDecoupage, match="coupée au delà de 20 000"):  # message unique de guards
        plis_validation(df, spec)


def test_validation_imposee_incompatible_avec_la_base_declaree(spec_reg) -> None:
    unites = Decoupage(Methode.UNITES, 0.2, 42, "machine", 5, EMPREINTE)
    with pytest.raises(ErreurDecoupage, match=r"validation imposée validation\.kfold ne convient pas"):
        spec_pour_base_test(spec_reg.avec(validation="validation.kfold"), unites)
    groupee = spec_pour_base_test(spec_reg.avec(validation=None), unites)
    assert groupee.structure is Structure.GROUPEE and groupee.validation is None


# ============================================================
# CLASSIFICATION GROUPÉE : TEST ET PLIS STRATIFIÉS PAR UNITÉS (I7)
# ============================================================
def _parc(en_panne: set[str], n_machines: int = 20, cycles: int = 10) -> pd.DataFrame:
    """Machines suivies sur `cycles` cycles ; panne aux trois derniers cycles des machines désignées."""
    machine = np.repeat([f"M{i:02d}" for i in range(n_machines)], cycles)
    cycle = np.tile(np.arange(cycles), n_machines)
    panne = np.isin(machine, sorted(en_panne)) & (cycle >= cycles - 3)
    x = np.random.default_rng(1).normal(size=machine.size) + panne
    return pd.DataFrame({"machine": machine, "cycle": cycle, "x": x, "panne": panne.astype(int)})


def _spec_parc(**changements: object) -> SpecProjet:
    roles = (("machine", Role.GROUPE), ("cycle", Role.DATE), ("x", Role.NUMERIQUE), ("panne", Role.CIBLE))
    return SpecProjet(Probleme.CLASSIFICATION, "panne", roles, "f1_macro", Structure.GROUPEE, **changements)


def _machines_test(df: pd.DataFrame, coffre: CoffreTest) -> set[str]:
    return set(df["machine"].to_numpy()[_test_de(coffre)])


def test_unites_stratifiees_test_avec_chaque_classe() -> None:
    df = _parc({"M01", "M05", "M11", "M17"})
    table = Table(df, nom="parc")
    spec = _spec_parc(n_plis=3)  # une machine en panne au test, trois à l'apprentissage
    decoupage, coffre = sceller(table, spec)
    test = _test_de(coffre)
    assert decoupage.methode is Methode.UNITES_STRATIFIEES and decoupage.colonne_groupe == "machine"
    assert set(df["panne"].to_numpy()[test]) == {0, 1}
    assert not _machines_test(df, coffre) & set(df["machine"].to_numpy()[coffre.positions_apprentissage()])
    assert controler_decoupage(table, spec, coffre) == ()
    assert not decoupage.conforme_contrat
    with pytest.raises(ErreurDecoupage, match="méthode « unites » du dépôt"):
        decoupage.vers_base_test()
    with pytest.raises(ErreurDecoupage, match="n'appartient pas au contrat"):
        Decoupage.depuis_base_test(
            {
                **sceller(table, _spec_parc(), methode="unites")[0].vers_base_test(),
                "methode": "unites_stratifiees",
            }
        )


def test_unites_du_contrat_sans_positif_au_test_signale() -> None:
    """Les pannes sont mises sur des machines que le tirage du dépôt laisse à l'apprentissage."""
    base = _parc(set())
    test = set(base["machine"].to_numpy()[positions_test(base, Methode.UNITES, 0.2, 42, "machine")])
    en_panne = set(sorted(set(base["machine"]) - test)[:4])
    table = Table(_parc(en_panne), nom="parc")
    spec = _spec_parc(n_plis=4)  # quatre machines en panne à l'apprentissage : une par pli
    _, coffre = sceller(table, spec, methode=Methode.UNITES)
    constats = controler_classes(table, spec, coffre)
    assert [c.code for c in constats] == ["I7.classe_absente_test"]
    assert constats[0].bloquant and "« 1 »" in constats[0].message and constats[0].ref_livre == "T02-1-1"
    _, stratifie = sceller(table, spec)
    assert set(table.donnees["panne"].to_numpy()[_test_de(stratifie)]) == {0, 1}
    assert "I7.classe_absente_test" not in [c.code for c in controler_classes(table, spec, stratifie)]


def test_session_bloque_un_test_sans_classe() -> None:
    base = _parc(set())
    test = set(base["machine"].to_numpy()[positions_test(base, Methode.UNITES, 0.2, 42, "machine")])
    table = Table(_parc(set(sorted(set(base["machine"]) - test)[:4])), nom="parc")
    spec = _spec_parc(n_plis=3)
    s3 = Session(auditeur=controler_decoupage).appliquer(Charger(table)).appliquer(Auditer())
    s3 = s3.appliquer(FixerObjectif(spec))
    s4 = s3.appliquer(Decouper(sceller(table, spec, methode="unites")[1]))
    assert [c.code for c in s4.bloquants()] == ["I7.classe_absente_test"]
    with pytest.raises(TransitionRefusee, match="classe_absente_test"):
        s4.appliquer(Pretraiter(spec.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))))
    assert s3.appliquer(Decouper(sceller(table, spec)[1])).bloquants() == ()


def test_classe_presente_au_test_seulement() -> None:
    df = _parc({"M03"})
    table = Table(df, nom="parc")
    test = np.flatnonzero(df["machine"].isin(["M03", "M04"]).to_numpy())
    coffre = CoffreTest.par_positions(table, test, decoupage=_spec_parc().decoupage())
    codes = [c.code for c in controler_classes(table, _spec_parc(), coffre)]
    assert "I7.classe_absente_apprentissage" in codes


def _coffre_parc(
    df: pd.DataFrame, spec: SpecProjet, test_machines: tuple[str, ...]
) -> tuple[Table, CoffreTest]:
    table = Table(df, nom="parc")
    test = np.flatnonzero(df["machine"].isin(test_machines).to_numpy())
    return table, CoffreTest.par_positions(table, test, decoupage=spec.decoupage())


def test_pli_sans_positif_signale() -> None:
    df = _parc({"M00", "M01", "M18"})
    spec = _spec_parc(validation="validation.plis_groupes")
    table, coffre = _coffre_parc(df, spec, ("M18", "M19"))
    constats = controler_classes(table, spec, coffre)
    assert [c.code for c in constats] == ["I7.classe_absente_pli"]
    assert "validation sans « 1 »" in constats[0].message and constats[0].ref_livre == "T02-2-1"
    deux = _spec_parc(n_plis=2)  # deux machines en panne à l'apprentissage : deux plis stratifiés suffisent
    table, coffre = _coffre_parc(df, deux, ("M18", "M19"))
    assert controler_classes(table, deux, coffre) == ()


def test_plis_impossibles_signales() -> None:
    spec = _spec_parc(n_plis=19)
    table, coffre = _coffre_parc(_parc({"M00", "M01", "M18"}), spec, ("M18", "M19"))
    constats = controler_classes(table, spec, coffre)
    assert [c.code for c in constats] == ["I7.plis_impossibles"] and "18 unités pour 19 plis" in constats[
        0
    ].message


def test_plis_groupes_stratifies_par_defaut() -> None:
    df = _parc({"M01", "M05", "M11", "M17"})
    spec = _spec_parc(n_plis=4)
    plis = plis_validation(df, spec)
    m, y = df["machine"].to_numpy(), df["panne"].to_numpy()
    assert len(plis) == 4
    for app, val in plis:
        assert not set(m[app]) & set(m[val]) and set(y[val]) == {0, 1}
    assert all(
        np.array_equal(v1, v2) for (_, v1), (_, v2) in zip(plis, plis_validation(df, spec), strict=True)
    )
    assert plis_validation(df, spec.avec(graine=7))[0][1].tolist() != plis[0][1].tolist()


def test_exclusion_unique_sans_constat_de_pli(table_petite, spec_reg) -> None:
    spec = _spec_cls(spec_reg, validation="validation.exclusion_unique")
    _, coffre = sceller(table_petite, spec)
    assert controler_classes(table_petite, spec, coffre) == ()


def test_controler_classes_sans_objet(table_petite, spec_reg, df_petit) -> None:
    _, coffre = sceller(table_petite, spec_reg)
    assert controler_classes(table_petite, spec_reg, coffre) == ()  # régression
    spec = _spec_cls(spec_reg)
    assert controler_classes(table_petite, spec, None) == () == controler_classes(table_petite, None, coffre)
    app, test = Table(df_petit.iloc[:40], nom="a.csv"), Table(df_petit.iloc[40:], nom="t.csv")
    _, fichier = sceller_fichier(app, test, spec)
    assert controler_classes(app, spec, fichier) == ()  # test imposé, non rouvert


def test_controler_decoupage_reunit_les_deux_controles() -> None:
    spec = _spec_parc()
    df = _parc({"M01", "M05"})
    table = Table(df, nom="parc")
    a_cheval = CoffreTest.par_positions(table, np.arange(0, len(df), 5), decoupage=spec.decoupage())
    tout = controler_decoupage(table, spec, a_cheval)
    assert tout == controler_structure(table, spec, a_cheval) + controler_classes(table, spec, a_cheval)
    assert "I4.unites_a_cheval" in [c.code for c in tout]


def test_classes_absentes_par_pli() -> None:
    y = np.array([0, 0, 1, 1, 0, 0])
    plis = ((np.array([0, 1, 2, 3]), np.array([4, 5])), (np.array([0, 1, 4, 5]), np.array([2, 3])))
    assert classes_absentes(y, plis) == ((frozenset(), frozenset({1})), (frozenset({1}), frozenset({0})))


def test_unites_stratifiees_refus() -> None:
    peu = _parc({"M00"}, n_machines=3)
    with pytest.raises(
        ErreurDecoupage, match=r"3 unités pour une part de test de 0\.2 : il en faut au moins 5"
    ):
        positions_test(peu, Methode.UNITES_STRATIFIEES, 0.2, 42, "machine", strate="panne")
    rare = _parc({"M00"})  # trois lignes en panne pour cinq parts
    with pytest.raises(ErreurDecoupage, match="classe la plus rare compte 3 lignes pour 5 parts"):
        positions_test(rare, Methode.UNITES_STRATIFIEES, 0.2, 42, "machine", strate="panne")


def test_cible_manquante_isolee_rejoint_la_classe_majoritaire() -> None:
    y = pd.Series(["a"] * 40 + ["b"] * 40 + [None])
    test = positions_test(pd.DataFrame({"y": y}), Methode.STRATIFIE, 0.25, 42, "y")
    assert test.size == 21  # ceil(0,25 × 81), sans échec de la strate isolée


def test_separer_sans_cible(df_petit) -> None:
    trous = df_petit.copy()
    trous.loc[[5, 9], "y_reg"] = np.nan
    avec, sans = separer_sans_cible(trous, "y_reg")
    assert len(avec) == 58 and list(sans.index) == [5, 9]


# ============================================================
# PART DE TEST DES UNITÉS STRATIFIÉES, PART OBTENUE DITE PAR UN CONSTAT D'INFORMATION
# ============================================================
def _decoupe_parc(part_test: float, n_machines: int = 20):
    df = _parc({"M01", "M05", "M11", "M17"}, n_machines=n_machines)
    table = Table(df, nom="parc")
    spec = _spec_parc(n_plis=3, part_test=part_test)
    decoupage, coffre = sceller(table, spec)
    return df, table, spec, decoupage, coffre


def test_une_part_de_test_de_0_4_met_40_pour_cent_des_lignes_en_test_et_non_50() -> None:
    """Une part de 0,4 ne met pas la moitié des lignes en test (`round(1 / 0,4)` donnait deux parts)."""
    df, _, _, decoupage, coffre = _decoupe_parc(0.4)
    assert decoupage.methode is Methode.UNITES_STRATIFIEES and decoupage.test_size == 0.4
    assert part_obtenue(coffre) == pytest.approx(0.4, abs=0.05)  # deux parts d'unités sur cinq
    assert set(df["panne"].to_numpy()[_test_de(coffre)]) == {0, 1}


@pytest.mark.parametrize("part", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7])
def test_la_part_obtenue_est_a_cinq_points_au_plus_de_la_part_visee(part: float) -> None:
    df, table, spec, _, coffre = _decoupe_parc(part)
    assert abs(part_obtenue(coffre) - part) <= 0.05 + 1e-9, (part, part_obtenue(coffre))
    assert set(df["panne"].to_numpy()[_test_de(coffre)]) == {0, 1}  # les deux classes sont au test
    assert not _machines_test(df, coffre) & set(df["machine"].to_numpy()[coffre.positions_apprentissage()])
    assert controler_structure(table, spec, coffre) == ()  # aucune unité à cheval


def test_la_part_de_0_2_garde_le_tirage_d_une_part_sur_cinq() -> None:
    """Une part sur cinq, comme avant : la réunion d'une seule part est l'ancien choix, au bit près."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = _parc({"M01", "M05", "M11", "M17"})
    classes, unites = df["panne"].to_numpy(), df["machine"]
    tirage = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    parts = [val for _, val in tirage.split(np.arange(len(df)), classes, groups=unites)]
    ancien = max(parts, key=lambda v: (len(np.unique(classes[v])), -abs(v.size - 0.2 * len(df))))
    nouveau = positions_test(df, Methode.UNITES_STRATIFIEES, 0.2, 42, "machine", strate="panne")
    assert np.array_equal(np.sort(ancien), nouveau)


def test_une_part_hors_de_portee_est_refusee_avec_les_parts_possibles() -> None:
    parc = _parc({"M01", "M05", "M11", "M17"})
    with pytest.raises(ErreurDecoupage, match=r"impossible par unités stratifiées") as erreur:
        positions_test(parc, Methode.UNITES_STRATIFIEES, 0.02, 42, "machine", strate="panne")
    message = str(erreur.value)
    assert "Parts les plus proches" in message and "0,10 (1 part sur 10)" in message
    assert "Choisir l'une d'elles" in message
    with pytest.raises(ErreurDecoupage, match=r"impossible par unités stratifiées"):
        positions_test(parc, Methode.UNITES_STRATIFIEES, 0.97, 42, "machine", strate="panne")


def test_une_part_que_les_donnees_ne_permettent_pas_se_replie_sur_la_plus_proche_possible() -> None:
    """Huit machines : dix parts sont impossibles pour 0,3, la part la plus proche qui l'est est retenue."""
    df = _parc({"M01", "M05"}, n_machines=8)
    test = positions_test(df, Methode.UNITES_STRATIFIEES, 0.3, 42, "machine", strate="panne")
    assert abs(len(test) / len(df) - 0.3) <= 0.1  # une machine pèse 12,5 % : trois sur huit au plus proche
    assert set(df["panne"].to_numpy()[test]) == {0, 1}
    assert split._fractions(0.3, split.ECART_PART_MAX)[0] == (3, 10)  # la plus proche à l'absolu : dix parts


def test_la_part_obtenue_se_lit_sur_le_coffre() -> None:
    table = Table(_parc(set()), nom="parc")
    coffre = CoffreTest.par_positions(table, np.arange(0, 200, 5), decoupage=_spec_parc().decoupage())
    assert part_obtenue(coffre) == pytest.approx(0.2)


def test_un_ecart_d_un_point_au_moins_est_dit_par_un_constat_d_information() -> None:
    table = Table(_parc(set()), nom="parc")
    spec = _spec_parc(part_test=0.2)
    conforme = CoffreTest.par_positions(table, np.arange(0, 200, 5), decoupage=spec.decoupage())
    assert controler_part_test(spec, conforme) == ()
    plus_grand = CoffreTest.par_positions(table, np.arange(0, 200, 4), decoupage=spec.decoupage())
    (constat,) = controler_part_test(spec, plus_grand)
    assert constat.code == "DECOUPAGE.part_test_obtenue" and constat.niveau.value == "info"
    assert "20,0 % visés, 25,0 % obtenus" in constat.message and "50 lignes sur 200" in constat.message
    assert constat.ref_livre == "T02-1-1" and not constat.bloquant
    assert controler_part_test(None, plus_grand) == controler_part_test(spec, None) == ()
    codes = [c.code for c in controler_decoupage(table, spec, plus_grand)]
    assert "DECOUPAGE.part_test_obtenue" not in codes  # une information : l'audit l'ajoute à part
    app, test = Table(_parc(set()).iloc[:150], nom="a.csv"), Table(_parc(set()).iloc[150:], nom="t.csv")
    _, fichier = sceller_fichier(app, test, spec)
    assert controler_part_test(spec, fichier) == ()  # un test fourni à part n'a pas de part visée


def test_l_audit_du_decoupage_dit_la_part_obtenue() -> None:
    from mecaml.core.audit import auditer

    df = _parc({"M01", "M05", "M11", "M17"})
    table = Table(df, nom="parc")
    spec = _spec_parc(n_plis=3, part_test=0.27)
    _, coffre = sceller(table, spec)
    constats = {c.code: c for c in auditer(table, spec, coffre)}
    constat = constats["DECOUPAGE.part_test_obtenue"]
    assert constat.niveau.value == "info" and "27,0 % visés" in constat.message
    assert not [c for c in constats.values() if c.bloquant]


def test_le_nombre_de_plis_possible_suit_les_unites_les_dates_et_la_classe_rare() -> None:
    import aides_reproductibilite as aides
    from mecaml.core.types import Probleme

    serie = aides.jeu_chronologique(n=30)
    assert split.plis_au_plus(serie, aides.spec_chronologique(None)) == 29  # dates distinctes moins une
    parc = pd.DataFrame(
        {"machine": ["A", "A", "B", "B", "C", "C", "D", "D"], "panne": [0, 1, 0, 0, 1, 0, 0, 0]}
    )
    spec = SpecProjet(
        Probleme.CLASSIFICATION,
        "panne",
        (("machine", Role.GROUPE), ("panne", Role.CIBLE)),
        "f1",
        Structure.GROUPEE,
    )
    assert split.plis_au_plus(parc, spec) == 2  # quatre unités, mais seulement deux lignes de la classe rare
    assert split.plis_au_plus(parc.iloc[:2], spec) == 1
