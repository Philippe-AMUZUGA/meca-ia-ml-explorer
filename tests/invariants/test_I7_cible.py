"""I7, cible : jamais en entrée ; lignes sans cible retirées et comptées ; cible incohérente avec le problème,
moins de deux classes ou classe plus petite que le nombre de plis : bloquant ; classe rare : alerte.

La stratification par défaut relève du découpage. Les cas dorés sur les jeux autorisés se lancent avec la
variable d'environnement MECAML_JEUX (dossier des classeurs, en lecture seule) ; sans elle, ils sont ignorés.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecaml.core.audit import auditer, proposer_roles, proposer_structure
from mecaml.core.guards import detecter_probleme, lignes_sans_cible
from mecaml.core.io import charger
from mecaml.core.session import (
    Auditer,
    Charger,
    Decouper,
    FixerObjectif,
    Pretraiter,
    Session,
    TransitionRefusee,
)
from mecaml.core.types import ChoixTechnique, CoffreTest, Niveau, Probleme, Role, SpecProjet, Structure, Table


def mesures(n: int = 300, graine: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    temperature = rng.normal(60.0, 5.0, n)
    return pd.DataFrame(
        {
            "temperature": temperature,
            "effort": rng.normal(12.0, 2.0, n),
            "y": 0.3 * temperature + rng.normal(0.0, 0.5, n),
        }
    )


def spec_pour(cible: str, probleme: Probleme, variables=("temperature", "effort"), **kw) -> SpecProjet:
    roles = (*((v, Role.NUMERIQUE) for v in variables), (cible, Role.CIBLE))
    return SpecProjet(probleme, cible, roles, "rmse" if probleme is Probleme.REGRESSION else "f1", **kw)


def constats(df: pd.DataFrame, spec: SpecProjet) -> dict:
    return {c.code: c for c in auditer(Table(df, nom="essai.csv"), spec, None)}


# ============================================================
# LA CIBLE N'EST JAMAIS UNE ENTRÉE
# ============================================================
def test_la_specification_refuse_la_cible_comme_variable() -> None:
    with pytest.raises(ValueError, match="autre rôle"):
        SpecProjet(Probleme.REGRESSION, "y", (("y", Role.NUMERIQUE), ("temperature", Role.NUMERIQUE)), "rmse")
    with pytest.raises(ValueError, match="réservé à la colonne cible"):
        SpecProjet(Probleme.REGRESSION, "y", (("effort", Role.CIBLE), ("y", Role.CIBLE)), "rmse")


def test_les_roles_proposes_ne_mettent_jamais_la_cible_en_variable() -> None:
    t = Table(mesures(), nom="essai.csv")
    roles = proposer_roles(t, "y")
    spec = SpecProjet(Probleme.REGRESSION, "y", roles, "rmse", proposer_structure(roles))
    assert "y" not in spec.variables() and spec.role_de("y") is Role.CIBLE


def test_l_audit_ne_juge_pas_la_cible_comme_une_variable() -> None:
    cs = constats(mesures(), spec_pour("y", Probleme.REGRESSION))
    assert not [c for c in cs if c.startswith(("I3.fuite:y", "I3.soeur:y"))]


# ============================================================
# LIGNES SANS CIBLE : RETIRÉES ET COMPTÉES
# ============================================================
def test_lignes_sans_cible_comptees_et_annoncees_comme_retirees() -> None:
    df = mesures()
    df.loc[[3, 10, 11, 40], "y"] = np.nan
    assert lignes_sans_cible(df, "y") == 4
    c = constats(df, spec_pour("y", Probleme.REGRESSION))["I7.cible_manquante"]
    assert c.niveau is Niveau.ALERTE and "4 ligne(s)" in c.message and "retirées" in c.message


def test_les_lignes_sans_cible_ne_faussent_ni_la_fuite_ni_les_classes() -> None:
    df = mesures()
    df["copie"] = df["y"] * 2.0
    df.loc[::3, "y"] = np.nan
    cs = constats(df, spec_pour("y", Probleme.REGRESSION, ("temperature", "effort", "copie")))
    assert cs["I3.fuite:copie"].bloquant


def test_cible_entierement_manquante_bloquante() -> None:
    df = mesures()
    df["y"] = np.nan
    assert constats(df, spec_pour("y", Probleme.REGRESSION))["I7.cible_vide"].bloquant


# ============================================================
# CIBLE INCOHÉRENTE AVEC LE PROBLÈME
# ============================================================
def test_cible_continue_en_classification_bloquante() -> None:
    c = constats(mesures(), spec_pour("y", Probleme.CLASSIFICATION))["I7.cible_continue"]
    assert c.niveau is Niveau.BLOQUANT and "grandeur continue" in c.message and "régression" in c.action


def test_cible_textuelle_en_regression_bloquante() -> None:
    df = mesures()
    df["etat"] = np.where(df["y"] > df["y"].median(), "haut", "bas")
    assert constats(df, spec_pour("etat", Probleme.REGRESSION))["I7.cible_textuelle"].bloquant


def test_cible_binaire_en_regression_alerte() -> None:
    df = mesures()
    df["panne"] = (df["y"] > df["y"].median()).astype(int)
    assert (
        constats(df, spec_pour("panne", Probleme.REGRESSION))["I7.cible_binaire_en_regression"].niveau
        is Niveau.ALERTE
    )


# ============================================================
# MOINS DE DEUX CLASSES, CLASSE PLUS PETITE QUE LES PLIS
# ============================================================
def test_une_seule_classe_bloquante() -> None:
    df = mesures()
    df["etat"] = "conforme"
    assert constats(df, spec_pour("etat", Probleme.CLASSIFICATION))["I7.moins_de_deux_classes"].bloquant


@pytest.mark.parametrize("n_plis", [3, 5, 10])
def test_classe_plus_petite_que_le_nombre_de_plis_bloquante(n_plis: int) -> None:
    df = mesures()
    df["etat"] = "ok"
    df.loc[: n_plis - 2, "etat"] = "defaut"
    c = constats(df, spec_pour("etat", Probleme.CLASSIFICATION, n_plis=n_plis))[
        "I7.classe_plus_petite_que_les_plis"
    ]
    assert c.niveau is Niveau.BLOQUANT and f"{n_plis} plis" in c.message


def test_classe_assez_grande_pour_les_plis_non_bloquante() -> None:
    df = mesures()
    df["etat"] = "ok"
    df.loc[:19, "etat"] = "defaut"
    cs = constats(df, spec_pour("etat", Probleme.CLASSIFICATION, n_plis=5))
    assert "I7.classe_plus_petite_que_les_plis" not in cs


def test_au_decoupage_la_classe_est_comptee_dans_l_apprentissage_et_bloque_la_session() -> None:
    df = mesures()
    df["etat"] = "ok"
    df.loc[[0, 5, 10, 15, 20, 25, 30, 35], "etat"] = "defaut"  # huit lignes, quatre vont en test
    t = Table(df, nom="essai.csv")
    s = spec_pour("etat", Probleme.CLASSIFICATION, n_plis=5, part_test=0.3, classe_positive="defaut")
    s3 = Session(auditeur=auditer).appliquer(Charger(t)).appliquer(Auditer()).appliquer(FixerObjectif(s))
    assert not s3.bloquants()  # huit lignes, cinq plis : le découpage n'est pas encore connu
    coffre = CoffreTest.par_positions(t, np.array([0, 5, 10, 15, *range(100, 200)]), decoupage=s.decoupage())
    s4 = s3.appliquer(Decouper(coffre))
    assert {c.code for c in s4.bloquants()} == {"I7.classe_plus_petite_que_les_plis", "I7.plis_impossibles"}
    with pytest.raises(TransitionRefusee, match="bloquants"):
        s4.appliquer(Pretraiter(s.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))))
    bon = CoffreTest.par_positions(t, np.array([0, 5, *range(100, 200)]), decoupage=s.decoupage())
    s3b = s4.appliquer(Decouper(bon))
    assert not s3b.bloquants()


# ============================================================
# CLASSE RARE : ALERTE
# ============================================================
def test_classe_rare_comme_panne_a_3_39_pour_cent() -> None:
    df = mesures(10_000)
    df["Panne"] = 0
    df.loc[:338, "Panne"] = 1
    c = constats(df, spec_pour("Panne", Probleme.CLASSIFICATION))["I7.classe_rare"]
    assert c.niveau is Niveau.ALERTE and "3,39" in c.message and "rappel" in c.action


def test_classes_equilibrees_aucune_alerte_de_rarete() -> None:
    df = mesures()
    df["etat"] = np.tile(["a", "b", "c"], len(df) // 3)
    assert "I7.classe_rare" not in constats(df, spec_pour("etat", Probleme.CLASSIFICATION))


# ============================================================
# CAS DORÉS SUR LES JEUX AUTORISÉS
# ============================================================
JEUX = Path(os.environ["MECAML_JEUX"]) if os.environ.get("MECAML_JEUX") else None


def jeu_autorise(identifiant: str) -> Table:
    if JEUX is None or not (JEUX / f"MecaIA-{identifiant}.xlsx").is_file():
        pytest.skip("MECAML_JEUX ne désigne pas un dossier contenant ce jeu")
    return charger(JEUX / f"MecaIA-{identifiant}.xlsx").table


def audit_avec(table: Table, cible: str, probleme: Probleme | None = None) -> dict:
    df = table.donnees
    probleme = probleme or detecter_probleme(df[cible])[0]
    roles = tuple(
        (c, Role.CIBLE if c == cible else Role.NUMERIQUE)
        for c in df.columns
        if c == cible or (pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ("unnamed: 0",))
    )
    spec = SpecProjet(probleme, cible, roles, "rmse" if probleme is Probleme.REGRESSION else "f1")
    return {c.code: c for c in auditer(table, spec, None)}


def test_dore_ai4i_panne_est_une_classe_rare() -> None:
    table = jeu_autorise("AI4I_maintenance")
    assert table.donnees["Panne"].mean() == pytest.approx(0.0339)
    c = audit_avec(table, "Panne")["I7.classe_rare"]
    assert c.niveau is Niveau.ALERTE and "3,39" in c.message


def test_dore_cible_continue_en_classification() -> None:
    table = jeu_autorise("Resistance_Composite")
    cs = audit_avec(table, "resistance_traction_MPa", Probleme.CLASSIFICATION)
    assert cs["I7.cible_continue"].niveau is Niveau.BLOQUANT


def test_dore_cible_textuelle_en_regression() -> None:
    table = jeu_autorise("Controle_Conformite")
    assert audit_avec(table, "etat_piece", Probleme.REGRESSION)["I7.cible_textuelle"].bloquant


def test_dore_classe_rare_de_rebuts() -> None:
    cs = audit_avec(jeu_autorise("Reduction_Rebuts"), "classe_qualite")
    assert cs["I7.classe_rare"].niveau is Niveau.ALERTE


def test_dore_problemes_detectes_sur_les_cibles_des_jeux() -> None:
    attendu = {
        ("Traitement_Acier", "resistance_traction_MPa"): Probleme.REGRESSION,
        ("Qualite_soudure", "qualite_soudure"): Probleme.CLASSIFICATION,
        ("Diagnostic_Vibratoire", "Defaut"): Probleme.CLASSIFICATION,
        ("AI4I_maintenance", "Panne"): Probleme.CLASSIFICATION,
        ("RUL", "RUL_jours"): Probleme.REGRESSION,
        ("Surveillance_Vibratoire", "alarme_criticite"): Probleme.CLASSIFICATION,
        ("TimeSeries_RUL", "Duree_Vie_Restante_Cycles"): Probleme.REGRESSION,
    }
    for (jeu, cible), probleme in attendu.items():
        assert detecter_probleme(jeu_autorise(jeu).donnees[cible])[0] is probleme, (jeu, cible)


# ============================================================
# CLASSE ABSENTE DU TEST OU D'UN PLI : BLOQUANT AU REJEU DE S4 (contrôle du découpage, rejoué par l'audit)
# ============================================================
def classes_a_trois_etats(n: int = 300) -> pd.DataFrame:
    df = mesures(n)
    df["etat"] = "a"
    df.loc[df.index % 5 == 0, "etat"] = "b"
    df["t"] = np.arange(n)
    return df


def audit_au_s4(df: pd.DataFrame, test: np.ndarray, **kw) -> dict:
    roles = (
        ("temperature", Role.NUMERIQUE),
        ("effort", Role.NUMERIQUE),
        ("etat", Role.CIBLE),
        *kw.pop("roles", ()),
    )
    s = SpecProjet(Probleme.CLASSIFICATION, "etat", roles, "exactitude", **kw)
    t = Table(df, nom="essai.csv")
    return {c.code: c for c in auditer(t, s, CoffreTest.par_positions(t, test, decoupage=s.decoupage()))}


def test_classe_absente_du_test_bloquante_au_rejeu_de_s4() -> None:
    df = classes_a_trois_etats()
    test = np.flatnonzero(df["etat"] == "a")[:60]  # aucune ligne « b » au test
    c = audit_au_s4(df, test)["I7.classe_absente_test"]
    assert c.niveau is Niveau.BLOQUANT and "« b »" in c.message and "part de test" in c.action


def test_classe_vue_au_test_seulement_bloquante() -> None:
    df = classes_a_trois_etats()
    df.loc[df.index % 5 == 0, "etat"] = "a"
    df.loc[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "etat"] = "c"
    test = np.arange(10, 20)  # les dix lignes « c » sont toutes au test
    c = audit_au_s4(df, test)["I7.classe_absente_apprentissage"]
    assert c.niveau is Niveau.BLOQUANT and "« c »" in c.message


def test_classe_absente_d_un_pli_de_validation_bloquante() -> None:
    """Série chronologique où la classe « b » n'apparaît qu'à la fin : les premiers plis n'en ont pas."""
    df = mesures(300)
    df["t"] = np.arange(300)
    df["etat"] = np.where(df["t"] < 200, "a", np.where(df["t"] % 3 == 0, "a", "b"))
    test = np.arange(250, 300)  # « a » et « b » y figurent toutes deux
    cs = audit_au_s4(df, test, roles=(("t", Role.DATE),), structure=Structure.CHRONOLOGIQUE)
    c = cs["I7.classe_absente_pli"]
    assert c.niveau is Niveau.BLOQUANT and "pli 1" in c.message and "indéfinies" in c.message
    assert "I7.classe_absente_test" not in cs


def test_classes_presentes_partout_aucun_constat_de_classe_manquante() -> None:
    df = classes_a_trois_etats()
    test = np.arange(0, 300, 4)
    cs = audit_au_s4(df, test)
    assert not [c for c in cs if c.startswith("I7.classe_absente") or c == "I7.plis_impossibles"]


def test_la_session_refuse_d_avancer_tant_qu_une_classe_manque_au_test() -> None:
    df = classes_a_trois_etats()
    t = Table(df, nom="essai.csv")
    s = SpecProjet(
        Probleme.CLASSIFICATION,
        "etat",
        (("temperature", Role.NUMERIQUE), ("effort", Role.NUMERIQUE), ("etat", Role.CIBLE)),
        "exactitude",
    )
    s3 = Session(auditeur=auditer).appliquer(Charger(t)).appliquer(Auditer()).appliquer(FixerObjectif(s))
    mauvais = CoffreTest.par_positions(t, np.flatnonzero(df["etat"] == "a")[:60], decoupage=s.decoupage())
    s4 = s3.appliquer(Decouper(mauvais))
    assert "I7.classe_absente_test" in {c.code for c in s4.bloquants()}
    with pytest.raises(TransitionRefusee, match="bloquants"):
        s4.appliquer(Pretraiter(s.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))))
