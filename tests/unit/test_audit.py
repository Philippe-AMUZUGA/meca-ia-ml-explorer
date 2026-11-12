"""Profil des colonnes et audit : rôles proposés, constats par thème, niveaux, codes, coffre."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core.audit import (
    PropositionRoles,
    auditer,
    mots,
    preuve_soeur,
    probleme_propose,
    profiler,
    proposer_cible,
    proposer_roles,
    proposer_structure,
    proposition_roles,
    scores_fuite,
)
from mecaml.core.io import ATTR_FORMULES, ATTR_LIGNES_VIDES, ATTR_NOTES
from mecaml.core.session import (
    Acquitter,
    Auditer,
    Charger,
    Decouper,
    Etat,
    FixerObjectif,
    Pretraiter,
    Revenir,
    Session,
    TransitionRefusee,
)
from mecaml.core.types import (
    ChoixTechnique,
    CoffreTest,
    Constat,
    Niveau,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
)

EXEMPLES = Path(__file__).resolve().parents[2] / "examples"
N = 240
CYCLES = N // 12
ROLES_DE_BASE = (
    ("id_piece", Role.IDENTIFIANT),
    ("temperature", Role.NUMERIQUE),
    ("effort", Role.NUMERIQUE),
    ("materiau", Role.CATEGORIELLE),
    ("y_reg", Role.CIBLE),
    ("y_cls", Role.IGNOREE),
)


def donnees(n: int = N, graine: int = 0) -> pd.DataFrame:
    """Douze machines suivies sur vingt cycles : identifiant, groupe, ordre, mesures, deux cibles."""
    rng = np.random.default_rng(graine)
    temperature = rng.normal(60.0, 5.0, n)
    effort = rng.normal(12.0, 2.0, n)
    y = 0.3 * temperature + 1.5 * effort + rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "id_piece": np.arange(n),
            "machine": np.repeat([f"M{i}" for i in range(12)], n // 12),
            "cycle": np.tile(np.arange(n // 12), 12),
            "temperature": temperature,
            "effort": effort,
            "materiau": rng.choice(["acier", "alu", "inox"], n),
            "y_reg": y,
            "y_cls": np.where(y > np.median(y), "haut", "bas"),
        }
    )


def tab(df: pd.DataFrame) -> Table:
    return Table(df, nom="essai.csv")


def spec(
    roles=ROLES_DE_BASE,
    probleme: Probleme = Probleme.REGRESSION,
    cible: str | None = "y_reg",
    structure: Structure = Structure.INDEPENDANTE,
    **kw,
) -> SpecProjet:
    metrique = (
        None if probleme is Probleme.SANS_CIBLE else ("rmse" if probleme is Probleme.REGRESSION else "f1")
    )
    return SpecProjet(
        probleme=probleme, cible=cible, roles=tuple(roles), metrique=metrique, structure=structure, **kw
    )


def par_code(constats) -> dict[str, Constat]:
    return {c.code: c for c in constats}


def niveaux(constats) -> dict[str, Niveau]:
    return {c.code: c.niveau for c in constats}


# ============================================================
# MOTS ET PROFIL
# ============================================================
def test_mots_des_noms_de_colonnes() -> None:
    assert mots("Piece_ID") == ("piece", "id")
    assert mots("ID_Machine") == ("id", "machine")
    assert mots("idPiece") == ("id", "piece")
    assert mots("Unnamed: 0") == ("unnamed", "0")
    assert mots("Température_moy_°C") == ("temperature", "moy", "c")
    assert mots("Stress/MPa") == ("stress", "mpa")


def frame_de_profil() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 60
    return pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "id_lot": [f"L{i:03d}" for i in range(n)],
            "ID_Machine": np.repeat(["a", "b", "c"], n // 3),
            "time": np.arange(n) * 0.5,
            "Cycle_Operation": np.tile(np.arange(n // 3), 3),
            "temps_cycle_min": rng.normal(10, 2, n),
            "force_N": rng.normal(100, 5, n),
            "materiau": rng.choice(["acier", "alu"], n),
            "commentaire": [f"remarque numéro {i}" for i in range(n)],
            "constante": 5.0,
            "vide": np.nan,
            "actif": rng.choice([True, False], n),
            "jour": pd.date_range("2026-01-01", periods=n),
            "unique_flottante": rng.normal(size=n),
            "nb_passes": rng.integers(0, 5, n),
        }
    )


def test_roles_proposes() -> None:
    profils = {p.nom: p for p in profiler(tab(frame_de_profil()))}
    attendu = {
        "Unnamed: 0": Role.IDENTIFIANT,
        "id_lot": Role.IDENTIFIANT,
        "ID_Machine": Role.GROUPE,
        "time": Role.DATE,
        "Cycle_Operation": Role.DATE,
        "temps_cycle_min": Role.NUMERIQUE,  # le mot « temps » ne suffit pas : l'ordre doit croître
        "force_N": Role.NUMERIQUE,
        "materiau": Role.CATEGORIELLE,
        "commentaire": Role.IDENTIFIANT,
        "constante": Role.IGNOREE,
        "vide": Role.IGNOREE,
        "actif": Role.CATEGORIELLE,
        "jour": Role.DATE,
        "unique_flottante": Role.NUMERIQUE,  # une grandeur continue n'est jamais un identifiant
        "nb_passes": Role.NUMERIQUE,
    }
    assert {nom: p.role_propose for nom, p in profils.items()} == attendu
    assert (
        profils["id_lot"].preuve_identifiant == "nom"
        and profils["commentaire"].preuve_identifiant == "unicite"
    )
    assert all(p.raison for p in profils.values())


def test_valeurs_du_profil() -> None:
    p = {p.nom: p for p in profiler(tab(frame_de_profil()))}
    assert (p["force_N"].famille, p["materiau"].famille, p["jour"].famille, p["actif"].famille) == (
        "nombre",
        "texte",
        "date",
        "booleen",
    )
    assert p["vide"].famille == "vide" and p["vide"].constante and p["vide"].n_manquants == 60
    assert p["materiau"].modalites and set(p["materiau"].modalites) <= {"acier", "alu"}
    assert p["force_N"].minimum < p["force_N"].moyenne < p["force_N"].maximum
    assert p["unique_flottante"].reelle and not p["nb_passes"].reelle and p["nb_passes"].n_distincts <= 5


def test_nombres_ecrits_en_texte_reconnus() -> None:
    df = pd.DataFrame({"a": ["12,5", "13,5", "11,0", "9,75"] * 5, "b": ["x", "y", "z", "w"] * 5})
    p = {p.nom: p for p in profiler(tab(df))}
    assert p["a"].nombres_en_texte and not p["b"].nombres_en_texte


def test_profil_sur_les_positions_demandees() -> None:
    t = tab(donnees())
    p = {p.nom: p for p in profiler(t, np.arange(10))}
    assert p["temperature"].n_lignes == 10


@pytest.mark.parametrize(
    ("colonnes", "attendu"),
    [
        (["time", "force", "Panne"], "Panne"),
        (["time", "Panne", "Panne_outil", "Panne_alimentation"], "Panne"),
        (["a", "b", "resultat"], "resultat"),
        (["classe_qualite", "x", "y"], "classe_qualite"),
        (["cible", "x", "y"], "cible"),
    ],
)
def test_cible_proposee(colonnes: list[str], attendu: str) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=30) for c in colonnes})
    assert proposer_cible(profiler(tab(df))) == attendu


def test_aucune_cible_proposee_si_rien_ne_se_distingue() -> None:
    df = pd.DataFrame({"id": range(30), "jour": pd.date_range("2026-01-01", periods=30)})
    assert proposer_cible(profiler(tab(df))) is None


def test_structure_proposee_par_les_roles() -> None:
    assert proposer_structure([("a", Role.NUMERIQUE)]) is Structure.INDEPENDANTE
    assert proposer_structure([("a", Role.GROUPE)]) is Structure.GROUPEE
    assert proposer_structure([("a", Role.DATE)]) is Structure.CHRONOLOGIQUE
    assert proposer_structure([("a", Role.DATE), ("b", Role.GROUPE)]) is Structure.GROUPEE_CHRONOLOGIQUE


def test_probleme_propose() -> None:
    t = tab(donnees())
    assert probleme_propose(t, "y_cls") is Probleme.CLASSIFICATION
    assert probleme_propose(t, "y_reg") is Probleme.REGRESSION


def test_roles_proposes_donnent_une_specification_sans_constat_bloquant() -> None:
    df = donnees()
    df["copie_cible"] = df["y_reg"] * 3.0 + 1.0
    df["cluster_gmm"] = np.arange(len(df)) % 3
    df["operateur_genre"] = np.where(np.arange(len(df)) % 2, "F", "H")
    df["y_reg_corrige"] = df["y_reg"] + 0.1
    df["constante"] = 7.0
    t = tab(df)
    roles = dict(proposer_roles(t, "y_reg"))
    for colonne in ("copie_cible", "cluster_gmm", "operateur_genre", "y_reg_corrige", "constante"):
        assert roles[colonne] is Role.IGNOREE
    assert roles["y_reg"] is Role.CIBLE and roles["id_piece"] is Role.IDENTIFIANT
    assert roles["machine"] is Role.GROUPE and roles["cycle"] is Role.DATE  # série par machine : unité suivie
    s = spec(tuple(roles.items()), structure=proposer_structure(tuple(roles.items())))
    assert not [c for c in auditer(t, s, None) if c.bloquant or c.niveau is Niveau.ALERTE]


def test_roles_proposes_une_seule_date_et_un_seul_groupe() -> None:
    df = pd.DataFrame(
        {
            "g1": np.repeat(["a", "b"], 20),
            "id_g2": np.tile(["u", "v"], 20),
            "t1": np.arange(40.0),
            "time": np.arange(40.0),
        }
    )
    roles = dict(proposer_roles(tab(df)))
    assert list(roles.values()).count(Role.DATE) == 1 and list(roles.values()).count(Role.GROUPE) <= 1


def test_roles_proposes_cible_inconnue() -> None:
    with pytest.raises(KeyError, match="absente"):
        proposer_roles(tab(donnees()), "inconnue")


# ============================================================
# AUDIT SANS SPÉCIFICATION (S2)
# ============================================================
def test_audit_s2_rend_des_constats_a_codes_uniques() -> None:
    constats = auditer(tab(donnees()), None, None)
    assert all(isinstance(c, Constat) for c in constats)
    codes = [c.code for c in constats]
    assert len(codes) == len(set(codes))
    assert "I3.identifiant:id_piece" in codes and niveaux(constats)["I3.identifiant:id_piece"] is Niveau.INFO


def test_audit_s2_ne_depend_d_aucune_cible() -> None:
    t = tab(donnees())
    assert {
        c.code for c in auditer(t, None, None) if c.code.startswith(("I3.fuite", "I3.soeur", "I7"))
    } == set()


def test_colonne_constante_ou_vide() -> None:
    df = donnees()
    df["constante"] = 3
    df["vide"] = np.nan
    c = par_code(auditer(tab(df), None, None))
    assert (
        c["DONNEES.constante:constante"].niveau is Niveau.ALERTE
        and "une seule valeur" in c["DONNEES.constante:constante"].message
    )
    assert (
        c["DONNEES.constante:vide"].niveau is Niveau.ALERTE and "vide" in c["DONNEES.constante:vide"].message
    )
    assert c["DONNEES.constante:vide"].action


def test_codes_distincts_pour_des_noms_qui_se_confondent() -> None:
    df = donnees()
    df["a b"] = 1
    df["a_b"] = 2
    df["é"] = 3
    constats = auditer(tab(df), None, None)
    codes = [c.code for c in constats if c.code.startswith("DONNEES.constante")]
    assert len(codes) == 3 and len(set(codes)) == 3


@pytest.mark.parametrize(
    ("part", "niveau"),
    [(0.02, Niveau.INFO), (0.05, Niveau.ALERTE), (0.30, Niveau.ALERTE)],
)
def test_valeurs_manquantes_niveau_selon_la_part(part: float, niveau: Niveau) -> None:
    df = donnees()
    df.loc[: int(part * len(df)) - 1, "temperature"] = np.nan
    c = par_code(auditer(tab(df), None, None))["DONNEES.manquants"]
    assert c.niveau is niveau and "temperature" in c.message


def test_valeurs_manquantes_massives() -> None:
    df = donnees()
    df.loc[: int(0.7 * len(df)), "effort"] = np.nan
    c = par_code(auditer(tab(df), None, None))
    assert c["DONNEES.manquants_massifs:effort"].niveau is Niveau.ALERTE
    assert (
        "70" in c["DONNEES.manquants_massifs:effort"].message and c["DONNEES.manquants_massifs:effort"].action
    )


def test_aucun_manquant_aucun_constat_de_manquants() -> None:
    assert not [c for c in auditer(tab(donnees()), None, None) if c.code.startswith("DONNEES.manquants")]


def test_doublons_niveau_selon_la_part() -> None:
    df = donnees()
    un = pd.concat([df, df.iloc[:1]], ignore_index=True)
    un["id_piece"] = np.arange(len(un))
    c = par_code(auditer(tab(un), None, None))["DONNEES.doublons"]
    assert c.niveau is Niveau.INFO and "1 ligne(s)" in c.message
    beaucoup = pd.concat([df, df.iloc[:20]], ignore_index=True)
    beaucoup["id_piece"] = np.arange(len(beaucoup))
    assert par_code(auditer(tab(beaucoup), None, None))["DONNEES.doublons"].niveau is Niveau.ALERTE


def test_doublons_a_l_identifiant_pres() -> None:
    df = donnees()
    df = pd.concat([df, df.iloc[:30]], ignore_index=True)
    df["id_piece"] = np.arange(len(df))  # identifiants tous différents : les lignes restent des copies
    assert "DONNEES.doublons" in par_code(auditer(tab(df), None, None))


def test_pas_de_doublon_pas_de_constat() -> None:
    assert "DONNEES.doublons" not in par_code(auditer(tab(donnees()), None, None))


def test_etiquette_issue_d_un_modele_et_attribut_personnel_en_alerte() -> None:
    df = donnees()
    df["cluster_gmm"] = np.arange(len(df)) % 4
    df["operateur_genre"] = np.where(np.arange(len(df)) % 2, "F", "H")
    df["age_operateur"] = np.arange(len(df)) % 40 + 20.0
    c = par_code(auditer(tab(df), None, None))
    for code in (
        "I3.etiquette_modele:cluster_gmm",
        "I3.personnel:operateur_genre",
        "I3.personnel:age_operateur",
    ):
        assert c[code].niveau is Niveau.ALERTE and c[code].action
    assert "I3.personnel:temperature" not in c and "I3.personnel:age_machine" not in c


def test_age_d_une_machine_n_est_pas_un_attribut_personnel() -> None:
    df = donnees()
    df["age_machine_annees"] = np.arange(len(df)) % 15 + 1.0
    assert "I3.personnel:age_machine_annees" not in par_code(auditer(tab(df), None, None))


def test_nombres_ecrits_en_texte_en_alerte() -> None:
    df = donnees()
    df["force"] = ["12,5", "13,5", "11,0", "9,75"] * (len(df) // 4)
    c = par_code(auditer(tab(df), None, None))["DONNEES.nombres_en_texte:force"]
    assert c.niveau is Niveau.ALERTE and "virgule décimale" in c.action


def test_trop_de_categories_en_alerte() -> None:
    df = donnees()
    df["atelier"] = [f"A{i % 80}" for i in range(len(df))]
    c = par_code(auditer(tab(df), None, None))["DONNEES.cardinalite:atelier"]
    assert c.niveau is Niveau.ALERTE and "80" in c.message


def test_repere_de_temps_signale_en_info() -> None:
    c = par_code(auditer(tab(donnees()), None, None))
    assert c["TEMPS.colonne_date:cycle"].niveau is Niveau.INFO


def test_petit_echantillon_signale() -> None:
    c = par_code(auditer(tab(donnees(60)), None, None))
    assert c["DONNEES.echantillon_petit"].niveau is Niveau.INFO
    assert "DONNEES.echantillon_petit" not in par_code(auditer(tab(donnees(240)), None, None))


def test_remarques_de_lecture_et_formules() -> None:
    df = donnees()
    df.attrs[ATTR_NOTES] = ("colonne 1 sans titre",)
    df.attrs[ATTR_FORMULES] = (4, 0)
    c = par_code(auditer(tab(df), None, None))
    assert (
        c["IO.remarques_lecture"].niveau is Niveau.INFO and "sans titre" in c["IO.remarques_lecture"].message
    )
    assert c["IO.formules_lues_en_valeurs"].niveau is Niveau.INFO
    df.attrs[ATTR_FORMULES] = (4, 3)
    c = par_code(auditer(tab(df), None, None))
    assert c["IO.formules_sans_valeur"].niveau is Niveau.ALERTE and "IO.formules_lues_en_valeurs" not in c


# ============================================================
# AUDIT AVEC SPÉCIFICATION (S3)
# ============================================================
def test_specification_saine_sans_constat_de_niveau_eleve() -> None:
    constats = auditer(tab(donnees()), spec(), None)
    assert not [c for c in constats if c.niveau is not Niveau.INFO]


def test_seules_les_colonnes_d_entree_sont_jugees() -> None:
    df = donnees()
    df["constante"] = 1
    df["cluster_gmm"] = np.arange(len(df)) % 3
    roles = (*ROLES_DE_BASE, ("constante", Role.IGNOREE), ("cluster_gmm", Role.IGNOREE))
    codes = niveaux(auditer(tab(df), spec(roles), None))
    assert not [c for c in codes if c.startswith(("DONNEES.constante", "I3.etiquette_modele"))]
    comme_variables = (*ROLES_DE_BASE, ("constante", Role.NUMERIQUE), ("cluster_gmm", Role.NUMERIQUE))
    codes = niveaux(auditer(tab(df), spec(comme_variables), None))
    assert (
        codes["DONNEES.constante:constante"] is Niveau.ALERTE
        and codes["I3.etiquette_modele:cluster_gmm"] is Niveau.ALERTE
    )


def test_identifiant_par_le_nom_utilise_comme_variable_bloque() -> None:
    roles = (*ROLES_DE_BASE[1:], ("id_piece", Role.NUMERIQUE))
    c = par_code(auditer(tab(donnees()), spec(roles), None))["I3.identifiant:id_piece"]
    assert (
        c.niveau is Niveau.BLOQUANT
        and "retiendrait les lignes par cœur" in c.message
        and "identifiant" in c.action
    )


def test_identifiant_par_l_unicite_seule_en_alerte() -> None:
    df = donnees()
    df["serie_pieces"] = np.arange(len(df)) * 7  # entiers tous distincts, sans mot d'identifiant dans le nom
    roles = (*ROLES_DE_BASE, ("serie_pieces", Role.NUMERIQUE))
    assert niveaux(auditer(tab(df), spec(roles), None))["I3.identifiant:serie_pieces"] is Niveau.ALERTE


def test_identifiant_declare_comme_tel_ne_leve_rien() -> None:
    assert not [c for c in auditer(tab(donnees()), spec(), None) if c.code.startswith("I3.identifiant")]


def test_unite_repetee_utilisee_comme_variable_en_alerte() -> None:
    df = donnees()
    df["ID_Machine"] = np.repeat(np.arange(12), len(df) // 12)
    roles = (*ROLES_DE_BASE, ("ID_Machine", Role.NUMERIQUE))
    c = par_code(auditer(tab(df), spec(roles), None))["I3.identifiant_repete:ID_Machine"]
    assert c.niveau is Niveau.ALERTE and "groupe" in c.action


def test_fuite_lineaire_bloquante() -> None:
    df = donnees()
    df["y_x100"] = df["y_reg"] * 100.0 + 5.0
    roles = (*ROLES_DE_BASE, ("y_x100", Role.NUMERIQUE))
    c = par_code(auditer(tab(df), spec(roles), None))["I3.fuite:y_x100"]
    assert (
        c.niveau is Niveau.BLOQUANT
        and "contient déjà la réponse" in c.message
        and "corrélation 1,000" in c.message
    )


def test_relation_monotone_seule_en_alerte_acquittable() -> None:
    df = donnees()
    df["exponentielle"] = np.exp((df["y_reg"] - df["y_reg"].mean()) / df["y_reg"].std() * 3.0)
    roles = (*ROLES_DE_BASE, ("exponentielle", Role.NUMERIQUE))
    c = par_code(auditer(tab(df), spec(roles), None))
    assert c["I3.relation_monotone:exponentielle"].niveau is Niveau.ALERTE
    assert "I3.fuite:exponentielle" not in c


def test_forte_correlation_legitime_ne_bloque_pas() -> None:
    df = donnees()
    df["quasi"] = df["y_reg"] + np.random.default_rng(5).normal(0.0, df["y_reg"].std() * 0.35, len(df))
    roles = (*ROLES_DE_BASE, ("quasi", Role.NUMERIQUE))
    assert not [c for c in auditer(tab(df), spec(roles), None) if c.code.startswith("I3.fuite")]


def test_fuite_categorielle_par_le_v_de_cramer() -> None:
    df = donnees()
    df["copie_texte"] = df["y_cls"].map({"haut": "oui", "bas": "non"})
    roles = (*ROLES_DE_BASE[:-2], ("y_cls", Role.CIBLE), ("copie_texte", Role.CATEGORIELLE))
    c = par_code(auditer(tab(df), spec(roles, Probleme.CLASSIFICATION, "y_cls"), None))[
        "I3.fuite:copie_texte"
    ]
    assert c.niveau is Niveau.BLOQUANT and "V de Cramér" in c.message


def test_fuite_d_une_grandeur_qui_separe_les_classes() -> None:
    df = donnees()
    df["seuil"] = np.where(df["y_cls"] == "haut", 100.0, 0.0) + np.random.default_rng(2).random(len(df))
    roles = (*ROLES_DE_BASE[:-2], ("y_cls", Role.CIBLE), ("seuil", Role.NUMERIQUE))
    c = par_code(auditer(tab(df), spec(roles, Probleme.CLASSIFICATION, "y_cls"), None))["I3.fuite:seuil"]
    assert c.niveau is Niveau.BLOQUANT and "séparation" in c.message


def test_nom_de_cible_soeur_sans_preuve_dans_les_valeurs_est_une_alerte_a_acquitter() -> None:
    df = donnees()
    df["y_reg_ambiante"] = np.random.default_rng(3).normal(size=len(df))
    roles = (*ROLES_DE_BASE, ("y_reg_ambiante", Role.NUMERIQUE))
    c = par_code(auditer(tab(df), spec(roles), None))["I3.soeur:y_reg_ambiante"]
    assert c.niveau is Niveau.ALERTE and "aucune de ses valeurs" in c.message and "Acquitter" in c.action


def test_la_cible_n_est_jamais_une_variable_du_calcul_de_fuite() -> None:
    df = donnees()
    scores = scores_fuite(tab(df), "y_reg", ["y_reg", "temperature"])
    assert "y_reg" not in scores and "temperature" in scores


def test_scores_sans_assez_de_lignes() -> None:
    assert scores_fuite(tab(donnees(60).iloc[:20].reset_index(drop=True)), "y_reg", ["temperature"]) == {}


def test_scores_a_graine_fixe_sur_grand_echantillon() -> None:
    rng = np.random.default_rng(0)
    n = 30_000
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = df["x"] * 2.0 + rng.normal(scale=0.1, size=n)
    t = tab(df)
    a, b = scores_fuite(t, "y", ["x"], graine=7), scores_fuite(t, "y", ["x"], graine=7)
    assert a == b and a["x"][1] > 0.99


def test_colonnes_non_citees_signalees_en_info() -> None:
    roles = tuple(r for r in ROLES_DE_BASE if r[0] not in ("effort", "y_cls"))
    c = par_code(auditer(tab(donnees()), spec(roles), None))["DONNEES.colonnes_non_citees"]
    assert (
        c.niveau is Niveau.INFO and "effort" in c.message and "y_cls" in c.message and "machine" in c.message
    )


def test_toutes_les_colonnes_citees_pas_de_constat_info() -> None:
    roles = (*ROLES_DE_BASE, ("machine", Role.GROUPE), ("cycle", Role.DATE))
    constats = auditer(tab(donnees()), spec(roles, structure=Structure.GROUPEE_CHRONOLOGIQUE), None)
    assert "DONNEES.colonnes_non_citees" not in par_code(constats)


def test_la_cible_est_controlee_avec_les_regles_i7() -> None:
    c = par_code(auditer(tab(donnees()), spec(probleme=Probleme.CLASSIFICATION), None))
    assert c["I7.cible_continue"].niveau is Niveau.BLOQUANT


def test_sans_cible_aucun_controle_de_cible_ni_de_fuite() -> None:
    roles = (*(r for r in ROLES_DE_BASE if r[0] != "y_reg"), ("y_reg", Role.NUMERIQUE))
    constats = auditer(tab(donnees()), spec(roles, Probleme.SANS_CIBLE, None), None)
    assert not [c for c in constats if c.code.startswith(("I7", "I3.fuite", "I3.soeur"))]


# ============================================================
# TEMPS
# ============================================================
def test_colonne_de_temps_avec_decoupage_aleatoire_en_alerte() -> None:
    roles = (*ROLES_DE_BASE, ("cycle", Role.DATE))
    c = par_code(auditer(tab(donnees()), spec(roles), None))["TEMPS.fuite_chronologique"]
    assert c.niveau is Niveau.ALERTE and "chronologique" in c.action


def test_structure_chronologique_pas_d_alerte_de_fuite_temporelle() -> None:
    roles = (*ROLES_DE_BASE, ("cycle", Role.DATE))
    constats = auditer(tab(donnees()), spec(roles, structure=Structure.CHRONOLOGIQUE), None)
    assert "TEMPS.fuite_chronologique" not in par_code(constats)


def test_repere_de_temps_utilise_comme_variable_en_alerte() -> None:
    roles = (*ROLES_DE_BASE, ("cycle", Role.NUMERIQUE))
    c = par_code(auditer(tab(donnees()), spec(roles), None))["TEMPS.date_variable:cycle"]
    assert c.niveau is Niveau.ALERTE


# ============================================================
# AVEC LE COFFRE (S4)
# ============================================================
def coffre_pour(t: Table, s: SpecProjet, positions) -> CoffreTest:
    return CoffreTest.par_positions(t, np.asarray(positions), decoupage=s.decoupage())


def test_groupes_disjoints_conformes() -> None:
    t = tab(donnees())
    roles = (*ROLES_DE_BASE, ("machine", Role.GROUPE))
    s = spec(roles, structure=Structure.GROUPEE)
    test = np.flatnonzero(t.donnees["machine"].isin(["M9", "M10", "M11"]))
    assert not [c for c in auditer(t, s, coffre_pour(t, s, test)) if c.code.startswith("I4")]


def test_groupe_a_cheval_bloquant() -> None:
    t = tab(donnees())
    s = spec((*ROLES_DE_BASE, ("machine", Role.GROUPE)), structure=Structure.GROUPEE)
    c = par_code(auditer(t, s, coffre_pour(t, s, np.arange(0, len(t.donnees), 5))))["I4.unites_a_cheval"]
    assert c.niveau is Niveau.BLOQUANT and "12 unité(s)" in c.message and "unités entières" in c.action


def test_chronologie_respectee() -> None:
    t = tab(donnees())
    s = spec((*ROLES_DE_BASE, ("cycle", Role.DATE)), structure=Structure.CHRONOLOGIQUE)
    test = np.flatnonzero(t.donnees["cycle"] >= CYCLES - 4)
    assert not [c for c in auditer(t, s, coffre_pour(t, s, test)) if c.code.startswith("I4")]


def test_chronologie_melangee_bloquante() -> None:
    t = tab(donnees())
    s = spec((*ROLES_DE_BASE, ("cycle", Role.DATE)), structure=Structure.CHRONOLOGIQUE)
    c = par_code(auditer(t, s, coffre_pour(t, s, np.arange(0, len(t.donnees), 5))))["I4.dates_a_cheval"]
    assert c.niveau is Niveau.BLOQUANT and "cycle" in c.message


def test_date_a_cheval_d_un_seul_cycle_bloquante() -> None:
    t = tab(donnees())
    s = spec((*ROLES_DE_BASE, ("cycle", Role.DATE)), structure=Structure.CHRONOLOGIQUE)
    test = np.flatnonzero(t.donnees["cycle"] >= CYCLES - 4)
    test = test[: len(test) // 2]  # le dernier cycle d'apprentissage rejoint des cycles de test
    assert "I4.dates_a_cheval" in par_code(auditer(t, s, coffre_pour(t, s, test)))


def test_groupee_et_chronologique_unites_entieres_d_un_seul_cote() -> None:
    t = tab(donnees())
    roles = (*ROLES_DE_BASE, ("machine", Role.GROUPE), ("cycle", Role.DATE))
    s = spec(roles, structure=Structure.GROUPEE_CHRONOLOGIQUE)
    bonne = np.flatnonzero(t.donnees["machine"].isin(["M9", "M10", "M11"]))
    assert not [c for c in auditer(t, s, coffre_pour(t, s, bonne)) if c.code.startswith("I4")]
    mauvaise = np.arange(0, len(t.donnees), 5)
    assert "I4.unites_a_cheval" in par_code(auditer(t, s, coffre_pour(t, s, mauvaise)))


def test_test_fourni_a_part_structure_non_verifiable() -> None:
    df = donnees()
    app, test = tab(df.iloc[:200].reset_index(drop=True)), tab(df.iloc[200:].reset_index(drop=True))
    s = spec((*ROLES_DE_BASE, ("cycle", Role.DATE)), structure=Structure.CHRONOLOGIQUE)
    coffre = CoffreTest.par_fichier(app, test, decoupage=s.decoupage())
    c = par_code(auditer(app, s, coffre))
    assert c["I4.non_verifiable"].niveau is Niveau.INFO and "I4.dates_a_cheval" not in c


def test_independante_aucun_controle_de_structure() -> None:
    t = tab(donnees())
    s = spec()
    assert not [
        c
        for c in auditer(t, s, coffre_pour(t, s, np.arange(0, len(t.donnees), 5)))
        if c.code.startswith("I4")
    ]


def test_classes_comptees_sur_les_lignes_d_apprentissage_seules() -> None:
    df = donnees()
    df["rare"] = "a"
    df.loc[[0, 5, 10, 15, 20, 25], "rare"] = "b"  # 6 lignes au total, dont 3 vont en test
    t = tab(df)
    roles = (*ROLES_DE_BASE[:-2], ("rare", Role.CIBLE), ("y_reg", Role.NUMERIQUE))
    s = spec(roles, Probleme.CLASSIFICATION, "rare")
    avant = par_code(auditer(t, s, None))
    assert "I7.classe_plus_petite_que_les_plis" not in avant
    test = np.array([0, 5, 10, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12])  # trois lignes « b » sur six
    apres = par_code(auditer(t, s, coffre_pour(t, s, test)))
    assert apres["I7.classe_plus_petite_que_les_plis"].niveau is Niveau.BLOQUANT


@pytest.mark.parametrize("structure", [Structure.INDEPENDANTE, Structure.GROUPEE, Structure.CHRONOLOGIQUE])
def test_l_audit_ne_depend_pas_des_lignes_de_test(structure: Structure) -> None:
    """Changer tout le contenu des lignes de test (sauf groupe et date) ne change aucun constat."""
    df = donnees()
    roles = (*ROLES_DE_BASE, ("machine", Role.GROUPE), ("cycle", Role.DATE))
    s = spec(roles, structure=structure)
    if structure is Structure.CHRONOLOGIQUE:
        test = np.flatnonzero(df["cycle"] >= CYCLES - 4)
    elif structure is Structure.GROUPEE:
        test = np.flatnonzero(df["machine"].isin(["M9", "M10", "M11"]))
    else:
        test = np.arange(0, len(df), 5)
    abime = df.copy()
    rng = np.random.default_rng(99)
    for colonne in abime.columns:
        if colonne in ("machine", "cycle"):
            continue
        if pd.api.types.is_numeric_dtype(abime[colonne]):
            abime[colonne] = abime[colonne].astype(float)
            abime.loc[test, colonne] = rng.normal(1e9, 1e6, len(test))
        else:
            abime.loc[test, colonne] = "zzz"
    t1, t2 = tab(df), tab(abime)
    c1 = auditer(t1, s, coffre_pour(t1, s, test))
    c2 = auditer(t2, s, coffre_pour(t2, s, test))
    assert c1 == c2


# ============================================================
# AVEC LA SESSION
# ============================================================
def session_jusqu_a_s3(t: Table, s: SpecProjet) -> Session:
    return Session(auditeur=auditer).appliquer(Charger(t)).appliquer(Auditer()).appliquer(FixerObjectif(s))


def test_la_session_bloque_tant_qu_un_constat_bloquant_subsiste() -> None:
    df = donnees()
    df["copie"] = df["y_reg"] * 2.0
    t = tab(df)
    s = spec((*ROLES_DE_BASE, ("copie", Role.NUMERIQUE)))
    s3 = session_jusqu_a_s3(t, s)
    assert [c.code for c in s3.bloquants()] == ["I3.fuite:copie"]
    s4 = s3.appliquer(Decouper(coffre_pour(t, s, np.arange(0, len(df), 5))))
    with pytest.raises(TransitionRefusee, match="bloquants"):
        s4.appliquer(Pretraiter(s.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))))
    corrigee = spec((*ROLES_DE_BASE, ("copie", Role.IGNOREE)))
    s3b = s3.appliquer(Revenir(Etat.AUDIT)).appliquer(FixerObjectif(corrigee))
    assert not s3b.bloquants()


def test_la_session_exige_l_acquittement_des_alertes() -> None:
    df = donnees()
    df["cluster_gmm"] = np.arange(len(df)) % 3
    t = tab(df)
    s = spec((*ROLES_DE_BASE, ("cluster_gmm", Role.NUMERIQUE)))
    s3 = session_jusqu_a_s3(t, s)
    assert [c.code for c in s3.alertes_en_attente()] == ["I3.etiquette_modele:cluster_gmm"]
    s4 = s3.appliquer(Decouper(coffre_pour(t, s, np.arange(0, len(df), 5))))
    pretraite = Pretraiter(s.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),)))
    with pytest.raises(TransitionRefusee, match="alertes à acquitter"):
        s4.appliquer(pretraite)
    assert (
        s4.appliquer(Acquitter(frozenset({"I3.etiquette_modele:cluster_gmm"}))).appliquer(pretraite).etat == 5
    )


def test_la_session_rejoue_l_audit_avec_le_coffre_sans_changer_les_codes() -> None:
    t = tab(donnees())
    s = spec()
    s3 = session_jusqu_a_s3(t, s)
    s4 = s3.appliquer(Decouper(coffre_pour(t, s, np.arange(0, len(t.donnees), 5))))
    assert {c.code for c in s4.constats} >= {c.code for c in s3.constats if c.niveau is not Niveau.INFO}


def test_lignes_sans_aucune_valeur_signalees_en_alerte() -> None:
    df = donnees()
    df.attrs[ATTR_LIGNES_VIDES] = 3
    c = par_code(auditer(tab(df), None, None))["DONNEES.lignes_vides"]
    assert c.niveau is Niveau.ALERTE and "3 ligne(s)" in c.message and "positions" in c.message
    assert "DONNEES.lignes_vides" not in par_code(auditer(tab(donnees()), None, None))


def test_classe_assez_nombreuse_en_apprentissage_aucune_alerte_apres_decoupage() -> None:
    """Au S4, les classes sont comptées telles quelles : pas de seconde réduction par la part de test."""
    df = donnees()
    df["rare"] = "a"
    df.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "rare"] = "b"
    t = tab(df)
    roles = (*ROLES_DE_BASE[:-2], ("rare", Role.CIBLE), ("y_reg", Role.NUMERIQUE))
    s = spec(roles, Probleme.CLASSIFICATION, "rare", part_test=0.7)
    test = np.arange(100, 220)  # les douze lignes « b » restent en apprentissage
    apres = par_code(auditer(t, s, coffre_pour(t, s, test)))
    avant = par_code(auditer(t, s, None))
    assert (
        "I7.classe_trop_petite_apres_decoupage" not in apres
        and "I7.classe_plus_petite_que_les_plis" not in apres
    )
    assert (
        "I7.classe_trop_petite_apres_decoupage" in avant
    )  # douze lignes, 70 % iront en test : trois pour cinq plis


# ============================================================
# IDENTIFIANTS, CIBLES SŒURS, UNITÉS SUIVIES, RENVOIS AU LIVRE
# ============================================================
def avec_colonne(nom: str, valeurs) -> pd.DataFrame:
    df = donnees()
    df[nom] = valeurs
    return df


@pytest.mark.parametrize("nom", ["temperature_ref", "index_viscosite", "ref_dilatation", "numero_pression"])
def test_un_flottant_n_est_jamais_un_identifiant_meme_au_nom_evocateur(nom: str) -> None:
    df = avec_colonne(nom, np.random.default_rng(1).normal(size=N))
    p = {p.nom: p for p in profiler(tab(df))}[nom]
    assert p.role_propose is Role.NUMERIQUE and p.preuve_identifiant is None
    roles = (*ROLES_DE_BASE, (nom, Role.NUMERIQUE))
    assert not [c for c in auditer(tab(df), spec(roles), None) if c.code.startswith("I3.identifiant")]


def test_un_entier_aux_valeurs_repetees_n_est_pas_bloque_par_son_nom() -> None:
    """`num_tours` : entier de 100 à 5 000 dont les valeurs se répètent."""
    df = avec_colonne("num_tours", np.random.default_rng(2).choice(np.arange(100, 5000, 50), N))
    p = {p.nom: p for p in profiler(tab(df))}["num_tours"]
    assert p.role_propose is Role.NUMERIQUE
    roles = (*ROLES_DE_BASE, ("num_tours", Role.NUMERIQUE))
    assert not [c for c in auditer(tab(df), spec(roles), None) if c.code == "I3.identifiant:num_tours"]


def test_mot_faible_et_entiers_distincts_mais_disperses_alerte_seulement() -> None:
    """`num_serie_rpm` : valeurs toutes distinctes mais étendue sans rapport avec leur nombre."""
    valeurs = np.random.default_rng(3).choice(np.arange(1000, 90_000), N, replace=False)
    df = avec_colonne("num_serie_rpm", valeurs)
    p = {p.nom: p for p in profiler(tab(df))}["num_serie_rpm"]
    assert p.role_propose is Role.IDENTIFIANT and p.preuve_identifiant == "unicite"
    roles = (*ROLES_DE_BASE, ("num_serie_rpm", Role.NUMERIQUE))
    assert niveaux(auditer(tab(df), spec(roles), None))["I3.identifiant:num_serie_rpm"] is Niveau.ALERTE


def test_mot_faible_et_entiers_consecutifs_bloquants() -> None:
    df = avec_colonne("Piece_Num", np.arange(500, 500 + N))
    roles = (*ROLES_DE_BASE, ("Piece_Num", Role.NUMERIQUE))
    assert par_code(auditer(tab(df), spec(roles), None))["I3.identifiant:Piece_Num"].niveau is Niveau.BLOQUANT


def test_mot_fort_et_texte_distinct_bloquant() -> None:
    df = avec_colonne("identifiant_lot", [f"LOT-{i * 7:05d}" for i in range(N)])
    roles = (*ROLES_DE_BASE, ("identifiant_lot", Role.CATEGORIELLE))
    assert par_code(auditer(tab(df), spec(roles), None))["I3.identifiant:identifiant_lot"].bloquant


def test_nom_d_identifiant_et_valeurs_en_partie_repetees_alerte() -> None:
    valeurs = np.random.default_rng(4).integers(0, int(0.7 * N), N)  # environ 60 % de valeurs distinctes
    df = avec_colonne("id_commande", valeurs)
    p = {p.nom: p for p in profiler(tab(df))}["id_commande"]
    assert p.preuve_identifiant == "nom_partiel" and p.role_propose is Role.NUMERIQUE
    roles = (*ROLES_DE_BASE, ("id_commande", Role.NUMERIQUE))
    assert niveaux(auditer(tab(df), spec(roles), None))["I3.identifiant:id_commande"] is Niveau.ALERTE


def frame_soeur(avec_implication: bool) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    df = donnees()
    df["Panne"] = rng.choice([0, 1], N, p=[0.9, 0.1])
    if avec_implication:
        df["Panne_outil"] = ((df["Panne"] == 1) & (rng.random(N) < 0.6)).astype(int)
    else:
        df["Panne_outil"] = rng.choice([0, 1], N, p=[0.8, 0.2])
    return df


def roles_panne() -> tuple[tuple[str, Role], ...]:
    sans_y = tuple(r for r in ROLES_DE_BASE if r[0] != "y_reg")
    return (*sans_y, ("Panne", Role.CIBLE), ("Panne_outil", Role.NUMERIQUE))


def test_preuve_soeur_une_modalite_implique_la_cible() -> None:
    df = frame_soeur(True)
    valeur, t, n = preuve_soeur(df["Panne_outil"], df["Panne"])
    assert (valeur, t) == (1, 1) and n == int(df["Panne_outil"].sum())


def test_pas_de_preuve_sans_implication_ni_sur_une_grandeur_continue() -> None:
    df = frame_soeur(False)
    assert preuve_soeur(df["Panne_outil"], df["Panne"]) is None
    assert preuve_soeur(df["temperature"], df["effort"]) is None


def test_pas_de_preuve_avec_trop_peu_de_lignes_ou_par_hasard() -> None:
    cible = pd.Series([0] * 95 + [1] * 5 + [0] * 100)
    assert preuve_soeur(pd.Series([0] * 95 + [1] * 2 + [0] * 103), cible) is None  # deux lignes : trop peu
    rare = pd.Series(["x"] * 6 + ["y"] * 294)
    zeros = pd.Series([0] * 291 + [1] * 9)  # « x » implique 0, mais 97 % des lignes valent 0 : sans preuve
    assert preuve_soeur(rare, zeros) is None


def test_soeur_nommee_avec_preuve_bloquante_et_sans_preuve_alerte() -> None:
    s = spec(roles_panne(), Probleme.CLASSIFICATION, "Panne")
    c = par_code(auditer(tab(frame_soeur(True)), s, None))["I3.soeur:Panne_outil"]
    assert c.niveau is Niveau.BLOQUANT and "implique" in c.message and "= 1" in c.message
    c = par_code(auditer(tab(frame_soeur(False)), s, None))["I3.soeur:Panne_outil"]
    assert c.niveau is Niveau.ALERTE and "Acquitter" in c.action


def test_soeur_acquittee_laisse_passer_la_session() -> None:
    s = spec(roles_panne(), Probleme.CLASSIFICATION, "Panne", classe_positive=1)
    s3 = session_jusqu_a_s3(tab(frame_soeur(False)), s)
    assert not s3.bloquants() and [c.code for c in s3.alertes_en_attente()] == ["I3.soeur:Panne_outil"]
    assert not s3.appliquer(Acquitter(frozenset({"I3.soeur:Panne_outil"}))).alertes_en_attente()


def test_temperature_ambiante_n_est_pas_une_soeur_de_temperature() -> None:
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"temperature": rng.normal(60, 5, N), "temperature_ambiante": rng.normal(20, 2, N)})
    df["effort"] = df["temperature"] * 0.5 + rng.normal(size=N)
    roles = (
        ("temperature", Role.CIBLE),
        ("temperature_ambiante", Role.NUMERIQUE),
        ("effort", Role.NUMERIQUE),
    )
    cs = par_code(auditer(tab(df), spec(roles, Probleme.REGRESSION, "temperature"), None))
    assert cs["I3.soeur:temperature_ambiante"].niveau is Niveau.ALERTE
    prop = proposition_roles(tab(df), "temperature")
    assert dict(prop.roles)["temperature_ambiante"] is Role.NUMERIQUE  # gardée : à l'utilisateur de trancher


def test_proposition_donne_la_raison_de_chaque_colonne_ecartee() -> None:
    df = frame_soeur(True)
    df["cluster_gmm"] = np.arange(N) % 3
    df["constante"] = 4.0
    df["copie"] = df["Panne"] * 3.0
    prop = proposition_roles(tab(df), "Panne")
    assert isinstance(prop, PropositionRoles) and set(prop.raisons) == set(df.columns)
    assert dict(prop.roles)["Panne_outil"] is Role.IGNOREE and "cible sœur" in prop.raison_de("Panne_outil")
    assert "modèle" in prop.raison_de("cluster_gmm") and "constante" in prop.raison_de("constante")
    assert "fuite" in prop.raison_de("copie") and prop.raison_de("Panne") == "cible choisie"
    assert dict(prop.roles) == dict(proposer_roles(tab(df), "Panne"))


def serie_par_unite(unite: str, entrelace: bool = False, repart: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n_unites, par_unite = 12, 20
    u = np.repeat([f"{unite[0]}{i:02d}" for i in range(n_unites)], par_unite)
    t = np.tile(np.arange(1, par_unite + 1), n_unites) if repart else np.arange(n_unites * par_unite)
    df = pd.DataFrame({unite: u, "temps_coupe_min": t, "effort": rng.normal(size=len(u))})
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True) if entrelace else df


@pytest.mark.parametrize("unite", ["outil", "machine", "eprouvette", "lot", "moteur", "pompe"])
def test_unite_suivie_reconnue_quand_le_repere_de_temps_repart(unite: str) -> None:
    prop = {p.nom: p.role_propose for p in profiler(tab(serie_par_unite(unite)))}
    assert prop[unite] is Role.GROUPE and prop["temps_coupe_min"] is Role.DATE
    assert proposer_structure(proposer_roles(tab(serie_par_unite(unite)))) is Structure.GROUPEE_CHRONOLOGIQUE


def test_unite_suivie_en_blocs_sans_redemarrage_du_temps() -> None:
    prop = {p.nom: p.role_propose for p in profiler(tab(serie_par_unite("machine", repart=False)))}
    assert prop["machine"] is Role.GROUPE and prop["temps_coupe_min"] is Role.DATE


def test_journal_ou_les_unites_alternent_n_est_pas_une_serie_par_unite() -> None:
    n = 240
    df = pd.DataFrame({"machine": np.tile(["m1", "m2", "m3"], n // 3), "temps_coupe_min": np.arange(n)})
    assert {p.nom: p.role_propose for p in profiler(tab(df))}["machine"] is Role.CATEGORIELLE


def test_lignes_melangees_ou_sans_temps_pas_d_unite() -> None:
    melange = {p.nom: p.role_propose for p in profiler(tab(serie_par_unite("outil", entrelace=True)))}
    assert melange["outil"] is Role.CATEGORIELLE
    sans_temps = serie_par_unite("outil").drop(columns="temps_coupe_min")
    assert {p.nom: p.role_propose for p in profiler(tab(sans_temps))}["outil"] is Role.CATEGORIELLE


def test_un_nom_hors_du_lexique_d_unites_reste_une_categorie() -> None:
    df = serie_par_unite("outil").rename(columns={"outil": "famille"})
    assert {p.nom: p.role_propose for p in profiler(tab(df))}["famille"] is Role.CATEGORIELLE


def test_une_etiquette_de_modele_n_est_jamais_proposee_comme_cible() -> None:
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        {"couple": rng.normal(size=N), "debit": rng.normal(size=N), "cluster_gmm": np.arange(N) % 7}
    )
    assert proposer_cible(profiler(tab(df))) != "cluster_gmm"


def test_les_renvois_des_constats_sont_des_identifiants_du_livre() -> None:
    from mecaml.core.livre import est_identifiant

    df = avec_colonne("cluster_gmm", np.arange(N) % 3)
    df["operateur_genre"] = np.where(np.arange(N) % 2, "F", "H")
    df["constante"] = 1
    df["copie"] = df["y_reg"] * 2.0
    df["Piece_Num"] = np.arange(N)
    df.loc[:20, "temperature"] = np.nan
    df = pd.concat([df, df.iloc[:15]], ignore_index=True)
    df.attrs[ATTR_LIGNES_VIDES] = 2
    df.attrs[ATTR_FORMULES] = (3, 3)
    roles = (
        *ROLES_DE_BASE,
        ("cluster_gmm", Role.NUMERIQUE),
        ("operateur_genre", Role.CATEGORIELLE),
        ("constante", Role.NUMERIQUE),
        ("copie", Role.NUMERIQUE),
        ("Piece_Num", Role.NUMERIQUE),
        ("cycle", Role.DATE),
    )
    constats = [*auditer(tab(df), spec(roles), None), *auditer(tab(df), None, None)]
    assert len({c.code.split(":")[0] for c in constats}) >= 8
    for c in constats:
        assert c.ref_livre is None or est_identifiant(c.ref_livre), (c.code, c.ref_livre)
    assert all(
        c.ref_livre is not None for c in constats if c.code.startswith(("I3.fuite", "DONNEES.doublons"))
    )


def test_la_graine_par_defaut_est_celle_du_contrat() -> None:
    import inspect

    from mecaml.core.types import GRAINE_CONTRAT

    for fonction in (scores_fuite, proposer_roles, proposition_roles):
        assert inspect.signature(fonction).parameters["graine"].default == GRAINE_CONTRAT


def test_effectif_minimal_de_la_preuve_de_soeur() -> None:
    """Quatre lignes positives sur cinq ne prouvent rien (même si le hasard est exclu), cinq oui."""
    cible = pd.Series([1] * 5 + [0] * 195)
    assert preuve_soeur(pd.Series([1] * 4 + [0] * 196), cible) is None
    assert preuve_soeur(pd.Series([1] * 5 + [0] * 195), cible) == (1, 1, 5)


# ============================================================
# ENTIER LU EN FLOTTANT, VALEURS ABERRANTES, CIBLE PASSÉE, PART DE TEST
# ============================================================
def roles_avec(**roles: Role) -> tuple[tuple[str, Role], ...]:
    """Rôles de base, modifiés ou complétés par ceux des colonnes nommées."""
    return tuple({**dict(ROLES_DE_BASE), **roles}.items())


def test_un_identifiant_entier_avec_une_case_vide_reste_un_identifiant() -> None:
    """`id_piece` de 1 à N avec une case vide est lu en flottant : il suit les règles des entiers."""
    complet = donnees()
    complet["id_piece"] = np.arange(1, N + 1)
    vide = complet.assign(id_piece=np.arange(1, N + 1, dtype=float))
    vide.loc[17, "id_piece"] = np.nan
    assert pd.api.types.is_float_dtype(vide["id_piece"])
    roles = roles_avec(id_piece=Role.NUMERIQUE)
    for df in (complet, vide):
        p = {p.nom: p for p in profiler(tab(df))}["id_piece"]
        assert p.role_propose is Role.IDENTIFIANT and p.preuve_identifiant == "nom" and not p.flottante
        assert dict(proposer_roles(tab(df), "y_reg"))["id_piece"] is Role.IDENTIFIANT
        assert par_code(auditer(tab(df), spec(roles), None))["I3.identifiant:id_piece"].bloquant


def test_une_unite_numerique_a_case_vide_reste_une_unite_suivie() -> None:
    """Numéros de machine lus en flottant (une case vide) : mêmes règles que des entiers."""
    df = serie_par_unite("machine")
    df["machine"] = np.repeat(np.arange(12), 20).astype(float)
    df.loc[3, "machine"] = np.nan
    prop = {p.nom: p.role_propose for p in profiler(tab(df))}
    assert prop["machine"] is Role.GROUPE and prop["temps_coupe_min"] is Role.DATE


def test_flottant_decimal_jamais_identifiant_et_compte_entier_a_case_vide_reste_une_mesure() -> None:
    decimal = donnees().assign(id_piece=np.arange(N) + 0.5)
    p = {p.nom: p for p in profiler(tab(decimal))}["id_piece"]
    assert p.role_propose is Role.NUMERIQUE and p.flottante and p.preuve_identifiant is None
    roles = roles_avec(id_piece=Role.NUMERIQUE)
    assert not [c for c in auditer(tab(decimal), spec(roles), None) if c.code.startswith("I3.identifiant")]
    # `num_tours` : entiers répétés, lus en flottant faute d'une case vide, restent une mesure
    tours = np.random.default_rng(2).choice(np.arange(100, 5000, 50), N).astype(float)
    tours[3] = np.nan
    df = avec_colonne("num_tours", tours)
    assert {p.nom: p for p in profiler(tab(df))}["num_tours"].role_propose is Role.NUMERIQUE
    roles = (*ROLES_DE_BASE, ("num_tours", Role.NUMERIQUE))
    assert not [c for c in auditer(tab(df), spec(roles), None) if c.code == "I3.identifiant:num_tours"]


def avec_pression(aberrantes: dict[int, float]) -> pd.DataFrame:
    """Une pression uniforme de 0 à 1 (rien hors des bornes), puis des valeurs posées à des lignes données."""
    df = donnees()
    df["pression"] = np.linspace(0.0, 1.0, N)
    for ligne, valeur in aberrantes.items():
        df.loc[ligne, "pression"] = valeur
    return df


def test_les_valeurs_hors_des_bornes_de_tukey_sont_une_information_du_profil_de_s2() -> None:
    df = avec_pression(dict(zip(range(8), [5.0, -3.0, 9.0, -4.0, 6.0, 7.0, -2.0, 8.0], strict=True)))
    p = {p.nom: p for p in profiler(tab(df))}
    assert p["pression"].n_hors_tukey == 8 and p["temperature"].n_hors_tukey < 5
    constat = par_code(auditer(tab(df), None, None))["DONNEES.valeurs_aberrantes"]  # S2 : sans spécification
    assert constat.niveau is Niveau.INFO and constat.ref_livre == "T03-1-2"
    assert "« pression » (8 valeurs, 3,3 %)" in constat.message and "Tukey" in constat.message
    roles = roles_avec(pression=Role.NUMERIQUE)
    assert "DONNEES.valeurs_aberrantes" in par_code(auditer(tab(df), spec(roles), None))  # S3 aussi


def test_peu_de_valeurs_hors_des_bornes_ne_font_aucun_constat() -> None:
    """Moins de cinq valeurs, ou aucune : la colonne n'est pas citée (une autre colonne peut l'être)."""
    for df in (avec_pression({0: 9.0, 1: -9.0}), avec_pression({})):
        constat = par_code(auditer(tab(df), None, None)).get("DONNEES.valeurs_aberrantes")
        assert constat is None or "pression" not in constat.message


def test_les_valeurs_aberrantes_se_comptent_sur_l_apprentissage_quand_le_coffre_existe() -> None:
    df = avec_pression(dict(zip(range(8), [5.0, -3.0, 9.0, -4.0, 6.0, 7.0, -2.0, 8.0], strict=True)))
    t, s = tab(df), spec(roles_avec(pression=Role.NUMERIQUE))
    assert "DONNEES.valeurs_aberrantes" in par_code(auditer(t, s, None))
    coffre = coffre_pour(t, s, np.arange(10))  # les huit valeurs posées sont au test : l'audit ne les lit pas
    assert "DONNEES.valeurs_aberrantes" not in par_code(auditer(t, s, coffre))


def parc_mixte() -> pd.DataFrame:
    """M00 à M03 : la cible décompte ; M04 à M07 : une température relevée à chaque cycle."""
    decompte = aides.jeu_groupe(n_groupes=4, longueur=30)
    mesure = aides.jeu_serie_mesuree(n_groupes=4, longueur=30)
    mesure["machine"] = [f"M{int(m[1:]) + 4:02d}" for m in mesure["machine"]]
    return pd.concat([decompte, mesure], ignore_index=True)


def projet_serie(df: pd.DataFrame, modeles=(), pretraitement=()) -> tuple[Table, SpecProjet]:
    return tab(df), aides.spec_groupe(modeles=modeles, pretraitement=pretraitement)


PERSISTANCE_DECLAREE = ChoixTechnique.creer("reference.persistance", cible_connue=True)
CIBLE_DECALEE_DECLAREE = ChoixTechnique.creer("chronologique.cible_decalee", cible_connue=True)


@pytest.mark.parametrize(
    ("choix", "code"),
    [
        (ChoixTechnique.creer("reference.persistance"), "I3.cible_passee:persistance"),
        (ChoixTechnique.creer("reference.persistance", cible_connue=False), "I3.cible_passee:persistance"),
        (ChoixTechnique.creer("chronologique.cible_decalee"), "I3.cible_passee:cible_decalee"),
    ],
)
def test_la_cible_passee_sans_declaration_est_un_constat_bloquant(choix: ChoixTechnique, code: str) -> None:
    modeles, pretraitement = ((choix,), ()) if choix.famille == "reference" else ((), (choix,))
    t, s = projet_serie(aides.jeu_serie_mesuree(), modeles, pretraitement)
    constat = par_code(auditer(t, s, None))[code]
    assert constat.niveau is Niveau.BLOQUANT and "cible_connue=True" in constat.action
    assert constat.ref_livre == "T12-2-1"
    assert not [
        c for c in auditer(t, aides.spec_groupe(), None) if c.code.startswith("I3.cible_")
    ]  # sans le choix


@pytest.mark.parametrize("declare", [PERSISTANCE_DECLAREE, CIBLE_DECALEE_DECLAREE])
def test_la_cible_passee_declaree_sur_une_cible_connue_ne_leve_rien(declare: ChoixTechnique) -> None:
    modeles, pretraitement = ((declare,), ()) if declare.famille == "reference" else ((), (declare,))
    t, s = projet_serie(aides.jeu_serie_mesuree(), modeles, pretraitement)
    assert not [c for c in auditer(t, s, None) if c.code.startswith("I3.cible_")]


@pytest.mark.parametrize(
    ("declare", "code"),
    [
        (PERSISTANCE_DECLAREE, "I3.cible_decompte:persistance"),
        (CIBLE_DECALEE_DECLAREE, "I3.cible_decompte:cible_decalee"),
    ],
)
def test_la_cible_passee_declaree_sur_une_cible_qui_decompte_est_un_constat_bloquant(
    declare: ChoixTechnique, code: str
) -> None:
    modeles, pretraitement = ((declare,), ()) if declare.famille == "reference" else ((), (declare,))
    t, s = projet_serie(aides.jeu_groupe(n_groupes=8, longueur=30), modeles, pretraitement)
    constat = par_code(auditer(t, s, None))[code]
    assert (
        constat.niveau is Niveau.BLOQUANT
        and "panne" in constat.message
        and "compte à rebours" in constat.message
    )


def test_le_compte_a_rebours_se_lit_sur_l_apprentissage_seul() -> None:
    df = parc_mixte()
    t, s = projet_serie(df, modeles=(PERSISTANCE_DECLAREE,))
    assert not [
        c for c in auditer(t, s, None) if c.code.startswith("I3.cible_")
    ]  # la moitié des pas seulement
    machines = df["machine"]
    avec_test_observable = coffre_pour(t, s, np.flatnonzero(machines.isin(["M04", "M05", "M06", "M07"])))
    assert "I3.cible_decompte:persistance" in par_code(auditer(t, s, avec_test_observable))
    avec_test_decompte = coffre_pour(t, s, np.flatnonzero(machines.isin(["M00", "M01", "M02", "M03"])))
    assert not [c for c in auditer(t, s, avec_test_decompte) if c.code.startswith("I3.cible_")]


def test_la_duree_de_vie_restante_des_exemples_leve_le_constat_meme_declaree() -> None:
    df = pd.read_excel(EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx")
    cible = "Duree_Vie_Restante_Cycles"
    capteurs = [c for c in df.columns if c not in ("ID_Machine", "Cycle_Operation_Heures", cible)]
    roles = (
        ("ID_Machine", Role.GROUPE),
        ("Cycle_Operation_Heures", Role.DATE),
        *((c, Role.NUMERIQUE) for c in capteurs),
        (cible, Role.CIBLE),
    )
    objectif = {"roles": roles, "cible": cible, "structure": Structure.GROUPEE_CHRONOLOGIQUE}
    sans_choix = spec(**objectif)
    assert not [c for c in auditer(tab(df), sans_choix, None) if c.code.startswith("I3.cible_")]
    avec_persistance = sans_choix.avec(modeles=(PERSISTANCE_DECLAREE,))
    constat = par_code(auditer(tab(df), avec_persistance, None))["I3.cible_decompte:persistance"]
    assert constat.bloquant and "panne" in constat.message and "baisse de 1 à chaque ligne" in constat.message


def test_l_audit_de_la_cible_passee_se_rejoue_avec_la_specification_complete() -> None:
    """À l'objectif (S3) la session purge les modèles et le prétraitement : l'audit n'en voit aucun.
    Les écrans de S5 et de S6 rejouent donc l'audit avec la spécification complète ; l'entraînement
    refuse de toute façon."""
    t, s = projet_serie(aides.jeu_groupe(n_groupes=8, longueur=30), modeles=(PERSISTANCE_DECLAREE,))
    assert not [c for c in session_jusqu_a_s3(t, s).bloquants() if c.code.startswith("I3.cible_")]
    assert "I3.cible_decompte:persistance" in par_code(auditer(t, s, None))
