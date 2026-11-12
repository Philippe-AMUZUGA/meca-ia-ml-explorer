"""I3, entrées interdites : identifiants exclus, fuite par la cible et cibles sœurs bloquantes,
étiquette issue d'un modèle et attribut personnel en alerte, détection en moins de 2 s.

Les cas dorés sur les jeux autorisés se lancent avec la variable d'environnement MECAML_JEUX
(dossier des classeurs, en lecture seule) ; sans elle, ils sont ignorés.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecaml.core.audit import auditer, profiler, proposer_cible, proposer_roles, proposer_structure
from mecaml.core.guards import detecter_probleme
from mecaml.core.io import charger
from mecaml.core.session import (
    Acquitter,
    Auditer,
    Charger,
    Decouper,
    FixerObjectif,
    Pretraiter,
    Session,
    TransitionRefusee,
)
from mecaml.core.types import ChoixTechnique, CoffreTest, Niveau, Probleme, Role, SpecProjet, Structure, Table

BUDGET_S = 2.0


# ============================================================
# OUTILS
# ============================================================
def table_de_mesures(n: int = 300, graine: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    temperature = rng.normal(60.0, 5.0, n)
    effort = rng.normal(12.0, 2.0, n)
    y = 0.3 * temperature + 1.5 * effort + rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "temperature": temperature,
            "effort": effort,
            "materiau": rng.choice(["acier", "alu", "inox"], n),
            "y": y,
            "classe": np.where(y > np.median(y), "haut", "bas"),
        }
    )


def specification(
    df: pd.DataFrame, cible: str, probleme: Probleme, supplementaires: dict[str, Role]
) -> SpecProjet:
    roles = {"temperature": Role.NUMERIQUE, "effort": Role.NUMERIQUE, "materiau": Role.CATEGORIELLE}
    roles[cible] = Role.CIBLE
    roles.update(supplementaires)
    for autre in ("y", "classe"):
        roles.setdefault(autre, Role.IGNOREE)
    roles = {c: r for c, r in roles.items() if c in df.columns}
    metrique = "rmse" if probleme is Probleme.REGRESSION else "f1"
    return SpecProjet(probleme, cible, tuple(roles.items()), metrique)


def constats(df: pd.DataFrame, cible: str, probleme: Probleme, supplementaires: dict[str, Role]):
    spec = specification(df, cible, probleme, supplementaires)
    return {c.code: c for c in auditer(Table(df, nom="essai.csv"), spec, None)}


# ============================================================
# IDENTIFIANTS EXCLUS
# ============================================================
@pytest.mark.parametrize(
    "nom", ["id_piece", "Piece_ID", "id_soudure", "Unnamed: 0", "numero_serie", "ref", "UUID"]
)
def test_identifiant_utilise_comme_variable_est_bloquant(nom: str) -> None:
    df = table_de_mesures()
    df[nom] = np.arange(len(df))
    cs = constats(df, "y", Probleme.REGRESSION, {nom: Role.NUMERIQUE})
    (c,) = [c for c in cs.values() if c.code.startswith("I3.identifiant")]
    assert c.niveau is Niveau.BLOQUANT and nom in c.message and c.action


@pytest.mark.parametrize("nom", ["id_piece", "Piece_ID", "id_soudure", "Unnamed: 0"])
def test_identifiant_declare_ou_ecarte_ne_leve_aucun_constat(nom: str) -> None:
    df = table_de_mesures()
    df[nom] = np.arange(len(df))
    for role in (Role.IDENTIFIANT, Role.IGNOREE):
        cs = constats(df, "y", Probleme.REGRESSION, {nom: role})
        assert not [c for c in cs if c.startswith("I3.")]


def test_identifiant_textuel_comme_variable_bloquant() -> None:
    df = table_de_mesures()
    df["id_lot"] = [f"P{i:05d}" for i in range(len(df))]
    cs = constats(df, "classe", Probleme.CLASSIFICATION, {"id_lot": Role.CATEGORIELLE})
    assert cs["I3.identifiant:id_lot"].niveau is Niveau.BLOQUANT


def test_identifiants_proposes_en_info_avant_l_objectif() -> None:
    df = table_de_mesures()
    df["Piece_ID"] = np.arange(len(df)) + 1000
    c = {c.code: c for c in auditer(Table(df, nom="essai.csv"), None, None)}["I3.identifiant:Piece_ID"]
    assert c.niveau is Niveau.INFO and "exclue des variables" in c.message


def test_une_colonne_de_mesure_n_est_pas_prise_pour_un_identifiant() -> None:
    cs = constats(table_de_mesures(), "y", Probleme.REGRESSION, {})
    assert not [c for c in cs if c.startswith("I3.identifiant")]


# ============================================================
# FUITE PAR LA CIBLE ET CIBLES SŒURS
# ============================================================
def test_fuite_par_correlation_parfaite_comme_taux_de_rebuts() -> None:
    df = table_de_mesures()
    df["taux_pct"] = 100.0 * (1.0 - df["y"] / df["y"].max())  # fonction affine de la cible
    cs = constats(df, "y", Probleme.REGRESSION, {"taux_pct": Role.NUMERIQUE})
    assert cs["I3.fuite:taux_pct"].niveau is Niveau.BLOQUANT
    assert "corrélation 1,000" in cs["I3.fuite:taux_pct"].message


def test_fuite_d_une_grandeur_qui_separe_les_classes() -> None:
    df = table_de_mesures()
    df["indice"] = (
        np.where(df["classe"] == "haut", 10.0, 1.0) + np.random.default_rng(1).random(len(df)) * 0.5
    )
    cs = constats(df, "classe", Probleme.CLASSIFICATION, {"indice": Role.NUMERIQUE})
    assert cs["I3.fuite:indice"].niveau is Niveau.BLOQUANT and "séparation" in cs["I3.fuite:indice"].message


def test_fuite_par_copie_categorielle_de_la_cible() -> None:
    df = table_de_mesures()
    df["controle_visuel"] = df["classe"].map({"haut": "KO", "bas": "OK"})
    cs = constats(df, "classe", Probleme.CLASSIFICATION, {"controle_visuel": Role.CATEGORIELLE})
    assert cs["I3.fuite:controle_visuel"].niveau is Niveau.BLOQUANT


def test_fuite_d_une_categorie_deduite_d_une_cible_continue() -> None:
    df = table_de_mesures()
    df["tranche"] = pd.cut(
        df["y"], [-np.inf, df["y"].quantile(1 / 3), df["y"].quantile(2 / 3), np.inf], labels=["A", "B", "C"]
    ).astype(str)
    cs = constats(df, "y", Probleme.REGRESSION, {"tranche": Role.CATEGORIELLE})
    assert cs["I3.fuite:tranche"].niveau is Niveau.BLOQUANT


@pytest.mark.parametrize("suffixe", ["outil", "hydraulique", "puissance", "surcharge", "alimentation"])
def test_cibles_soeurs_panne_avec_preuve_dans_les_valeurs(suffixe: str) -> None:
    """`Panne_<mode>` vaut 1 seulement quand `Panne` vaut 1 : la valeur de la colonne implique la cible."""
    rng = np.random.default_rng(0)
    df = table_de_mesures()
    df["Panne"] = rng.choice([0, 1], len(df), p=[0.9, 0.1])
    df[f"Panne_{suffixe}"] = ((df["Panne"] == 1) & (rng.random(len(df)) < 0.6)).astype(int)
    cs = constats(df, "Panne", Probleme.CLASSIFICATION, {f"Panne_{suffixe}": Role.NUMERIQUE})
    c = cs[f"I3.soeur:Panne_{suffixe}"]
    assert c.niveau is Niveau.BLOQUANT and "implique" in c.message


@pytest.mark.parametrize("suffixe", ["outil", "ambiante", "x"])
def test_nom_de_soeur_sans_implication_alerte_avec_recours(suffixe: str) -> None:
    """Un nom dérivé de la cible ne suffit pas : sans preuve dans les valeurs, alerte que l'on acquitte."""
    rng = np.random.default_rng(1)
    df = table_de_mesures()
    df["Panne"] = rng.choice([0, 1], len(df), p=[0.9, 0.1])
    df[f"Panne_{suffixe}"] = rng.choice([0, 1], len(df), p=[0.8, 0.2])
    c = constats(df, "Panne", Probleme.CLASSIFICATION, {f"Panne_{suffixe}": Role.NUMERIQUE})[
        f"I3.soeur:Panne_{suffixe}"
    ]
    assert c.niveau is Niveau.ALERTE and "Acquitter" in c.action


def test_cible_soeur_dans_l_autre_sens() -> None:
    df = table_de_mesures()
    df["Panne"] = np.tile([0, 1], len(df) // 2)
    df["Panne_outil"] = np.tile([1, 0], len(df) // 2)
    cs = constats(df, "Panne_outil", Probleme.CLASSIFICATION, {"Panne": Role.NUMERIQUE})
    assert cs["I3.soeur:Panne"].niveau is Niveau.BLOQUANT


def test_une_forte_correlation_legitime_ne_bloque_pas() -> None:
    df = table_de_mesures()
    df["capteur_bruite"] = df["y"] + np.random.default_rng(4).normal(0.0, df["y"].std() * 0.3, len(df))
    cs = constats(df, "y", Probleme.REGRESSION, {"capteur_bruite": Role.NUMERIQUE})
    assert not [c for c in cs.values() if c.bloquant]


def test_la_fuite_est_cherchee_sur_les_lignes_d_apprentissage_seules() -> None:
    """Une variable qui copie la cible sur les seules lignes de test n'est pas vue : le test reste scellé."""
    df = table_de_mesures()
    df["copie_en_test"] = np.random.default_rng(7).normal(size=len(df))
    test = np.arange(0, len(df), 5)
    df.loc[test, "copie_en_test"] = df.loc[test, "y"]
    t = Table(df, nom="essai.csv")
    s = specification(df, "y", Probleme.REGRESSION, {"copie_en_test": Role.NUMERIQUE})
    coffre = CoffreTest.par_positions(t, test, decoupage=s.decoupage())
    assert not [c for c in auditer(t, s, coffre) if c.code.startswith("I3.fuite")]


# ============================================================
# ÉTIQUETTE ISSUE D'UN MODÈLE ET ATTRIBUT PERSONNEL
# ============================================================
@pytest.mark.parametrize("nom", ["cluster_gmm", "cluster_kmeans", "classe_predite", "proba_defaut"])
def test_etiquette_issue_d_un_modele_en_alerte(nom: str) -> None:
    df = table_de_mesures()
    df[nom] = np.random.default_rng(2).integers(0, 4, len(df))
    cs = constats(df, "y", Probleme.REGRESSION, {nom: Role.NUMERIQUE})
    assert cs[f"I3.etiquette_modele:{nom}"].niveau is Niveau.ALERTE


@pytest.mark.parametrize("nom", ["operateur_genre", "sexe", "nationalite", "age_operateur"])
def test_attribut_personnel_en_alerte(nom: str) -> None:
    df = table_de_mesures()
    df[nom] = np.random.default_rng(2).choice(["a", "b"], len(df))
    cs = constats(df, "y", Probleme.REGRESSION, {nom: Role.CATEGORIELLE})
    assert cs[f"I3.personnel:{nom}"].niveau is Niveau.ALERTE


def test_alerte_a_acquitter_pour_poursuivre_et_bloquant_a_corriger() -> None:
    df = table_de_mesures()
    df["cluster_gmm"] = np.random.default_rng(2).integers(0, 4, len(df))
    t = Table(df, nom="essai.csv")
    s = specification(df, "y", Probleme.REGRESSION, {"cluster_gmm": Role.NUMERIQUE})
    s3 = Session(auditeur=auditer).appliquer(Charger(t)).appliquer(Auditer()).appliquer(FixerObjectif(s))
    assert not s3.bloquants() and [c.code for c in s3.alertes_en_attente()] == [
        "I3.etiquette_modele:cluster_gmm"
    ]
    s4 = s3.appliquer(
        Decouper(CoffreTest.par_positions(t, np.arange(0, len(df), 5), decoupage=s.decoupage()))
    )
    suite = Pretraiter(s.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),)))
    with pytest.raises(TransitionRefusee, match="alertes à acquitter"):
        s4.appliquer(suite)
    s5 = s4.appliquer(Acquitter(frozenset({"I3.etiquette_modele:cluster_gmm"}))).appliquer(suite)
    assert s5.etat == 5


# ============================================================
# DÉTECTION EN MOINS DE 2 S
# ============================================================
def test_detection_en_moins_de_2_s_a_20000_lignes_et_60_colonnes() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame(rng.normal(size=(n, 56)), columns=[f"x{i:02d}" for i in range(56)])
    df["materiau"] = rng.choice(["a", "b", "c"], n)
    df["cible"] = df["x00"] * 2.0 + rng.normal(scale=0.5, size=n)
    df["grandeur_deduite"] = df["cible"] * 100.0
    df["Piece_ID"] = np.arange(n)
    t = Table(df, nom="essai.csv")
    roles = (
        *((c, Role.NUMERIQUE) for c in df.columns if c.startswith("x")),
        ("materiau", Role.CATEGORIELLE),
        ("cible", Role.CIBLE),
        ("grandeur_deduite", Role.NUMERIQUE),
        ("Piece_ID", Role.NUMERIQUE),
    )
    s = SpecProjet(Probleme.REGRESSION, "cible", roles, "rmse")
    debut = time.perf_counter()
    cs = {c.code: c for c in auditer(t, s, None)}
    duree = time.perf_counter() - debut
    assert cs["I3.fuite:grandeur_deduite"].bloquant and cs["I3.identifiant:Piece_ID"].bloquant
    assert duree < BUDGET_S, f"audit en {duree:.2f} s"


# ============================================================
# CAS DORÉS SUR LES JEUX AUTORISÉS
# ============================================================
JEUX = Path(os.environ["MECAML_JEUX"]) if os.environ.get("MECAML_JEUX") else None

# (jeu, cible ou None, constats bloquants attendus quand toutes les colonnes servent de variables)
DORES = [
    ("Traitement_Acier", "resistance_traction_MPa", set()),
    ("Plaque_Trouee_FEA", "Stress/MPa", {"I3.identifiant:Unnamed_0"}),
    (
        "Qualite_soudure",
        "qualite_soudure",
        {"I3.identifiant:id_soudure", "I3.fuite:controle_visuel_resultat"},
    ),
    (
        "Reduction_Rebuts",
        "classe_qualite",
        {"I3.fuite:taux_rebuts_pct", "I3.fuite:indice_performance_qualite"},
    ),
    ("Suivi_Production_Machines", "etat_sante_machine", {"I3.fuite:indice_sante_machine_0_100"}),
    ("Surveillance_Vibratoire", "duree_vie_restante_RUL_h", set()),
    ("Criticite_Fissures", "etat_fissure", set()),
    ("Diagnostic_Vibratoire", "Defaut", {"I3.identifiant:Piece_ID"}),
    ("Qualite_Surface_Usinee", "classe_qualite_surface", {"I3.fuite:rugosite_ra_um"}),
    ("RUL", "RUL_jours", set()),
    ("Resistance_Composite", "resistance_traction_MPa", set()),
    ("TimeSeries_RUL", "Duree_Vie_Restante_Cycles", set()),
    ("AI4I_maintenance", "Panne", set()),
    (
        "AI4I_maintenance_TypePanne",
        "Panne",
        # « alimentation » (défaillances aléatoires) n'implique pas `Panne` : alerte, non blocage
        {f"I3.soeur:Panne_{m}" for m in ("outil", "hydraulique", "puissance", "surcharge")},
    ),
    ("Controle_Conformite", "etat_piece", {"I3.identifiant:id_piece"}),
    ("Usinage_CNC", "rugosite_Ra_um", set()),
    ("Classification_TypeDePanne", "type_panne", set()),
    ("Identification_Conformite", "presence_defaut", set()),
    ("Resistance_Traction_Composite", "resistance_traction_MPa", set()),
    ("Etude_de_defaut", "defaut", set()),
    ("Anomalies_Pompe", None, set()),
    ("Banc_MotoPompe", None, set()),
]


def jeu_autorise(identifiant: str) -> Table:
    if JEUX is None or not (JEUX / f"MecaIA-{identifiant}.xlsx").is_file():
        pytest.skip("MECAML_JEUX ne désigne pas un dossier contenant ce jeu")
    return charger(JEUX / f"MecaIA-{identifiant}.xlsx").table


def specification_naive(table: Table, cible: str | None) -> SpecProjet:
    """Toutes les colonnes comme variables, sans rien écarter : ce qu'un utilisateur pressé ferait."""
    df = table.donnees
    roles = []
    for c in df.columns:
        if c == cible:
            roles.append((c, Role.CIBLE))
        elif pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            roles.append((c, Role.NUMERIQUE))
        else:
            roles.append((c, Role.CATEGORIELLE))
    if cible is None:
        return SpecProjet(Probleme.SANS_CIBLE, None, tuple(roles), "silhouette")
    probleme = detecter_probleme(df[cible])[0]
    return SpecProjet(probleme, cible, tuple(roles), "rmse" if probleme is Probleme.REGRESSION else "f1")


@pytest.mark.parametrize(("jeu", "cible", "bloquants"), DORES, ids=[d[0] for d in DORES])
def test_cas_dores_bloquants_exacts_et_audit_en_moins_de_2_s(
    jeu: str, cible: str | None, bloquants: set[str]
) -> None:
    table = jeu_autorise(jeu)
    debut = time.perf_counter()
    auditer(table, None, None)
    duree_s2 = time.perf_counter() - debut
    spec = specification_naive(table, cible)
    debut = time.perf_counter()
    constats_s3 = auditer(table, spec, None)
    duree_s3 = time.perf_counter() - debut
    assert {c.code for c in constats_s3 if c.bloquant} == bloquants
    assert duree_s2 < BUDGET_S and duree_s3 < BUDGET_S, f"S2 {duree_s2:.2f} s, S3 {duree_s3:.2f} s"


def test_cas_dores_alertes_attendues() -> None:
    rebuts = {
        c.code: c
        for c in auditer(
            jeu_autorise("Reduction_Rebuts"),
            specification_naive(jeu_autorise("Reduction_Rebuts"), "classe_qualite"),
            None,
        )
    }
    assert rebuts["I3.personnel:operateur_genre"].niveau is Niveau.ALERTE
    banc = jeu_autorise("Banc_MotoPompe")
    codes = {c.code: c for c in auditer(banc, specification_naive(banc, None), None)}
    assert codes["I3.etiquette_modele:cluster_gmm"].niveau is Niveau.ALERTE
    fea = jeu_autorise("Plaque_Trouee_FEA")
    codes = {c.code: c for c in auditer(fea, specification_naive(fea, "Stress/MPa"), None)}
    assert codes["I3.relation_monotone:Disp_mm"].niveau is Niveau.ALERTE  # autre sortie du même calcul
    assert codes["I3.relation_monotone:d0"].niveau is Niveau.ALERTE  # la loi étudiée : à acquitter


@pytest.mark.parametrize(("jeu", "cible", "_"), DORES, ids=[d[0] for d in DORES])
def test_roles_proposes_donnent_un_audit_sans_bloquant(jeu: str, cible: str | None, _: set[str]) -> None:
    table = jeu_autorise(jeu)
    roles = proposer_roles(table, cible)
    structure = proposer_structure(roles)
    if cible is None:
        spec = SpecProjet(Probleme.SANS_CIBLE, None, roles, "silhouette", structure)
    else:
        probleme = detecter_probleme(table.donnees[cible])[0]
        spec = SpecProjet(
            probleme, cible, roles, "rmse" if probleme is Probleme.REGRESSION else "f1", structure
        )
    assert not [c.code for c in auditer(table, spec, None) if c.bloquant]


def test_dore_soeur_sans_preuve_dans_les_valeurs_reste_une_alerte() -> None:
    table = jeu_autorise("AI4I_maintenance_TypePanne")
    cs = {c.code: c for c in auditer(table, specification_naive(table, "Panne"), None)}
    assert cs["I3.soeur:Panne_alimentation"].niveau is Niveau.ALERTE
    assert all(cs[f"I3.soeur:Panne_{m}"].bloquant for m in ("outil", "hydraulique", "puissance", "surcharge"))


def test_dore_unites_suivies_proposees() -> None:
    """`outil` (Usure_Outil) et `machine` (Temperature_Broche) : séries par unité, découpage par unités."""
    for jeu, unite, repere in (
        ("Usure_Outil", "outil", "temps_coupe_min"),
        ("Temperature_Broche", "machine", "horodatage"),
    ):
        table = jeu_autorise(jeu)
        roles = dict(proposer_roles(table))
        assert roles[unite] is Role.GROUPE and roles[repere] is Role.DATE, jeu
        assert proposer_structure(tuple(roles.items())) is Structure.GROUPEE_CHRONOLOGIQUE


def test_dore_etiquette_de_modele_jamais_proposee_comme_cible() -> None:
    table = jeu_autorise("Banc_MotoPompe")
    assert proposer_cible(profiler(table)) != "cluster_gmm"


def test_dore_flottants_physiques_jamais_identifiants() -> None:
    """Aucun flottant des 22 jeux n'est proposé comme identifiant ni comme groupe."""
    for jeu, _cible, _ in DORES:
        for p in profiler(jeu_autorise(jeu)):
            if p.flottante:
                assert p.role_propose not in (Role.IDENTIFIANT, Role.GROUPE), (jeu, p.nom)
