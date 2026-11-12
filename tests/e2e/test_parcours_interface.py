"""Parcours scripté de bout en bout par l'interface : les gestes d'un utilisateur, de l'accueil à l'export.

Chaque parcours passe par les écrans (mêmes appels que leurs boutons), puis le projet exporté est rejoué par
la ligne de commande du cœur : le rejeu doit retrouver les scores de l'interface à 1e-9. Les quatre jeux
canoniques se lisent dans le dossier désigné par `MECAML_JEUX` ; les exemples du dépôt sont toujours joués.
"""

from __future__ import annotations

import os
import socket
import time
import tkinter as tk
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as jeux
import aides_interface as ai
import aides_reproductibilite as aides
from mecaml import cli
from mecaml.core import catalog, train
from mecaml.core.session import Etat
from mecaml.core.types import ChoixTechnique
from mecaml.ui.app import Application
from mecaml.ui.textes import t, titre_technique

EXEMPLES = Path(__file__).resolve().parents[2] / "examples"
JEUX = Path(os.environ["MECAML_JEUX"]) if os.environ.get("MECAML_JEUX") else None
FICHIERS_EXPORTES = {"modele.joblib", "projet.json", "rapport.html", "rapport.xlsx", "script_projet.py"}

CT = ChoixTechnique.creer
DUMMY_REG, DUMMY_CLS = CT("reference.dummy_regression"), CT("reference.dummy_classification")
RIDGE, ARBRE_REG = CT("regression.ridge"), CT("regression.arbre", max_depth=4)
LOGISTIQUE, ARBRE_CLS = CT("classification.logistique"), CT("classification.arbre", max_depth=4)
SANS_CIBLE = (
    CT("sans_cible.kmeans", n_clusters=3),
    CT("sans_cible.acp", n_components=2),
    CT("sans_cible.dbscan", eps=0.6, min_samples=5),
    CT("sans_cible.foret_isolement", contamination=0.05),
    CT("sans_cible.facteur_local", contamination=0.05),
)


@pytest.fixture(scope="module")
def app() -> Iterator[Application]:
    fenetre = ai.ouvrir_application()
    yield fenetre
    ai.fermer(fenetre)


def rejouer(dossier: Path, donnees: Path) -> list[str]:
    """Rejoue le projet exporté par la ligne de commande du cœur ; rend les écarts avec les attendus."""
    projet = cli.lire_projet(dossier / "projet.json")
    execution = cli.rejouer(projet, dossier, donnees=donnees, importances=False)
    return cli.verifier_attendus(execution, projet["attendus"])


def parcours(app: Application, chemin: Path, tmp: Path, prefixe: str, **options) -> None:
    """Parcours complet : de l'accueil à l'export, puis rejeu du projet exporté."""
    sans_cible = options.get("sans_cible", False)
    final = Etat.DIAGNOSTIC if sans_cible else Etat.TESTE
    atteint = ai.mener_jusqu_a(app, chemin, final, prefixe=prefixe, **options)
    assert atteint is final
    session = app.ctrl.session
    if sans_cible:
        assert session.coffre is None and session.spec.cible is None
    else:
        assert session.coffre.ouvertures == 1
        assert str(app._rail[9].cget("state")) == "normal", (
            "le rail dégrise l'export dès que le test est ouvert"
        )
    dossier = tmp / prefixe.rstrip("_")
    assert ai.exporter(app, dossier, prefixe=prefixe) == sorted(FICHIERS_EXPORTES)
    assert app.ctrl.etat is Etat.EXPORTE
    assert rejouer(dossier, chemin) == []


# ============================================================
# EXEMPLES DU DÉPÔT : toujours joués
# ============================================================
def test_exemple_regression_acier_sans_reseau(
    app: Application, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parcours complet avec toute connexion interdite : l'application n'ouvre aucun accès réseau."""
    tentatives: list[str] = []

    def refuser(*args, **kwargs):
        tentatives.append("connexion")
        raise AssertionError("accès réseau interdit")

    monkeypatch.setattr(socket.socket, "connect", refuser)
    monkeypatch.setattr(socket.socket, "connect_ex", refuser)
    monkeypatch.setattr(socket, "getaddrinfo", refuser)
    monkeypatch.setattr(socket, "create_connection", refuser)
    parcours(app, EXEMPLES / "MecaIA-Traitement_Acier.xlsx", tmp_path, "acier_")
    spec = app.ctrl.session.spec
    assert spec.cible == "resistance_traction_MPa" and spec.probleme.value == "regression"
    assert tentatives == []
    rapport = (tmp_path / "acier" / "rapport.html").read_text(encoding="utf-8")
    assert 'src="http' not in rapport and 'href="http' not in rapport


def test_exemple_classification_conformite(app: Application, tmp_path: Path) -> None:
    parcours(
        app, EXEMPLES / "MecaIA-Controle_Conformite.xlsx", tmp_path, "conformite_",
        modeles=(DUMMY_CLS, LOGISTIQUE, ARBRE_CLS),
    )  # fmt: skip
    spec = app.ctrl.session.spec
    assert spec.cible == "etat_piece" and spec.classe_positive == "rebut" and spec.metrique == "f1"
    assert app.ctrl.session.resultat_test.scores["f1"] > 0.0


def test_exemple_serie_chronologique_et_groupee(app: Application, tmp_path: Path) -> None:
    parcours(
        app, EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx", tmp_path, "serie_",
        modeles=(DUMMY_REG, RIDGE, ARBRE_REG),
    )  # fmt: skip
    spec = app.ctrl.session.spec
    assert spec.structure.groupee and spec.structure.chronologique
    assert any(c.id == "chronologique.variables_temporelles" for c in spec.pretraitement)
    assert app.ctrl.decoupage.methode.value.startswith("unites")


def test_exemple_sans_cible_pompe(app: Application, tmp_path: Path) -> None:
    parcours(
        app, EXEMPLES / "MecaIA-Anomalies_Pompe.xlsx", tmp_path, "pompe_",
        sans_cible=True, modeles=SANS_CIBLE,
    )  # fmt: skip
    assert app.ctrl.session.sans_cible
    app.afficher(8)
    assert t("ecran.s8.sans_objet") in app.ecrans[8].etat.cget("text")


# ============================================================
# CIBLES CODÉES EN ENTIERS : 0 et 1 (pannes, défauts), chiffres (classes de défauts)
# ============================================================
def _cible_entiere(
    app: Application, chemin: Path, tmp: Path, prefixe: str, classes: int | None = None, **options
) -> None:
    """Parcours complet par l'interface : diagnostic (S7), test (S8), export (S9) et rejeu à 1e-9."""
    atteint = ai.mener_jusqu_a(app, chemin, Etat.TESTE, cible="y", prefixe=prefixe, **options)
    assert atteint is Etat.TESTE, app.banniere.texte
    assert "diagnostic" in app.ctrl.vues, "le diagnostic hors pli de S7 s'est affiché"
    session = app.ctrl.session
    assert np.asarray(session.resultats.modele(session.decision.modele_id).predictions).dtype.kind == "i"
    if classes is not None:
        assert len(session.resultats.classes) == classes
    dossier = tmp / prefixe.rstrip("_")
    assert ai.exporter(app, dossier, prefixe=prefixe) == sorted(FICHIERS_EXPORTES), app.banniere.texte
    assert app.ctrl.etat is Etat.EXPORTE
    assert rejouer(dossier, chemin) == []


def test_cible_entiere_binaire_0_et_1_de_l_interface_au_rejeu(app: Application, tmp_path: Path) -> None:
    jeu = jeux.jeu_classification(n=300, rare=0.15)
    jeu["y"] = (jeu["y"] == "defaut").astype(int)
    chemin = aides.ecrire_csv(jeu, tmp_path / "MecaIA-Defauts_01.csv")
    _cible_entiere(app, chemin, tmp_path, "binaire_", classes=2, modeles=(DUMMY_CLS, LOGISTIQUE, ARBRE_CLS))
    assert app.ctrl.session.spec.classe_positive == 1


def test_cible_entiere_en_chiffres_de_l_interface_au_rejeu(app: Application, tmp_path: Path) -> None:
    jeu = jeux.jeu_classification(n=300, classes=4)
    jeu["y"] = jeu["y"].str.removeprefix("classe_").astype(int)
    chemin = aides.ecrire_csv(jeu, tmp_path / "MecaIA-Marquages_chiffres.csv")
    _cible_entiere(app, chemin, tmp_path, "chiffres_", classes=4, modeles=(DUMMY_CLS, LOGISTIQUE))


def test_cible_entiere_chronologique_de_l_interface_au_rejeu(app: Application, tmp_path: Path) -> None:
    jeu = aides.jeu_chronologique_entier().drop(columns="t")
    jeu.insert(0, "date", pd.date_range("2025-01-01", periods=len(jeu), freq="D").strftime("%Y-%m-%d"))
    chemin = aides.ecrire_csv(jeu, tmp_path / "MecaIA-Serie_pannes.csv")
    _cible_entiere(
        app, chemin, tmp_path, "serie_entiere_", classes=2, modeles=(DUMMY_CLS, LOGISTIQUE, ARBRE_CLS)
    )
    assert app.ctrl.session.spec.structure.chronologique


# ============================================================
# QUATRE JEUX CANONIQUES : avec MECAML_JEUX
# ============================================================
SANS_JEUX = pytest.mark.skipif(JEUX is None, reason="MECAML_JEUX non défini : jeux canoniques absents")


@SANS_JEUX
def test_canonique_RUL(app: Application, tmp_path: Path) -> None:
    parcours(app, JEUX / "MecaIA-RUL.xlsx", tmp_path, "RUL_", modeles=(DUMMY_REG, RIDGE, ARBRE_REG))
    assert app.ctrl.session.spec.cible == "RUL_jours"


@SANS_JEUX
def test_canonique_Qualite_soudure(app: Application, tmp_path: Path) -> None:
    parcours(
        app, JEUX / "MecaIA-Qualite_soudure.xlsx", tmp_path, "Qualite_soudure_",
        modeles=(DUMMY_CLS, LOGISTIQUE, ARBRE_CLS),
    )  # fmt: skip
    spec = app.ctrl.session.spec
    assert spec.cible == "qualite_soudure" and spec.classe_positive == "non-conforme"
    assert all(role.value != "cible" for nom, role in spec.roles if nom != "qualite_soudure")


@SANS_JEUX
def test_canonique_Anomalies_Pompe(app: Application, tmp_path: Path) -> None:
    parcours(
        app,
        JEUX / "MecaIA-Anomalies_Pompe.xlsx",
        tmp_path,
        "Anomalies_Pompe_",
        sans_cible=True,
        modeles=SANS_CIBLE,
    )
    assert app.ctrl.session.sans_cible


@SANS_JEUX
def test_canonique_TimeSeries_RUL(app: Application, tmp_path: Path) -> None:
    parcours(
        app, JEUX / "MecaIA-TimeSeries_RUL.xlsx", tmp_path, "TimeSeries_RUL_",
        modeles=(DUMMY_REG, RIDGE, ARBRE_REG),
    )  # fmt: skip
    session = app.ctrl.session
    assert session.spec.structure.value == "groupee_chronologique"
    assert session.coffre.n_test > 0 and session.resultats.validation.startswith("validation.plis_groupes")


# ============================================================
# GARDE DE LA CIBLE PASSÉE, REJOUÉE AUX ÉCRANS S5 ET S6
# ============================================================
def _ouvrir_reglage_valide(ecran) -> None:
    """Valide la fenêtre de réglage ouverte par l'écran, comme le ferait un clic sur « Valider »."""
    dialogues = [w for w in ecran.liste.winfo_children() if isinstance(w, tk.Toplevel)]
    assert dialogues
    dialogues[-1].valider()


def test_la_cible_passee_est_refusee_aux_ecrans_s5_et_s6(app: Application) -> None:
    ai.mener_jusqu_a(app, EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx", Etat.PRETRAITEMENT)
    app.afficher(6)
    ai.attendre(app)
    s6 = app.ecrans[6]
    offerts = {titre_technique(e.id) for e in s6._offertes()}
    assert titre_technique("reference.persistance") not in offerts, "la case décochée cache la persistance"
    # Sans déclaration : le constat bloquant se lit avant l'entraînement, et l'entraînement est refusé.
    s6.liste.definir((DUMMY_REG, RIDGE))
    s6.liste.ajouter_choix(CT("reference.persistance"))
    ai.attendre(app)
    assert any(c.startswith("I3.cible_passee") for c in s6.constats.codes())
    s6.entrainer()
    ai.attendre(app)
    assert "cible_connue=True" in app.banniere.texte
    assert app.ctrl.etat is Etat.PRETRAITEMENT and app.ctrl.session.resultats is None
    # Déclarée par la case : le réglage pose cible_connue=True ; le compte à rebours est refusé quand même.
    s6.var_cible_connue.set(True)
    s6.liste.definir((DUMMY_REG, RIDGE))
    s6.liste.ouvrir_reglage(catalog.entree("reference.persistance"), None, None)
    _ouvrir_reglage_valide(s6)
    ai.attendre(app)
    assert s6.liste.choix[-1].parametres_dict() == {"cible_connue": True}
    assert any(c.startswith("I3.cible_decompte") for c in s6.constats.codes())
    s6.entrainer()
    ai.attendre(app)
    assert "compte à rebours" in app.banniere.texte.lower()
    assert app.ctrl.etat is Etat.PRETRAITEMENT and app.ctrl.session.resultats is None


def test_la_cible_decalee_du_pretraitement_est_gardee_a_l_ecran_s5(app: Application) -> None:
    ai.mener_jusqu_a(app, EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx", Etat.DECOUPE)
    app.afficher(5)
    ai.attendre(app)
    s5 = app.ecrans[5]
    assert titre_technique("chronologique.cible_decalee") not in {
        titre_technique(e.id) for e in s5._offertes()
    }
    s5.liste.ajouter_choix(CT("chronologique.cible_decalee"))
    ai.attendre(app)
    assert any(c.startswith("I3.cible_passee") for c in s5.constats.codes())
    s5.appliquer()
    ai.attendre(app)
    assert app.banniere.texte and app.ctrl.etat is Etat.DECOUPE


@SANS_JEUX
def test_la_persistance_declaree_est_permise_sur_une_serie_mesuree(app: Application) -> None:
    """Température de broche : grandeur mesurée, non compte à rebours ; la persistance sert de repère."""
    ai.mener_jusqu_a(app, JEUX / "MecaIA-Temperature_Broche.xlsx", Etat.PRETRAITEMENT)
    app.afficher(6)
    ai.attendre(app)
    s6 = app.ecrans[6]
    s6.var_cible_connue.set(True)
    s6.liste.definir((DUMMY_REG, RIDGE))
    s6.liste.ouvrir_reglage(catalog.entree("reference.persistance"), None, None)
    _ouvrir_reglage_valide(s6)
    ai.attendre(app)
    s6.entrainer()
    ai.attendre(app)
    assert app.ctrl.etat is Etat.MODELES, app.banniere.texte
    assert "reference.persistance" in {m.id for m in app.ctrl.session.resultats.modeles}


# ============================================================
# MODE CHALLENGE, ANNULATION, RÉACTIVITÉ, RÉSEAU COUPÉ
# ============================================================
def test_mode_challenge_a_deux_feuilles(app: Application, tmp_path: Path) -> None:
    jeu = pd.read_excel(EXEMPLES / "MecaIA-Traitement_Acier.xlsx")
    chemin = tmp_path / "MecaIA-Challenge_Acier.xlsx"
    with pd.ExcelWriter(chemin) as ecrivain:
        jeu.iloc[:120].to_excel(ecrivain, sheet_name="apprentissage", index=False)
        jeu.iloc[120:].to_excel(ecrivain, sheet_name="test", index=False)
    atteint = ai.mener_jusqu_a(app, chemin, Etat.TESTE)
    assert atteint is Etat.TESTE
    coffre = app.ctrl.session.coffre
    assert coffre.mode_challenge and coffre.n_test == 30 and coffre.n_apprentissage == 120
    assert app.ctrl.decoupage.methode.value == "fichier"
    assert "à part" in app.ecrans[4].note.cget("text")
    assert app.ecrans[1].case_challenge.winfo_exists()


def test_annuler_un_entrainement_laisse_la_session_intacte_et_grise_les_boutons(app: Application) -> None:
    ai.mener_jusqu_a(app, EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx", Etat.PRETRAITEMENT)
    app.afficher(6)
    ai.attendre(app)
    s6 = app.ecrans[6]
    lents = (
        DUMMY_REG,
        RIDGE,
        CT("regression.foret", n_estimators=300),
        CT("regression.gradient_boosting"),
    )
    s6.liste.definir(lents)
    avant = app.ctrl.session
    s6.entrainer()
    assert app.occupe
    assert all(str(b.cget("state")) == "disabled" for b, _ in s6._boutons), "boutons grisés pendant le calcul"
    assert str(app.bouton_annuler.cget("state")) == "normal"
    app.pilote.annuler()
    ai.attendre(app)
    assert str(app.bouton_annuler.cget("state")) == "disabled"
    assert app.statut.get() == t("pied.annule") and app.ctrl.session is avant
    assert app.ctrl.etat is Etat.PRETRAITEMENT and not app.banniere.texte


def test_la_boucle_tk_reste_reactive_pendant_un_entrainement(app: Application) -> None:
    """Pendant l'entraînement, la boucle de Tk ne reste jamais figée 200 ms."""
    ai.chauffer(app)
    ai.mener_jusqu_a(app, EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx", Etat.PRETRAITEMENT)
    app.afficher(6)
    ai.attendre(app)
    modeles = (
        DUMMY_REG,
        RIDGE,
        CT("regression.foret", n_estimators=60),
        CT("regression.gradient_histogramme", max_iter=40),
    )
    app.ecrans[6].liste.definir(modeles)
    instants: list[float] = []

    def battement() -> None:
        instants.append(time.perf_counter())
        if app.occupe:
            app.after(10, battement)

    app.ecrans[6].entrainer()
    app.after(10, battement)
    ai.attendre(app)
    ecarts = [b - a for a, b in pairwise(instants)]
    assert len(ecarts) > 100, "l'entraînement doit durer assez pour être mesuré"
    pendant = ecarts[:-1]
    assert max(pendant) < 0.2, f"boucle figée {max(pendant):.3f} s pendant l'entraînement"
    assert ecarts[-1] < 1.0, f"mise à jour de l'écran après l'entraînement : {ecarts[-1]:.3f} s"


def test_les_resultats_de_l_interface_sont_ceux_du_coeur(app: Application) -> None:
    """Mêmes données, spécification et coffre : le tableau de l'interface est celui de `train.entrainer`."""
    ai.mener_jusqu_a(app, EXEMPLES / "MecaIA-Traitement_Acier.xlsx", Etat.MODELES)
    session = app.ctrl.session
    direct = train.entrainer(session.spec, session.table, session.coffre)
    colonnes = ["modele", "moyenne", "ecart_type", "gain_vs_reference", "plis_valides"]
    assert session.resultats.tableau()[colonnes].equals(direct.tableau()[colonnes])


def test_rouvrir_le_meme_test_est_dit_optimiste(app: Application, tmp_path: Path) -> None:
    """Le même jeu rouvert dans un nouveau projet : le test est compté déjà vu, et l'écran S8 le dit."""
    jeu = pd.read_excel(EXEMPLES / "MecaIA-Traitement_Acier.xlsx").iloc[::2]
    chemin = tmp_path / "MecaIA-Acier_une_ligne_sur_deux.xlsx"
    jeu.to_excel(chemin, index=False)
    modeles = (DUMMY_REG, RIDGE)
    ai.mener_jusqu_a(app, chemin, Etat.TESTE, modeles=modeles)
    assert not app.ctrl.session.score_test_optimiste
    assert t("ecran.s8.optimiste") not in app.ecrans[8].etat.cget("text")
    ai.mener_jusqu_a(app, chemin, Etat.TESTE, modeles=modeles)
    assert app.ctrl.session.score_test_optimiste
    assert t("ecran.s8.optimiste") in app.ecrans[8].etat.cget("text")
