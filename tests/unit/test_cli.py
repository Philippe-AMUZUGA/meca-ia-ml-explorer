"""Ligne de commande : le projet JSON se rejoue par les mêmes actions de session que l'application."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml import __version__, cli
from mecaml.core import codegen, report, split, train
from mecaml.core.io import charger
from mecaml.core.session import Etat
from mecaml.core.types import ChoixTechnique, Probleme, SpecProjet

CT = ChoixTechnique.creer
RIDGE = CT("regression.ridge")
ARBRE = CT("regression.arbre", max_depth=3)
LOGISTIQUE = CT("classification.logistique")
ARBRE_CLS = CT("classification.arbre", max_depth=3)
OPT = "-" * 2  # préfixe des options longues, écrit sans double trait d'union littéral
RACINE = Path(__file__).resolve().parents[2]


def _ecrire(dossier: Path, jeu: pd.DataFrame, spec: SpecProjet, **rubriques: Any) -> Path:
    """Fichier de données et projet JSON écrits dans le dossier ; le chemin du projet est rendu."""
    dossier.mkdir(parents=True, exist_ok=True)
    aides.ecrire_csv(jeu, dossier / "jeu.csv")
    projet = {
        "format": cli.FORMAT_DU_PROJET,
        "donnees": {"fichier": "jeu.csv", "feuille": None},
        "projet": spec.vers_dict(),
        **rubriques,
    }
    chemin = dossier / "projet.json"
    chemin.write_text(json.dumps(projet, indent=2), encoding="utf-8")
    return chemin


def _spec_regression(**changements: Any) -> SpecProjet:
    return jeux.spec_regression(modeles=(RIDGE, ARBRE), **changements)


def _spec_classification() -> SpecProjet:
    return jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS))


@pytest.fixture(scope="module")
def base_partagee(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dossier des projets que les tests lisent sans jamais les modifier."""
    return tmp_path_factory.mktemp("cli")


@pytest.fixture(scope="module")
def regression(base_partagee: Path) -> Path:
    return _ecrire(
        base_partagee / "reg",
        jeux.jeu_regression(n=120),
        _spec_regression(),
        acquittements=["DONNEES.manquants"],
    )


@pytest.fixture(scope="module")
def classification(base_partagee: Path) -> Path:
    return _ecrire(
        base_partagee / "cls",
        jeux.jeu_classification(n=300),
        _spec_classification(),
        acquittements=["DONNEES.manquants"],
    )


def _rejouer(chemin: Path, **options: Any) -> cli.Execution:
    return cli.rejouer(cli.lire_projet(chemin), chemin.parent, importances=False, **options)


# ============================================================
# LECTURE DU PROJET
# ============================================================
def test_un_projet_illisible_est_refuse_avec_un_message_qui_dit_quoi_faire(tmp_path: Path) -> None:
    with pytest.raises(cli.ErreurProjet, match="Projet illisible"):
        cli.lire_projet(tmp_path / "absent.json")
    (tmp_path / "casse.json").write_text("{ pas du json", encoding="utf-8")
    with pytest.raises(cli.ErreurProjet, match="pas du JSON valide"):
        cli.lire_projet(tmp_path / "casse.json")
    (tmp_path / "autre.json").write_text('{"format": 7}', encoding="utf-8")
    with pytest.raises(cli.ErreurProjet, match="au format 1"):
        cli.lire_projet(tmp_path / "autre.json")
    (tmp_path / "incomplet.json").write_text('{"format": 1, "donnees": {}}', encoding="utf-8")
    with pytest.raises(cli.ErreurProjet, match="pas de rubrique"):
        cli.lire_projet(tmp_path / "incomplet.json")
    (tmp_path / "liste.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(cli.ErreurProjet, match="au format 1"):
        cli.lire_projet(tmp_path / "liste.json")


# ============================================================
# REJEU : DE S0 À S8, PUIS S9
# ============================================================
def test_le_rejeu_mene_la_session_jusqu_au_test_et_ouvre_le_test_une_fois(regression: Path) -> None:
    execution = _rejouer(regression)
    session = execution.session
    assert session.etat is Etat.TESTE and session.coffre.ouvertures == 1 and not session.test_optimiste
    assert execution.export is None and session.decision.modele_id == "regression.ridge"
    assert execution.decoupage.methode is split.Methode.ALEATOIRE
    assert session.acquittes == frozenset({"DONNEES.manquants"})


def test_le_rejeu_donne_les_scores_de_l_application(regression: Path) -> None:
    execution = _rejouer(regression)
    table = execution.chargement.table
    spec = SpecProjet.depuis_dict(cli.lire_projet(regression)["projet"])
    decoupage, coffre = split.sceller(table, spec)
    direct = train.entrainer(spec, table, coffre)
    for modele in direct.modeles:
        voulu = {k: list(v) for k, v in modele.scores_plis.items()}
        obtenu = execution.session.resultats.modele(modele.id).scores_plis
        assert aides.ecart_max(voulu, {k: list(v) for k, v in obtenu.items()}) == 0.0
    assert execution.decoupage == decoupage


def test_deux_rejeux_donnent_les_memes_empreintes(regression: Path) -> None:
    a, b = _rejouer(regression), _rejouer(regression)
    assert a.session.resultat_test.empreinte_predictions == b.session.resultat_test.empreinte_predictions
    assert a.session.resultat_test.scores == b.session.resultat_test.scores


def test_le_rejeu_exporte_dans_un_dossier_cree_au_besoin(regression: Path, tmp_path: Path) -> None:
    sortie = tmp_path / "a" / "b" / "sortie"
    execution = _rejouer(regression, sortie=sortie)
    assert execution.session.etat is Etat.EXPORTE and execution.session.export is execution.export
    assert sorted(p.name for p in sortie.iterdir()) == sorted(report.NOMS.values())


def test_le_rejeu_arrete_sur_une_alerte_que_le_projet_n_acquitte_pas(tmp_path: Path) -> None:
    chemin = _ecrire(tmp_path, jeux.jeu_regression(n=120), _spec_regression())  # aucun acquittement
    with pytest.raises(
        cli.ErreurProjet, match=r"1 alerte\(s\) à acquitter.*DONNEES.manquants.*acquittements"
    ):
        _rejouer(chemin)
    assert _rejouer(chemin, tout_acquitter=True).session.etat is Etat.TESTE
    projet = cli.lire_projet(chemin)
    projet["acquittements"] = ["CODE.inexistant"]  # nommer un code absent n'acquitte rien
    with pytest.raises(cli.ErreurProjet, match=r"DONNEES\.manquants"):
        cli.rejouer(projet, chemin.parent)


def test_la_decision_du_projet_est_suivie(classification: Path) -> None:
    projet = cli.lire_projet(classification)
    projet["decision"] = {
        "modele": "classification.arbre",
        "seuil": None,
        "justification": "Le plus lisible.",
    }
    execution = cli.rejouer(projet, classification.parent, importances=False)
    choix = execution.session.decision
    assert choix.modele_id == "classification.arbre" and choix.justification == "Le plus lisible."
    projet["decision"] = {"modele": "classification.logistique", "seuil": 0.3}
    avec_seuil = cli.rejouer(projet, classification.parent, importances=False).session
    assert avec_seuil.decision.seuil == 0.3 and avec_seuil.resultat_test.seuil == 0.3
    projet["decision"] = {"modele": "classification.foret"}
    with pytest.raises(ValueError, match="absent de la comparaison"):
        cli.rejouer(projet, classification.parent, importances=False)


def test_le_donnees_peuvent_etre_remplacees_par_celles_de_la_ligne_de_commande(
    regression: Path, tmp_path: Path
) -> None:
    copie = tmp_path / "ailleurs" / "autre_nom.csv"
    copie.parent.mkdir()
    copie.write_bytes((regression.parent / "jeu.csv").read_bytes())
    execution = _rejouer(regression, donnees=copie)
    assert execution.session.table.nom == "autre_nom.csv" and execution.session.etat is Etat.TESTE
    projet = cli.lire_projet(regression)
    projet["donnees"] = {}
    with pytest.raises(cli.ErreurProjet, match="quel fichier lire"):
        cli.rejouer(projet, regression.parent)


def test_le_rejeu_trace_ses_etapes(regression: Path) -> None:
    messages: list[str] = []
    _rejouer(regression, avancer=messages.append)
    assert messages == [
        "Lecture des données",
        "Objectif et découpage",
        "Prétraitement et entraînement",
        "Ouverture du jeu de test",
    ]


# ============================================================
# REJEU D'UN PROJET EXPORTÉ : LES SCORES ATTENDUS
# ============================================================
@pytest.fixture(scope="module")
def exporte(regression: Path, base_partagee: Path) -> Path:
    sortie = base_partagee / "export"
    _rejouer(regression, sortie=sortie)
    return sortie / "projet.json"


def test_un_projet_exporte_se_rejoue_sur_ses_scores_attendus(exporte: Path) -> None:
    projet = cli.lire_projet(exporte)
    assert "base_test" in projet["decoupage"] and projet["acquittements"] == ["DONNEES.manquants"]
    execution = cli.rejouer(
        projet, exporte.parent, donnees=exporte.parent.parent / "reg" / "jeu.csv", importances=False
    )
    assert cli.verifier_attendus(execution, projet["attendus"]) == []
    assert execution.decoupage.conforme_contrat  # le découpage relu est celui de la base de test déclarée


def test_les_ecarts_au_dela_de_la_tolerance_sont_rapportes_et_en_dessous_tolerees(exporte: Path) -> None:
    projet = cli.lire_projet(exporte)
    execution = cli.rejouer(
        projet, exporte.parent, donnees=exporte.parent.parent / "reg" / "jeu.csv", importances=False
    )
    attendus = copy.deepcopy(projet["attendus"])
    attendus["validation"]["regression.ridge"]["rmse"][1] += 1e-10
    assert cli.verifier_attendus(execution, attendus) == []
    attendus["validation"]["regression.ridge"]["rmse"][1] += 1e-6
    ecarts = cli.verifier_attendus(execution, attendus)
    assert len(ecarts) == 1 and "regression.ridge, rmse, pli 2" in ecarts[0]
    attendus = copy.deepcopy(projet["attendus"])
    attendus["test"]["scores"]["rmse"] += 0.5
    attendus["empreinte_test"] = "0" * 64
    del attendus["validation"]["regression.arbre"]
    ecarts = " | ".join(cli.verifier_attendus(execution, attendus))
    assert (
        "test, rmse" in ecarts
        and "empreinte du test" in ecarts
        and "modèle regression.arbre absent" in ecarts
    )


@pytest.mark.parametrize(
    ("cle", "intitule"),
    [
        (("empreinte_donnees",), "empreinte des données"),
        (("empreinte_test",), "empreinte du test"),
        (("empreinte_plis",), "empreinte des plis"),
        (("test", "empreinte_predictions"), "empreinte des prédictions"),
    ],
)
def test_chaque_empreinte_gardee_par_le_projet_est_comparee(
    exporte: Path, cle: tuple[str, ...], intitule: str
) -> None:
    projet = cli.lire_projet(exporte)
    execution = cli.rejouer(
        projet, exporte.parent, donnees=exporte.parent.parent / "reg" / "jeu.csv", importances=False
    )
    attendus = copy.deepcopy(projet["attendus"])
    cible = attendus
    for nom in cle[:-1]:
        cible = cible[nom]
    cible[cle[-1]] = "0" * 64
    ecarts = cli.verifier_attendus(execution, attendus)
    assert len(ecarts) == 1 and ecarts[0].startswith(intitule)


def test_un_rejeu_sur_d_autres_donnees_ne_reproduit_pas_les_empreintes(exporte: Path, tmp_path: Path) -> None:
    jeu = pd.read_csv(exporte.parent.parent / "reg" / "jeu.csv")
    jeu.loc[5, "d"] += 1.0
    modifie = aides.ecrire_csv(jeu, tmp_path / "modifie.csv")
    projet = cli.lire_projet(exporte)
    execution = cli.rejouer(projet, exporte.parent, donnees=modifie, importances=False)
    ecarts = cli.verifier_attendus(execution, projet["attendus"])
    assert any("empreinte des données" in e for e in ecarts) and len(ecarts) > 2


def test_une_base_de_test_alteree_est_refusee(exporte: Path) -> None:
    projet = cli.lire_projet(exporte)
    projet["decoupage"]["base_test"]["empreinte_sha256"] = "0" * 64
    with pytest.raises(split.ErreurDecoupage, match="diffère de la base déclarée"):
        cli.rejouer(
            projet, exporte.parent, donnees=exporte.parent.parent / "reg" / "jeu.csv", importances=False
        )


def test_un_fichier_different_du_fichier_publie_arrete_l_export(regression: Path, tmp_path: Path) -> None:
    projet = cli.lire_projet(regression)
    projet["donnees"]["sha256_publie"] = "0" * 64
    sortie = tmp_path / "sortie"
    with pytest.raises(codegen.ErreurExport, match="n'est pas le fichier publié"):
        cli.rejouer(projet, regression.parent, sortie=sortie, importances=False)
    assert not list(sortie.iterdir())


def test_un_test_fourni_a_part_traverse_l_export_et_le_rejeu(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=120)
    dossier = tmp_path / "fichier"
    dossier.mkdir()
    aides.ecrire_csv(jeu.iloc[:100], dossier / "app.csv")
    aides.ecrire_csv(jeu.iloc[100:], dossier / "test.csv")
    spec = _spec_regression()
    projet = {
        "format": 1,
        "donnees": {"fichier": "app.csv", "fichier_test": "test.csv"},
        "projet": spec.vers_dict(),
        "acquittements": ["DONNEES.manquants"],
    }
    execution = cli.rejouer(projet, dossier, sortie=dossier / "sortie", importances=False)
    assert execution.decoupage.methode is split.Methode.FICHIER and execution.session.coffre.mode_challenge
    exporte = cli.lire_projet(dossier / "sortie" / "projet.json")
    assert exporte["donnees"]["fichier_test"] == "test.csv" and "base_test" not in exporte["decoupage"]
    nouveau = cli.rejouer(exporte, dossier, importances=False)
    assert cli.verifier_attendus(nouveau, exporte["attendus"]) == []
    script = (dossier / "sortie" / codegen.NOM_DU_SCRIPT).read_text(encoding="utf-8")
    assert f"{OPT}test" in script


def test_un_projet_sans_cible_se_rejoue_et_s_exporte_depuis_s7(tmp_path: Path) -> None:
    spec = jeux.spec_sans_cible(
        modeles=(CT("sans_cible.kmeans", n_clusters=2), CT("sans_cible.acp", n_components=2))
    )
    chemin = _ecrire(tmp_path, jeux.jeu_sans_cible(n=120), spec, decision={"modele": "sans_cible.kmeans"})
    execution = _rejouer(chemin, sortie=tmp_path / "sortie")
    assert execution.session.etat is Etat.EXPORTE and execution.session.resultat_test is None
    assert execution.decoupage is None and execution.session.decision.modele_id == "sans_cible.kmeans"
    exporte = cli.lire_projet(tmp_path / "sortie" / "projet.json")
    assert cli.verifier_attendus(cli.rejouer(exporte, tmp_path, importances=False), exporte["attendus"]) == []


# ============================================================
# PROJET PROPOSÉ D'APRÈS LES DONNÉES
# ============================================================
def _table(tmp_path: Path, jeu: pd.DataFrame):
    return charger(aides.ecrire_csv(jeu, tmp_path / "jeu.csv")).table


def test_le_projet_propose_pour_une_regression(tmp_path: Path) -> None:
    table = _table(tmp_path, jeux.jeu_regression(n=120))
    projet = cli.proposer_projet(table, cible="y")
    spec = SpecProjet.depuis_dict(projet["projet"])
    assert spec.probleme is Probleme.REGRESSION and spec.cible == "y" and spec.metrique == "rmse"
    assert [c.id for c in spec.modeles] == [c.id for c in train.proposer_modeles(Probleme.REGRESSION)]
    assert {c.id for c in spec.pretraitement} >= {
        "manquants.simple",
        "encodage.one_hot",
        "transformation.standard",
    }
    assert projet["a_confirmer"] == [] and projet["decoupage"] is None
    json.dumps(projet)  # sérialisable tel quel


def test_le_projet_propose_signale_la_classe_positive_a_confirmer(tmp_path: Path) -> None:
    table = _table(tmp_path, jeux.jeu_classification(n=300))
    projet = cli.proposer_projet(table)
    spec = SpecProjet.depuis_dict(projet["projet"])
    assert spec.probleme is Probleme.CLASSIFICATION and spec.metrique == "f1"
    assert spec.classe_positive in {"defaut", "conforme"}
    assert projet["a_confirmer"] and "classe positive proposée" in projet["a_confirmer"][0]


def test_le_projet_propose_pour_plus_de_deux_classes_et_sans_cible(tmp_path: Path) -> None:
    table = _table(tmp_path, jeux.jeu_classification(n=300, classes=3))
    spec = SpecProjet.depuis_dict(cli.proposer_projet(table, cible="y")["projet"])
    assert spec.metrique == "exactitude_equilibree" and spec.classe_positive is None
    (tmp_path / "sc").mkdir()
    sans = _table(tmp_path / "sc", jeux.jeu_sans_cible(n=100))
    spec = SpecProjet.depuis_dict(cli.proposer_projet(sans, sans_cible=True)["projet"])
    assert spec.probleme is Probleme.SANS_CIBLE and spec.cible is None and spec.metrique == "silhouette"


def test_le_projet_propose_voit_les_unites_suivies_dans_le_temps(tmp_path: Path) -> None:
    table = _table(tmp_path, jeux.jeu_serie_mesuree())
    spec = SpecProjet.depuis_dict(cli.proposer_projet(table, cible="y")["projet"])
    assert spec.colonne_groupe == "machine" and spec.colonne_date == "cycle"
    assert spec.structure.groupee and spec.structure.chronologique


def test_un_projet_propose_se_rejoue_sans_modification(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=120)
    table = _table(tmp_path, jeu)
    projet = cli.proposer_projet(table, cible="y")
    projet["acquittements"] = ["DONNEES.manquants"]
    execution = cli.rejouer(projet, tmp_path, importances=False)
    assert execution.session.etat is Etat.TESTE and execution.session.resultats.reference() is not None


# ============================================================
# POINT D'ENTRÉE ET CODES DE RETOUR
# ============================================================
def test_rejouer_en_ligne_de_commande_affiche_le_resume_et_rend_zero(
    regression: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.principal(
        ["rejouer", str(regression), "-o", str(tmp_path / "sortie"), f"{OPT}sans-importances"]
    )
    sortie = capsys.readouterr()
    assert code == 0
    assert (
        "Comparaison des modèles (rmse, validation.kfold)" in sortie.out
        and "Modèle retenu : regression.ridge" in sortie.out
    )
    assert "Export dans" in sortie.out and "rapport.html" in sortie.out
    assert "... Lecture des données" in sortie.err


def test_une_alerte_non_acquittee_rend_un_et_un_message_sur_la_sortie_d_erreur(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chemin = _ecrire(tmp_path, jeux.jeu_regression(n=120), _spec_regression())
    assert cli.principal(["rejouer", str(chemin), f"{OPT}sans-importances"]) == 1
    erreur = capsys.readouterr().err
    assert erreur.count("Erreur :") == 1 and "DONNEES.manquants" in erreur
    assert cli.principal(["rejouer", str(chemin), "-a", f"{OPT}sans-importances"]) == 0


def test_la_verification_rend_zero_si_le_rejeu_reproduit_et_un_sinon(
    exporte: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    donnees = str(exporte.parent.parent / "reg" / "jeu.csv")
    assert cli.principal(["rejouer", str(exporte), "-d", donnees, "-v", f"{OPT}sans-importances"]) == 0
    assert "reproduit les scores attendus" in capsys.readouterr().out
    projet = cli.lire_projet(exporte)
    projet["attendus"]["test"]["scores"]["rmse"] += 1.0
    abime = tmp_path / "abime.json"
    abime.write_text(json.dumps(projet), encoding="utf-8")
    assert cli.principal(["rejouer", str(abime), "-d", donnees, "-v", f"{OPT}sans-importances"]) == 1
    assert "Le rejeu diffère du projet" in capsys.readouterr().err


def test_un_fichier_absent_rend_un_avec_un_message_actionnable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chemin = _ecrire(tmp_path, jeux.jeu_regression(n=60), _spec_regression())
    code = cli.principal(["rejouer", str(chemin), "-d", str(tmp_path / "absent.csv")])
    assert code == 1 and "Fichier introuvable" in capsys.readouterr().err


def test_l_usage_incorrect_rend_deux() -> None:
    with pytest.raises(SystemExit) as sortie:
        cli.principal(["rejouer"])
    assert sortie.value.code == 2
    with pytest.raises(SystemExit) as sortie:
        cli.principal([])
    assert sortie.value.code == 2


def test_la_version_et_l_aide(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as sortie:
        cli.principal([f"{OPT}version"])
    assert sortie.value.code == 0 and f"MécaML Explorer {__version__}" in capsys.readouterr().out


def test_auditer_affiche_les_constats_par_niveau(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fichier = aides.ecrire_csv(jeux.jeu_regression(n=120), tmp_path / "jeu.csv")
    assert cli.principal(["auditer", str(fichier)]) == 0
    sortie = capsys.readouterr().out
    assert "BLOQUANT (0)" in sortie and "ALERTE (1)" in sortie and "DONNEES.manquants" in sortie
    assert sortie.index("BLOQUANT") < sortie.index("ALERTE") < sortie.index("INFO")
    projet = _ecrire(tmp_path / "p", jeux.jeu_regression(n=120), _spec_regression())
    assert cli.principal(["auditer", str(fichier), "-p", str(projet)]) == 0


def test_proposer_ecrit_un_projet_ou_l_affiche(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fichier = aides.ecrire_csv(jeux.jeu_classification(n=300), tmp_path / "jeu.csv")
    cible = tmp_path / "projet.json"
    assert cli.principal(["proposer", str(fichier), "-c", "y", "-s", str(cible)]) == 0
    sortie = capsys.readouterr()
    assert (
        "à relire avant de le rejouer" in sortie.out
        and "A confirmer : classe positive proposée" in sortie.err
    )
    assert cli.lire_projet(cible)["projet"]["cible"] == "y"
    assert cli.principal(["proposer", str(fichier), "-c", "y"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == 1


def test_le_module_se_lance_en_processus_neuf() -> None:
    environnement = {**os.environ, "PYTHONPATH": str(RACINE / "src"), "PYTHONIOENCODING": "utf-8"}
    sortie = subprocess.run(
        [sys.executable, "-m", "mecaml.cli", f"{OPT}version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environnement,
        timeout=120,
        check=False,
    )
    assert sortie.returncode == 0 and f"MécaML Explorer {__version__}" in sortie.stdout


def test_verifier_sans_scores_attendus_est_refuse_avant_tout_calcul(
    regression: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.principal(["rejouer", str(regression), "-v"]) == 1
    erreur = capsys.readouterr().err
    assert "n'a pas de scores attendus" in erreur and "... Lecture des données" not in erreur


def test_un_projet_propose_qui_garde_des_points_a_confirmer_le_dit_au_rejeu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fichier = aides.ecrire_csv(jeux.jeu_classification(n=300), tmp_path / "jeu.csv")
    cible = tmp_path / "projet.json"
    assert cli.principal(["proposer", str(fichier), "-c", "y", "-s", str(cible)]) == 0
    capsys.readouterr()
    assert cli.principal(["rejouer", str(cible), "-a", f"{OPT}sans-importances"]) == 0
    assert "A confirmer avant de s'y fier : classe positive proposée" in capsys.readouterr().err


def test_une_erreur_de_bibliotheque_est_dite_en_francais_par_la_ligne_de_commande(
    regression: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from sklearn.metrics import confusion_matrix

    def casser(*_args: Any, **_kwargs: Any) -> None:
        confusion_matrix([0, 1, 1], [0, None, 1])

    monkeypatch.setattr(cli, "rejouer", casser)
    assert cli.principal(["rejouer", str(regression), f"{OPT}sans-importances"]) == 1
    erreur = capsys.readouterr().err
    assert erreur.startswith("Erreur : Une erreur imprévue d'une bibliothèque de calcul (ValueError)")
    assert "classification metrics" not in erreur.lower()


def test_un_classeur_absent_a_cote_du_projet_cite_l_option_d(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chemin = _ecrire(tmp_path, jeux.jeu_regression(n=60), _spec_regression())
    (tmp_path / "jeu.csv").unlink()
    assert cli.principal(["rejouer", str(chemin), f"{OPT}sans-importances"]) == 1
    erreur = capsys.readouterr().err
    assert "Fichier introuvable" in erreur and "option -d" in erreur


def test_le_nombre_de_plis_propose_ne_depasse_pas_le_nombre_d_unites(tmp_path: Path) -> None:
    """Quatre machines : le test en prend une, trois restent à l'apprentissage, donc trois plis au plus."""
    chemin = aides.ecrire_csv(jeux.jeu_groupe(n_groupes=4, longueur=14), tmp_path / "parc.csv")
    projet = cli.proposer_projet(charger(chemin).table)
    assert projet["projet"]["structure"].startswith("groupee") and projet["projet"]["n_plis"] == 3
    assert any("plis" in note for note in projet["a_confirmer"])
    execution = cli.rejouer(projet, tmp_path, donnees=chemin, tout_acquitter=True, importances=False)
    assert execution.session.etat is Etat.TESTE


def test_un_jeu_assez_grand_garde_les_cinq_plis_proposes(tmp_path: Path) -> None:
    chemin = aides.ecrire_csv(jeux.jeu_regression(n=200), tmp_path / "jeu.csv")
    projet = cli.proposer_projet(charger(chemin).table)
    assert projet["projet"]["n_plis"] == 5 and projet["a_confirmer"] == []
