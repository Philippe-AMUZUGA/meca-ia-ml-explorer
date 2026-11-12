"""Contrôleur de l'interface (sans Tk) : il enchaîne les appels du cœur et ne change aucun résultat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mecaml.core import train
from mecaml.core.session import Acquitter, Etat, FixerObjectif, Pretraiter, Revenir, TransitionRefusee
from mecaml.core.tasks import ErreurAnnulation, EtatTache, Tache
from mecaml.core.types import ChoixTechnique, Niveau, SpecProjet
from mecaml.ui.controleur import Controleur, Vue

EXEMPLES = Path(__file__).resolve().parents[2] / "examples"
ACIER = EXEMPLES / "MecaIA-Traitement_Acier.xlsx"
SERIE = EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx"
POMPE = EXEMPLES / "MecaIA-Anomalies_Pompe.xlsx"
RIDGE = ChoixTechnique.creer("regression.ridge")


def executer(travail: Any) -> Any:
    """Exécute un travail du contrôleur dans une vraie tâche (un fil), avec un délai borné."""
    tache = Tache(travail, nom="essai").demarrer()
    assert tache.attendre(180), "travail trop long"
    return tache.resultat()


def jusqu_a_s3(chemin: Path, **proposition: Any) -> Controleur:
    ctrl = Controleur()
    ctrl.adopter_chargee(executer(ctrl.travail_charger(chemin)))
    ctrl.adopter_audit(executer(ctrl.travail_auditer()))
    spec = executer(ctrl.travail_proposer(**proposition))
    ctrl.adopter(executer(ctrl.travail_actions(FixerObjectif(spec))))
    return ctrl


@pytest.fixture()
def acier_decoupe() -> Controleur:
    ctrl = jusqu_a_s3(ACIER, cible=None)
    ctrl.adopter_scelle(executer(ctrl.travail_sceller(ctrl.session.spec, None)))
    return ctrl


def test_le_chargement_rend_une_session_en_s1_avec_ce_qu_il_faut_en_dire() -> None:
    ctrl = Controleur()
    chargee = executer(ctrl.travail_charger(ACIER))
    assert chargee.session.etat is Etat.DONNEES
    assert chargee.chargement.table.n_lignes == 150
    assert chargee.chargement.sha256
    ctrl.adopter_chargee(chargee)
    assert ctrl.etat is Etat.DONNEES and ctrl.chargement is chargee.chargement


def test_un_nouveau_fichier_est_un_nouveau_projet(acier_decoupe: Controleur) -> None:
    ctrl = Controleur(session=acier_decoupe.session, chargement=acier_decoupe.chargement)
    chargee = executer(ctrl.travail_charger(SERIE))
    assert chargee.session.etat is Etat.DONNEES
    assert chargee.session.spec is None and chargee.session.coffre is None
    assert chargee.chargement.table.nom == SERIE.name


def test_l_audit_et_le_profil(acier_decoupe: Controleur) -> None:
    ctrl = Controleur()
    ctrl.adopter_chargee(executer(ctrl.travail_charger(ACIER)))
    audit = executer(ctrl.travail_auditer())
    assert audit.session.etat is Etat.AUDIT
    assert {p.nom for p in audit.profils} == set(ctrl.table.colonnes)


def test_la_proposition_vient_du_coeur() -> None:
    ctrl = Controleur()
    ctrl.adopter_chargee(executer(ctrl.travail_charger(ACIER)))
    ctrl.adopter_audit(executer(ctrl.travail_auditer()))
    spec = executer(ctrl.travail_proposer(cible=None))
    assert isinstance(spec, SpecProjet)
    assert spec.cible == "resistance_traction_MPa" and spec.metrique == "rmse"
    sans = executer(ctrl.travail_proposer(None, sans_cible=True))
    assert not sans.probleme.avec_cible and sans.cible is None


def test_un_refus_de_la_session_remonte_tel_quel() -> None:
    ctrl = Controleur()
    ctrl.adopter_chargee(executer(ctrl.travail_charger(ACIER)))
    tache = Tache(ctrl.travail_actions(Revenir(Etat.AUDIT)), nom="refus").demarrer()
    tache.attendre(30)
    assert tache.etat is EtatTache.ECHEC and isinstance(tache.erreur, TransitionRefusee)
    assert "antérieur" in tache.erreur.motif


def test_le_scelle_est_confie_a_la_session(acier_decoupe: Controleur) -> None:
    session = acier_decoupe.session
    assert session.etat is Etat.DECOUPE and session.coffre is not None
    assert session.coffre.ouvertures == 0
    assert acier_decoupe.decoupage.lignes_test == session.coffre.n_test


def test_le_parcours_du_controleur_donne_les_resultats_du_coeur(acier_decoupe: Controleur) -> None:
    """Chaque étape du contrôleur égale l'appel direct du cœur : aucun calcul n'a été ajouté ni changé."""
    ctrl = Controleur(session=acier_decoupe.session, chargement=acier_decoupe.chargement)
    ctrl.decoupage = acier_decoupe.decoupage
    spec = ctrl.session.spec
    pretraitement = executer(ctrl.travail_proposer_pretraitement())
    ctrl.adopter(executer(ctrl.travail_actions(Pretraiter(spec.avec(pretraitement=pretraitement)))))
    assert ctrl.etat is Etat.PRETRAITEMENT
    modeles = train.proposer_modeles(spec.probleme)
    ctrl.adopter(executer(ctrl.travail_entrainer(ctrl.session.spec.avec(modeles=modeles))))
    assert ctrl.etat is Etat.MODELES
    direct = train.entrainer(ctrl.session.spec, ctrl.table, ctrl.session.coffre)
    assert (
        ctrl.session.resultats.tableau()
        .drop(columns="duree_s")
        .equals(direct.tableau().drop(columns="duree_s"))
    )
    identifiant = next(m.id for m in ctrl.session.resultats.classement() if m.famille != "reference")
    vue = executer(ctrl.travail_diagnostic(identifiant))
    assert isinstance(vue, Vue) and vue.figures and vue.tableaux
    ctrl.adopter(executer(ctrl.travail_decider(identifiant, None, "essai")))
    assert ctrl.etat is Etat.DIAGNOSTIC and ctrl.session.decision.justification == "essai"
    ctrl.adopter(executer(ctrl.travail_tester()))
    assert ctrl.etat is Etat.TESTE and ctrl.session.coffre.ouvertures == 1
    assert ctrl.vue_du_test().tableaux


def test_adopter_oublie_ce_qui_depend_d_un_etat_invalide(acier_decoupe: Controleur) -> None:
    ctrl = Controleur(session=acier_decoupe.session, chargement=acier_decoupe.chargement)
    ctrl.decoupage = acier_decoupe.decoupage
    ctrl.vues["diagnostic"] = Vue()
    ctrl.adopter(ctrl.session.appliquer(Revenir(Etat.OBJECTIF)))
    assert ctrl.decoupage is None and "diagnostic" not in ctrl.vues
    ctrl.adopter(ctrl.session.appliquer(Revenir(Etat.VIDE)))
    assert ctrl.chargement is None and ctrl.profils == ()


def test_le_chargement_annule_n_est_pas_un_echec() -> None:
    ctrl = Controleur()
    tache = Tache(ctrl.travail_charger(ACIER), nom="annulation")
    tache.annulation.demander()
    tache.demarrer().attendre(30)
    assert tache.etat is EtatTache.ANNULEE and isinstance(tache.erreur, ErreurAnnulation)


def _serie_a_s5() -> Controleur:
    ctrl = Controleur()
    ctrl.adopter_chargee(executer(ctrl.travail_charger(SERIE)))
    ctrl.adopter_audit(executer(ctrl.travail_auditer()))
    spec = executer(ctrl.travail_proposer(cible=None))
    ctrl.adopter(executer(ctrl.travail_actions(FixerObjectif(spec))))
    ctrl.adopter_scelle(executer(ctrl.travail_sceller(ctrl.session.spec, None)))
    ctrl.adopter(
        ctrl.session.appliquer(Acquitter(frozenset(c.code for c in ctrl.session.alertes_en_attente())))
    )
    return ctrl


def test_la_cible_passee_sans_declaration_se_lit_avant_l_entrainement() -> None:
    """L'audit rejoué sur la spécification complète lève le bloquant ; aucun résultat n'est fabriqué."""
    ctrl = _serie_a_s5()
    assert ctrl.session.spec.structure.groupee
    spec = ctrl.session.spec.avec(modeles=(ChoixTechnique.creer("reference.persistance"), RIDGE))
    verification = executer(ctrl.travail_verifier(spec))
    codes = {c.code: c for c in verification.constats}
    assert any(c.startswith("I3.cible_passee") for c in codes)
    assert all(codes[c].niveau is Niveau.BLOQUANT for c in codes if c.startswith("I3.cible_passee"))
    tache = Tache(ctrl.travail_entrainer(spec), nom="refus").demarrer()
    tache.attendre(60)
    assert tache.etat is EtatTache.ECHEC
    assert "cible_connue=True" in str(tache.erreur)
    assert ctrl.session.resultats is None


def test_le_compte_a_rebours_est_refuse_meme_declare() -> None:
    ctrl = _serie_a_s5()
    declare = ChoixTechnique.creer("reference.persistance", cible_connue=True)
    spec = ctrl.session.spec.avec(modeles=(declare, RIDGE))
    verification = executer(ctrl.travail_verifier(spec))
    assert any(c.code.startswith("I3.cible_decompte") for c in verification.constats)
    tache = Tache(ctrl.travail_entrainer(spec), nom="refus").demarrer()
    tache.attendre(60)
    assert tache.etat is EtatTache.ECHEC and "compte" in str(tache.erreur).lower()


def test_sans_cible_le_diagnostic_lit_les_modeles_sans_cible() -> None:
    ctrl = Controleur()
    ctrl.adopter_chargee(executer(ctrl.travail_charger(POMPE)))
    ctrl.adopter_audit(executer(ctrl.travail_auditer()))
    spec = executer(ctrl.travail_proposer(None, sans_cible=True))
    ctrl.adopter(executer(ctrl.travail_actions(FixerObjectif(spec))))
    ctrl.adopter_scelle(executer(ctrl.travail_sceller(ctrl.session.spec, None)))
    assert ctrl.session.coffre is None and ctrl.decoupage is None
    modeles = (ChoixTechnique.creer("sans_cible.kmeans", n_clusters=3),)
    ctrl.adopter(executer(ctrl.travail_actions(Pretraiter(ctrl.session.spec))))
    ctrl.adopter(executer(ctrl.travail_entrainer(ctrl.session.spec.avec(modeles=modeles))))
    vue = executer(ctrl.travail_diagnostic("sans_cible.kmeans"))
    assert {n for n, _ in vue.figures} == {"plan", "profils"} and vue.tableaux


def test_une_erreur_de_bibliotheque_s_affiche_en_francais_et_nomme_l_etape() -> None:
    from sklearn.metrics import confusion_matrix

    from mecaml.ui.worker import message_d_erreur

    try:
        confusion_matrix([0, 1, 1], [0, None, 1])
    except Exception as erreur:
        texte = message_d_erreur(erreur, "Diagnostic")
    assert texte.startswith("Une erreur imprévue") and "« Diagnostic »" in texte
    assert "classification metrics" not in texte.lower()
    assert (
        message_d_erreur(TransitionRefusee(Etat.DONNEES, Acquitter(frozenset()), "le test reste scellé."))
        == "Le test reste scellé."
    )
