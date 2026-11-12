"""Un test de fumée par écran : l'écran s'ouvre dans l'état voulu, se remplit et ne lève aucune exception."""

from __future__ import annotations

import aides_interface as ai
from mecaml.core import libelles
from mecaml.core.session import Etat
from mecaml.ui.app import ECRANS, Application
from mecaml.ui.choix import DialogueTechnique
from mecaml.ui.ressources import exemples
from mecaml.ui.textes import t


def ouvrir(app: Application, instantanes: dict, etat: Etat, numero: int) -> None:
    """Remet le contrôleur dans l'état voulu, puis ouvre l'écran comme le fait la navigation."""
    ai.restaurer(app, instantanes[etat])
    app.banniere.masquer()
    app.afficher(numero)
    ai.attendre(app)


def test_les_dix_ecrans_existent_et_se_rafraichissent_a_vide(app: Application) -> None:
    assert len(ECRANS) == 10 and sorted(app.ecrans) == list(range(10))
    for numero, ecran in app.ecrans.items():
        ecran.rafraichir()
        assert numero == ecran.NUMERO
    assert app.courant == 0


def test_a_vide_seul_l_accueil_et_les_donnees_sont_ouvrables(app: Application) -> None:
    app.afficher(0)
    ouvrables = [k for k in range(10) if app.peut_afficher(k)]
    assert ouvrables == [0, 1]
    app.afficher(5)
    assert app.courant == 0


def test_s0_accueil_propose_les_exemples(app: Application) -> None:
    app.afficher(0)
    ecran = app.ecrans[0]
    boutons = [w for w in ecran.exemples.winfo_children() if w.winfo_class() == "TButton"]
    assert len(boutons) == len(exemples()) == 4
    textes = {w.cget("text") for w in ecran.corps.winfo_children() if w.winfo_class() == "TLabel"}
    assert {t("ecran.accueil.principe"), t("ecran.accueil.confiance")} <= textes
    assert "maîtrise de bout en bout" in t("ecran.accueil.principe")


def test_s1_donnees_apercu_et_resume(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.DONNEES, 1)
    ecran = app.ecrans[1]
    assert len(ecran.apercu.lignes()) == 100
    assert "150" in ecran.resume.cget("text") and "Traitement_Acier" in ecran.resume.cget("text")
    assert not ecran.case_challenge.winfo_ismapped()
    assert str(ecran.liste_feuilles.cget("values"))


def test_s2_audit_profil_et_constats(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.AUDIT, 2)
    ecran = app.ecrans[2]
    assert len(ecran.profil.lignes()) == app.ctrl.table.n_colonnes
    roles = {ligne[0]: ligne[2] for ligne in ecran.profil.lignes()}
    assert roles["resistance_traction_MPa"] and roles["type_alliage"] == t("role.categorielle")


def test_s3_objectif_les_champs_redonnent_la_specification_de_la_session(
    app: Application, instantanes: dict
) -> None:
    ouvrir(app, instantanes, Etat.OBJECTIF, 3)
    ecran = app.ecrans[3]
    assert ecran.var_cible.get() == "resistance_traction_MPa"
    assert len(ecran.tab_roles.lignes()) == app.ctrl.table.n_colonnes
    reconstruite = ecran.construire_spec()
    voulue = app.ctrl.session.spec.avec(modeles=(), pretraitement=())
    assert reconstruite.objectif() == voulue.objectif()
    assert reconstruite.decoupage() == voulue.decoupage()


def test_s4_decoupage_effectifs_et_scelle(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.DECOUPE, 4)
    ecran = app.ecrans[4]
    coffre = app.ctrl.session.coffre
    texte = ecran.resume.cget("text")
    assert str(coffre.n_test) in texte and "scellé" in texte
    assert ecran.var_methode.get()


def test_s5_pretraitement_reprend_la_chaine_de_la_session(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.PRETRAITEMENT, 5)
    ecran = app.ecrans[5]
    assert ecran.liste.choix == app.ctrl.session.spec.pretraitement
    assert len(ecran.liste.tableau.lignes()) == len(ecran.liste.choix)
    assert not ecran.var_cible_connue.get()


def test_s6_modeles_tableau_comparatif_et_figure(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.MODELES, 6)
    ecran = app.ecrans[6]
    resultat = app.ctrl.session.resultats
    assert len(ecran.tableau.lignes()) == len(resultat.modeles)
    assert ecran.figure.figure is not None
    statuts = {ligne[6] for ligne in ecran.tableau.lignes()}
    assert t("ecran.s6.statut.reference") in statuts


def test_s7_diagnostic_modele_figures_et_decision(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.DIAGNOSTIC, 7)
    ecran = app.ecrans[7]
    assert ecran.var_modele.get() and ecran.planches["diagnostic"].figures
    assert app.ctrl.session.decision.modele_id in ecran._ids.values()


def test_s8_test_scores_et_ouverture_unique(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.TESTE, 8)
    ecran = app.ecrans[8]
    assert app.ctrl.session.coffre.ouvertures == 1
    assert "1" in ecran.etat.cget("text")
    assert ecran.planche.tableaux and ecran.planche.figures
    bouton_ouvrir = ecran._boutons[0][0]
    assert str(bouton_ouvrir.cget("state")) == "disabled", "le test déjà ouvert ne se rouvre pas ici"


def test_s9_export_fichiers_ecrits(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.EXPORTE, 9)
    ecran = app.ecrans[9]
    assert len(ecran.tableau.lignes()) == 5
    assert all(len(ligne[2]) == 32 for ligne in ecran.tableau.lignes())


def test_le_rail_suit_la_session(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.AUDIT, 2)
    etats = {k: str(b.cget("state")) for k, b in app._rail.items()}
    assert etats[3] == "normal" and etats[4] == "disabled" and etats[9] == "disabled"
    ouvrir(app, instantanes, Etat.TESTE, 8)
    assert str(app._rail[9].cget("state")) == "normal"


def test_la_navigation_refuse_une_etape_non_atteinte(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.DONNEES, 1)
    app.afficher(6)
    assert app.courant == 1


def test_un_refus_de_la_session_s_affiche_dans_le_bandeau(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.TESTE, 8)
    app.afficher(4)
    ecran = app.ecrans[4]
    ecran.var_graine.set("42")
    ecran.sceller()
    ai.attendre(app)
    assert "test a été ouvert" in app.banniere.texte or "figés" in app.banniere.texte
    assert app.ctrl.etat is Etat.TESTE


def test_une_erreur_de_saisie_ne_lance_aucun_calcul(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.OBJECTIF, 3)
    ecran = app.ecrans[3]
    ecran.var_plis.set("beaucoup")
    ecran.fixer()
    assert not app.occupe and app.banniere.texte
    ecran.var_plis.set("5")


def test_le_reglage_d_une_technique_pose_les_parametres_et_les_colonnes(app: Application) -> None:
    from mecaml.core import catalog

    recu = []
    dialogue = DialogueTechnique(app, catalog.entree("regression.ridge"), None, ["a", "b"], recu.append)
    dialogue.champs["alpha"].set("0.5")
    dialogue.liste.selection_set(1)
    dialogue.valider()
    assert recu[0].id == "regression.ridge"
    assert recu[0].parametres_dict() == {"alpha": 0.5} and recu[0].colonnes == ("b",)


def test_un_parametre_hors_grille_ou_invalide_est_refuse_par_le_coeur(app: Application) -> None:
    from mecaml.core import catalog

    recu = []
    dialogue = DialogueTechnique(app, catalog.entree("regression.ridge"), None, [], recu.append)
    dialogue.champs["alpha"].set("1e999")
    dialogue.valider()
    assert recu == [] and dialogue.erreur.cget("text")
    dialogue.destroy()


def test_la_confirmation_de_l_ouverture_du_test_est_demandee(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.DIAGNOSTIC, 8)
    ouvertures = app.ctrl.session.coffre.ouvertures
    demandes = []
    ancienne = app._confirmer
    app._confirmer = lambda texte: demandes.append(texte) or False
    try:
        app.ecrans[8].ouvrir()
    finally:
        app._confirmer = ancienne
    assert demandes and not app.occupe and app.ctrl.session.coffre.ouvertures == ouvertures


def test_s6_recherche_d_hyperparametres_classe_les_essais_et_reporte_le_meilleur(
    app: Application, instantanes: dict
) -> None:
    ouvrir(app, instantanes, Etat.MODELES, 6)
    ecran = app.ecrans[6]
    panneau = ecran.recherche
    assert (
        "regression.ridge" in panneau._ids.values()
        and "reference.dummy_regression" not in panneau._ids.values()
    )
    panneau.var_modele.set(next(titre for titre, i in panneau._ids.items() if i == "regression.ridge"))
    panneau.chercher()
    ai.attendre(app)
    resultat = app.ctrl.recherches[-1]
    assert len(panneau.tableau.lignes()) == len(resultat.essais) == 5
    metrique = libelles.metrique_courte(resultat.metrique)
    assert t("ecran.s6.recherche.optimiste", metrique=metrique) == panneau.avis.cget("text")
    panneau.utiliser()
    reglage = next(c for c in ecran.liste.choix if c.id == "regression.ridge")
    assert reglage == resultat.meilleur
    assert app.ctrl.session.spec.modeles != ecran.liste.choix or reglage in app.ctrl.session.spec.modeles


def test_les_recherches_suivent_les_plis_de_leur_comparaison(app: Application, instantanes: dict) -> None:
    ouvrir(app, instantanes, Etat.MODELES, 6)
    app.ecrans[6].recherche.var_modele.set(next(iter(app.ecrans[6].recherche._ids)))
    app.ecrans[6].recherche.chercher()
    ai.attendre(app)
    assert app.ctrl.recherches
    app.ctrl.adopter(instantanes[Etat.PRETRAITEMENT].session)
    assert app.ctrl.recherches == []


# ============================================================
# RÉGLAGES DES MODÈLES : libellés, plages, valeurs par défaut, repli « avancé », refus hors bornes
# ============================================================
def _textes_du_dialogue(dialogue) -> list[str]:
    from tkinter import ttk

    nous = [w for w in _descendants(dialogue) if isinstance(w, ttk.Label)]
    return [str(w.cget("text")) for w in nous]


def _descendants(widget) -> list:
    sortie = []
    for enfant in widget.winfo_children():
        sortie.append(enfant)
        sortie.extend(_descendants(enfant))
    return sortie


def test_le_reglage_d_un_modele_montre_libelle_plage_defaut_et_aide(app: Application) -> None:
    from mecaml.core import catalog, hyperparametres

    dialogue = DialogueTechnique(app, catalog.entree("regression.foret"), None, [], lambda _c: None)
    textes = _textes_du_dialogue(dialogue)
    assert hyperparametres.libelle("regression.foret", "n_estimators") in textes
    assert t("choix.plage", bornes="1 à 500", valeur="100") in textes
    assert hyperparametres.aide("regression.foret", "n_estimators") in textes
    assert {h.nom for h in hyperparametres.principaux("regression.foret")} <= set(dialogue.champs)
    assert (
        dialogue.champs["n_estimators"].get() == "100"
        and dialogue.champs["max_depth"].get() == "aucune limite"
    )
    dialogue.destroy()


def test_les_reglages_avances_sont_replies_puis_se_deplient(app: Application) -> None:
    from mecaml.core import catalog, hyperparametres

    dialogue = DialogueTechnique(app, catalog.entree("regression.foret"), None, [], lambda _c: None)
    app.update()
    assert not dialogue._cadre_avance.winfo_ismapped()
    assert "ccp_alpha" in dialogue.champs and hyperparametres.avances("regression.foret")
    dialogue._basculer_avance()
    app.update()
    assert dialogue._cadre_avance.winfo_ismapped()
    dialogue.destroy()


def test_un_reglage_hors_bornes_est_refuse_avec_un_message_et_rien_n_est_pose(app: Application) -> None:
    from mecaml.core import catalog

    recu = []
    dialogue = DialogueTechnique(app, catalog.entree("regression.foret"), None, [], recu.append)
    dialogue.champs["n_estimators"].set("0")
    dialogue.valider()
    message = dialogue.erreur.cget("text")
    assert recu == [] and "Nombre d'arbres" in message and "hors bornes" in message and "1 à 500" in message
    dialogue.champs["n_estimators"].set("beaucoup")
    dialogue.valider()
    assert "n'est pas un nombre" in dialogue.erreur.cget("text") and recu == []
    dialogue.champs["n_estimators"].set("200")
    dialogue.valider()
    assert recu[0].parametres_dict() == {"n_estimators": 200}


def test_valider_sans_rien_changer_ne_pose_aucun_reglage_et_les_defauts_se_retablissent(
    app: Application,
) -> None:
    from mecaml.core import catalog

    recu = []
    dialogue = DialogueTechnique(app, catalog.entree("classification.arbre"), None, [], recu.append)
    dialogue.champs["max_depth"].set("9")
    dialogue.champs["class_weight"].set("balanced")
    dialogue.retablir_defauts()
    assert dialogue.champs["max_depth"].get() == "5" and dialogue.champs["class_weight"].get() == "aucun"
    dialogue.valider()
    assert recu[0].parametres_dict() == {}


def test_un_reglage_avance_et_un_choix_se_posent_comme_les_autres(app: Application) -> None:
    from mecaml.core import catalog

    recu = []
    dialogue = DialogueTechnique(app, catalog.entree("classification.foret"), None, [], recu.append)
    dialogue.champs["class_weight"].set("balanced_subsample")
    dialogue.champs["ccp_alpha"].set("0,01")
    dialogue.champs["bootstrap"].set("non")
    dialogue.valider()
    assert recu[0].parametres_dict() == {
        "bootstrap": False,
        "ccp_alpha": 0.01,
        "class_weight": "balanced_subsample",
    }


def test_la_liste_de_modeles_d_un_vote_se_choisit_dans_le_catalogue(app: Application) -> None:
    from mecaml.core import catalog

    recu = []
    dialogue = DialogueTechnique(app, catalog.entree("classification.vote"), None, [], recu.append)
    boite, ids = dialogue._listes["base"]
    assert "classification.vote" not in ids and "classification.empilement" not in ids
    assert all(i.startswith("classification.") for i in ids)
    boite.selection_clear(0, "end")
    boite.selection_set(ids.index("classification.foret"))
    boite.selection_set(ids.index("classification.svc"))
    dialogue.valider()
    assert recu[0].parametres_dict() == {"base": ("classification.foret", "classification.svc")}


def test_un_reglage_deja_pose_est_relu_dans_la_fenetre(app: Application) -> None:
    from mecaml.core import catalog
    from mecaml.core.types import ChoixTechnique

    choix = ChoixTechnique.creer(
        "regression.reseau_neurones", hidden_layer_sizes=(20, 10), early_stopping=True
    )
    dialogue = DialogueTechnique(app, catalog.entree(choix.id), choix, [], lambda _c: None)
    assert (
        dialogue.champs["hidden_layer_sizes"].get() == "20, 10"
        and dialogue.champs["early_stopping"].get() == "oui"
    )
    dialogue.destroy()


def test_une_technique_sans_reglages_declares_garde_sa_grille(app: Application) -> None:
    from mecaml.core import catalog

    dialogue = DialogueTechnique(app, catalog.entree("transformation.polynomiale"), None, [], lambda _c: None)
    assert list(dialogue.champs) == ["degree"] and not dialogue.reglages
    dialogue.destroy()
