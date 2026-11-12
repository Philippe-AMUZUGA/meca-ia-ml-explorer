"""Contrats du cœur : machine à états, types gelés, coffre de test (I1), registre."""

from __future__ import annotations

import importlib
import json
import pickle
from dataclasses import FrozenInstanceError, dataclass
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from mecaml.core.catalog import (
    GRILLE_MAX_COMBINAISONS,
    REGISTRE,
    Cout,
    EntreeCatalogue,
    construire_registre,
    entree,
    entrees,
)
from mecaml.core.session import (
    ACTIONS,
    Acquitter,
    Action,
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Etat,
    Exporter,
    FixerObjectif,
    Pretraiter,
    Revenir,
    Session,
    Tester,
    TransitionRefusee,
)
from mecaml.core.types import (
    GRAINE_CONTRAT,
    LIGNES_MAX_LENTES,
    METRIQUES_CLASSE_POSITIVE,
    VALIDATIONS,
    ChoixTechnique,
    CoffreTest,
    Constat,
    Decision,
    ErreurScelle,
    Niveau,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
    identifiant_metrique,
    validations_permises,
)

S = Etat
AVANCES = tuple(a for a in ACTIONS if a is not Revenir)
MEDIANE = (ChoixTechnique.creer("manquants.mediane"),)

# ============================================================
# ORACLE DES TRANSITIONS (écrit à la main, indépendant du code)
# ============================================================
_AVANT_S5 = {Charger, Auditer, FixerObjectif, Decouper}
ATTENDU_AVEC_CIBLE: dict[Etat, set[type]] = {
    S.VIDE: {Charger},
    S.DONNEES: {Charger, Auditer},
    S.AUDIT: {Charger, Auditer, Acquitter, FixerObjectif},
    S.OBJECTIF: {Charger, Auditer, Acquitter, FixerObjectif, Decouper},
    S.DECOUPE: _AVANT_S5 | {Acquitter, Pretraiter},
    S.PRETRAITEMENT: _AVANT_S5 | {Pretraiter, Entrainer},
    S.MODELES: _AVANT_S5 | {Pretraiter, Entrainer, Decider},
    S.DIAGNOSTIC: _AVANT_S5 | {Pretraiter, Entrainer, Decider, Tester},
    S.TESTE: {Pretraiter, Entrainer, Decider, Exporter},
    S.EXPORTE: {Pretraiter, Entrainer, Decider, Exporter},
}
ATTENDU_SANS_CIBLE: dict[Etat, set[type]] = {
    **{e: ATTENDU_AVEC_CIBLE[e] for e in (S.VIDE, S.DONNEES, S.AUDIT, S.OBJECTIF, S.DECOUPE)},
    S.PRETRAITEMENT: _AVANT_S5 | {Pretraiter, Entrainer},
    S.MODELES: _AVANT_S5 | {Pretraiter, Entrainer, Decider},
    S.DIAGNOSTIC: _AVANT_S5 | {Pretraiter, Entrainer, Decider, Exporter},
    S.EXPORTE: _AVANT_S5 | {Pretraiter, Entrainer, Decider, Exporter},
}
# Reprise après ouverture du test : seuls prétraitement, modèles et décision se refont.
ATTENDU_APRES_TEST: dict[Etat, set[type]] = {
    S.PRETRAITEMENT: {Pretraiter, Entrainer},
    S.MODELES: {Pretraiter, Entrainer, Decider},
    S.DIAGNOSTIC: {Pretraiter, Entrainer, Decider, Tester},
}


def _etat_attendu(session: Session, action: object) -> Etat:
    if isinstance(action, Acquitter):
        return session.etat
    return type(action).produit


def _verifier_paire(parcours, session: Session, type_action: type, attendus: set[type]) -> None:
    action = parcours.action_valide(type_action, session)
    etat_avant = session.etat
    if type_action in attendus:
        suivante = session.appliquer(action)
        assert suivante.etat == _etat_attendu(session, action)
        assert suivante.table is parcours.table or suivante.etat == S.VIDE
    else:
        with pytest.raises(TransitionRefusee) as refus:
            session.appliquer(action)
        assert refus.value.motif
        assert session.etat == etat_avant  # la session refusée n'a pas bougé


@pytest.mark.parametrize("type_action", AVANCES, ids=lambda t: t.__name__)
@pytest.mark.parametrize("etat", list(Etat), ids=lambda e: e.name)
def test_paires_avec_cible(parcours, etat: Etat, type_action: type) -> None:
    _verifier_paire(parcours, parcours.jusqua(etat), type_action, ATTENDU_AVEC_CIBLE[etat])


@pytest.mark.parametrize("type_action", AVANCES, ids=lambda t: t.__name__)
@pytest.mark.parametrize("etat", [e for e in Etat if e != S.TESTE], ids=lambda e: e.name)
def test_paires_sans_cible(parcours_sans_cible, etat: Etat, type_action: type) -> None:
    session = parcours_sans_cible.jusqua(etat)
    _verifier_paire(parcours_sans_cible, session, type_action, ATTENDU_SANS_CIBLE[etat])


@pytest.mark.parametrize("type_action", AVANCES, ids=lambda t: t.__name__)
@pytest.mark.parametrize("etat", sorted(ATTENDU_APRES_TEST), ids=lambda e: e.name)
def test_paires_apres_ouverture_du_test(parcours, etat: Etat, type_action: type) -> None:
    session = parcours.jusqua(S.TESTE).appliquer(Revenir(etat))
    assert session.test_vu
    _verifier_paire(parcours, session, type_action, ATTENDU_APRES_TEST[etat])


@pytest.mark.parametrize("vers", list(Etat), ids=lambda e: e.name)
@pytest.mark.parametrize("depuis", list(Etat), ids=lambda e: e.name)
def test_paires_retour_arriere(parcours, depuis: Etat, vers: Etat) -> None:
    session = parcours.jusqua(depuis)
    test_vu = depuis >= S.TESTE
    permis = vers < depuis and (not test_vu or vers == S.VIDE or vers >= S.PRETRAITEMENT)
    if permis:
        suivante = session.appliquer(Revenir(vers))
        assert suivante.etat == vers
    else:
        with pytest.raises(TransitionRefusee):
            session.appliquer(Revenir(vers))


def test_sans_cible_S8_inatteignable(parcours_sans_cible) -> None:
    s7 = parcours_sans_cible.jusqua(S.DIAGNOSTIC)
    with pytest.raises(TransitionRefusee, match="sans cible"):
        s7.appliquer(parcours_sans_cible.action_valide(Tester, s7))
    assert s7.appliquer(Exporter(("rapport.html",))).etat == S.EXPORTE


# ============================================================
# RÈGLES 1 ET 2 : AVANCE, REPRISE, RETOUR, INVALIDATION DE L'AVAL
# ============================================================
def test_chaine_complete_partage_la_table(parcours) -> None:
    s9 = parcours.jusqua(S.EXPORTE)
    assert s9.table is parcours.table
    assert parcours.table.est_intacte()
    assert [nom for nom, _ in s9.historique] == [
        "Charger", "Auditer", "FixerObjectif", "Decouper", "Pretraiter",
        "Entrainer", "Decider", "Tester", "Exporter",
    ]  # fmt: skip
    assert s9.resultat_test == {"n_lignes": 12}
    assert s9.coffre.ouvertures == 1 and not s9.score_test_optimiste
    assert s9.tests_vus == {parcours.table.empreinte}


def test_revenir_invalide_l_aval(parcours) -> None:
    s7 = parcours.jusqua(S.DIAGNOSTIC)
    s4 = s7.appliquer(Revenir(S.DECOUPE))
    assert s4.coffre is s7.coffre and s4.spec.objectif() == s7.spec.objectif()
    assert s4.resultats is None and s4.decision is None
    s2 = s7.appliquer(Revenir(S.AUDIT))
    assert s2.spec is None and s2.coffre is None and s2.table is s7.table
    s0 = s7.appliquer(Revenir(S.VIDE))
    assert s0.table is None and s0.etat == S.VIDE and s0.auditeur is s7.auditeur


@pytest.mark.parametrize(
    ("vers", "pretraitement_garde", "modeles_gardes"),
    [
        (S.MODELES, True, True),
        (S.PRETRAITEMENT, True, False),
        (S.DECOUPE, False, False),
        (S.OBJECTIF, False, False),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_revenir_purge_la_specification(parcours, vers, pretraitement_garde, modeles_gardes) -> None:
    s7 = parcours.jusqua(S.DIAGNOSTIC)
    assert s7.spec.pretraitement == MEDIANE and s7.spec.modeles
    s = s7.appliquer(Revenir(vers))
    assert s.spec.pretraitement == (MEDIANE if pretraitement_garde else ())
    assert s.spec.modeles == (s7.spec.modeles if modeles_gardes else ())


def test_refaire_une_etape_purge_la_specification(parcours, spec_reg, modeles_reg) -> None:
    s7 = parcours.jusqua(S.DIAGNOSTIC)
    s5 = s7.appliquer(Pretraiter(s7.spec))  # la spécification transmise porte encore les modèles
    assert s5.spec.pretraitement == MEDIANE and s5.spec.modeles == ()
    s3 = s7.appliquer(FixerObjectif(s7.spec))
    assert s3.spec.pretraitement == () and s3.spec.modeles == ()
    with pytest.raises(TransitionRefusee, match="aucun modèle"):
        s5.appliquer(Entrainer({"m": 1.0}))  # les anciens modèles ne reviennent pas en silence
    assert s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(modeles=modeles_reg))).etat == S.MODELES


def test_refaire_une_etape_vaut_retour_puis_avance(parcours, spec_reg) -> None:
    s7 = parcours.jusqua(S.DIAGNOSTIC)
    autre = spec_reg.avec(cible="effort", roles=tuple(
        (c, Role.CIBLE if c == "effort" else (Role.NUMERIQUE if c == "y_reg" else r))
        for c, r in spec_reg.roles
    ))  # fmt: skip
    s3 = s7.appliquer(FixerObjectif(autre))
    assert s3.etat == S.OBJECTIF and s3.spec == autre
    assert s3.coffre is None and s3.resultats is None and s3.decision is None


def test_charger_une_autre_table_repart_de_S1(parcours, df_petit) -> None:
    s6 = parcours.jusqua(S.MODELES)
    autre = Table(df_petit.head(30), nom="autre.csv")
    s1 = s6.appliquer(Charger(autre))
    assert s1.etat == S.DONNEES and s1.table is autre
    assert s1.spec is None and s1.constats == () and s1.auditeur is s6.auditeur


def test_session_gelee(parcours) -> None:
    s = parcours.jusqua(S.DONNEES)
    with pytest.raises(FrozenInstanceError):
        s.etat = S.EXPORTE  # type: ignore[misc]


@dataclass(frozen=True)
class ActionEtrangere(Action):
    """Sous-classe non enregistrée dans la session."""


@pytest.mark.parametrize(
    "action", ["Charger", Action(), ActionEtrangere()], ids=["texte", "base", "etrangere"]
)
def test_action_inconnue_refusee_par_erreur_typee(parcours, action) -> None:
    s = parcours.jusqua(S.DONNEES)
    with pytest.raises(TypeError, match="Action inconnue"):
        s.appliquer(action)  # type: ignore[arg-type]
    assert not s.peut(action)  # type: ignore[arg-type]


def test_peut_reflete_les_regles_d_etat(parcours) -> None:
    s3 = parcours.jusqua(S.OBJECTIF)
    assert s3.peut(parcours.action_valide(Decouper, s3))
    assert not s3.peut(parcours.action_valide(Tester, s3))


# ============================================================
# CONTENU DES ACTIONS
# ============================================================
def test_objectif_avec_colonne_absente_refuse(parcours, spec_reg) -> None:
    s2 = parcours.jusqua(S.AUDIT)
    with pytest.raises(TransitionRefusee, match="absentes"):
        s2.appliquer(FixerObjectif(spec_reg.avec(roles=(*spec_reg.roles, ("fantome", Role.NUMERIQUE)))))


def test_roles_permutes_meme_specification(parcours, fabrique_parcours, espion, spec_reg) -> None:
    permutee = spec_reg.avec(roles=tuple(reversed(spec_reg.roles)))
    assert permutee == spec_reg and hash(permutee) == hash(spec_reg)
    assert permutee.cle_audit() == spec_reg.cle_audit()
    auditeur = espion()
    p = fabrique_parcours(spec_reg, auditeur)
    s4 = p.jusqua(S.DECOUPE)
    n_appels = len(auditeur.appels)
    s5 = s4.appliquer(Pretraiter(permutee.avec(pretraitement=MEDIANE)))
    assert s5.etat == S.PRETRAITEMENT
    p.jusqua(S.OBJECTIF).appliquer(FixerObjectif(permutee))
    # S5 rejoue l'audit (+1), le parcours jusqu'à S3 en fait deux (+2), puis aucun rejeu parasite
    assert len(auditeur.appels) == n_appels + 3


def test_decouper_ne_change_pas_l_objectif(parcours, spec_reg, coffre_de) -> None:
    s3 = parcours.jusqua(S.OBJECTIF)
    with pytest.raises(TransitionRefusee, match="objectif"):
        s3.appliquer(Decouper(coffre_de(s3.table), spec=spec_reg.avec(metrique="mae")))
    nouvelle = spec_reg.avec(part_test=0.25)
    s4 = s3.appliquer(Decouper(coffre_de(s3.table, nouvelle.decoupage()), spec=nouvelle))
    assert s4.spec.part_test == 0.25


def test_I4_coffre_rattache_au_decoupage(parcours, spec_reg, coffre_de) -> None:
    s3 = parcours.jusqua(S.OBJECTIF)
    chrono = spec_reg.avec(structure=Structure.CHRONOLOGIQUE)
    with pytest.raises(TransitionRefusee, match="autre découpage"):
        s3.appliquer(Decouper(coffre_de(s3.table), spec=chrono))  # coffre construit pour l'indépendance
    assert s3.appliquer(Decouper(coffre_de(s3.table, chrono.decoupage()), spec=chrono)).etat == S.DECOUPE


def test_decouper_controle_le_coffre(parcours, parcours_sans_cible, df_petit, coffre_de) -> None:
    s3 = parcours.jusqua(S.OBJECTIF)
    with pytest.raises(TransitionRefusee, match="exige un coffre"):
        s3.appliquer(Decouper(None))
    etrangere = Table(df_petit, nom="copie.csv")
    with pytest.raises(TransitionRefusee, match="construit sur la table"):
        s3.appliquer(Decouper(coffre_de(etrangere)))
    deja_ouvert = coffre_de(s3.table)
    deja_ouvert.ouvrir(Decision("x.y", "rmse", 1.0))
    with pytest.raises(TransitionRefusee, match="déjà été ouvert"):
        s3.appliquer(Decouper(deja_ouvert))
    s3_sc = parcours_sans_cible.jusqua(S.OBJECTIF)
    with pytest.raises(TransitionRefusee, match="sans cible"):
        s3_sc.appliquer(Decouper(coffre_de(s3_sc.table)))


def test_pretraiter_et_entrainer_ne_changent_que_leur_part(parcours, spec_reg, modeles_reg) -> None:
    s4 = parcours.jusqua(S.DECOUPE)
    with pytest.raises(TransitionRefusee, match="seul le prétraitement"):
        s4.appliquer(Pretraiter(spec_reg.avec(n_plis=3)))
    s5 = parcours.jusqua(S.PRETRAITEMENT)
    with pytest.raises(TransitionRefusee, match="seuls les modèles"):
        s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(pretraitement=(), modeles=modeles_reg)))
    s6 = s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(modeles=modeles_reg)))
    assert s6.spec.modeles == modeles_reg
    with pytest.raises(TransitionRefusee, match="résultats"):
        s5.appliquer(Entrainer(None, spec=s5.spec.avec(modeles=modeles_reg)))


def test_I6_reference_naive_exigee(parcours, parcours_sans_cible) -> None:
    s5 = parcours.jusqua(S.PRETRAITEMENT)
    sans_reference = (ChoixTechnique.creer("regression.ridge", alpha=1.0),)
    with pytest.raises(TransitionRefusee, match="référence naïve"):
        s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(modeles=sans_reference)))
    s5_sc = parcours_sans_cible.jusqua(S.PRETRAITEMENT)
    assert s5_sc.appliquer(parcours_sans_cible.action_valide(Entrainer, s5_sc)).etat == S.MODELES


def test_I7_metrique_fixee_en_S3(parcours) -> None:
    s6 = parcours.jusqua(S.MODELES)
    with pytest.raises(TransitionRefusee, match="métrique fixée en S3"):
        s6.appliquer(Decider(Decision("reference.dummy_regression", metrique="r2", score_validation=0.9)))


def test_decision_sur_un_modele_compare(parcours) -> None:
    s6 = parcours.jusqua(S.MODELES)
    with pytest.raises(TransitionRefusee, match="pas parmi les modèles comparés"):
        s6.appliquer(Decider(Decision("regression.lasso", metrique="rmse", score_validation=0.9)))
    assert s6.appliquer(Decider(Decision("regression.ridge", "rmse", 0.9))).etat == S.DIAGNOSTIC


def test_exporter_sans_test_refuse_avec_cible(parcours) -> None:
    s7 = parcours.jusqua(S.DIAGNOSTIC)
    with pytest.raises(TransitionRefusee, match="jeu de test"):
        s7.appliquer(Exporter(("rapport.html",)))


# ============================================================
# RÈGLE 3 : REJEU DE L'AUDIT ; GARDE DES CONSTATS
# ============================================================
def test_rejeu_de_l_audit(fabrique_parcours, espion, spec_reg, coffre_de) -> None:
    auditeur = espion()
    p = fabrique_parcours(spec_reg, auditeur)
    s3 = p.jusqua(S.OBJECTIF)
    assert auditeur.appels == [None, spec_reg] and auditeur.coffres == [None, None]
    s3.appliquer(FixerObjectif(spec_reg))  # même objectif, même découpage, sans coffre : pas de rejeu
    assert len(auditeur.appels) == 2
    s4 = s3.appliquer(p.action_valide(Decouper, s3))  # nouveau coffre : rejeu avec le coffre
    assert len(auditeur.appels) == 3 and auditeur.coffres[-1] is s4.coffre
    s7 = p.jusqua(S.DIAGNOSTIC, depart=s4)
    assert len(auditeur.appels) == 5  # S5 et S6 rejouent l'audit avec la spécification complète, S7 non
    assert auditeur.appels[3].pretraitement and auditeur.appels[4].modeles
    plus_grand = spec_reg.avec(part_test=0.3)
    s4.appliquer(Decouper(coffre_de(s4.table, plus_grand.decoupage()), spec=plus_grand))
    assert auditeur.appels[-1] == plus_grand and len(auditeur.appels) == 6
    s7.appliquer(Revenir(S.OBJECTIF))  # retour en S3 : le coffre disparaît, l'audit est rejoué sans lui
    assert len(auditeur.appels) == 7 and auditeur.coffres[-1] is None
    autres_roles = spec_reg.avec(
        roles=tuple((c, Role.CATEGORIELLE if c == "y_cls" else r) for c, r in spec_reg.roles)
    )
    s3.appliquer(FixerObjectif(autres_roles))
    assert len(auditeur.appels) == 8
    s4.appliquer(Revenir(S.AUDIT))  # retour à l'audit : profil seul, sans objectif
    assert auditeur.appels[-1] is None and len(auditeur.appels) == 9


def _regle_alertes(table: Table, spec: SpecProjet | None, coffre: CoffreTest | None) -> tuple[Constat, ...]:
    constats = [Constat(Niveau.ALERTE, "A1.manquants:effort", "30 % de valeurs manquantes.", "Imputer.")]
    if spec is None or spec.part_test < 0.3:
        constats.append(Constat(Niveau.ALERTE, "A2.petit_test", "Test de moins de 30 lignes.", "Agrandir."))
    constats.append(Constat(Niveau.INFO, "I.profil", "Profil calculé."))
    return tuple(constats)


def test_alertes_a_acquitter_avant_de_quitter_S4(fabrique_parcours, espion, spec_reg, coffre_de) -> None:
    p = fabrique_parcours(spec_reg, espion(_regle_alertes))
    s2 = p.jusqua(S.AUDIT)
    with pytest.raises(TransitionRefusee, match="sans alerte"):
        s2.appliquer(Acquitter(frozenset({"I.profil"})))
    s2 = s2.appliquer(Acquitter(frozenset({"A1.manquants:effort"})))
    s4 = s2.appliquer(FixerObjectif(spec_reg)).appliquer(Decouper(coffre_de(s2.table)))
    assert s4.acquittes == {"A1.manquants:effort"}
    with pytest.raises(TransitionRefusee, match="alertes à acquitter"):
        s4.appliquer(p.action_valide(Pretraiter, s4))
    plus_grand = spec_reg.avec(part_test=0.3)
    s4b = s4.appliquer(Decouper(coffre_de(s4.table, plus_grand.decoupage()), spec=plus_grand))
    assert [c.code for c in s4b.alertes_en_attente()] == []  # A2 a disparu au rejeu, A1 reste acquittée
    assert s4b.appliquer(p.action_valide(Pretraiter, s4b)).etat == S.PRETRAITEMENT


def _regle_identifiant(
    table: Table, spec: SpecProjet | None, coffre: CoffreTest | None
) -> tuple[Constat, ...]:
    if spec is not None and spec.role_de("id_piece") in (Role.NUMERIQUE, Role.CATEGORIELLE):
        return (
            Constat(
                Niveau.BLOQUANT,
                "I3.identifiant:id_piece",
                "Colonne identifiante utilisée comme variable.",
                "Lui donner le rôle IDENTIFIANT.",
                ref_livre="C04",
            ),
        )
    return ()


def test_I3_bloquant_leve_par_l_objectif(fabrique_parcours, espion, spec_reg, coffre_de) -> None:
    fautive = spec_reg.avec(
        roles=(*[r for r in spec_reg.roles if r[0] != "id_piece"], ("id_piece", Role.NUMERIQUE))
    )
    p = fabrique_parcours(fautive, espion(_regle_identifiant))
    s4 = p.jusqua(S.DECOUPE)  # découper reste permis : c'est une action corrective
    assert [c.code for c in s4.bloquants()] == ["I3.identifiant:id_piece"]
    with pytest.raises(TransitionRefusee, match="bloquants"):
        s4.appliquer(p.action_valide(Pretraiter, s4))
    s3 = s4.appliquer(FixerObjectif(spec_reg))
    assert s3.bloquants() == ()
    s5 = s3.appliquer(Decouper(coffre_de(s3.table))).appliquer(Pretraiter(spec_reg))
    assert s5.etat == S.PRETRAITEMENT


def test_auditeur_invalide_detecte(fabrique_parcours, spec_reg) -> None:
    p = fabrique_parcours(spec_reg, lambda table, spec, coffre: ("pas un constat",))
    with pytest.raises(TypeError, match="Constat"):
        p.jusqua(S.AUDIT)


def test_codes_de_constat_uniques(fabrique_parcours, spec_reg) -> None:
    double = Constat(Niveau.INFO, "I.profil", "Profil calculé.")
    p = fabrique_parcours(spec_reg, lambda table, spec, coffre: (double, double))
    with pytest.raises(ValueError, match="en double"):
        p.jusqua(S.AUDIT)


# ============================================================
# I1 : COFFRE DE TEST SCELLÉ ; TESTS DÉJÀ VUS
# ============================================================
def test_I1_tester_ouvre_une_fois_et_compte(parcours) -> None:
    s8 = parcours.jusqua(S.TESTE)
    assert s8.coffre.ouvertures == 1 and s8.test_vu and not s8.score_test_optimiste
    s8b = s8.appliquer(Revenir(S.MODELES)).appliquer(parcours.action_valide(Decider, s8))
    s8b = s8b.appliquer(parcours.action_valide(Tester, s8b))
    assert s8b.coffre is s8.coffre and s8b.coffre.ouvertures == 2 and s8b.score_test_optimiste
    with pytest.raises(TransitionRefusee, match="découpage ne se refait pas"):
        s8b.appliquer(Revenir(S.DECOUPE))


def test_I1_test_deja_vu_survit_au_nouveau_projet(parcours) -> None:
    s8 = parcours.jusqua(S.TESTE)
    s1 = s8.appliquer(Revenir(S.VIDE)).appliquer(Charger(parcours.table))
    assert s1.tests_vus == {parcours.table.empreinte}
    s8b = parcours.jusqua(S.TESTE, depart=s1)  # nouveau coffre, même table
    assert s8b.coffre is not s8.coffre and s8b.coffre.ouvertures == 1
    assert s8b.score_test_optimiste
    copie = Table(parcours.table.donnees.copy(), nom="meme_contenu.csv")  # même empreinte, autre objet
    s8c = parcours.jusqua(S.TESTE, depart=s8.appliquer(Revenir(S.VIDE)).appliquer(Charger(copie)))
    assert s8c.score_test_optimiste


def test_I1_autre_table_non_signalee(parcours, df_petit) -> None:
    s8 = parcours.jusqua(S.TESTE)
    autre = Table(df_petit.assign(effort=df_petit["effort"] + 1.0), nom="autre_campagne.csv")
    s8b = parcours.jusqua(S.TESTE, depart=s8.appliquer(Revenir(S.VIDE)).appliquer(Charger(autre)))
    assert not s8b.score_test_optimiste
    assert s8b.tests_vus == {parcours.table.empreinte, autre.empreinte}


def test_I1_evaluation_en_echec_compte_l_ouverture(parcours) -> None:
    s7 = parcours.jusqua(S.DIAGNOSTIC)

    def evaluer_en_echec(lignes: pd.DataFrame) -> None:
        raise RuntimeError("calcul interrompu")

    with pytest.raises(RuntimeError):
        s7.appliquer(Tester(evaluer_en_echec))
    assert s7.etat == S.DIAGNOSTIC and s7.coffre.ouvertures == 1
    s8 = s7.appliquer(parcours.action_valide(Tester, s7))
    assert s8.coffre.ouvertures == 2 and s8.score_test_optimiste


@pytest.mark.parametrize("jeton", [None, "decision", {"modele_id": "x"}, 1.0])
def test_I1_ouverture_sans_decision_refusee(table_petite, coffre_de, jeton) -> None:
    coffre = coffre_de(table_petite)
    with pytest.raises(ErreurScelle):
        coffre.ouvrir(jeton)  # type: ignore[arg-type]
    assert coffre.ouvertures == 0


def test_I1_seconde_ouverture_comptee(table_petite, coffre_de) -> None:
    coffre = coffre_de(table_petite)
    decision = Decision("regression.ridge", "rmse", 1.2)
    premiere = coffre.ouvrir(decision)
    assert coffre.ouvertures == 1 and not coffre.score_optimiste
    seconde = coffre.ouvrir(decision)
    assert coffre.ouvertures == 2 and coffre.score_optimiste
    pd.testing.assert_frame_equal(premiere, seconde)
    assert list(premiere.index) == list(range(0, 60, 5))


def test_I1_aucune_methode_ne_rend_le_test_avant_ouverture(table_petite, coffre_de) -> None:
    coffre = coffre_de(table_petite)
    positions_test = set(range(0, 60, 5))
    for nom in dir(coffre):
        if nom.startswith("_") or nom == "ouvrir":
            continue
        valeur = getattr(coffre, nom)
        if callable(valeur):
            try:
                valeur = valeur()
            except TypeError:
                continue  # méthode à arguments (repose_sur, constructeurs)
        if isinstance(valeur, pd.DataFrame):
            assert positions_test.isdisjoint(valeur.index), nom
        if isinstance(valeur, np.ndarray):
            assert positions_test.isdisjoint(valeur.tolist()), nom
    assert coffre.ouvertures == 0
    assert "id_piece" not in repr(coffre)


def test_I1_apprentissage_disjoint_et_complet(table_petite, coffre_de) -> None:
    coffre = coffre_de(table_petite)
    app = coffre.apprentissage()
    assert coffre.n_apprentissage == len(app) == 48 and coffre.n_test == 12
    assert not coffre.positions_apprentissage().flags.writeable
    test = coffre.ouvrir(Decision("x.y", "rmse", 0.0))
    assert set(app.index).isdisjoint(test.index)
    assert len(set(app.index) | set(test.index)) == table_petite.n_lignes


def test_I1_coffre_ni_construit_directement_ni_serialise(table_petite, coffre_de) -> None:
    with pytest.raises(TypeError):
        CoffreTest()
    with pytest.raises(TypeError, match="sérialise"):
        pickle.dumps(coffre_de(table_petite))
    with pytest.raises(TypeError, match="decoupage"):
        CoffreTest.par_positions(table_petite, [0, 1], decoupage=())


@pytest.mark.parametrize(
    "positions",
    [[], [0, 0], [-1], [60], [0.5], list(range(60))],
    ids=["vide", "double", "negative", "hors_table", "non_entiere", "toutes"],
)
def test_I1_positions_de_test_invalides(table_petite, spec_reg, positions) -> None:
    with pytest.raises(ValueError):
        CoffreTest.par_positions(table_petite, np.asarray(positions), decoupage=spec_reg.decoupage())


def test_I1_mode_challenge_fichier_de_test(df_petit, spec_reg) -> None:
    app = Table(df_petit.iloc[:40], nom="challenge.xlsx", feuille="apprentissage")
    test = Table(df_petit.iloc[40:], nom="challenge.xlsx", feuille="test")
    coffre = CoffreTest.par_fichier(app, test, decoupage=spec_reg.decoupage())
    assert coffre.mode_challenge and coffre.n_test == 20 and len(coffre.apprentissage()) == 40
    assert coffre.repose_sur(app) and not coffre.repose_sur(test)
    assert coffre.empreinte_test == test.empreinte
    lignes = coffre.ouvrir(Decision("x.y", "rmse", 0.0))
    pd.testing.assert_frame_equal(lignes, test.donnees)
    assert lignes is not test.donnees
    lignes.loc[lignes.index[0], "effort"] = -1.0  # une évaluation qui écrit ne touche pas le test
    assert test.est_intacte()
    with pytest.raises(ValueError, match="colonnes différentes"):
        CoffreTest.par_fichier(
            app, Table(df_petit.iloc[40:, :-1], nom="t.csv"), decoupage=spec_reg.decoupage()
        )


def test_I1_ouverture_par_positions_ne_touche_pas_la_table(table_petite, coffre_de) -> None:
    lignes = coffre_de(table_petite).ouvrir(Decision("x.y", "rmse", 0.0))
    lignes.loc[lignes.index[0], "effort"] = -1.0
    assert table_petite.est_intacte()


# ============================================================
# TYPES GELÉS
# ============================================================
def test_table_empreinte_et_identite(df_petit) -> None:
    table = Table(df_petit, nom="a.csv")
    assert table.est_intacte() and table.forme == (60, 8) and table.colonnes[0] == "id_piece"
    assert "temperature" not in repr(table)
    assert table != Table(df_petit, nom="a.csv") and table == table
    assert table.empreinte == Table(df_petit.copy(), nom="b.csv").empreinte
    df_petit.loc[3, "effort"] = -1.0
    assert not table.est_intacte()


def test_table_refus(df_petit) -> None:
    with pytest.raises(TypeError):
        Table(df_petit.to_numpy(), nom="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non textuels"):
        Table(pd.DataFrame({0: [1, 2]}), nom="x")
    with pytest.raises(ValueError, match="en double"):
        Table(pd.DataFrame([[1, 2]], columns=["a", "a"]), nom="x")


def test_table_vide_a_une_empreinte() -> None:
    assert Table(pd.DataFrame({"a": []}), nom="vide.csv").est_intacte()


def test_table_types_pandas3(df_petit) -> None:
    df = df_petit.astype({"materiau": "object"})
    assert Table(df, nom="o.csv").empreinte != Table(df_petit, nom="s.csv").empreinte


@pytest.mark.parametrize(
    ("changements", "motif"),
    [
        ({"probleme": "regression"}, "Probleme"),
        ({"cible": None}, "exige une cible"),
        ({"metrique": None}, "métrique"),
        ({"roles": (("y_reg", Role.CIBLE), ("effort", Role.CIBLE))}, "réservé"),
        ({"roles": (("y_reg", Role.NUMERIQUE),)}, "autre rôle"),
        ({"roles": (("a", Role.NUMERIQUE), ("a", Role.CATEGORIELLE))}, "plusieurs rôles"),
        ({"roles": (("a", "numerique"),)}, "rôle invalide"),
        ({"roles": [("a", Role.NUMERIQUE)]}, "tuple"),
        ({"structure": Structure.GROUPEE, "roles": (("y_reg", Role.CIBLE),)}, "GROUPE"),
        ({"structure": Structure.CHRONOLOGIQUE, "roles": (("y_reg", Role.CIBLE),)}, "DATE"),
        ({"part_test": 0.0}, "part_test"),
        ({"part_test": 1.0}, "part_test"),
        ({"part_test": float("nan")}, "part_test"),
        ({"part_test": True}, "part_test"),
        ({"n_plis": 1}, "n_plis"),
        ({"n_plis": 2.0}, "n_plis"),
        ({"graine": -1}, "graine"),
        ({"pretraitement": ("manquants.mediane",)}, "ChoixTechnique"),
    ],
)
def test_spec_refus(spec_reg, changements, motif) -> None:
    with pytest.raises((ValueError, TypeError), match=motif):
        spec_reg.avec(**changements)


def test_spec_sans_cible_refuse_une_cible(spec_sc) -> None:
    with pytest.raises(ValueError, match="sans cible"):
        spec_sc.avec(cible="y_reg")


def test_spec_graine_par_defaut_du_contrat(spec_reg) -> None:
    assert GRAINE_CONTRAT == 42 and spec_reg.graine == 42
    assert spec_reg.decoupage()[3] == 42
    lu = spec_reg.vers_dict()
    del lu["graine"]
    assert SpecProjet.depuis_dict(lu).graine == 42  # projet écrit sans graine : graine du contrat


# ============================================================
# VALIDATION IMPOSÉE ET CLASSE POSITIVE
# ============================================================
def _classification(spec_reg: SpecProjet, **changements: object) -> SpecProjet:
    """Classification binaire de `y_cls` (conforme, non_conforme) sur la table synthétique."""
    roles = tuple(
        (c, Role.CIBLE if c == "y_cls" else (Role.NUMERIQUE if c == "y_reg" else r))
        for c, r in spec_reg.roles
    )
    base = spec_reg.avec(
        probleme=Probleme.CLASSIFICATION,
        cible="y_cls",
        roles=roles,
        metrique="f1",
        classe_positive="non_conforme",
    )
    return base.avec(**changements) if changements else base


@pytest.mark.parametrize(
    ("structure", "validation", "motif"),
    [
        (Structure.GROUPEE, "validation.kfold", "même unité"),
        (Structure.GROUPEE, "validation.exclusion_unique", "même unité"),
        (Structure.CHRONOLOGIQUE, "validation.kfold", "postérieures"),
        (Structure.GROUPEE_CHRONOLOGIQUE, "validation.serie_temporelle", "même unité"),
        (Structure.INDEPENDANTE, "validation.plis_groupes", "structure indépendante"),
        (Structure.INDEPENDANTE, "validation.serie_temporelle", "structure indépendante"),
        (Structure.INDEPENDANTE, "validation.kfold_stratifie", "stratifie"),
        (Structure.GROUPEE, "validation.plis_groupes_stratifies", "stratifie"),
        (Structure.INDEPENDANTE, "validation.recherche_grille", "inconnue"),
        (Structure.INDEPENDANTE, "kfold", "inconnue"),
    ],
    ids=lambda v: str(v),
)
def test_I4_validation_imposee_incompatible_refusee(spec_reg, structure, validation, motif) -> None:
    with pytest.raises(ValueError, match=motif) as refus:
        spec_reg.avec(structure=structure, validation=validation)
    assert "None" in str(refus.value) or "inconnue" in motif  # le message propose une issue


@pytest.mark.parametrize(
    ("structure", "validation"),
    [
        (Structure.INDEPENDANTE, None),
        (Structure.INDEPENDANTE, "validation.kfold"),
        (Structure.INDEPENDANTE, "validation.exclusion_unique"),
        (Structure.GROUPEE, "validation.plis_groupes"),
        (Structure.GROUPEE, "validation.groupes_aleatoires"),
        (Structure.CHRONOLOGIQUE, "validation.serie_temporelle"),
        (Structure.GROUPEE_CHRONOLOGIQUE, "validation.plis_groupes"),
    ],
    ids=lambda v: str(v),
)
def test_I4_validation_compatible_acceptee(spec_reg, structure, validation) -> None:
    spec = spec_reg.avec(structure=structure, validation=validation)
    assert spec.validation == validation and spec.decoupage()[-1] == validation


def test_validation_stratifiee_en_classification_et_sans_cible(spec_reg, spec_sc) -> None:
    for structure, v in ((Structure.INDEPENDANTE, "validation.kfold_stratifie"),
                         (Structure.GROUPEE, "validation.plis_groupes_stratifies")):  # fmt: skip
        assert _classification(spec_reg, structure=structure, validation=v).validation == v
    with pytest.raises(ValueError, match="sans cible"):
        spec_sc.avec(validation="validation.kfold")
    assert validations_permises(Structure.GROUPEE, Probleme.REGRESSION) == {
        "validation.plis_groupes", "validation.groupes_aleatoires"
    }  # fmt: skip
    assert validations_permises(Structure.INDEPENDANTE, Probleme.SANS_CIBLE) == frozenset()


def test_validation_fixee_au_decoupage(parcours, spec_reg, coffre_de) -> None:
    s3 = parcours.jusqua(S.OBJECTIF)
    kfold = spec_reg.avec(validation="validation.kfold")
    with pytest.raises(TransitionRefusee, match="autre découpage"):
        s3.appliquer(Decouper(coffre_de(s3.table), spec=kfold))  # coffre construit pour la validation déduite
    s4 = s3.appliquer(Decouper(coffre_de(s3.table, kfold.decoupage()), spec=kfold))
    assert s4.spec.validation == "validation.kfold"
    with pytest.raises(TransitionRefusee, match="seul le prétraitement"):
        s4.appliquer(Pretraiter(spec_reg.avec(pretraitement=MEDIANE)))  # la validation ne change plus en S5


def test_validation_et_classe_positive_survivent_au_json(spec_reg) -> None:
    spec = _classification(
        spec_reg, structure=Structure.GROUPEE, validation="validation.plis_groupes_stratifies"
    )
    relue = SpecProjet.depuis_dict(json.loads(json.dumps(spec.vers_dict())))
    assert relue == spec and relue.validation == spec.validation and relue.classe_positive == "non_conforme"
    entiers = _classification(spec_reg, classe_positive=np.int64(1))
    assert type(entiers.classe_positive) is int and SpecProjet.depuis_dict(entiers.vers_dict()) == entiers


def test_identifiants_du_socle_existent_dans_le_registre_et_les_metriques() -> None:
    from mecaml.core import metrics

    validations_du_registre = {e.id for e in entrees(famille="validation")}
    assert validations_du_registre >= VALIDATIONS, sorted(VALIDATIONS - validations_du_registre)
    ids_metriques = {m.id for m in metrics.METRIQUES}
    assert ids_metriques >= METRIQUES_CLASSE_POSITIVE, sorted(METRIQUES_CLASSE_POSITIVE - ids_metriques)


@pytest.mark.parametrize(
    ("metrique", "exigee"),
    [("f1", True), ("precision", True), ("rappel", True), ("precision_moyenne", True),
     ("cout:fp=1,fn=10", True), ("exactitude", False), ("exactitude_equilibree", False), ("roc_auc", False)],
)  # fmt: skip
def test_classe_positive_exigee_selon_la_metrique(spec_reg, metrique, exigee) -> None:
    assert _classification(spec_reg, metrique=metrique).exige_classe_positive() is exigee
    assert identifiant_metrique(metrique) in METRIQUES_CLASSE_POSITIVE or not exigee


def test_classe_positive_refus_de_type_et_de_probleme(spec_reg) -> None:
    with pytest.raises(ValueError, match="qu'en classification"):
        spec_reg.avec(classe_positive="non_conforme")
    with pytest.raises(TypeError, match="classe positive"):
        _classification(spec_reg, classe_positive=["non_conforme"])
    with pytest.raises(TypeError, match="classe positive"):
        _classification(spec_reg, classe_positive="  ")
    assert type(_classification(spec_reg, classe_positive=np.str_("conforme")).classe_positive) is str


def test_I7_classe_positive_confirmee_en_S3(parcours, spec_reg) -> None:
    s2 = parcours.jusqua(S.AUDIT)
    with pytest.raises(TransitionRefusee, match="désigner la classe positive") as refus:
        s2.appliquer(FixerObjectif(_classification(spec_reg, classe_positive=None)))
    assert (
        "conforme" in refus.value.motif and "non_conforme" in refus.value.motif
    )  # les deux classes proposées
    with pytest.raises(TransitionRefusee, match="n'est pas une valeur de la cible"):
        s2.appliquer(FixerObjectif(_classification(spec_reg, classe_positive="rebut")))
    s3 = s2.appliquer(FixerObjectif(_classification(spec_reg)))
    assert s3.spec.classe_positive == "non_conforme"
    sans_besoin = _classification(spec_reg, classe_positive=None, metrique="exactitude")
    assert s2.appliquer(FixerObjectif(sans_besoin)).etat == S.OBJECTIF


def test_I7_pas_de_classe_positive_en_multiclasse(parcours, spec_reg) -> None:
    roles = tuple(
        (c, Role.CIBLE if c == "materiau" else (Role.NUMERIQUE if c == "y_reg" else r))
        for c, r in spec_reg.roles
    )
    multiclasse = spec_reg.avec(
        probleme=Probleme.CLASSIFICATION, cible="materiau", roles=roles, metrique="f1"
    )
    s2 = parcours.jusqua(S.AUDIT)
    assert s2.appliquer(FixerObjectif(multiclasse)).etat == S.OBJECTIF  # trois classes : F1 moyen
    # en désigner une serait ignoré en silence : refus, avec le moyen de s'en sortir
    with pytest.raises(TransitionRefusee, match=r"3 classes.*moyennes macro.*regrouper les classes") as refus:
        s2.appliquer(FixerObjectif(multiclasse.avec(classe_positive="inox")))
    assert "ne pas fixer de classe positive" in refus.value.motif


def test_classe_positive_fait_partie_de_l_objectif(fabrique_parcours, espion, spec_reg) -> None:
    auditeur = espion()
    spec = _classification(spec_reg)
    p = fabrique_parcours(spec, auditeur)
    s3 = p.jusqua(S.OBJECTIF)
    n = len(auditeur.appels)
    s3b = s3.appliquer(FixerObjectif(spec.avec(classe_positive="conforme")))
    assert len(auditeur.appels) == n + 1  # changer la classe positive rejoue l'audit
    assert s3b.spec.objectif() != s3.spec.objectif()


def test_sources_uniques_du_socle() -> None:
    module_entree = importlib.import_module(
        "mecaml.core.catalog.entree"
    )  # `catalog.entree` est aussi une fonction
    assert LIGNES_MAX_LENTES == 20_000 and module_entree.LIGNES_MAX_LENTES is LIGNES_MAX_LENTES
    vues: list[int] = []

    def fabrique(graine: int, **parametres: object) -> object:
        vues.append(graine)
        return object()

    lourde = EntreeCatalogue(
        id="regression.essai_lourd",
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=fabrique,
        grille_bornee=(),
        cout=Cout.LOURD,
        ref_livre="T05-1-1",
        texte_id="regression.essai_lourd",
    )
    lourde.construire()
    assert vues == [GRAINE_CONTRAT]
    assert not lourde.coupee(LIGNES_MAX_LENTES) and lourde.coupee(LIGNES_MAX_LENTES + 1)
    legere = EntreeCatalogue(**{**{k: getattr(lourde, k) for k in EntreeCatalogue.__dataclass_fields__},
                                "cout": Cout.LEGER})  # fmt: skip
    assert not legere.coupee(10**6)


def test_spec_lecture_et_cles(spec_reg) -> None:
    spec = spec_reg.avec(structure=Structure.GROUPEE_CHRONOLOGIQUE)
    assert spec.colonne_groupe == "machine" and spec.colonne_date == "cycle"
    assert spec_reg.colonne_groupe is None and spec_reg.colonne_date is None
    assert spec.variables() == ("effort", "materiau", "temperature")
    assert spec.cle_audit() != spec_reg.cle_audit() and spec.objectif() == spec_reg.objectif()
    assert "y_reg" in spec.colonnes_citees()
    assert [nom for nom, _ in spec.roles] == sorted(nom for nom, _ in spec.roles)


def test_spec_aller_retour_json_et_hachage(spec_reg) -> None:
    spec = spec_reg.avec(
        structure=Structure.GROUPEE,
        pretraitement=(ChoixTechnique.creer("manquants.mediane", colonnes=["effort"]),),
        modeles=(
            ChoixTechnique.creer("regression.ridge", alpha=0.5, fit_intercept=True),
            ChoixTechnique.creer("regression.mlp", hidden_layer_sizes=(10, 5), solver="lbfgs"),
            ChoixTechnique.creer("regression.grille", valeurs=[1, [2, 3]], aucune=None),
        ),
    )
    texte = json.dumps(spec.vers_dict(), ensure_ascii=False)
    relue = SpecProjet.depuis_dict(json.loads(texte))
    assert relue == spec and hash(relue) == hash(spec)
    assert spec.modeles[1].parametres_dict()["hidden_layer_sizes"] == (10, 5)
    assert spec.modeles[2].parametres_dict()["valeurs"] == (1, (2, 3))


def test_choix_technique_normalise(spec_reg) -> None:
    a = ChoixTechnique("regression.ridge", parametres=(("b", [1, 2]), ("a", np.float64(0.5))))
    b = ChoixTechnique("regression.ridge", parametres={"a": 0.5, "b": (1, 2)})
    assert a == b and hash(a) == hash(b) and a.parametres == (("a", 0.5), ("b", (1, 2)))
    assert type(a.parametres_dict()["a"]) is float and a.famille == "regression"


@pytest.mark.parametrize(
    ("parametres", "erreur", "motif"),
    [
        ({"x": {"a": 1}}, TypeError, "non sérialisable"),
        ({"x": {1, 2}}, TypeError, "non sérialisable"),
        ({"x": object()}, TypeError, "non sérialisable"),
        ({"x": float("nan")}, ValueError, "non fini"),
        ({"x": [1.0, float("inf")]}, ValueError, "non fini"),
    ],
    ids=["dict", "ensemble", "objet", "nan", "infini_imbrique"],
)
def test_choix_technique_parametres_refuses(parametres, erreur, motif) -> None:
    with pytest.raises(erreur, match=motif):
        ChoixTechnique.creer("regression.ridge", **parametres)


@pytest.mark.parametrize(
    "arguments",
    [
        {"niveau": "alerte", "code": "A", "message": "m", "action": "a"},
        {"niveau": Niveau.INFO, "code": "a b", "message": "m"},
        {"niveau": Niveau.INFO, "code": "A", "message": "  "},
        {"niveau": Niveau.ALERTE, "code": "A", "message": "m"},
        {"niveau": Niveau.BLOQUANT, "code": "A", "message": "m", "action": ""},
    ],
    ids=["niveau_texte", "code_espace", "message_vide", "alerte_sans_action", "bloquant_sans_action"],
)
def test_constat_refus(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        Constat(**arguments)


@pytest.mark.parametrize(
    ("arguments", "motif"),
    [
        ({"modele_id": "", "metrique": "rmse", "score_validation": 1.0}, "Decision : `modele_id` est vide"),
        ({"modele_id": "m.a", "metrique": "", "score_validation": 1.0}, "Decision : `metrique` est vide"),
        ({"modele_id": "m.a", "metrique": "rmse", "score_validation": float("inf")}, "nombre fini"),
        ({"modele_id": "m.a", "metrique": "rmse", "score_validation": "0.9"}, "nombre fini"),
        ({"modele_id": "m.a", "metrique": "f1", "score_validation": 0.9, "seuil": 1.5}, "entre 0 et 1"),
    ],
    ids=["modele_vide", "metrique_vide", "score_infini", "score_texte", "seuil_hors_bornes"],
)
def test_decision_refus(arguments, motif) -> None:
    with pytest.raises(ValueError, match=motif):
        Decision(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [{"id": "ridge"}, {"id": "Regression.Ridge"}, {"id": "a.b", "parametres": (("x", 1), ("x", 2))}],
    ids=["sans_famille", "majuscules", "parametre_double"],
)
def test_choix_technique_refus(arguments) -> None:
    with pytest.raises(ValueError):
        ChoixTechnique(**arguments)


def test_enumerations() -> None:
    assert Probleme.REGRESSION.avec_cible and not Probleme.SANS_CIBLE.avec_cible
    assert Structure.GROUPEE_CHRONOLOGIQUE.groupee and Structure.GROUPEE_CHRONOLOGIQUE.chronologique
    assert not Structure.INDEPENDANTE.groupee
    assert Constat(Niveau.BLOQUANT, "B", "m", "a").bloquant


# ============================================================
# REGISTRE DES TECHNIQUES
# ============================================================
def _module(nom: str, entrees_module: object) -> ModuleType:
    module = ModuleType(nom)
    module.ENTREES = entrees_module  # type: ignore[attr-defined]
    return module


def test_registre_explicite_coherent() -> None:
    assert isinstance(REGISTRE, tuple)
    assert all(isinstance(e, EntreeCatalogue) for e in REGISTRE)
    assert len({e.id for e in REGISTRE}) == len(REGISTRE)


def test_entree_dummy_construit_un_estimateur(entree_dummy) -> None:
    estimateur = entree_dummy.construire(graine=0, strategy="median")
    assert estimateur.get_params()["strategy"] == "median"
    assert entree_dummy.famille == "reference" and entree_dummy.combinaisons == 2
    assert entree_dummy.grille() == {"strategy": ["mean", "median"]}


def test_registre_assemble_filtre_et_cherche(entree_dummy) -> None:
    registre = construire_registre([_module("catalog.dummy", (entree_dummy,))])
    assert entrees(registre, probleme=Probleme.REGRESSION) == (entree_dummy,)
    assert entrees(registre, probleme=Probleme.CLASSIFICATION) == ()
    assert entrees(registre, famille="reference") == (entree_dummy,)
    assert entree("reference.dummy_regression", registre) is entree_dummy
    with pytest.raises(KeyError, match="inconnue"):
        entree("regression.absente", registre)


def test_registre_refus(entree_dummy) -> None:
    with pytest.raises(ValueError, match="double"):
        construire_registre([_module("a", (entree_dummy,)), _module("b", (entree_dummy,))])
    with pytest.raises(TypeError, match="ENTREES"):
        construire_registre([ModuleType("vide")])
    with pytest.raises(TypeError, match="invalide"):
        construire_registre([_module("c", ("pas une entrée",))])


@pytest.mark.parametrize(
    ("changements", "erreur"),
    [
        ({"id": "dummy"}, ValueError),
        ({"id": "inconnue.dummy"}, ValueError),
        ({"problemes": frozenset()}, ValueError),
        ({"problemes": frozenset({"regression"})}, TypeError),
        ({"fabrique": "DummyRegressor"}, TypeError),
        ({"cout": "leger"}, TypeError),
        ({"ref_livre": ""}, ValueError),
        ({"grille_bornee": (("a", ()),)}, ValueError),
        ({"grille_bornee": (("a", (1,)), ("a", (2,)))}, ValueError),
        ({"grille_bornee": (("a", tuple(range(GRILLE_MAX_COMBINAISONS + 1))),)}, ValueError),
    ],
)
def test_entree_catalogue_refus(entree_dummy, changements, erreur) -> None:
    champs = {k: getattr(entree_dummy, k) for k in EntreeCatalogue.__dataclass_fields__}
    with pytest.raises(erreur):
        EntreeCatalogue(**{**champs, **changements})


def test_cout_enumere() -> None:
    assert {c.value for c in Cout} == {"leger", "moyen", "lourd"}
