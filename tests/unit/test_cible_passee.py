"""Cible passée : seuils du compte à rebours épinglés en valeur, verrou de session, référence naïve.

Les frontières sont écrites en valeur absolue (9 pas, 90 %, un demi pas) et non relativement aux
constantes du module : changer un seuil doit faire échouer un test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core import cible_passee, train
from mecaml.core.audit import auditer
from mecaml.core.session import (
    Auditer,
    Charger,
    Decouper,
    Entrainer,
    FixerObjectif,
    Pretraiter,
    Session,
    TransitionRefusee,
)
from mecaml.core.types import REFERENCES_NAIVES, ChoixTechnique, Probleme, Table

CT = ChoixTechnique.creer
PERSISTANCE = CT("reference.persistance", cible_connue=True)
PERSISTANCE_NON_DECLAREE = CT("reference.persistance")
CIBLE_DECALEE_NON_DECLAREE = CT("chronologique.cible_decalee")
DUMMY = CT("reference.dummy_regression")
RIDGE = CT("regression.ridge")


# ============================================================
# COMPTE À REBOURS : LES SEUILS, EN VALEUR ABSOLUE
# ============================================================
def _serie(pas) -> pd.DataFrame:
    """Une machine, un cycle par ligne ; la cible suit exactement la suite de pas donnée."""
    pas = np.asarray(pas, dtype=float)
    y = 1000.0 + np.concatenate([[0.0], np.cumsum(pas)])
    return pd.DataFrame({"machine": "M00", "cycle": np.arange(len(y)), "y": y})


def _decompte(donnees: pd.DataFrame):
    return cible_passee.decompte(donnees, aides.spec_groupe())


def test_la_duree_de_vie_restante_est_un_compte_a_rebours_quel_que_soit_le_pas() -> None:
    assert _decompte(_serie([-1.0] * 40)) == (-1.0, 1.0)
    assert _decompte(_serie([-0.5] * 40)) == (-0.5, 1.0)  # la même cible, en jours
    assert _decompte(_serie([-2.0] * 40)) == (-2.0, 1.0)  # un cycle sur deux


@pytest.mark.parametrize(
    "pas",
    [
        [0.0] * 40,  # cible constante
        [0.0] * 19 + [5.0] + [0.0] * 20,  # escalier montant : médiane nulle
        [0.0] * 19 + [-5.0] + [0.0] * 20,  # escalier descendant : médiane nulle, jamais un compte à rebours
        [1.0] * 40,  # âge qui croît
    ],
)
def test_une_cible_constante_par_morceaux_ou_croissante_ne_se_juge_pas(pas) -> None:
    assert _decompte(_serie(pas)) is None


def test_neuf_pas_ne_se_jugent_pas_dix_pas_se_jugent() -> None:
    assert _decompte(_serie([-1.0] * 9)) is None
    assert _decompte(_serie([-1.0] * 10)) == (-1.0, 1.0)


@pytest.mark.parametrize(("reguliers", "attendu"), [(89, False), (90, True), (91, True)])
def test_la_part_des_pas_reguliers_se_coupe_a_90_pour_cent(reguliers: int, attendu: bool) -> None:
    pas = [-1.0] * reguliers + [5.0] * (100 - reguliers)  # les autres pas montent, ils ne comptent pas
    mesure = _decompte(_serie(pas))
    assert (mesure is not None) is attendu
    if attendu:
        assert mesure == (-1.0, reguliers / 100)


@pytest.mark.parametrize(("ecart", "attendu"), [(0.4, True), (0.5, True), (0.6, False)])
def test_un_pas_s_ecarte_au_plus_d_un_demi_pas_de_la_mediane(ecart: float, attendu: bool) -> None:
    """Le quart des pas à `ecart` en dessous de la médiane, le quart à `ecart` au dessus."""
    pas = [-1.0] * 50 + [-1.0 - ecart] * 25 + [-1.0 + ecart] * 25
    assert (_decompte(_serie(pas)) is not None) is attendu


def test_des_pas_bien_plus_negatifs_que_la_mediane_ne_comptent_pas() -> None:
    """La tolérance est bilatérale : le cinquième des pas à -10 sort du compte, qui tombe à 80 %."""
    assert _decompte(_serie([-1.0] * 80 + [-10.0] * 20)) is None
    assert _decompte(_serie([-1.0] * 80 + [5.0] * 20)) is None  # et de même au dessus
    assert _decompte(_serie([-1.0] * 90 + [-10.0] * 10)) == (-1.0, 0.9)


@pytest.mark.parametrize(("ecart_type", "attendu"), [(0.2, True), (0.3, False)])
def test_le_bruit_de_la_cible_se_tolere_jusqu_a_0_2_pas_et_non_au_dela(
    ecart_type: float, attendu: bool
) -> None:
    """Un bruit de mesure de la cible : écart type de 0,2 pas détecté, de 0,3 pas non détecté."""
    bruit = np.random.default_rng(0).normal(0.0, ecart_type, 4001)
    donnees = _serie([-1.0] * 4000)
    donnees["y"] = donnees["y"] + bruit
    assert (_decompte(donnees) is not None) is attendu


def test_les_pas_se_comptent_dans_chaque_unite_et_non_d_une_unite_a_l_autre() -> None:
    """Quatre machines de 30 cycles : les pas se calculent dans chaque unité, pas d'une unité à l'autre."""
    donnees = aides.jeu_groupe(n_groupes=4, longueur=30)
    mesure = _decompte(donnees)
    assert mesure is not None and mesure[0] == pytest.approx(-1.0, abs=0.05) and mesure[1] == 1.0


def test_limite_connue_une_duree_de_vie_plafonnee_a_125_cycles_n_est_pas_reperee() -> None:
    """Caractérisation : la garde ne repère qu'un compte à rebours régulier ; la déclaration engage le projet.

    Plafonnée à 125 cycles sur 200, la cible perd un cycle à chaque cycle dans 63 % des pas seulement.
    """
    lignes = []
    for machine in range(4):
        for cycle in range(200):
            lignes.append({"machine": f"M{machine:02d}", "cycle": cycle, "y": float(min(200 - cycle, 125))})
    plafonnee = pd.DataFrame(lignes)
    assert _decompte(plafonnee) is None
    libre = plafonnee.assign(y=[float(200 - c) for c in plafonnee["cycle"]])
    assert _decompte(libre) == (-1.0, 1.0)


def test_une_cible_constante_declaree_ne_leve_aucun_constat() -> None:
    spec = aides.spec_groupe(modeles=(PERSISTANCE,))
    assert cible_passee.constats(spec, _serie([0.0] * 40)) == ()
    assert [c.code for c in cible_passee.constats(spec, _serie([-1.0] * 40))] == [
        "I3.cible_decompte:persistance"
    ]
    assert cible_passee.constats(spec, _serie([-1.0] * 9)) == ()  # trop peu de pas pour juger


# ============================================================
# SESSION : LE VERROU TIENT AUX ÉTAPES S5 ET S6
# ============================================================
def _s4(donnees: pd.DataFrame, groupes_test: tuple[str, ...]):
    table = Table(donnees, "essai")
    spec = aides.spec_groupe()
    coffre = aides.coffre_par_groupes(table, spec, groupes_test)
    s3 = (
        Session(auditeur=auditer)
        .appliquer(Charger(table))
        .appliquer(Auditer())
        .appliquer(FixerObjectif(spec))
    )
    return s3.appliquer(Decouper(coffre))


def _s5(donnees: pd.DataFrame, groupes_test: tuple[str, ...] = ("M06", "M07")):
    s4 = _s4(donnees, groupes_test)
    return s4.appliquer(Pretraiter(s4.spec))


def test_s5_refuse_une_cible_decalee_non_declaree() -> None:
    s4 = _s4(aides.jeu_serie_mesuree(), ("M06", "M07"))
    avec_decalage = s4.spec.avec(pretraitement=(CIBLE_DECALEE_NON_DECLAREE,))
    with pytest.raises(TransitionRefusee, match=r"I3\.cible_passee:cible_decalee"):
        s4.appliquer(Pretraiter(avec_decalage))
    declaree = s4.spec.avec(pretraitement=(CT("chronologique.cible_decalee", cible_connue=True),))
    assert s4.appliquer(Pretraiter(declaree)).etat.name == "PRETRAITEMENT"


def test_s5_refuse_une_cible_decalee_declaree_sur_un_compte_a_rebours() -> None:
    s4 = _s4(aides.jeu_groupe(n_groupes=8, longueur=30), ("M06", "M07"))
    declaree = s4.spec.avec(pretraitement=(CT("chronologique.cible_decalee", cible_connue=True),))
    with pytest.raises(TransitionRefusee, match=r"I3\.cible_decompte:cible_decalee"):
        s4.appliquer(Pretraiter(declaree))


def test_s6_refuse_une_persistance_non_declaree_ou_sur_un_compte_a_rebours() -> None:
    s5 = _s5(aides.jeu_serie_mesuree())
    with pytest.raises(TransitionRefusee, match=r"I3\.cible_passee:persistance"):
        s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(modeles=(DUMMY, PERSISTANCE_NON_DECLAREE))))
    declaree = s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(modeles=(DUMMY, PERSISTANCE))))
    assert declaree.etat.name == "MODELES"  # le témoin : une série mesurée, déclarée, passe
    s5_rul = _s5(aides.jeu_groupe(n_groupes=8, longueur=30))
    with pytest.raises(TransitionRefusee, match=r"I3\.cible_decompte:persistance"):
        s5_rul.appliquer(Entrainer({"m": 1.0}, spec=s5_rul.spec.avec(modeles=(DUMMY, PERSISTANCE))))


# ============================================================
# RÉFÉRENCE NAÏVE : LA MOYENNE, JAMAIS LA PERSISTANCE
# ============================================================
def test_les_references_naives_sont_la_moyenne_et_la_classe_majoritaire() -> None:
    assert dict(REFERENCES_NAIVES) == {
        Probleme.REGRESSION: "reference.dummy_regression",
        Probleme.CLASSIFICATION: "reference.dummy_classification",
    }
    assert dict(REFERENCES_NAIVES) == train.REFERENCE


@pytest.mark.parametrize(
    "modeles",
    [
        (PERSISTANCE, DUMMY, RIDGE),
        (DUMMY, PERSISTANCE, RIDGE),
        (PERSISTANCE, RIDGE),
    ],
    ids=["persistance_en_tete", "moyenne_en_tete", "moyenne_ajoutee_d_office"],
)
def test_les_gains_se_lisent_contre_la_moyenne_quel_que_soit_l_ordre_des_modeles(modeles) -> None:
    table = Table(aides.jeu_serie_mesuree(), "temperatures")
    spec = aides.spec_groupe(modeles=modeles)
    coffre = aides.coffre_par_groupes(table, spec, ("M06", "M07"))
    res = train.entrainer(spec, table, coffre)
    assert res.reference().id == "reference.dummy_regression"
    moyenne = res.valeur(res.modele("reference.dummy_regression"))
    for id_modele in ("regression.ridge", "reference.persistance"):
        assert res.gain(id_modele) == pytest.approx(moyenne - res.valeur(res.modele(id_modele)))
    assert res.gain("reference.persistance") > 0  # la persistance bat la moyenne : un gain, pas la base


def test_une_comparaison_avec_la_seule_persistance_est_refusee_par_la_session() -> None:
    s5 = _s5(aides.jeu_serie_mesuree())
    with pytest.raises(TransitionRefusee, match="référence naïve"):
        s5.appliquer(Entrainer({"m": 1.0}, spec=s5.spec.avec(modeles=(PERSISTANCE, RIDGE))))
    avec_moyenne = s5.spec.avec(modeles=(PERSISTANCE, DUMMY))
    assert s5.appliquer(Entrainer({"m": 1.0}, spec=avec_moyenne)).etat.name == "MODELES"


# ============================================================
# DEUX CHOIX DE MÊME TECHNIQUE : TOUS DOIVENT DÉCLARER
# ============================================================
def test_deux_choix_de_meme_technique_dont_un_seul_declare_sont_refuses() -> None:
    un_seul = aides.spec_groupe(
        modeles=(CT("reference.persistance", cible_connue=True), CT("reference.persistance", pas=2))
    )
    codes = [c.code for c in cible_passee.constats(un_seul, _serie([0.0] * 40))]
    assert codes == ["I3.cible_passee:persistance"]
    les_deux = aides.spec_groupe(
        modeles=(
            CT("reference.persistance", cible_connue=True),
            CT("reference.persistance", cible_connue=True, pas=2),
        )
    )
    assert cible_passee.constats(les_deux, _serie([0.0] * 40)) == ()
