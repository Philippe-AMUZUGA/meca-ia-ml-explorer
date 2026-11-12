"""Décision : modèle retenu, seuil, mises en garde ; tout se joue sur la validation, avant le test."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml.core import decision, evaluate, metrics, train
from mecaml.core.session import (
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Etat,
    FixerObjectif,
    Pretraiter,
    Session,
)
from mecaml.core.types import ChoixTechnique, Decision

CT = ChoixTechnique.creer
RIDGE = CT("regression.ridge")
ARBRE = CT("regression.arbre", max_depth=3)
LOGISTIQUE = CT("classification.logistique")
ARBRE_CLS = CT("classification.arbre", max_depth=3)


def _mener(dossier: Path, jeu: pd.DataFrame, fabrique, **options) -> aides.Projet:
    return aides.mener_le_projet(aides.ecrire_csv(jeu, dossier / "jeu.csv"), fabrique, **options)


@pytest.fixture(scope="module")
def regression(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("decision_regression")
    return _mener(dossier, jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE, ARBRE)))


@pytest.fixture(scope="module")
def classification(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    dossier = tmp_path_factory.mktemp("decision_classification")
    return _mener(
        dossier,
        jeux.jeu_classification(n=300),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS, CT("classification.svc"))),
    )


@pytest.fixture(scope="module")
def bruit(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    """Une cible sans rapport avec les variables : aucun modèle ne bat la moyenne."""
    rng = np.random.default_rng(3)
    n = 100
    jeu = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "d": rng.uniform(0, 10, n),
            "c": rng.choice(["x", "y", "z"], n),
            "y": rng.normal(size=n),
        }
    )
    dossier = tmp_path_factory.mktemp("decision_bruit")
    return _mener(
        dossier, jeu, lambda t: jeux.spec_regression(modeles=(RIDGE, CT("regression.foret", n_estimators=20)))
    )


def _sans(resultat: train.Resultat, **champs) -> train.Resultat:
    return dataclasses.replace(resultat, **champs)


# ============================================================
# MODÈLES QUE L'ON PEUT RETENIR
# ============================================================
def test_la_reference_naive_ne_se_retient_jamais_meme_quand_elle_gagne(bruit: aides.Projet) -> None:
    res = bruit.resultat
    assert res.classement()[0].id == "reference.dummy_regression"  # sur du bruit, la moyenne gagne
    retenables = decision.modeles_a_retenir(res)
    assert [m.id for m in retenables] and "reference.dummy_regression" not in {m.id for m in retenables}
    assert decision.meilleur_modele(res).id == retenables[0].id
    with pytest.raises(decision.ErreurDecision, match="référence naïve"):
        decision.decider(res, "reference.dummy_regression")


def test_le_meilleur_modele_est_le_premier_du_classement_hors_reference(regression: aides.Projet) -> None:
    res = regression.resultat
    attendu = next(m for m in res.classement() if m.famille != "reference")
    assert decision.meilleur_modele(res).id == attendu.id == "regression.ridge"
    assert [m.id for m in decision.modeles_a_retenir(res)] == [
        m.id for m in res.classement() if m.famille != "reference"
    ]


def test_sans_modele_a_score_defini_il_n_y_a_rien_a_retenir(regression: aides.Projet) -> None:
    echec = train.ResultatModele(RIDGE, {}, 0.0, erreur="ValueError : essai")
    reference = regression.resultat.reference()
    res = _sans(regression.resultat, modeles=(reference, echec))
    assert decision.modeles_a_retenir(res) == ()
    with pytest.raises(decision.ErreurDecision, match="Aucun modèle n'a de score défini"):
        decision.meilleur_modele(res)
    with pytest.raises(decision.ErreurDecision, match="a échoué"):
        decision.decider(res, "regression.ridge")


def test_un_modele_sans_score_pour_la_metrique_ne_se_retient_pas(regression: aides.Projet) -> None:
    sans_score = train.ResultatModele(RIDGE, {"rmse": (math.nan, math.nan)}, 0.0)
    res = _sans(regression.resultat, modeles=(regression.resultat.reference(), sans_score))
    with pytest.raises(decision.ErreurDecision, match="aucun score défini"):
        decision.decider(res, "regression.ridge")


def test_un_modele_hors_comparaison_est_signale_en_francais(regression: aides.Projet) -> None:
    with pytest.raises(decision.ErreurDecision, match="absent de la comparaison"):
        decision.decider(regression.resultat, "regression.foret")


# ============================================================
# DÉCISION SANS SEUIL
# ============================================================
def test_la_decision_reprend_le_score_du_tableau_de_validation(regression: aides.Projet) -> None:
    res = regression.resultat
    choix = decision.decider(res, "regression.arbre")
    assert isinstance(choix, Decision)
    assert choix.modele_id == "regression.arbre" and choix.metrique == res.metrique == "rmse"
    assert choix.score_validation == res.valeur(res.modele("regression.arbre")) and choix.seuil is None
    assert "rmse" in choix.justification and "référence naïve" in choix.justification


def test_une_justification_donnee_est_gardee_telle_quelle(regression: aides.Projet) -> None:
    choix = decision.decider(regression.resultat, "regression.ridge", justification="  Le plus simple.  ")
    assert choix.justification == "Le plus simple."
    assert decision.proposer(regression.resultat) == decision.decider(regression.resultat, "regression.ridge")


def test_la_decision_est_acceptee_par_la_session(regression: aides.Projet) -> None:
    session = Session()
    for action in (
        Charger(regression.table),
        Auditer(),
        FixerObjectif(regression.spec),
        Decouper(regression.coffre, spec=regression.spec),
        Pretraiter(regression.spec),
        Entrainer(regression.resultat, spec=regression.resultat.spec),
        Decider(decision.proposer(regression.resultat)),
    ):
        session = session.appliquer(action)
    assert session.etat is Etat.DIAGNOSTIC and session.decision.modele_id == "regression.ridge"


# ============================================================
# SEUIL
# ============================================================
def _score_au_seuil_a_la_main(res: train.Resultat, id_modele: str, seuil: float) -> float:
    diag = evaluate.diagnostiquer_validation(res, id_modele, seuil=seuil)
    valeurs = []
    for k in range(len(res.plis)):
        m = diag.plis == k
        try:
            valeurs.append(
                metrics.calculer(
                    res.metrique,
                    y_vrai=diag.y_vrai[m],
                    y_pred=diag.y_pred[m],
                    scores=diag.scores[m],
                    classes=diag.classes,
                    classe_positive=diag.classe_positive,
                    seuil=seuil,
                )
            )
        except metrics.ErreurMetrique:
            continue
    return float(np.mean(valeurs))


@pytest.mark.parametrize("seuil", [0.15, 0.5, 0.8])
def test_le_score_d_une_decision_a_seuil_est_recalcule_pli_par_pli(
    classification: aides.Projet, seuil: float
) -> None:
    res = classification.resultat
    choix = decision.decider(res, "classification.logistique", seuil=seuil)
    assert choix.seuil == seuil
    assert choix.score_validation == pytest.approx(
        _score_au_seuil_a_la_main(res, "classification.logistique", seuil)
    )
    assert decision.score_au_seuil(res, "classification.logistique", seuil) == choix.score_validation


def test_un_seuil_change_le_score_et_les_phrases_le_disent(classification: aides.Projet) -> None:
    res = classification.resultat
    bas = decision.decider(res, "classification.logistique", seuil=0.1)
    haut = decision.decider(res, "classification.logistique", seuil=0.9)
    assert bas.score_validation != haut.score_validation
    assert "Seuil réglé à 0,100" in bas.justification


@pytest.mark.parametrize("seuil", [-0.1, 1.5, True])
def test_un_seuil_se_donne_entre_zero_et_un(classification: aides.Projet, seuil) -> None:
    with pytest.raises(decision.ErreurDecision, match="entre 0 et 1"):
        decision.decider(classification.resultat, "classification.logistique", seuil=seuil)


def test_un_seuil_suppose_des_probabilites(classification: aides.Projet) -> None:
    with pytest.raises(decision.ErreurDecision, match="ne donne pas de probabilités"):
        decision.decider(classification.resultat, "classification.svc", seuil=0.5)  # fonction de décision
    with pytest.raises(decision.ErreurDecision, match="ne donne pas de probabilités"):
        decision.seuil_minimisant_le_cout(classification.resultat, "classification.svc")


def test_un_seuil_suppose_deux_classes(regression: aides.Projet, tmp_path: Path) -> None:
    with pytest.raises(decision.ErreurDecision, match="deux classes"):
        decision.decider(regression.resultat, "regression.ridge", seuil=0.5)
    multi = _mener(
        tmp_path,
        jeux.jeu_classification(n=240, classes=3),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,), binaire=False),
    )
    with pytest.raises(decision.ErreurDecision, match="deux classes"):
        decision.decider(multi.resultat, "classification.logistique", seuil=0.5)
    assert decision.proposer(multi.resultat).seuil is None


def test_le_seuil_du_plus_petit_cout_minimise_le_cout_de_la_courbe(tmp_path: Path) -> None:
    projet = _mener(
        tmp_path,
        jeux.jeu_classification(n=300),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,), metrique="cout:fp=1,fn=10"),
    )
    res = projet.resultat
    propose = decision.seuil_minimisant_le_cout(res, "classification.logistique")
    assert (propose.cout_fp, propose.cout_fn) == (1.0, 10.0)  # les coûts viennent de la métrique du projet
    assert propose.cout == pytest.approx(propose.courbe["cout"].min())
    assert propose.seuil in set(propose.courbe["seuil"])
    # un faux négatif coûte dix fois un faux positif : le seuil optimal est bas
    egal = decision.seuil_minimisant_le_cout(res, "classification.logistique", cout_fp=1.0, cout_fn=1.0)
    assert propose.seuil < egal.seuil
    choix = decision.decider(res, "classification.logistique", seuil=propose.seuil)
    assert choix.score_validation == pytest.approx(
        _score_au_seuil_a_la_main(res, "classification.logistique", propose.seuil)
    )


def test_a_cout_egal_le_seuil_le_plus_proche_de_un_demi_l_emporte(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n = 200
    a = np.r_[rng.normal(-6, 0.5, n // 2), rng.normal(6, 0.5, n // 2)]
    jeu = pd.DataFrame({"a": a, "b": rng.normal(size=n), "c": rng.choice(["x", "y"], n)})
    jeu["y"] = np.where(a > 0, "defaut", "conforme")
    projet = _mener(
        tmp_path,
        jeu,
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,), metrique="cout:fp=1,fn=1"),
    )
    propose = decision.seuil_minimisant_le_cout(projet.resultat, "classification.logistique", points=201)
    assert propose.cout == 0.0  # deux paquets séparés : un seuil sans aucune erreur existe
    sans_erreur = propose.courbe[propose.courbe["cout"] == 0.0]["seuil"]
    assert abs(propose.seuil - 0.5) == min(abs(sans_erreur - 0.5))
    courbe = pd.DataFrame({"seuil": [0.1, 0.3, 0.45, 0.7, 0.9], "cout": [2.0, 0.5, 0.5, 0.5, 1.0]})
    assert (
        decision.choisir_le_seuil(courbe)["seuil"] == 0.45
    )  # trois seuils à coût égal : le plus proche de 0,5
    courbe.loc[3, "cout"] = 0.499999  # un coût plus petit, même de peu, l'emporte (les coûts sont des k / n)
    assert decision.choisir_le_seuil(courbe)["seuil"] == 0.7
    courbe.loc[3, "cout"] = 0.5 - 1e-14  # l'écart d'un arrondi n'est pas un écart
    assert decision.choisir_le_seuil(courbe)["seuil"] == 0.45


def test_le_seuil_du_cout_suppose_un_modele_retenable(classification: aides.Projet) -> None:
    with pytest.raises(decision.ErreurDecision, match="référence naïve"):
        decision.seuil_minimisant_le_cout(classification.resultat, "reference.dummy_classification")


# ============================================================
# MISES EN GARDE
# ============================================================
def test_un_modele_qui_ne_bat_pas_la_reference_est_signale(bruit: aides.Projet) -> None:
    notes = decision.avertissements(bruit.resultat, "regression.ridge")
    assert any("ne fait pas mieux que la référence naïve" in n for n in notes)


def test_un_modele_nettement_meilleur_n_est_pas_signale(regression: aides.Projet) -> None:
    notes = decision.avertissements(regression.resultat, "regression.ridge")
    assert not any("référence naïve" in n for n in notes)


def test_deux_modeles_que_les_plis_ne_departagent_pas_sont_signales(tmp_path: Path) -> None:
    projet = _mener(
        tmp_path,
        jeux.jeu_regression(n=120),
        lambda t: jeux.spec_regression(
            modeles=(CT("regression.ridge", alpha=1.0), CT("regression.lasso", alpha=0.0001))
        ),
    )
    notes = decision.avertissements(projet.resultat, decision.meilleur_modele(projet.resultat).id)
    assert any("ne se distinguent pas" in n for n in notes)


def test_une_metrique_definie_sur_une_partie_des_plis_est_signalee(tmp_path: Path) -> None:
    df = aides.jeu_pannes(n_machines=20, longueur=12)
    en_panne = df["machine"].isin(["M00", "M01", "M02"]) & (df.groupby("machine").cumcount() >= 9)
    df["y"] = np.where(
        en_panne, "panne", "marche"
    )  # trois machines en panne pour trois plis : un pli en manque
    projet = _mener(tmp_path, df, aides.spec_classification_groupee)
    modele = projet.resultat.modele("classification.logistique")
    assert modele.plis_valides("f1") < len(projet.resultat.plis)
    notes = decision.avertissements(projet.resultat, "classification.logistique")
    assert any("n'est défini que sur" in n and "plis sur 3" in n for n in notes)


def test_une_classe_positive_proposee_est_signalee_et_une_classe_confirmee_non(tmp_path: Path) -> None:
    propose = _mener(
        tmp_path,
        jeux.jeu_classification(n=240),
        lambda t: jeux.spec_classification(
            modeles=(LOGISTIQUE,), metrique="exactitude", classe_positive=None
        ),
    )
    assert propose.resultat.classe_positive_source not in (None, "specification")
    assert any(
        "proposition non confirmée" in n
        for n in decision.avertissements(propose.resultat, "classification.logistique")
    )
    (tmp_path / "c").mkdir()
    confirmee = _mener(
        tmp_path / "c",
        jeux.jeu_classification(n=240),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE,)),
    )
    assert confirmee.resultat.classe_positive_source == "specification"
    assert not any(
        "proposition non confirmée" in n
        for n in decision.avertissements(confirmee.resultat, "classification.logistique")
    )


def test_un_avertissement_de_calcul_du_modele_est_repris(tmp_path: Path) -> None:
    projet = _mener(
        tmp_path,
        jeux.jeu_classification(n=240),
        lambda t: jeux.spec_classification(modeles=(CT("classification.reseau_neurones", max_iter=2),)),
    )
    notes = decision.avertissements(projet.resultat, "classification.reseau_neurones")
    assert any(n.startswith("Avertissement de calcul") for n in notes)


# ============================================================
# SANS CIBLE
# ============================================================
def test_sans_cible_la_decision_porte_sur_la_metrique_du_projet(tmp_path: Path) -> None:
    projet = _mener(
        tmp_path,
        jeux.jeu_sans_cible(n=120),
        lambda t: jeux.spec_sans_cible(
            modeles=(CT("sans_cible.kmeans", n_clusters=2), CT("sans_cible.acp", n_components=2))
        ),
    )
    res = projet.resultat
    assert res.reference() is None
    choix = decision.decider(res, "sans_cible.kmeans")
    assert choix.metrique == "silhouette" and choix.score_validation == pytest.approx(
        res.valeur(res.modele("sans_cible.kmeans"))
    )
    with pytest.raises(decision.ErreurDecision, match="aucun score défini"):
        decision.decider(res, "sans_cible.acp")  # l'analyse en composantes n'a pas de silhouette
    assert decision.avertissements(res, "sans_cible.kmeans") == ()
    assert decision.meilleur_modele(res).id == "sans_cible.kmeans"


# ============================================================
# FIGURE
# ============================================================
def test_la_figure_de_comparaison_distingue_la_reference(regression: aides.Projet) -> None:
    fig = decision.figure_comparaison(regression.resultat)
    axe = fig.axes[0]
    assert len(axe.patches) == len(regression.resultat.modeles) == 3
    couleurs = {tuple(p.get_facecolor()) for p in axe.patches}
    assert len(couleurs) == 2  # la référence a sa couleur
    assert "Comparaison des modèles" in axe.get_title() and "plus petit est meilleur" in axe.get_xlabel()
    assert axe.get_xlim()[1] >= max(regression.resultat.valeur(m) for m in regression.resultat.modeles)


def test_la_figure_de_comparaison_d_une_metrique_a_maximiser(classification: aides.Projet) -> None:
    fig = decision.figure_comparaison(classification.resultat)
    assert "plus grand est meilleur" in fig.axes[0].get_xlabel()
