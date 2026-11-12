"""Entraînement : validation croisée par structure, résultats, référence naïve, échecs, bornes, recherche."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core import train
from mecaml.core.catalog import entree
from mecaml.core.session import (
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    FixerObjectif,
    Pretraiter,
    Session,
    Tester,
)
from mecaml.core.tasks import Annulation, ErreurAnnulation, EtatTache
from mecaml.core.train import ErreurBorne, ErreurBudget, ErreurEntrainement
from mecaml.core.types import ChoixTechnique, CoffreTest, Decision, Probleme, Role, Structure, Table

CT = ChoixTechnique.creer
RIDGE, FORET = CT("regression.ridge"), CT("regression.foret", n_estimators=20)


def _regression(modeles=(RIDGE, FORET), **changements):
    df = aides.jeu_regression()
    table = Table(df, "regression")
    spec = aides.spec_regression(modeles=modeles, **changements)
    return df, table, spec, aides.coffre_par_pas(table, spec)


def _classification(classes: int = 2, modeles=None, **changements):
    modeles = modeles or (CT("classification.logistique"), CT("classification.foret", n_estimators=20))
    df = aides.jeu_classification(classes=classes)
    table = Table(df, "classification")
    spec = aides.spec_classification(modeles=modeles, binaire=classes == 2, **changements)
    return df, table, spec, aides.coffre_par_pas(table, spec)


# ============================================================
# RÉGRESSION DE BOUT EN BOUT
# ============================================================
def test_regression_de_bout_en_bout() -> None:
    _, table, spec, coffre = _regression()
    res = train.entrainer(spec, table, coffre)
    assert [m.id for m in res.modeles] == [
        "reference.dummy_regression",
        "regression.ridge",
        "regression.foret",
    ]
    assert res.spec.modeles[0].famille == "reference" and res.metrique == "rmse" and res.meilleur == "bas"
    assert res.validation == "validation.kfold" and len(res.plis) == spec.n_plis
    assert (res.positions == coffre.positions_apprentissage()).all()
    assert len(res.y_vrai) == coffre.n_apprentissage and res.n_sans_cible == 0
    # les plis de validation couvrent chaque ligne une fois, sans recouvrement
    validation = np.sort(np.concatenate([v for _, v in res.plis]))
    assert (validation == np.arange(coffre.n_apprentissage)).all()
    assert all(not set(a) & set(v) for a, v in res.plis)
    ridge, reference = res.modele("regression.ridge"), res.reference()
    assert ridge.moyenne("rmse") < 1.0 < 1.5 < reference.moyenne("rmse")
    assert res.gain("regression.ridge") > 1.5 and res.gain("reference.dummy_regression") == 0
    assert res.classement()[0].id in ("regression.ridge", "regression.foret")
    assert (
        set(ridge.scores_plis) == {"r2", "mae", "rmse", "mape"}
        and len(ridge.scores_plis["rmse"]) == spec.n_plis
    )


def test_les_scores_par_pli_se_recalculent_depuis_les_predictions_hors_pli() -> None:
    _, table, spec, coffre = _regression()
    res = train.entrainer(spec, table, coffre)
    ridge = res.modele("regression.ridge")
    assert not np.isnan(ridge.predictions).any()
    for k, (_, val) in enumerate(res.plis):
        erreur = res.y_vrai[val] - ridge.predictions[val]
        assert ridge.scores_plis["rmse"][k] == pytest.approx(math.sqrt(np.mean(erreur**2)))
        assert ridge.scores_plis["mae"][k] == pytest.approx(np.mean(np.abs(erreur)))
    assert ridge.ecart_type("rmse") == pytest.approx(np.std(ridge.scores_plis["rmse"], ddof=1))
    assert ridge.moyenne("rmse") == pytest.approx(np.mean(ridge.scores_plis["rmse"]))


def test_tableau_comparatif() -> None:
    _, table, spec, coffre = _regression()
    res = train.entrainer(spec, table, coffre)
    tableau = res.tableau()
    assert list(tableau.columns) == [
        "modele",
        "famille",
        "moyenne",
        "ecart_type",
        "gain_vs_reference",
        "plis_valides",
        "duree_s",
        "statut",
    ]
    assert tableau["modele"].iloc[-1] == "reference.dummy_regression"  # la plus mauvaise en dernier (RMSE)
    assert tableau["statut"].iloc[-1] == "référence" and set(tableau["statut"].iloc[:-1]) == {"ajusté"}
    assert (tableau["moyenne"].diff().dropna() >= 0).all()  # RMSE : du plus petit au plus grand


def test_reproductibilite_meme_graine_memes_scores() -> None:
    _, table, spec, coffre = _regression()
    a, b = train.entrainer(spec, table, coffre), train.entrainer(spec, table, coffre)
    for ma, mb in zip(a.modeles, b.modeles, strict=True):
        assert ma.scores_plis == mb.scores_plis
    autre = aides.spec_regression(modeles=spec.modeles, graine=7)
    c = train.entrainer(autre, table, aides.coffre_par_pas(table, autre))
    assert c.modele("regression.foret").scores_plis != a.modele("regression.foret").scores_plis


def test_les_lignes_de_test_n_influencent_aucun_score() -> None:
    """I1 côté apprentissage : changer les valeurs du test ne change aucun score de validation."""
    df, table, spec, coffre = _regression()
    poison = df.copy()
    test = coffre_test_positions(coffre, len(df))
    poison.loc[poison.index[test], ["a", "b", "d", "y"]] = 1e9
    table_poison = Table(poison, "poison")
    res = train.entrainer(spec, table, coffre)
    res_poison = train.entrainer(spec, table_poison, aides.coffre_par_pas(table_poison, spec))
    for m, mp in zip(res.modeles, res_poison.modeles, strict=True):
        assert m.scores_plis == mp.scores_plis, m.id
    assert np.abs(res_poison.y_vrai).max() < 1e3


def coffre_test_positions(coffre: CoffreTest, n: int) -> np.ndarray:
    """Positions du test, déduites du complément de l'apprentissage (le coffre ne les donne pas)."""
    return np.setdiff1d(np.arange(n), coffre.positions_apprentissage())


# ============================================================
# CLASSIFICATION
# ============================================================
def test_classification_binaire_classe_positive_et_reference() -> None:
    _, table, spec, coffre = _classification()
    res = train.entrainer(spec, table, coffre)
    assert res.classes == ("conforme", "defaut") and res.classe_positive == "defaut"
    assert res.validation == "validation.kfold_stratifie" and res.meilleur == "haut"
    reference, logistique = res.reference(), res.modele("classification.logistique")
    assert reference.id == "reference.dummy_classification"
    assert reference.moyenne("f1") == 0.0  # la référence prédit toujours la classe majoritaire
    assert reference.moyenne("roc_auc") == pytest.approx(0.5)
    assert reference.moyenne("exactitude") == pytest.approx(0.7, abs=0.05)
    assert logistique.moyenne("roc_auc") > 0.9 and logistique.moyenne("f1") > 0.7
    assert res.gain("classification.logistique") > 0.7
    # plis stratifiés : la part de la classe rare est stable d'un pli à l'autre
    parts = [np.mean(res.y_vrai[v] == "defaut") for _, v in res.plis]
    assert max(parts) - min(parts) < 0.05
    assert (
        logistique.scores_continus.shape == (coffre.n_apprentissage,)
        and not np.isnan(logistique.scores_continus).any()
    )
    assert set(logistique.predictions) <= {"conforme", "defaut"}


def test_classification_multiclasse() -> None:
    _, table, spec, coffre = _classification(classes=3)
    res = train.entrainer(spec, table, coffre)
    assert len(res.classes) == 3 and res.classe_positive is None
    logistique = res.modele("classification.logistique")
    assert logistique.scores_continus.shape == (coffre.n_apprentissage, 3)
    assert logistique.moyenne("roc_auc") > 0.85 and logistique.moyenne("f1") > 0.6
    assert res.reference().moyenne("exactitude") < logistique.moyenne("exactitude")


def test_metrique_de_cout_parametree() -> None:
    _, table, spec, coffre = _classification(metrique="cout:fp=1,fn=5")
    res = train.entrainer(spec, table, coffre)
    assert res.metrique == "cout:fp=1,fn=5" and res.meilleur == "bas"
    logistique = res.modele("classification.logistique")
    assert "cout:fp=1,fn=5" in logistique.scores_plis and not math.isnan(res.valeur(logistique))
    # prédire toujours « conforme » rate toutes les pièces défectueuses : 30 % de faux négatifs à 5
    assert res.valeur(res.reference()) == pytest.approx(0.3 * 5, abs=0.3)
    assert res.gain("classification.logistique") > 0


def test_classe_rare_plus_petite_que_les_plis_refusee() -> None:
    df = aides.jeu_classification()
    df.loc[df.index[:3], "y"] = "rarissime"
    table = Table(df, "rare")
    spec = aides.spec_classification(modeles=(CT("classification.logistique"),), binaire=False)
    coffre = CoffreTest.par_positions(table, np.arange(200, 300), decoupage=spec.decoupage())
    with pytest.raises(ErreurEntrainement, match=r"rarissime|classe la plus rare"):
        train.entrainer(spec, table, coffre)


# ============================================================
# STRUCTURES : GROUPES ET TEMPS
# ============================================================
def test_validation_par_groupes_aucune_machine_a_cheval() -> None:
    df = aides.jeu_groupe()
    table = Table(df, "groupes")
    spec = aides.spec_groupe(modeles=(RIDGE,))
    coffre = aides.coffre_par_groupes(table, spec, ("M08", "M09"))
    res = train.entrainer(spec, table, coffre)
    assert res.validation == "validation.plis_groupes" and len(res.plis) == 3
    apprentissage = coffre.apprentissage()
    machines = apprentissage["machine"].to_numpy()
    for app, val in res.plis:
        assert not set(machines[app]) & set(machines[val])
    assert not set(machines) & {"M08", "M09"}  # le test reste hors de la validation
    assert res.modele("regression.ridge").moyenne("rmse") < res.reference().moyenne("rmse")


def test_validation_chronologique_la_validation_suit_l_apprentissage() -> None:
    df = aides.jeu_groupe(n_groupes=1, longueur=60)
    table = Table(df, "serie")
    roles = (("machine", Role.IGNOREE), *aides.ROLES_GROUPE[1:])
    spec = aides.spec_groupe(modeles=(RIDGE,), roles=roles, structure=Structure.CHRONOLOGIQUE)
    coffre = CoffreTest.par_positions(table, np.arange(48, 60), decoupage=spec.decoupage())
    res = train.entrainer(spec, table, coffre)
    cycles = coffre.apprentissage()["cycle"].to_numpy()
    assert res.validation == "validation.serie_temporelle"
    for app, val in res.plis:
        assert cycles[app].max() < cycles[val].min()
    ridge = res.modele("regression.ridge")
    assert np.isnan(ridge.predictions).any(), "la fenêtre croissante ne valide pas le tout début de la série"
    assert not np.isnan(ridge.predictions[res.plis[-1][1]]).any()


def test_variables_temporelles_dans_la_validation_par_groupes() -> None:
    df = aides.jeu_groupe()
    table = Table(df, "groupes")
    pretraitement = (
        CT("chronologique.variables_temporelles", retards=(1, 2), fenetres=(3,)),
        CT("transformation.standard"),
    )
    spec = aides.spec_groupe(modeles=(RIDGE,), pretraitement=pretraitement)
    coffre = aides.coffre_par_groupes(table, spec, ("M08", "M09"))
    res = train.entrainer(spec, table, coffre)
    sans = train.entrainer(
        aides.spec_groupe(modeles=(RIDGE,)),
        table,
        aides.coffre_par_groupes(table, aides.spec_groupe(), ("M08", "M09")),
    )
    assert res.modele("regression.ridge").ok
    assert not np.isnan(res.modele("regression.ridge").predictions).any()
    # la dérive de chaque machine rend les retards très informatifs : la variable construite aide
    assert res.modele("regression.ridge").moyenne("rmse") <= sans.modele("regression.ridge").moyenne("rmse")


# ============================================================
# SANS CIBLE
# ============================================================
def test_modeles_sans_cible() -> None:
    df = aides.jeu_sans_cible()
    table = Table(df, "amas")
    reference = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    modeles = (
        CT("sans_cible.kmeans", n_clusters=2),
        CT("sans_cible.melange_gaussien", n_components=2),
        CT("sans_cible.dbscan", eps=1.0, min_samples=5),
        CT("sans_cible.acp", n_components=2),
        CT("sans_cible.foret_isolement", contamination=0.05, n_estimators=50),
        CT("sans_cible.facteur_local", contamination=0.05),
        CT("sans_cible.svm_une_classe", nu=0.05),
    )
    res = train.entrainer(aides.spec_sans_cible(modeles=modeles), table, etiquettes_reference=reference)
    assert res.reference() is None and res.plis == () and res.validation == "aucune"
    assert all(m.ok for m in res.modeles), [m.erreur for m in res.modeles if not m.ok]
    kmeans = res.modele("sans_cible.kmeans")
    assert kmeans.moyenne("silhouette") > 0.3 and kmeans.moyenne("rand_ajuste") == pytest.approx(1.0)
    assert kmeans.details["n_groupes"] == 2 and set(kmeans.etiquettes.tolist()) == {0, 1}
    assert res.modele("sans_cible.melange_gaussien").moyenne("rand_ajuste") == pytest.approx(1.0)
    assert res.modele("sans_cible.dbscan").details["n_groupes"] >= 2
    acp = res.modele("sans_cible.acp")
    assert 0.5 < acp.moyenne("variance_expliquee") <= 1.0 and len(acp.details["variance_par_composante"]) == 2
    for id_anomalie in ("foret_isolement", "facteur_local", "svm_une_classe"):
        m = res.modele(f"sans_cible.{id_anomalie}")
        assert 0.0 <= m.moyenne("taux_anomalies") <= 0.3, id_anomalie
        assert m.details["part_anomalies"] == pytest.approx(m.moyenne("taux_anomalies"))
        assert "silhouette" not in m.scores_plis
    assert res.modele("sans_cible.kmeans").pipeline is not None and res.metrique == "silhouette"


def test_etiquettes_de_reference_de_mauvaise_longueur() -> None:
    table = Table(aides.jeu_sans_cible(), "amas")
    spec = aides.spec_sans_cible(modeles=(CT("sans_cible.kmeans", n_clusters=2),))
    with pytest.raises(ErreurEntrainement, match="étiquettes de référence"):
        train.entrainer(spec, table, etiquettes_reference=[0, 1])


# ============================================================
# ÉCHECS, AVERTISSEMENTS, ANNULATION, BUDGET
# ============================================================
def test_un_modele_defaillant_est_consigne_et_les_autres_continuent() -> None:
    # Réglages permis un à un, refusés ensemble par la bibliothèque : le modèle échoue, les autres continuent.
    echec = CT("regression.ridge", solver="svd", positive=True)
    _, table, spec, coffre = _regression(modeles=(CT("regression.lineaire"), echec))
    res = train.entrainer(spec, table, coffre)
    ridge = res.modele("regression.ridge")
    assert not ridge.ok and ridge.erreur and ridge.scores_plis == {}
    assert res.modele("regression.lineaire").ok and res.reference().ok
    assert res.classement()[-1].id == "regression.ridge"
    assert res.gain("regression.ridge") is None
    ligne = res.tableau().iloc[-1]
    assert ligne["statut"].startswith("échec : ") and math.isnan(ligne["moyenne"])


def test_un_reglage_hors_bornes_est_consigne_avec_son_message_francais() -> None:
    _, table, spec, coffre = _regression(modeles=(RIDGE, CT("regression.lasso", max_iter=0)))
    res = train.entrainer(spec, table, coffre)
    lasso = res.modele("regression.lasso")
    assert not lasso.ok and "Itérations maximales" in lasso.erreur and "hors bornes" in lasso.erreur
    assert res.modele("regression.ridge").ok


def test_la_reference_defaillante_arrete_la_comparaison() -> None:
    _, table, spec, coffre = _regression(
        modeles=(CT("reference.dummy_regression", strategy="inconnue"), RIDGE)
    )
    with pytest.raises(ErreurEntrainement, match="référence naïve a échoué"):
        train.entrainer(spec, table, coffre)


def test_les_avertissements_de_scikit_learn_sont_consignes_pas_leves() -> None:
    _, table, spec, coffre = _regression(modeles=(CT("regression.reseau_neurones", max_iter=3),))
    res = train.entrainer(spec, table, coffre)  # filterwarnings=error : une alerte non captée échouerait
    reseau = res.modele("regression.reseau_neurones")
    assert reseau.ok and any("ConvergenceWarning" in a for a in reseau.avertissements)


def test_annulation_entre_deux_modeles_et_plis() -> None:
    _, table, spec, coffre = _regression()
    annulation = Annulation()
    vus = []

    def avancer(p) -> None:
        vus.append(p)
        if len(vus) == 2:
            annulation.demander()

    with pytest.raises(ErreurAnnulation):
        train.entrainer(spec, table, coffre, annulation=annulation, progression=avancer)
    assert len(vus) == 2  # aucune unité de plus après la demande : 3 modèles x 5 plis = 15 prévues
    assert vus[0].total == 15 and vus[0].etape == "Entraînement"


def test_budget_epuise() -> None:
    _, table, spec, coffre = _regression()
    with pytest.raises(ErreurBudget, match="Budget de 0 s épuisé"):
        train.entrainer(spec, table, coffre, budget_s=0.0)


def test_entrainement_dans_un_fil() -> None:
    _, table, spec, coffre = _regression()
    tache = train.lancer_entrainement(spec, table, coffre)
    assert tache.attendre(60) and tache.etat is EtatTache.TERMINEE
    res = tache.resultat()
    assert res.modele("regression.ridge").ok
    assert [e.fait for e in tache.evenements()] == list(range(1, 16))
    ref = train.entrainer(spec, table, coffre)
    assert res.modele("regression.ridge").scores_plis == ref.modele("regression.ridge").scores_plis


def test_annulation_d_un_entrainement_dans_un_fil() -> None:
    _, table, spec, coffre = _regression()
    tache = train.lancer_entrainement(spec, table, coffre)
    tache.annuler()
    assert tache.attendre(60)
    assert tache.etat in (EtatTache.ANNULEE, EtatTache.TERMINEE)  # l'annulation peut arriver après la fin
    if tache.etat is EtatTache.ANNULEE:
        with pytest.raises(ErreurAnnulation):
            tache.resultat()


# ============================================================
# REFUS D'ENTRÉE
# ============================================================
def test_refus_sans_coffre_ou_avec_un_mauvais_coffre() -> None:
    df, table, spec, coffre = _regression()
    with pytest.raises(ErreurEntrainement, match="mis sous scellé"):
        train.entrainer(spec, table, None)
    autre = Table(df.copy(), "autre")
    with pytest.raises(ErreurEntrainement, match="pas été construit sur cette table"):
        train.entrainer(spec, autre, coffre)
    spec_autre_plis = spec.avec(n_plis=3)
    with pytest.raises(ErreurEntrainement, match="autre découpage"):
        train.entrainer(spec_autre_plis, table, coffre)


@pytest.mark.parametrize(
    ("modeles", "metrique", "motif"),
    [
        ((), "rmse", "Aucun modèle"),
        ((CT("classification.logistique"),), "rmse", "n'est pas un modèle applicable"),
        ((CT("selection.variance"),), "rmse", "n'est pas un modèle applicable"),
        ((CT("regression.absente"),), "rmse", "Technique inconnue"),
        ((RIDGE,), "f1", "ne s'applique pas"),
        ((RIDGE,), "inconnue", "Métrique inconnue"),
    ],
)
def test_refus_du_projet(modeles: tuple, metrique: str, motif: str) -> None:
    df = aides.jeu_regression()
    table = Table(df, "regression")
    spec = aides.spec_regression(modeles=modeles, metrique=metrique)
    with pytest.raises(ValueError, match=motif):
        train.entrainer(spec, table, aides.coffre_par_pas(table, spec))


def test_cible_non_numerique_en_regression_et_une_seule_classe() -> None:
    df = aides.jeu_regression()
    df["y"] = np.where(df["y"] > 0, "haut", "bas")
    table = Table(df, "texte")
    spec = aides.spec_regression(modeles=(RIDGE,))
    with pytest.raises(ErreurEntrainement, match="n'est pas numérique"):
        train.entrainer(spec, table, aides.coffre_par_pas(table, spec))
    df2 = aides.jeu_classification().assign(y="conforme")
    table2 = Table(df2, "une classe")
    spec2 = aides.spec_classification(modeles=(CT("classification.logistique"),))
    with pytest.raises(ErreurEntrainement, match="qu'une classe"):
        train.entrainer(spec2, table2, aides.coffre_par_pas(table2, spec2))


def test_lignes_sans_cible_retirees_et_comptees() -> None:
    df = aides.jeu_regression()
    df.loc[df.index[[1, 2, 3, 4, 6]], "y"] = np.nan  # positions 1 à 6 hors multiples de 5 : apprentissage
    table = Table(df, "manques")
    spec = aides.spec_regression(modeles=(RIDGE,))
    coffre = aides.coffre_par_pas(table, spec)
    res = train.entrainer(spec, table, coffre)
    assert res.n_sans_cible == 5 and len(res.y_vrai) == coffre.n_apprentissage - 5
    assert not np.isnan(res.y_vrai).any() and len(res.positions) == len(res.y_vrai)
    assert not {1, 2, 3, 4, 6} & set(res.positions.tolist())


def test_colonne_absente_et_variables_absentes() -> None:
    df = aides.jeu_regression()
    table = Table(df.drop(columns=["d"]), "incomplet")
    spec = aides.spec_regression(modeles=(RIDGE,))
    with pytest.raises(ErreurEntrainement, match="Colonnes absentes"):
        train.entrainer(spec, table, aides.coffre_par_pas(table, spec))


# ============================================================
# RECHERCHE D'HYPERPARAMÈTRES
# ============================================================
def test_recherche_sur_grille() -> None:
    _, table, spec, coffre = _regression(modeles=())
    recherche = train.rechercher(spec, RIDGE, table, coffre)
    grille = entree("regression.ridge").grille()["alpha"]
    assert len(recherche.essais) == len(grille) and recherche.mode == "grille"
    assert {e.choix.parametres_dict()["alpha"] for e in recherche.essais} == set(grille)
    classement = recherche.classement()
    scores = [e.moyenne("rmse") for e in classement]
    assert scores == sorted(scores) and recherche.meilleur == classement[0].choix
    assert recherche.meilleur.parametres_dict()["alpha"] in grille


def test_recherche_aleatoire_et_conservation_des_autres_parametres() -> None:
    _, table, spec, coffre = _regression(modeles=())
    choix = CT("regression.foret", n_estimators=10)
    recherche = train.rechercher(spec, choix, table, coffre, mode="aleatoire", n_iter=3)
    assert len(recherche.essais) == 3
    # la grille porte aussi n_estimators ; max_depth vient de la grille, rien d'autre n'est perdu
    assert all("max_depth" in e.choix.parametres_dict() for e in recherche.essais)
    deux_fois = train.rechercher(spec, choix, table, coffre, mode="aleatoire", n_iter=3)
    assert [e.choix for e in recherche.essais] == [e.choix for e in deux_fois.essais]  # tirage reproductible


@pytest.mark.parametrize(
    ("choix", "options", "erreur", "motif"),
    [
        (CT("regression.lineaire"), {}, ErreurEntrainement, "aucun hyperparamètre"),
        (RIDGE, {"mode": "autre"}, ErreurEntrainement, "Mode de recherche inconnu"),
        (RIDGE, {"mode": "aleatoire", "n_iter": 51}, ErreurBorne, "de 1 à 50"),
        (RIDGE, {"mode": "aleatoire", "n_iter": 0}, ErreurBorne, "de 1 à 50"),
    ],
)
def test_recherche_refus(choix, options, erreur, motif) -> None:
    _, table, spec, coffre = _regression(modeles=())
    with pytest.raises(erreur, match=motif):
        train.rechercher(spec, choix, table, coffre, **options)


def test_recherche_sans_cible_refusee() -> None:
    table = Table(aides.jeu_sans_cible(), "amas")
    with pytest.raises(ErreurEntrainement, match="avec une cible"):
        train.rechercher(aides.spec_sans_cible(), CT("sans_cible.kmeans"), table)


def test_recherche_annulable_et_bornee_en_temps() -> None:
    _, table, spec, coffre = _regression(modeles=())
    annulation = Annulation()
    annulation.demander()
    with pytest.raises(ErreurAnnulation):
        train.rechercher(spec, RIDGE, table, coffre, annulation=annulation)
    with pytest.raises(ErreurBudget):
        train.rechercher(spec, RIDGE, table, coffre, budget_s=0.0)


# ============================================================
# MODÈLE FINAL ET ÉVALUATION SUR DES LIGNES
# ============================================================
def test_modele_final_et_evaluation_sur_le_test_ouvert() -> None:
    df, table, spec, coffre = _regression()
    res = train.entrainer(spec, table, coffre)
    final = train.ajuster_final(res.spec, RIDGE, table, coffre)
    assert final.classes is None and final.pipeline.predict(
        df[list(aides.colonnes_modele())].head(3)
    ).shape == (3,)
    test = coffre.ouvrir(Decision("regression.ridge", "rmse", res.valeur(res.modele("regression.ridge"))))
    scores = train.evaluer_lignes(final, res.spec, test)
    attendu = math.sqrt(
        np.mean((test["y"].to_numpy() - final.pipeline.predict(test[list(aides.colonnes_modele())])) ** 2)
    )
    assert scores["rmse"] == pytest.approx(attendu) and scores["r2"] > 0.9
    # l'erreur sur le test est du même ordre que l'erreur de validation croisée
    assert scores["rmse"] < 3 * res.modele("regression.ridge").moyenne("rmse")


def test_evaluer_lignes_refuse_sans_cible_et_ignore_les_cibles_manquantes() -> None:
    df, table, spec, coffre = _regression()
    final = train.ajuster_final(spec, RIDGE, table, coffre)
    test = df.iloc[::5].copy()
    test.iloc[0, test.columns.get_loc("y")] = np.nan
    assert not math.isnan(train.evaluer_lignes(final, spec, test)["rmse"])
    with pytest.raises(ErreurEntrainement, match="suppose une cible"):
        train.evaluer_lignes(final, aides.spec_sans_cible(), test)


def test_modele_final_de_classification_garde_les_classes() -> None:
    df, table, spec, coffre = _classification()
    final = train.ajuster_final(spec, CT("classification.logistique"), table, coffre)
    assert final.classes == ("conforme", "defaut") and final.classe_positive == "defaut"
    scores = train.evaluer_lignes(final, spec, df.iloc[::5])
    assert scores["roc_auc"] > 0.9 and scores["rappel"] > 0.5


# ============================================================
# BORNES (I9)
# ============================================================
def test_techniques_lourdes_refusees_au_dela_de_vingt_mille_lignes() -> None:
    spec = aides.spec_classification(modeles=(CT("classification.svc"),))
    train.verifier_bornes(spec, train.LIGNES_MAX_LOURD)  # la borne elle-même est permise
    with pytest.raises(ErreurBorne, match="coupée au delà de 20 000 lignes"):
        train.verifier_bornes(spec, train.LIGNES_MAX_LOURD + 1)
    for id_technique in ("regression.svr", "sans_cible.svm_une_classe"):
        lourd = aides.spec_regression(modeles=(CT(id_technique),))
        with pytest.raises(ErreurBorne, match="plus légère"):
            train.verifier_bornes(lourd, 20_001)
    with pytest.raises(ErreurBorne):
        train.verifier_bornes(aides.spec_regression(validation="validation.exclusion_unique"), 20_001)
    leger = aides.spec_regression(modeles=(CT("regression.foret"),))
    train.verifier_bornes(leger, 200_000)


def test_entrainer_refuse_un_svc_sur_trop_de_lignes_avant_tout_calcul() -> None:
    rng = np.random.default_rng(0)
    n = 26_000
    df = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": "x", "y": rng.choice(["p", "q"], n)}
    )
    table = Table(df, "grand")
    spec = aides.spec_classification(modeles=(CT("classification.svc"),), classe_positive="q")
    coffre = CoffreTest.par_positions(table, np.arange(0, n, 5), decoupage=spec.decoupage())
    assert coffre.n_apprentissage > train.LIGNES_MAX_LOURD
    with pytest.raises(ErreurBorne, match=r"classification\.svc est coupée"):
        train.entrainer(spec, table, coffre)


def test_le_nombre_de_plis_trop_grand_pour_la_classe_rare() -> None:
    _, table, spec, coffre = _classification(n_plis=10)
    res = train.entrainer(spec, table, coffre)  # 30 % de rares : assez pour 10 plis
    assert len(res.plis) == 10


# ============================================================
# CATALOGUE FILTRÉ, PROPOSITIONS, ASSEMBLAGE
# ============================================================
def test_catalogue_filtre_et_comparaison_proposee() -> None:
    for probleme in Probleme:
        dispo = {e.id for e in train.modeles_disponibles(probleme)}
        proposes = train.proposer_modeles(probleme)
        assert {c.id for c in proposes} <= dispo
        assert all(
            e.famille in ("reference", "regression", "classification", "sans_cible")
            for e in train.modeles_disponibles(probleme)
        )
        assert (proposes[0].famille == "reference") == probleme.avec_cible
    assert "regression.ridge" in {e.id for e in train.modeles_disponibles(Probleme.REGRESSION)}
    assert "classification.svc" not in {e.id for e in train.modeles_disponibles(Probleme.REGRESSION)}


def test_construire_pipeline_assemble_pretraitement_et_modele() -> None:
    spec = aides.spec_regression()
    pipeline = train.construire_pipeline(spec, RIDGE)
    assert [nom for nom, _ in pipeline.steps] == ["pretraitement", "modele"]
    assert type(pipeline[-1]).__name__ == "Ridge"
    avec_selection = aides.spec_regression(
        pretraitement=(*aides.PRETRAITEMENT_STANDARD, CT("selection.variance"))
    )
    assert [nom for nom, _ in train.construire_pipeline(avec_selection, RIDGE).steps][-2:] == [
        "selection_0_variance",
        "modele",
    ]


def test_avec_reference_n_ajoute_qu_en_cas_de_besoin() -> None:
    spec = aides.spec_regression(modeles=(RIDGE,))
    avec = train.avec_reference(spec)
    assert [c.id for c in avec.modeles] == ["reference.dummy_regression", "regression.ridge"]
    assert train.avec_reference(avec) is avec
    assert (
        train.avec_reference(aides.spec_sans_cible(modeles=(CT("sans_cible.kmeans"),))).modeles[0].famille
        == "sans_cible"
    )
    classification = train.avec_reference(aides.spec_classification(modeles=(CT("classification.arbre"),)))
    assert classification.modeles[0].id == "reference.dummy_classification"


def test_prechauffage() -> None:
    train.prechauffer()


# ============================================================
# AVEC LA SESSION : DE S0 À S8 AVEC UN VRAI RÉSULTAT
# ============================================================
def test_parcours_complet_de_la_session_avec_un_vrai_entrainement() -> None:
    _, table, spec, coffre = _regression()
    s = Session().appliquer(Charger(table)).appliquer(Auditer())
    s = s.appliquer(FixerObjectif(spec)).appliquer(Decouper(coffre)).appliquer(Pretraiter(spec))
    res = train.entrainer(spec, table, coffre)
    s = s.appliquer(Entrainer(res, spec=res.spec))
    assert s.etat.name == "MODELES" and s.resultats is res
    assert s.spec.modeles[0].famille == "reference"  # I6 : la comparaison de la session contient la référence
    meilleur = res.classement()[0]
    final = train.ajuster_final(s.spec, meilleur.choix, table, s.coffre)
    s = s.appliquer(Decider(Decision(meilleur.id, "rmse", res.valeur(meilleur))))
    s = s.appliquer(Tester(lambda lignes: train.evaluer_lignes(final, s.spec, lignes)))
    assert s.etat.name == "TESTE" and not s.score_test_optimiste
    assert s.resultat_test["rmse"] < 3 * res.valeur(meilleur)
    assert s.coffre.ouvertures == 1


def test_la_session_refuse_une_decision_hors_de_la_comparaison() -> None:
    _, table, spec, coffre = _regression()
    s = Session().appliquer(Charger(table)).appliquer(Auditer())
    s = s.appliquer(FixerObjectif(spec)).appliquer(Decouper(coffre)).appliquer(Pretraiter(spec))
    res = train.entrainer(spec, table, coffre)
    s = s.appliquer(Entrainer(res, spec=res.spec))
    with pytest.raises(Exception, match="pas parmi les modèles comparés"):
        s.appliquer(Decider(Decision("regression.lasso", "rmse", 1.0)))
