"""Lecture des modèles sans cible : composantes, regroupements, anomalies."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml.core import metrics, train, unsupervised
from mecaml.core.types import ChoixTechnique, Decision

CT = ChoixTechnique.creer
MODELES = (
    CT("sans_cible.kmeans", n_clusters=2),
    CT("sans_cible.dbscan", eps=0.6, min_samples=5),
    CT("sans_cible.dbscan", eps=50.0, min_samples=5),
    CT("sans_cible.acp", n_components=2),
    CT("sans_cible.foret_isolement", contamination=0.05),
    CT("sans_cible.facteur_local", contamination=0.05),
    CT("sans_cible.svm_une_classe", nu=0.05),
    CT("sans_cible.melange_gaussien", n_components=2),
)


@pytest.fixture(scope="module")
def projet(tmp_path_factory: pytest.TempPathFactory) -> aides.Projet:
    jeu = jeux.jeu_sans_cible(n=200)
    chemin = aides.ecrire_csv(jeu, tmp_path_factory.mktemp("sans_cible") / "jeu.csv")
    # Deux réglages de DBSCAN portent le même identifiant : on ne garde que le premier de chaque technique.
    modeles = tuple({c.id: c for c in reversed(MODELES)}.values())[::-1]
    return aides.mener_le_projet(chemin, lambda t: jeux.spec_sans_cible(modeles=modeles))


def _resultat_a_un_groupe(tmp_path: Path) -> tuple[aides.Projet, str]:
    """DBSCAN à très grand rayon : un seul groupe, donc pas de silhouette ni de modèle à retenir."""
    chemin = aides.ecrire_csv(jeux.jeu_sans_cible(n=120), tmp_path / "jeu.csv")
    modeles = (CT("sans_cible.dbscan", eps=50.0, min_samples=5),)

    def sans_decision(_resultat):
        return Decision("sans_cible.dbscan", "silhouette", 0.0)

    projet = aides.mener_le_projet(
        chemin, lambda t: jeux.spec_sans_cible(modeles=modeles), decider=sans_decision
    )
    return projet, "sans_cible.dbscan"


# ============================================================
# ANALYSE EN COMPOSANTES PRINCIPALES
# ============================================================
def test_les_composantes_sont_celles_du_modele_ajuste(projet: aides.Projet) -> None:
    analyse = unsupervised.analyser_composantes(projet.resultat, "sans_cible.acp", projet.table)
    final = projet.resultat.modele("sans_cible.acp").pipeline[-1]
    assert np.array_equal(analyse.variance, final.explained_variance_ratio_)
    assert analyse.cumulee[-1] == pytest.approx(analyse.variance.sum()) and 0 < analyse.cumulee[-1] <= 1
    assert analyse.coordonnees.shape == (projet.table.n_lignes, 2)
    assert list(analyse.poids.columns) == ["CP1", "CP2"]
    assert np.allclose(analyse.poids.to_numpy().T, final.components_)
    assert set(analyse.poids.index) == {"u", "v", "bruit"}  # les variables que le modèle reçoit


def test_les_composantes_se_lisent_sur_une_analyse_en_composantes_seulement(projet: aides.Projet) -> None:
    with pytest.raises(unsupervised.ErreurAnalyse, match="pas une analyse en composantes principales"):
        unsupervised.analyser_composantes(projet.resultat, "sans_cible.kmeans", projet.table)


# ============================================================
# REGROUPEMENT
# ============================================================
def test_le_regroupement_decrit_ses_groupes(projet: aides.Projet) -> None:
    analyse = unsupervised.analyser_regroupement(projet.resultat, "sans_cible.kmeans", projet.table)
    assert analyse.n_groupes == 2 and analyse.effectifs.sum() == projet.table.n_lignes
    assert analyse.effectifs.name == "effectif"
    manuel = projet.table.donnees[list(projet.spec.variables())].groupby(analyse.etiquettes).mean()
    pd.testing.assert_frame_equal(analyse.profils, manuel)
    assert set(analyse.silhouette_par_groupe.index) == {0, 1}
    assert (
        (analyse.silhouette_par_groupe > 0.5) & (analyse.silhouette_par_groupe <= 1)
    ).all()  # deux amas nets
    assert analyse.plan.shape == (projet.table.n_lignes, 2)


def test_le_bruit_de_dbscan_n_est_pas_un_groupe(projet: aides.Projet) -> None:
    analyse = unsupervised.analyser_regroupement(projet.resultat, "sans_cible.dbscan", projet.table)
    assert -1 in analyse.effectifs.index or analyse.n_groupes == len(analyse.effectifs)
    assert analyse.n_groupes == int((analyse.effectifs.index != -1).sum())
    assert -1 not in analyse.silhouette_par_groupe.index


def test_un_seul_groupe_n_a_pas_de_silhouette(tmp_path: Path) -> None:
    projet_unique, nom = _resultat_a_un_groupe(tmp_path)
    analyse = unsupervised.analyser_regroupement(projet_unique.resultat, nom, projet_unique.table)
    assert analyse.n_groupes == 1 and analyse.silhouette_par_groupe.empty


def test_un_modele_sans_etiquettes_n_a_pas_de_regroupement(projet: aides.Projet) -> None:
    with pytest.raises(unsupervised.ErreurAnalyse, match="pas d'étiquettes de groupe"):
        unsupervised.analyser_regroupement(projet.resultat, "sans_cible.acp", projet.table)


# ============================================================
# ANOMALIES
# ============================================================
@pytest.mark.parametrize(
    "modele", ["sans_cible.foret_isolement", "sans_cible.facteur_local", "sans_cible.svm_une_classe"]
)
def test_les_anomalies_sont_les_lignes_au_plus_grand_score(projet: aides.Projet, modele: str) -> None:
    analyse = unsupervised.analyser_anomalies(projet.resultat, modele, projet.table)
    assert analyse.scores.shape == (projet.table.n_lignes,)
    assert analyse.taux == pytest.approx(projet.resultat.modele(modele).moyenne("taux_anomalies"))
    anormales = analyse.etiquettes == -1
    assert anormales.any() and analyse.scores[anormales].mean() > analyse.scores[~anormales].mean()
    tete = analyse.plus_anormales(projet.table.donnees[["u", "v", "bruit"]], nombre=5)
    assert list(tete.columns)[:2] == ["position", "score_anomalie"] and len(tete) == 5
    assert tete["score_anomalie"].is_monotonic_decreasing
    assert tete["score_anomalie"].iloc[0] == analyse.scores.max()
    assert projet.table.donnees["u"].iloc[tete["position"].iloc[0]] == tete["u"].iloc[0]


def test_un_modele_qui_n_est_pas_un_detecteur_n_a_pas_de_score_d_anomalie(projet: aides.Projet) -> None:
    with pytest.raises(unsupervised.ErreurAnalyse, match="pas un détecteur"):
        unsupervised.analyser_anomalies(projet.resultat, "sans_cible.kmeans", projet.table)


# ============================================================
# AIGUILLAGE ET REFUS
# ============================================================
def test_la_lecture_est_choisie_d_apres_le_modele(projet: aides.Projet) -> None:
    assert isinstance(
        unsupervised.analyser(projet.resultat, "sans_cible.acp", projet.table),
        unsupervised.AnalyseComposantes,
    )
    assert isinstance(
        unsupervised.analyser(projet.resultat, "sans_cible.kmeans", projet.table),
        unsupervised.AnalyseRegroupement,
    )
    assert isinstance(
        unsupervised.analyser(projet.resultat, "sans_cible.facteur_local", projet.table),
        unsupervised.AnalyseAnomalies,
    )
    melange = unsupervised.analyser(projet.resultat, "sans_cible.melange_gaussien", projet.table)
    assert isinstance(melange, unsupervised.AnalyseRegroupement) and melange.n_groupes == 2
    with pytest.raises(unsupervised.ErreurAnalyse, match="absent de la comparaison"):
        unsupervised.analyser(projet.resultat, "sans_cible.classification_hierarchique", projet.table)
    inconnu = train.ResultatModele(CT("sans_cible.autre"), {}, 0.0, pipeline=object())
    resultat = dataclasses.replace(projet.resultat, modeles=(*projet.resultat.modeles, inconnu))
    with pytest.raises(unsupervised.ErreurAnalyse, match="pas de lecture sans cible connue"):
        unsupervised.analyser(resultat, "sans_cible.autre", projet.table)


def test_une_comparaison_avec_cible_n_a_pas_de_lecture_sans_cible(tmp_path: Path) -> None:
    avec_cible = aides.mener_le_projet(
        aides.ecrire_csv(jeux.jeu_regression(n=80), tmp_path / "jeu.csv"),
        lambda t: jeux.spec_regression(modeles=(CT("regression.ridge"),)),
    )
    with pytest.raises(unsupervised.ErreurAnalyse, match="projet sans cible"):
        unsupervised.analyser(avec_cible.resultat, "regression.ridge", avec_cible.table)


def test_le_plan_de_lecture_d_une_seule_variable_a_une_ordonnee_nulle() -> None:
    plan = unsupervised._plan(np.arange(10, dtype=float).reshape(-1, 1), graine=42)
    assert plan.shape == (10, 2) and (plan[:, 1] == 0).all()


# ============================================================
# FIGURES
# ============================================================
def test_la_figure_des_composantes(projet: aides.Projet) -> None:
    analyse = unsupervised.analyser_composantes(projet.resultat, "sans_cible.acp", projet.table)
    fig = unsupervised.figure_composantes(analyse)
    axe = fig.axes[0]
    assert len(axe.patches) == 2 and len(axe.lines) == 1 and "variance expliquée" in axe.get_ylabel()


def test_les_figures_du_regroupement(projet: aides.Projet) -> None:
    analyse = unsupervised.analyser_regroupement(projet.resultat, "sans_cible.kmeans", projet.table)
    plan = unsupervised.figure_plan(analyse.plan, analyse.etiquettes, "Plan")
    legende = [t.get_text() for t in plan.axes[0].get_legend().get_texts()]
    assert legende == ["groupe 0", "groupe 1"] and "première composante" in plan.axes[0].get_xlabel()
    profils = unsupervised.figure_profils(analyse)
    assert profils.axes[0].get_title() == "Profil des groupes"
    assert len(profils.axes[0].texts) == analyse.profils.size


def test_la_figure_du_plan_distingue_le_bruit_et_les_anomalies(projet: aides.Projet) -> None:
    densite = unsupervised.analyser_regroupement(projet.resultat, "sans_cible.dbscan", projet.table)
    plan = unsupervised.figure_plan(densite.plan, densite.etiquettes, "DBSCAN")
    textes = [t.get_text() for t in plan.axes[0].get_legend().get_texts()]
    assert ("bruit" in textes) == bool((densite.etiquettes == -1).any())
    analyse = unsupervised.analyser_anomalies(projet.resultat, "sans_cible.foret_isolement", projet.table)
    fig = unsupervised.figure_plan(analyse.plan, analyse.etiquettes, "Anomalies", anomalies=True)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["normales", "anormales"]
    histogramme = unsupervised.figure_anomalies(analyse)
    assert histogramme.axes[0].get_title() == "Forêt d'isolement" and len(histogramme.axes[0].patches) >= 40


def test_le_taux_d_anomalies_est_celui_de_la_metrique(projet: aides.Projet) -> None:
    analyse = unsupervised.analyser_anomalies(projet.resultat, "sans_cible.foret_isolement", projet.table)
    assert analyse.taux == pytest.approx(metrics.calculer("taux_anomalies", etiquettes=analyse.etiquettes))
