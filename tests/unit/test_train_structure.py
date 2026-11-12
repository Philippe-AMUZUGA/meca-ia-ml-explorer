"""Entraînement et structure : validation, classe positive, lignes sans cible, cible passée."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core import cible_passee, train
from mecaml.core.preprocess import colonnes_d_entree
from mecaml.core.train import ErreurEntrainement
from mecaml.core.types import (
    ChoixTechnique,
    CoffreTest,
    Decision,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
)

CT = ChoixTechnique.creer
RIDGE = CT("regression.ridge")
LOGISTIQUE = CT("classification.logistique")


# ============================================================
# VALIDATION : CELLE DE LA SPÉCIFICATION, JAMAIS CONTOURNÉE
# ============================================================
def test_aucun_argument_ne_permet_de_changer_la_validation() -> None:
    for fonction in (train.entrainer, train.rechercher, train.plis_validation, train.verifier_bornes):
        assert "validation" not in inspect.signature(fonction).parameters, fonction.__name__


@pytest.mark.parametrize(
    ("changements", "motif"),
    [
        ({"validation": "validation.kfold"}, "optimiste"),  # KFold sur des machines
        ({"validation": "validation.exclusion_unique"}, "optimiste"),  # exclusion d'une ligne à la fois
        # KFold sur une série chronologique
        ({"validation": "validation.kfold", "structure": Structure.CHRONOLOGIQUE}, "optimiste"),
        ({"validation": "validation.serie_temporelle", "structure": Structure.GROUPEE}, "choisir parmi"),
    ],
)
def test_une_validation_incompatible_avec_la_structure_est_refusee_des_la_specification(
    changements: dict, motif: str
) -> None:
    with pytest.raises(ValueError, match=motif):
        aides.spec_groupe(**changements)


def test_la_validation_imposee_gouverne_les_plis_et_se_rejoue_depuis_le_dictionnaire() -> None:
    df = aides.jeu_regression(n=60)
    table = Table(df, "petit")
    spec = aides.spec_regression(modeles=(RIDGE,), validation="validation.exclusion_unique")
    assert type(spec).depuis_dict(spec.vers_dict()).validation == "validation.exclusion_unique"
    coffre = aides.coffre_par_pas(table, spec)
    res = train.entrainer(spec, table, coffre)
    assert res.validation == "validation.exclusion_unique"
    assert len(res.plis) == coffre.n_apprentissage and all(len(v) == 1 for _, v in res.plis)
    assert res.modele("regression.ridge").moyenne("rmse") < res.reference().moyenne("rmse")


def _jeu_pannes(machines_en_panne: int = 12, n_machines: int = 12, longueur: int = 10) -> pd.DataFrame:
    """Machines suivies sur des cycles ; la panne est l'état des derniers cycles des premières machines."""
    df = aides.jeu_groupe(n_groupes=n_machines, longueur=longueur)
    en_panne = df["machine"].isin([f"M{k:02d}" for k in range(machines_en_panne)]) & (df["y"] <= 3)
    df["y"] = np.where(en_panne, "panne", "marche")
    return df


def _spec_pannes(**changements: object):
    roles = (
        ("machine", Role.GROUPE),
        ("cycle", Role.DATE),
        ("x1", Role.NUMERIQUE),
        ("x2", Role.NUMERIQUE),
        ("y", Role.CIBLE),
    )
    base = {
        "probleme": Probleme.CLASSIFICATION,
        "cible": "y",
        "roles": roles,
        "metrique": "f1",
        "classe_positive": "panne",
        "structure": Structure.GROUPEE_CHRONOLOGIQUE,
        "n_plis": 3,
        "pretraitement": (CT("transformation.standard"),),
        "modeles": (LOGISTIQUE,),
    }
    return SpecProjet(**{**base, **changements})


def test_classification_groupee_plis_stratifies_par_defaut_unites_disjointes() -> None:
    df = _jeu_pannes()
    table, spec = Table(df, "pannes"), _spec_pannes()
    coffre = aides.coffre_par_groupes(table, spec, ("M10", "M11"))
    res = train.entrainer(spec, table, coffre)
    assert res.validation == "validation.plis_groupes_stratifies"
    machines = coffre.apprentissage()["machine"].to_numpy()
    for app, val in res.plis:
        assert not set(machines[app]) & set(machines[val])  # aucune machine à cheval
        assert set(res.y_vrai[val]) == {"panne", "marche"}  # chaque pli valide les deux classes
    assert res.modele("classification.logistique").plis_valides("f1") == 3


def test_validation_imposee_groupes_aleatoires_pour_une_classification_groupee() -> None:
    df = _jeu_pannes()
    table = Table(df, "pannes")
    spec = _spec_pannes(validation="validation.groupes_aleatoires")
    res = train.entrainer(spec, table, aides.coffre_par_groupes(table, spec, ("M10", "M11")))
    assert res.validation == "validation.groupes_aleatoires" and len(res.plis) == 3


def test_un_pli_sans_la_classe_positive_est_indefini_et_compte_comme_tel() -> None:
    """Pannes sur trois machines seulement : des plis sans ligne positive ont un F1 indéfini, non nul."""
    df = _jeu_pannes(machines_en_panne=3, n_machines=12)
    table = Table(df, "rares")
    spec = _spec_pannes(n_plis=4)
    res = train.entrainer(spec, table, aides.coffre_par_groupes(table, spec, ("M10", "M11")))
    logistique = res.modele("classification.logistique")
    sans_positif = [k for k, (_, v) in enumerate(res.plis) if not np.any(res.y_vrai[v] == "panne")]
    assert sans_positif, "le jeu est construit pour que des plis n'aient aucune panne"
    assert logistique.plis_valides("f1") == len(res.plis) - len(sans_positif)
    for k in sans_positif:
        assert math.isnan(logistique.scores_plis["f1"][k]) and math.isnan(logistique.scores_plis["rappel"][k])
    assert any("Aucune ligne de la classe positive" in a for a in logistique.avertissements)
    ligne = res.tableau().set_index("modele").loc["classification.logistique"]
    assert ligne["plis_valides"] == logistique.plis_valides("f1") < len(res.plis)
    assert not math.isnan(logistique.moyenne("f1"))  # la moyenne porte sur les plis définis


# ============================================================
# CLASSE POSITIVE : FIXÉE PAR LA SPÉCIFICATION, PROPOSITION TRACÉE
# ============================================================
def _binaire(**changements: object):
    df = aides.jeu_classification()
    table = Table(df, "classification")
    spec = aides.spec_classification(modeles=(LOGISTIQUE,), **changements)
    return table, spec, aides.coffre_par_pas(table, spec)


def test_une_metrique_qui_depend_de_la_classe_positive_exige_qu_elle_soit_fixee() -> None:
    table, spec, coffre = _binaire(classe_positive=None)
    with pytest.raises(ErreurEntrainement, match=r"dépend de la classe positive.*proposition : 'defaut'"):
        train.entrainer(spec, table, coffre)


def test_une_metrique_independante_laisse_une_proposition_dont_l_origine_est_dite() -> None:
    table, spec, coffre = _binaire(classe_positive=None, metrique="exactitude")
    res = train.entrainer(spec, table, coffre)
    assert res.classe_positive == "defaut" and res.classe_positive_source == "lexique"


def test_la_classe_positive_de_la_specification_l_emporte_et_change_les_mesures() -> None:
    table, spec_defaut, coffre = _binaire()
    _, spec_saine, _ = _binaire(classe_positive="conforme")
    sur_defaut = train.entrainer(spec_defaut, table, coffre)
    sur_saine = train.entrainer(spec_saine, table, coffre)
    assert sur_defaut.classe_positive == "defaut" and sur_defaut.classe_positive_source == "specification"
    assert sur_saine.classe_positive == "conforme" and sur_saine.classe_positive_source == "specification"
    a, b = sur_defaut.modele("classification.logistique"), sur_saine.modele("classification.logistique")
    assert a.moyenne("rappel") != pytest.approx(b.moyenne("rappel"))
    assert a.moyenne("exactitude") == pytest.approx(b.moyenne("exactitude"))  # indépendante du choix


def test_classe_positive_hors_de_la_cible_refusee() -> None:
    table, spec, coffre = _binaire(classe_positive="inconnue")
    with pytest.raises(ErreurEntrainement, match="n'est pas une classe de la cible"):
        train.entrainer(spec, table, coffre)


def test_etiquettes_zero_un_et_multiclasse() -> None:
    df = aides.jeu_classification()
    df["y"] = (df["y"] == "defaut").astype(int)
    table = Table(df, "zero_un")
    spec = aides.spec_classification(modeles=(LOGISTIQUE,), classe_positive=None, metrique="exactitude")
    res = train.entrainer(spec, table, aides.coffre_par_pas(table, spec))
    assert res.classe_positive == 1 and res.classe_positive_source == "zero_un"
    triple = Table(aides.jeu_classification(classes=3), "trois")
    spec3 = aides.spec_classification(modeles=(LOGISTIQUE,), binaire=False)
    res3 = train.entrainer(spec3, triple, aides.coffre_par_pas(triple, spec3))
    assert res3.classe_positive is None and res3.classe_positive_source is None


def test_une_classe_positive_declaree_au_dela_de_deux_classes_est_refusee_et_non_ignoree() -> None:
    triple = Table(aides.jeu_classification(classes=3), "trois")
    spec = aides.spec_classification(modeles=(LOGISTIQUE,), binaire=False, classe_positive="classe_0")
    with pytest.raises(
        ErreurEntrainement, match=r"3 classes.*moyennes macro.*regrouper les classes"
    ) as refus:
        train.entrainer(spec, triple, aides.coffre_par_pas(triple, spec))
    assert "retirer la classe positive" in str(refus.value)


# ============================================================
# LIGNES SANS CIBLE : LEUR TEMPS COMPLÈTE LES SÉRIES
# ============================================================
def _serie_avec_trous():
    df = aides.jeu_groupe(n_groupes=6, longueur=14)
    df.loc[(df["machine"] == "M01") & (df["cycle"] == 5), "y"] = np.nan
    df.loc[(df["machine"] == "M05") & (df["cycle"].isin([2, 3])), "y"] = np.nan  # M05 est au test
    table = Table(df, "trous")
    temporelles = CT("chronologique.variables_temporelles", retards=(1,))
    spec = aides.spec_groupe(modeles=(RIDGE,), pretraitement=(temporelles,))
    return df, table, spec, aides.coffre_par_groupes(table, spec, ("M04", "M05"))


def test_le_retard_d_une_ligne_est_la_valeur_de_la_ligne_precedente_notee_ou_non() -> None:
    df, table, spec, coffre = _serie_avec_trous()
    res = train.entrainer(spec, table, coffre)
    assert res.n_sans_cible == 1  # M01 au cycle 5 ; les trous de M05 sont au test, hors de l'apprentissage
    modele = train.ajuster_final(spec, RIDGE, table, coffre)
    colonnes = list(colonnes_d_entree(spec))
    lignes = coffre.apprentissage()
    lignes = lignes[lignes["y"].notna()]
    transformateur = modele.pipeline[0]
    sortie = pd.DataFrame(
        transformateur.transform(lignes[colonnes]),
        columns=transformateur.get_feature_names_out(),
        index=lignes.index,
    )
    m01 = lignes["machine"] == "M01"
    apres = m01 & (lignes["cycle"] == 6)
    avant = df[(df["machine"] == "M01") & (df["cycle"] == 5)]["x1"].iloc[0]
    assert sortie.loc[apres, "x1_retard_1"].iloc[0] == pytest.approx(avant)  # et non la valeur du cycle 4
    assert modele.pipeline[0].named_transformers_["chronologique_0"]["variables"].contexte is not None


def test_les_lignes_de_test_sans_cible_sont_predites_puis_comptees() -> None:
    _, table, spec, coffre = _serie_avec_trous()
    modele = train.ajuster_final(spec, RIDGE, table, coffre)
    test = coffre.ouvrir(Decision("regression.ridge", "rmse", 1.0))
    assert test["y"].isna().sum() == 2
    evaluation = train.evaluer_lignes(modele, spec, test)
    assert evaluation.n_sans_cible == 2 and evaluation.n_lignes == len(test) - 2
    assert evaluation["rmse"] > 0 and isinstance(evaluation, dict)
    # les lignes sans cible servent de contexte au retard de la suivante : les retirer change le score
    sans_trous = test[test["y"].notna()]
    assert train.evaluer_lignes(modele, spec, sans_trous)["rmse"] != pytest.approx(evaluation["rmse"])


# ============================================================
# CIBLE PASSÉE : DÉCLARÉE CONNUE À LA PRÉVISION, JAMAIS AJOUTÉE D'OFFICE, REFUSÉE SI ELLE DÉCOMPTE
# ============================================================
PERSISTANCE = CT("reference.persistance", cible_connue=True)
CIBLE_DECALEE = CT("chronologique.cible_decalee", cible_connue=True)
EXEMPLES = Path(__file__).resolve().parents[2] / "examples"


def _serie_unique(longueur: int = 80):
    """Une machine dont la température est relevée à chaque cycle : la cible du cycle précédent est connue."""
    df = aides.jeu_serie_mesuree(n_groupes=1, longueur=longueur)
    roles = (("machine", Role.IGNOREE), *aides.ROLES_GROUPE[1:])
    return df, roles


def _coffre_fin(table: Table, spec, n_test: int):
    n = table.n_lignes
    return CoffreTest.par_positions(table, np.arange(n - n_test, n), decoupage=spec.decoupage())


def _parc_mesure(**changements):
    """Huit machines, température relevée à chaque cycle ; M06 et M07 sont au test."""
    table = Table(aides.jeu_serie_mesuree(), "temperatures")
    spec = aides.spec_groupe(**changements)
    return table, spec, aides.coffre_par_groupes(table, spec, ("M06", "M07"))


def test_aucune_reference_de_la_cible_passee_n_est_ajoutee_d_office() -> None:
    _, roles = _serie_unique()
    chronologique = aides.spec_groupe(
        modeles=(RIDGE,), roles=roles, structure=Structure.CHRONOLOGIQUE, pretraitement=()
    )
    attendu = ["reference.dummy_regression", "regression.ridge"]
    assert [c.id for c in train.avec_reference(chronologique).modeles] == attendu
    assert [c.id for c in train.avec_reference(aides.spec_groupe(modeles=(RIDGE,))).modeles] == attendu
    proposes = [c.id for c in train.proposer_modeles(Probleme.REGRESSION, Structure.CHRONOLOGIQUE)]
    assert proposes == [c.id for c in train.proposer_modeles(Probleme.REGRESSION)]
    assert proposes[0] == "reference.dummy_regression" and "reference.persistance" not in proposes
    table, spec, coffre = _parc_mesure(modeles=(RIDGE,))
    assert [m.id for m in train.entrainer(spec, table, coffre).modeles] == attendu


def test_la_persistance_declaree_sur_une_cible_connue_bat_la_moyenne_sans_la_remplacer() -> None:
    table, spec, coffre = _parc_mesure(modeles=(PERSISTANCE, RIDGE))
    assert cible_passee.decompte(coffre.apprentissage(), spec) is None  # la température ne décompte pas
    res = train.entrainer(spec, table, coffre)
    assert [m.id for m in res.modeles] == [
        "reference.dummy_regression",
        "reference.persistance",
        "regression.ridge",
    ]
    assert res.reference().id == "reference.dummy_regression"  # les gains se lisent contre la moyenne
    persistance, moyenne = res.modele("reference.persistance"), res.modele("reference.dummy_regression")
    assert persistance.ok and persistance.moyenne("rmse") < 0.6 * moyenne.moyenne("rmse")


def test_la_declaration_est_dans_la_specification_et_survit_a_son_aller_retour() -> None:
    spec = aides.spec_groupe(modeles=(PERSISTANCE,), pretraitement=(CIBLE_DECALEE,))
    relue = SpecProjet.depuis_dict(spec.vers_dict())
    assert relue == spec
    assert dict(relue.modeles[0].parametres) == {"cible_connue": True}
    assert dict(relue.pretraitement[0].parametres) == {"cible_connue": True}


@pytest.mark.parametrize("refus", [{}, {"cible_connue": False}], ids=["absente", "fausse"])
@pytest.mark.parametrize("choix", ["persistance", "cible_decalee"])
def test_sans_declaration_la_cible_passee_est_refusee_avant_tout_calcul(choix: str, refus: dict) -> None:
    if choix == "persistance":
        table, spec, coffre = _parc_mesure(modeles=(CT("reference.persistance", **refus), RIDGE))
    else:
        decalee = CT("chronologique.cible_decalee", **refus)
        table, spec, coffre = _parc_mesure(modeles=(RIDGE,), pretraitement=(decalee,))
    with pytest.raises(ErreurEntrainement, match=r"cible_connue=True") as erreur:
        train.entrainer(spec, table, coffre)
    assert "reference.persistance" in str(erreur.value) or "chronologique.cible_decalee" in str(erreur.value)
    with pytest.raises(ErreurEntrainement, match=r"cible_connue=True"):
        train.ajuster_final(spec, RIDGE, table, coffre)
    if choix == "cible_decalee":
        with pytest.raises(ErreurEntrainement, match=r"cible_connue=True"):
            train.rechercher(spec, RIDGE, table, coffre)


@pytest.mark.parametrize("choix", ["persistance", "cible_decalee"])
def test_une_cible_qui_decompte_est_refusee_meme_declaree(choix: str) -> None:
    table = Table(aides.jeu_groupe(n_groupes=10, longueur=30), "durees_de_vie")
    if choix == "persistance":
        spec = aides.spec_groupe(modeles=(PERSISTANCE, RIDGE))
    else:
        spec = aides.spec_groupe(modeles=(RIDGE,), pretraitement=(CIBLE_DECALEE,))
    coffre = aides.coffre_par_groupes(table, spec, ("M08", "M09"))
    with pytest.raises(ErreurEntrainement, match=r"panne") as erreur:
        train.entrainer(spec, table, coffre)
    message = str(erreur.value)
    assert (
        "compte à rebours" in message
        and "même déclaré" in message
        and "baisse de 1 à chaque ligne" in message
    )
    with pytest.raises(ErreurEntrainement, match=r"panne"):
        train.ajuster_final(spec, RIDGE, table, coffre)


def test_le_compte_a_rebours_se_mesure_sur_le_pas_median_de_chaque_unite() -> None:
    spec = aides.spec_groupe()
    descend = aides.jeu_groupe(n_groupes=6, longueur=20)
    pas, part = cible_passee.decompte(descend, spec)
    assert pas == pytest.approx(-1.0, abs=0.05) and part > 0.95
    assert cible_passee.decompte(aides.jeu_serie_mesuree(), spec) is None  # une température ne décompte pas
    monte = descend.assign(y=-descend["y"])
    assert cible_passee.decompte(monte, spec) is None  # un pas positif n'est pas un compte à rebours
    # moins de PAS_MIN pas : rien à juger
    court = aides.jeu_groupe(n_groupes=1, longueur=cible_passee.PAS_MIN)
    assert cible_passee.decompte(court, spec) is None
    # une seule série, sans colonne d'unité : même mesure
    roles = (("machine", Role.IGNOREE), *aides.ROLES_GROUPE[1:])
    seule = aides.spec_groupe(roles=roles, structure=Structure.CHRONOLOGIQUE)
    pas_seul, part_seule = cible_passee.decompte(aides.jeu_groupe(n_groupes=1, longueur=40), seule)
    assert pas_seul == pytest.approx(-1.0, abs=0.05) and part_seule > 0.95
    # le rang des lignes ne compte pas : les cycles se lisent dans l'ordre de la date de chaque machine
    melange = descend.sample(frac=1.0, random_state=3)
    assert cible_passee.decompte(melange, spec) == (pas, part)
    # un compte à rebours que le bruit défait n'en est plus un
    bruite = descend.assign(y=descend["y"] + np.random.default_rng(0).normal(0, 3.0, len(descend)))
    assert cible_passee.decompte(bruite, spec) is None


def _duree_de_vie_restante(**changements):
    """Jeu de durée de vie restante des exemples : 20 machines, la cible baisse d'un cycle à chaque cycle."""
    df = pd.read_excel(EXEMPLES / "MecaIA-TimeSeries_RUL_extrait.xlsx")
    cible = "Duree_Vie_Restante_Cycles"
    capteurs = [c for c in df.columns if c not in ("ID_Machine", "Cycle_Operation_Heures", cible)]
    roles = (
        ("ID_Machine", Role.GROUPE),
        ("Cycle_Operation_Heures", Role.DATE),
        *((c, Role.NUMERIQUE) for c in capteurs),
        (cible, Role.CIBLE),
    )
    base = {
        "probleme": Probleme.REGRESSION,
        "cible": cible,
        "roles": roles,
        "metrique": "rmse",
        "structure": Structure.GROUPEE_CHRONOLOGIQUE,
        "n_plis": 3,
        "graine": 42,
        "pretraitement": (CT("transformation.standard"),),
        "modeles": (RIDGE,),
    }
    spec = SpecProjet(**{**base, **changements})
    table = Table(df, "duree_de_vie_restante")
    positions = np.flatnonzero(df["ID_Machine"].isin([17, 18, 19, 20]).to_numpy())
    return table, spec, CoffreTest.par_positions(table, positions, decoupage=spec.decoupage())


def test_sur_la_duree_de_vie_restante_la_persistance_ne_prend_plus_la_tete_du_classement() -> None:
    table, spec, coffre = _duree_de_vie_restante(modeles=(RIDGE, CT("regression.arbre", max_depth=6)))
    res = train.entrainer(spec, table, coffre)
    assert [m.id for m in res.modeles] == [
        "reference.dummy_regression",
        "regression.ridge",
        "regression.arbre",
    ]
    assert res.classement()[0].id == "regression.ridge"  # et non une référence qui lit la cible passée
    assert 30 < res.modele("regression.ridge").moyenne("rmse") < 40  # le ridge sur les capteurs seuls


def test_sur_la_duree_de_vie_restante_la_cible_passee_est_refusee_meme_declaree() -> None:
    avec_persistance = _duree_de_vie_restante(modeles=(PERSISTANCE, RIDGE))
    avec_cible_decalee = _duree_de_vie_restante(modeles=(RIDGE,), pretraitement=(CIBLE_DECALEE,))
    for table, spec, coffre in (avec_persistance, avec_cible_decalee):
        assert cible_passee.decompte(coffre.apprentissage(), spec) == (-1.0, 1.0)  # 100 % des pas, de 1 cycle
        with pytest.raises(ErreurEntrainement, match=r"panne"):
            train.entrainer(spec, table, coffre)
    table, spec, coffre = _duree_de_vie_restante(modeles=(CT("reference.persistance"), RIDGE))
    with pytest.raises(ErreurEntrainement, match=r"cible_connue=True"):
        train.entrainer(spec, table, coffre)


def test_la_persistance_est_refusee_sans_structure_chronologique() -> None:
    table = Table(aides.jeu_regression(), "regression")
    spec = aides.spec_regression(modeles=(PERSISTANCE, RIDGE))
    with pytest.raises(ErreurEntrainement, match="persistance suppose des lignes ordonnées"):
        train.entrainer(spec, table, aides.coffre_par_pas(table, spec))


def test_la_persistance_lit_le_pas_precedent_meme_sans_cible_et_non_la_derniere_valeur_connue() -> None:
    df, roles = _serie_unique(30)
    df.loc[10, "y"] = np.nan
    spec = aides.spec_groupe(
        modeles=(RIDGE, PERSISTANCE), roles=roles, structure=Structure.CHRONOLOGIQUE, pretraitement=()
    )
    colonnes = list(colonnes_d_entree(spec))
    connues = df[df["y"].notna()]
    avec = train.construire_pipeline(spec, PERSISTANCE, contexte=df[df["y"].isna()][colonnes])
    sans = train.construire_pipeline(spec, PERSISTANCE)
    predictions = {}
    for nom, pipeline in (("avec", avec), ("sans", sans)):
        pipeline.fit(connues[colonnes], connues["y"])
        predictions[nom] = pd.Series(pipeline.predict(connues[colonnes]), index=connues.index)
    # sans contexte, le pas précédent de la ligne 11 serait la valeur de la ligne 9, d'un autre pas de temps
    assert predictions["sans"][11] == pytest.approx(df.loc[9, "y"])
    assert predictions["avec"][11] != pytest.approx(df.loc[9, "y"])
    # ailleurs, les deux lectures concordent : le pas précédent est la ligne d'avant
    for pas in (5, 20):
        assert predictions["avec"][pas] == pytest.approx(df.loc[pas - 1, "y"])
        assert predictions["sans"][pas] == pytest.approx(df.loc[pas - 1, "y"])


def test_la_cible_courante_n_atteint_jamais_le_modele() -> None:
    """Changer la cible d'une ligne ne change pas sa prédiction, seulement celle de la ligne suivante."""
    df, roles = _serie_unique()
    table = Table(df, "serie")
    decale = CT("chronologique.cible_decalee", retards=(1, 2), cible_connue=True)
    spec = aides.spec_groupe(
        modeles=(RIDGE,),
        roles=roles,
        structure=Structure.CHRONOLOGIQUE,
        pretraitement=(decale, CT("transformation.standard")),
    )
    coffre = _coffre_fin(table, spec, 16)
    assert spec.cible in colonnes_d_entree(spec)  # le transformeur reçoit la cible pour en décaler le passé
    modele = train.ajuster_final(spec, RIDGE, table, coffre)
    test = coffre.ouvrir(Decision("regression.ridge", "rmse", 1.0)).reset_index(drop=True)
    normal = modele.pipeline.predict(test[list(colonnes_d_entree(spec))])
    modifie = test.copy()
    ligne = 8
    modifie.loc[ligne, "y"] += 1000.0
    nouveau = modele.pipeline.predict(modifie[list(colonnes_d_entree(spec))])
    assert nouveau[ligne] == pytest.approx(normal[ligne]), (
        "la valeur courante de la cible est entrée dans le modèle"
    )
    assert nouveau[ligne + 1] != pytest.approx(
        normal[ligne + 1]
    )  # le passé de la cible sert à la ligne suivante
    assert np.allclose(np.delete(nouveau, [ligne + 1, ligne + 2]), np.delete(normal, [ligne + 1, ligne + 2]))


def test_la_cible_decalee_declaree_aide_et_le_modele_ne_voit_que_des_retards() -> None:
    df, roles = _serie_unique()
    table = Table(df, "serie")
    seul = aides.spec_groupe(
        modeles=(RIDGE,),
        roles=roles,
        structure=Structure.CHRONOLOGIQUE,
        pretraitement=(CT("transformation.standard"),),
    )
    avec = seul.avec(pretraitement=(CIBLE_DECALEE, CT("transformation.standard")))
    coffre = _coffre_fin(table, seul, 16)
    sans_passe = train.entrainer(seul, table, coffre).modele("regression.ridge")
    avec_passe = train.entrainer(avec, table, coffre).modele("regression.ridge")
    assert avec_passe.moyenne("rmse") < sans_passe.moyenne("rmse")
    cible_decalee = train.ajuster_final(avec, RIDGE, table, coffre).pipeline[0]
    assert "y_retard_1" in cible_decalee.get_feature_names_out()
    assert "y" not in cible_decalee.get_feature_names_out()  # la valeur courante ne sort pas du prétraitement
