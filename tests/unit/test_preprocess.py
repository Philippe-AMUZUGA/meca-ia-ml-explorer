"""Prétraitement par rôle : branches, refus actionnables, proposition, variables temporelles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

import aides_apprentissage as aides
from mecaml.core.catalog import REGISTRE
from mecaml.core.preprocess import (
    STADES,
    ErreurPretraitement,
    colonnes_d_entree,
    controles_pretraitement,
    etapes_pretraitement,
    normaliser_manquants,
    proposer_pretraitement,
)
from mecaml.core.types import ChoixTechnique, Probleme, Role, Structure

CT = ChoixTechnique.creer


def _pipeline(spec) -> Pipeline:
    return Pipeline([*etapes_pretraitement(spec), ("modele", Ridge())])


def _entrees(spec, df: pd.DataFrame) -> pd.DataFrame:
    return df[list(colonnes_d_entree(spec))]


# ============================================================
# CONSTRUCTION DE BASE
# ============================================================
def test_pretraitement_standard_imputation_encodage_echelle() -> None:
    df = aides.jeu_regression()
    spec = aides.spec_regression()
    pipeline = _pipeline(spec).fit(_entrees(spec, df), df["y"])
    sortie = pipeline[:-1].transform(_entrees(spec, df))
    assert sortie.shape == (len(df), 3 + 3)  # trois mesures, trois modalités
    assert not np.isnan(sortie).any()
    noms = list(pipeline[:-1].get_feature_names_out())
    assert {"a", "b", "d", "c_x", "c_y", "c_z"} == set(noms)
    # les mesures sont centrées et réduites sur l'apprentissage
    mesures = pd.DataFrame(sortie, columns=noms)[["a", "b", "d"]]
    assert np.allclose(mesures.mean(), 0, atol=1e-9) and np.allclose(mesures.std(ddof=0), 1, atol=1e-9)


def test_les_etapes_ne_sont_pas_ajustees_a_la_construction() -> None:
    """I2 : la construction ne lit aucune donnée ; l'ajustement se fait dans Pipeline.fit."""
    etapes = dict(etapes_pretraitement(aides.spec_regression()))
    with pytest.raises(NotFittedError):
        check_is_fitted(etapes["pretraitement"])


def test_variables_sans_role_et_cible_n_entrent_pas() -> None:
    spec = aides.spec_regression(roles=(*aides.ROLES_REGRESSION, ("ignoree", Role.IGNOREE)))
    assert colonnes_d_entree(spec) == ("a", "b", "c", "d")
    df = aides.jeu_regression().assign(ignoree="cache")
    sortie = _pipeline(spec).fit(df, df["y"])[:-1].transform(df)
    assert sortie.shape[1] == 6


def test_branches_par_suite_d_etapes_identique() -> None:
    """Deux imputations différentes sur deux groupes de colonnes : deux branches numériques distinctes."""
    pretraitement = (
        CT("manquants.simple", colonnes=("a",), strategy="mean"),
        CT("manquants.simple", colonnes=("b", "d"), strategy="median"),
        CT("encodage.ordinal"),
    )
    spec = aides.spec_regression(pretraitement=pretraitement)
    ct = dict(etapes_pretraitement(spec))["pretraitement"]
    noms = [nom for nom, _, _ in ct.transformers]
    assert len(noms) == 3 and sum(n.startswith("numerique") for n in noms) == 2
    colonnes = {tuple(c) for _, _, c in ct.transformers}
    assert colonnes == {("a",), ("b", "d"), ("c",)}


def test_ordre_des_etapes_independant_de_l_ordre_de_saisie() -> None:
    melange = (CT("transformation.standard"), CT("encodage.one_hot"), CT("manquants.simple"))
    df = aides.jeu_regression()
    spec_a = aides.spec_regression(pretraitement=melange)
    spec_b = aides.spec_regression(pretraitement=tuple(reversed(melange)))
    sortie_a = _pipeline(spec_a).fit(_entrees(spec_a, df), df["y"])[:-1].transform(_entrees(spec_a, df))
    sortie_b = _pipeline(spec_b).fit(_entrees(spec_b, df), df["y"])[:-1].transform(_entrees(spec_b, df))
    assert np.allclose(sortie_a, sortie_b)


def test_modalite_inconnue_a_la_transformation_ne_casse_pas() -> None:
    df = aides.jeu_regression()
    spec = aides.spec_regression()
    pipeline = _pipeline(spec).fit(_entrees(spec, df), df["y"])
    nouveau = df.head(5).assign(c="titane")
    sortie = pipeline[:-1].transform(_entrees(spec, nouveau))
    assert sortie.shape == (5, 6) and not np.isnan(sortie).any()


@pytest.mark.parametrize("dtype", ["object", "str", "string", "category"])
def test_types_pandas_des_colonnes_categorielles(dtype: str) -> None:
    """pandas 3 : texte en `object`, `str`, `string` (valeurs NA) ou `category`."""
    df = aides.jeu_regression()
    df["c"] = df["c"].astype(dtype)
    df.loc[df.index[::9], "c"] = None
    pretraitement = (
        CT("manquants.simple", colonnes=("c",), strategy="most_frequent"),
        *aides.PRETRAITEMENT_STANDARD,
    )
    spec = aides.spec_regression(pretraitement=pretraitement)
    sortie = _pipeline(spec).fit(_entrees(spec, df), df["y"])[:-1].transform(_entrees(spec, df))
    assert sortie.shape == (len(df), 6) and not np.isnan(sortie).any()


def test_indicateur_de_valeur_manquante_s_ajoute_a_l_imputation() -> None:
    df = aides.jeu_regression()
    pretraitement = (CT("manquants.simple"), CT("manquants.indicateur"), CT("encodage.one_hot"))
    spec = aides.spec_regression(pretraitement=pretraitement)
    pipeline = _pipeline(spec).fit(_entrees(spec, df), df["y"])
    sortie = pipeline[:-1].transform(_entrees(spec, df))
    assert sortie.shape[1] == 3 + 3 + 3  # valeurs imputées, un indicateur par mesure, modalités
    assert not np.isnan(sortie).any()
    indicateur_a = sortie[:, 3]  # colonne d'indicateur de `a`
    assert indicateur_a.sum() == df["a"].isna().sum()


def test_knn_polynomial_et_discretisation_sur_les_mesures() -> None:
    df = aides.jeu_regression()
    for technique, attendu in [
        (CT("transformation.polynomiale", degree=2), 3 + 6),  # 3 mesures + 3 carrés + 3 produits
        (CT("transformation.discretisation", n_bins=4), 3),
        (CT("transformation.puissance"), 3),
        (CT("transformation.minmax"), 3),
        (CT("transformation.robuste"), 3),
    ]:
        spec = aides.spec_regression(pretraitement=(CT("manquants.knn"), CT("encodage.one_hot"), technique))
        sortie = _pipeline(spec).fit(_entrees(spec, df), df["y"])[:-1].transform(_entrees(spec, df))
        assert sortie.shape[1] == attendu + 3, technique.id


# ============================================================
# SÉLECTION DE VARIABLES
# ============================================================
def test_selection_apres_le_prétraitement_avec_le_bon_score() -> None:
    df = aides.jeu_classification()
    pretraitement = (*aides.PRETRAITEMENT_STANDARD, CT("selection.k_meilleures", k=2))
    spec = aides.spec_classification(pretraitement=pretraitement)
    etapes = etapes_pretraitement(spec)
    assert [nom for nom, _ in etapes] == ["pretraitement", "selection_0_k_meilleures"]
    assert etapes[1][1].score_func.__name__ == "f_classif"
    regression = aides.spec_regression(
        pretraitement=(*aides.PRETRAITEMENT_STANDARD, CT("selection.k_meilleures", k=2))
    )
    assert (
        dict(etapes_pretraitement(regression))["selection_0_k_meilleures"].score_func.__name__
        == "f_regression"
    )
    sortie = Pipeline(etapes).fit(df[["a", "b", "c"]], df["y"]).transform(df[["a", "b", "c"]])
    assert sortie.shape[1] == 2


def test_plusieurs_selections_dans_un_ordre_fixe() -> None:
    pretraitement = (
        CT("selection.depuis_modele"),
        CT("selection.variance"),
        *aides.PRETRAITEMENT_STANDARD,
    )
    etapes = etapes_pretraitement(aides.spec_regression(pretraitement=pretraitement))
    assert [nom for nom, _ in etapes] == [
        "pretraitement",
        "selection_0_variance",
        "selection_1_depuis_modele",
    ]


def test_selection_supervisee_refusee_sans_cible() -> None:
    spec = aides.spec_sans_cible(pretraitement=(CT("selection.k_meilleures"),))
    with pytest.raises(ErreurPretraitement, match="ne s'applique pas"):
        etapes_pretraitement(spec)
    assert etapes_pretraitement(aides.spec_sans_cible(pretraitement=(CT("selection.variance"),)))


# ============================================================
# REFUS ACTIONNABLES
# ============================================================
@pytest.mark.parametrize(
    ("pretraitement", "motif"),
    [
        ((CT("encodage.one_hot", colonnes=("a",)),), "réservé aux colonnes catégorielles"),
        ((CT("transformation.standard", colonnes=("c",)),), "réservé aux colonnes numériques"),
        ((CT("manquants.knn", colonnes=("c",)),), "réservé aux colonnes numériques"),
        ((CT("transformation.standard", colonnes=("y",)),), "pas des variables du modèle"),
        ((CT("transformation.standard", colonnes=("inconnue",)),), "pas des variables du modèle"),
        ((CT("transformation.standard"), CT("transformation.minmax"), CT("encodage.one_hot")), "même étape"),
        ((CT("encodage.one_hot"), CT("encodage.ordinal")), "même étape"),
        ((CT("transformation.standard"),), "n'est pas encodée"),
        ((CT("technique.absente"),), "Technique inconnue"),
        ((CT("regression.ridge"),), "n'est pas une technique de prétraitement"),
        ((CT("transformation.polynomiale", degree=4), CT("encodage.one_hot")), None),
    ],
)
def test_refus_du_pretraitement(pretraitement: tuple, motif: str | None) -> None:
    spec = aides.spec_regression(pretraitement=pretraitement)
    if motif is None:
        etapes_pretraitement(spec)  # degré 4 sur 3 colonnes : 34 colonnes, accepté
        return
    with pytest.raises(ErreurPretraitement, match=motif):
        etapes_pretraitement(spec)


def test_expansion_polynomiale_demesuree_refusee() -> None:
    n = 60
    roles = (*((f"x{k}", Role.NUMERIQUE) for k in range(n)), ("y", Role.CIBLE))
    spec = aides.spec_regression(roles=roles, pretraitement=(CT("transformation.polynomiale", degree=4),))
    with pytest.raises(ErreurPretraitement, match="produirait"):
        etapes_pretraitement(spec)


def test_aucune_variable_d_entree() -> None:
    spec = aides.spec_regression(roles=(("y", Role.CIBLE),), pretraitement=())
    with pytest.raises(ErreurPretraitement, match="Aucune variable"):
        etapes_pretraitement(spec)


def test_variables_temporelles_exigent_une_structure_chronologique() -> None:
    spec = aides.spec_groupe(
        structure=Structure.GROUPEE, pretraitement=(CT("chronologique.variables_temporelles"),)
    )
    with pytest.raises(ErreurPretraitement, match="structure chronologique"):
        etapes_pretraitement(spec)


# ============================================================
# VARIABLES TEMPORELLES
# ============================================================
def test_variables_temporelles_branche_a_part() -> None:
    df = aides.jeu_groupe()
    pretraitement = (
        CT("chronologique.variables_temporelles", retards=(1, 2), fenetres=(3,)),
        CT("transformation.standard"),
    )
    spec = aides.spec_groupe(pretraitement=pretraitement)
    assert colonnes_d_entree(spec) == ("x1", "x2", "cycle", "machine")
    pipeline = _pipeline(spec).fit(_entrees(spec, df), df["y"])
    sortie = pipeline[:-1].transform(_entrees(spec, df))
    noms = list(pipeline[:-1].get_feature_names_out())
    assert sortie.shape == (len(df), 2 + 2 * 4)  # deux mesures, puis quatre variables construites par mesure
    assert "x1_retard_1" in noms and "x2_ecart_type_3" in noms
    assert not np.isnan(sortie).any()  # les manques du début de série sont imputés
    assert "cycle" not in noms and "machine" not in noms  # la date et l'unité n'entrent pas dans le modèle


def test_variables_temporelles_sans_le_futur_ni_les_autres_unites() -> None:
    """Le test est précédé de l'historique de l'apprentissage : jamais l'inverse (I5)."""
    df = aides.jeu_groupe(n_groupes=4, longueur=12)
    spec = aides.spec_groupe(pretraitement=(CT("chronologique.variables_temporelles", retards=(1,)),))
    entrees = _entrees(spec, df)
    app, test = entrees[entrees["cycle"] < 8], entrees[entrees["cycle"] >= 8]
    pipeline = _pipeline(spec).fit(app, df.loc[app.index, "y"])
    sortie = pd.DataFrame(
        pipeline[:-1].transform(test), columns=pipeline[:-1].get_feature_names_out(), index=test.index
    )
    # le premier cycle du test (8) reçoit comme retard la dernière valeur de l'apprentissage (cycle 7)
    attendu = df.loc[(df["cycle"] == 7), "x1"].to_numpy()
    obtenu = sortie.loc[test["cycle"] == 8, "x1_retard_1"].to_numpy()
    assert np.allclose(np.sort(obtenu), np.sort(attendu))


# ============================================================
# PROPOSITION ET CONTRÔLES
# ============================================================
def test_proposition_selon_les_donnees_d_apprentissage() -> None:
    df = aides.jeu_regression()
    spec = aides.spec_regression(pretraitement=())
    ids = [(c.id, c.parametres_dict().get("strategy")) for c in proposer_pretraitement(spec, df)]
    assert (
        ids
        == [
            ("encodage.one_hot", None),
            ("transformation.standard", None),
        ]
        or ("manquants.simple", "median") in ids
    )
    assert ("manquants.simple", "median") in ids  # `a` a des valeurs manquantes
    sans_manque = aides.jeu_regression(manquants=False)
    ids = [c.id for c in proposer_pretraitement(spec, sans_manque)]
    assert "manquants.simple" not in ids and "encodage.one_hot" in ids and "transformation.standard" in ids
    # la proposition construit un pipeline qui s'ajuste
    proposee = spec.avec(pretraitement=proposer_pretraitement(spec, df))
    assert _pipeline(proposee).fit(_entrees(proposee, df), df["y"]).predict(_entrees(proposee, df)).shape == (
        len(df),
    )


def test_proposition_categorielle_avec_manquants() -> None:
    df = aides.jeu_regression(manquants=False)
    df.loc[df.index[::7], "c"] = None
    spec = aides.spec_regression(pretraitement=())
    choix = proposer_pretraitement(spec, df)
    imputations = [c for c in choix if c.id == "manquants.simple"]
    assert len(imputations) == 1 and imputations[0].parametres_dict() == {"strategy": "most_frequent"}
    assert imputations[0].colonnes == ("c",)


def test_proposition_chronologique_ajoute_les_variables_temporelles() -> None:
    df = aides.jeu_groupe()
    spec = aides.spec_groupe(pretraitement=())
    ids = [c.id for c in proposer_pretraitement(spec, df)]
    assert "chronologique.variables_temporelles" in ids
    proposee = spec.avec(pretraitement=proposer_pretraitement(spec, df))
    sortie = _pipeline(proposee).fit(_entrees(proposee, df), df["y"])[:-1].transform(_entrees(proposee, df))
    assert not np.isnan(sortie).any()


def test_controles_signalent_les_manquants_sans_imputation() -> None:
    df = aides.jeu_regression()
    sans = aides.spec_regression(pretraitement=(CT("encodage.one_hot"),))
    avis = controles_pretraitement(sans, df)
    assert len(avis) == 1 and "'a'" in avis[0] and "imputation" in avis[0]
    assert controles_pretraitement(aides.spec_regression(), df) == []
    incoherent = aides.spec_regression(pretraitement=(CT("encodage.one_hot", colonnes=("a",)),))
    assert "réservé" in controles_pretraitement(incoherent, df)[0]
    assert "absentes" in controles_pretraitement(aides.spec_regression(), df.drop(columns=["a"]))[0]


# ============================================================
# COHÉRENCE AVEC LE CATALOGUE
# ============================================================
def test_chaque_technique_de_pretraitement_a_une_place() -> None:
    en_catalogue = {e.id for e in REGISTRE if e.famille in ("manquants", "encodage", "transformation")}
    assert set(STADES) == en_catalogue
    for id_technique, stade in STADES.items():
        assert stade.rang > 0 and stade.defaut in stade.permis, id_technique
    # imputation, puis encodage, puis transformations
    assert (
        STADES["manquants.simple"].rang
        < STADES["encodage.one_hot"].rang
        < STADES["transformation.standard"].rang
    )
    assert STADES["transformation.polynomiale"].rang < STADES["transformation.standard"].rang


def test_probleme_sans_cible_construit_ses_etapes() -> None:
    spec = aides.spec_sans_cible()
    assert spec.probleme is Probleme.SANS_CIBLE
    assert [nom for nom, _ in etapes_pretraitement(spec)] == ["pretraitement"]


# ============================================================
# ROBUSTESSE : ORDRE DES COLONNES, MARQUEURS DE MANQUANT
# ============================================================
def test_l_ordre_des_colonnes_a_la_prediction_ne_change_rien() -> None:
    """Le pipeline choisit les colonnes par nom : des colonnes permutées ne se mélangent pas."""
    df = aides.jeu_regression()
    spec = aides.spec_regression()
    pipeline = _pipeline(spec).fit(_entrees(spec, df), df["y"])
    normal = pipeline.predict(df[["a", "b", "c", "d"]].head(20))
    permute = pipeline.predict(df[["d", "c", "b", "a"]].head(20))
    assert np.allclose(normal, permute)


def test_normaliser_manquants() -> None:
    avant = pd.DataFrame(
        {
            "texte": pd.array(["x", None, "z"], dtype="string"),
            "entier": pd.array([1, None, 3], dtype="Int64"),
            "objet": pd.Series(["p", None, "r"], dtype=object),
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "vrai": [True, False, True],
            "flottant": [1.0, np.nan, 3.0],
        }
    )
    apres = normaliser_manquants(avant)
    assert apres["entier"].dtype == np.dtype("float64") and np.isnan(apres["entier"].iloc[1])
    assert apres["texte"].dtype == object and apres["texte"].isna().tolist() == [False, True, False]
    assert apres["objet"].isna().tolist() == [False, True, False]
    assert apres["date"].dtype == avant["date"].dtype and apres["vrai"].dtype == bool
    assert avant["entier"].dtype == "Int64", "l'entrée n'est pas modifiée"


# ============================================================
# ÉCRÊTAGE DE TUKEY, CIBLE DÉCALÉE, PERSISTANCE
# ============================================================
def test_bornes_et_valeurs_aberrantes_de_tukey() -> None:
    from mecaml.core.catalog.ecretage_tukey import bornes_de_tukey, valeurs_aberrantes

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0, np.nan])
    basse, haute = bornes_de_tukey(x, 1.5)
    q1, q3 = np.percentile(x[:-1], [25, 75])
    assert basse[0] == pytest.approx(q1 - 1.5 * (q3 - q1)) and haute[0] == pytest.approx(q3 + 1.5 * (q3 - q1))
    masque = valeurs_aberrantes(x)
    assert masque.tolist() == [False] * 8 + [True, False]  # 100 est aberrante, le manquant ne l'est pas
    assert bornes_de_tukey(np.full((3, 1), np.nan))[1][0] == np.inf  # colonne vide : rien n'est aberrant


def test_ecretage_de_tukey_apprend_ses_bornes_sur_l_apprentissage_seul() -> None:
    from mecaml.core.catalog.ecretage_tukey import bornes_de_tukey

    df = aides.jeu_regression(manquants=False)
    spec = aides.spec_regression(
        pretraitement=(CT("transformation.ecretage_tukey", k=1.5), CT("encodage.one_hot"))
    )
    colonnes = list(colonnes_d_entree(spec))
    apprentissage, test = df.iloc[:150], df.iloc[150:].copy()
    pipeline = _pipeline(spec).fit(apprentissage[colonnes], apprentissage["y"])
    test.iloc[0, test.columns.get_loc("a")] = 1e6
    sortie = pd.DataFrame(
        pipeline[:-1].transform(test[colonnes]), columns=pipeline[:-1].get_feature_names_out()
    )
    haute = bornes_de_tukey(apprentissage[["a", "b", "d"]].to_numpy(), 1.5)[1]
    assert sortie.loc[0, "a"] == pytest.approx(haute[0])  # ramenée à la borne de l'apprentissage
    assert sortie["a"].max() <= haute[0] + 1e-12


def test_la_cible_decalee_ajoute_la_cible_aux_colonnes_recues_sans_la_faire_entrer_dans_le_modele() -> None:
    sans = aides.spec_groupe()
    avec = aides.spec_groupe(pretraitement=(CT("chronologique.cible_decalee", cible_connue=True),))
    assert "y" not in colonnes_d_entree(sans) and "y" in colonnes_d_entree(avec)
    df = aides.jeu_groupe()
    pipeline = _pipeline(avec).fit(_entrees(avec, df), df["y"])
    noms = list(pipeline[:-1].get_feature_names_out())
    assert noms == [
        "x1",
        "x2",
        "y_retard_1",
    ]  # les mesures, puis le passé de la cible ; jamais « y » lui-même


def test_la_cible_decalee_est_refusee_hors_structure_chronologique_et_sur_une_autre_colonne() -> None:
    avec = (CT("chronologique.cible_decalee", cible_connue=True),)
    with pytest.raises(ErreurPretraitement, match="structure chronologique"):
        etapes_pretraitement(aides.spec_regression(pretraitement=avec))
    with pytest.raises(ErreurPretraitement, match="décale la cible du projet, et elle seule"):
        etapes_pretraitement(
            aides.spec_groupe(
                pretraitement=(CT("chronologique.cible_decalee", colonnes=("x1",), cible_connue=True),)
            )
        )
    classification = aides.spec_classification(pretraitement=avec)
    with pytest.raises(ErreurPretraitement, match="ne s'applique pas"):
        etapes_pretraitement(classification)


def test_la_cible_decalee_sans_declaration_est_refusee_par_le_pretraitement() -> None:
    """Sans `cible_connue=True`, ni la cible décalée ni la persistance ne construisent d'étapes."""
    from mecaml.core.preprocess import etapes_persistance

    for refus in ({}, {"cible_connue": False}):
        spec = aides.spec_groupe(pretraitement=(CT("chronologique.cible_decalee", **refus),))
        with pytest.raises(ErreurPretraitement, match=r"cible_connue=True"):
            etapes_pretraitement(spec)
        sans_declaration = aides.spec_groupe(modeles=(CT("reference.persistance", **refus),))
        with pytest.raises(ErreurPretraitement, match=r"cible_connue=True"):
            etapes_persistance(sans_declaration)
    with pytest.raises(ErreurPretraitement, match=r"cible_connue=True"):
        etapes_persistance(aides.spec_groupe())  # aucune persistance dans la spécification


def test_etapes_de_la_persistance_ne_lisent_que_le_passe_de_la_cible() -> None:
    from mecaml.core.preprocess import etapes_persistance

    spec = aides.spec_groupe(
        modeles=(CT("reference.persistance", cible_connue=True),),
        pretraitement=(CT("transformation.standard"),),
    )
    df = aides.jeu_groupe()
    etapes = etapes_persistance(spec)
    assert [nom for nom, _ in etapes] == ["pretraitement"]
    transformateur = etapes[0][1].fit(_entrees(spec, df), df["y"])
    sortie = transformateur.transform(_entrees(spec, df))
    assert sortie.shape == (
        len(df),
        1,
    )  # une colonne : la cible décalée, brute, sans la mise à l'échelle du projet
    cycle_1 = (df["cycle"] == 1) & (df["machine"] == "M00")
    suivante = df[(df["cycle"] == 0) & (df["machine"] == "M00")]["y"].iloc[0]
    assert sortie[cycle_1.to_numpy()][0, 0] == pytest.approx(suivante)
