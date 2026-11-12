"""Script équivalent : sérialiseur, code repris du paquet, ceinture à l'export, matrice des méthodes.

Le sérialiseur écrit les 61 estimateurs du catalogue et se relit à l'identique (classes et paramètres).
Le code repris du paquet est extrait par analyse syntaxique : ses dépendances viennent avec lui, ce
qui n'est pas autonome est refusé. Chaque méthode de découpage et de validation est ensuite rejouée
par le script, importé dans ce processus, contre les scores de l'application (les jeux de l'invariant
I8 passent, eux, par un processus neuf).
"""

from __future__ import annotations

import ast
import functools
import inspect
import math
import re
import sys
import types as outils_types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml.core import codegen, decision, features_ts, split
from mecaml.core import types as types_du_coeur
from mecaml.core.catalog import REGISTRE
from mecaml.core.guards import ErreurDonnees
from mecaml.core.types import ChoixTechnique, Probleme, SpecProjet, Structure, Table

CT = ChoixTechnique.creer
OPT = "-" * 2  # préfixe des options du script, écrit sans double trait d'union littéral
RIDGE = CT("regression.ridge")
ARBRE = CT("regression.arbre", max_depth=3)
LOGISTIQUE = CT("classification.logistique")
ARBRE_CLS = CT("classification.arbre", max_depth=3)


# ============================================================
# ÉCRITURE DE CODE : LARGEUR, ÉLÉMENTS, RELECTURE
# ============================================================
def test_un_document_court_tient_sur_une_ligne() -> None:
    doc = codegen._Appel("Ridge", [("alpha", "1.0"), ("random_state", "GRAINE")])
    assert codegen.rendre(doc) == "Ridge(alpha=1.0, random_state=GRAINE)"


def test_un_document_long_met_un_element_par_ligne_dans_la_largeur() -> None:
    colonnes = [f"colonne_numero_{k:03d}" for k in range(40)]
    doc = codegen._Appel("Pipeline", [(None, codegen._Suite("[", "]", [f'"{c}"' for c in colonnes]))])
    texte = codegen.rendre(doc, 4, len("return "))
    assert max(len(ligne) for ligne in texte.splitlines()) <= codegen.LARGEUR
    assert texte.count("\n") > 10
    assert ast.literal_eval(texte.replace("Pipeline(", "(").replace("\n", " ").strip()) == (colonnes,)


def test_un_tuple_d_un_element_garde_sa_virgule_et_un_tuple_vide_reste_vide() -> None:
    assert codegen.rendre(codegen._Suite("(", ")", ["1"])) == "(1,)"
    assert codegen.rendre(codegen._Suite("(", ")", [])) == "()"
    assert codegen.rendre(codegen._Suite("[", "]", [])) == "[]"


def test_un_dictionnaire_imbrique_se_relit_a_l_identique() -> None:
    projet = {"a": [1, 2.5, None, True], "b": {"c": "texte « accentué »", "d": ["x"] * 30}}
    ecrivain = codegen._Ecrivain(42, codegen._Imports())
    texte = codegen.rendre(ecrivain.valeur(projet))
    assert ast.literal_eval(texte) == projet
    assert max(len(ligne) for ligne in texte.splitlines()) <= codegen.LARGEUR


def test_les_imports_sont_fusionnes_et_ranges_par_groupe() -> None:
    imports = codegen._Imports()
    for instruction in (
        "import numpy as np",
        "import json",
        "from sklearn.linear_model import Ridge",
        "from sklearn.linear_model import Lasso",
        "from pathlib import Path",
        "import pandas as pd",
    ):
        imports.instruction(instruction)
    assert imports.lignes() == [
        "import json",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import pandas as pd",
        "from sklearn.linear_model import Lasso, Ridge",
    ]


# ============================================================
# SÉRIALISEUR : CHAQUE ESTIMATEUR DU CATALOGUE SE RELIT À L'IDENTIQUE
# ============================================================
SPECIAUX = {
    "chronologique.variables_temporelles": {
        "colonne_date": "t",
        "colonne_groupe": "m",
        "colonnes": ("a", "b"),
        "fenetres": (3,),
        "demi_vies": (2.0,),
    },
    "chronologique.cible_decalee": {"colonne_date": "t", "colonnes": ("y",), "cible_connue": True},
    "reference.persistance": {"cible_connue": True},
}
ENTREES_A_ECRIRE = [e for e in REGISTRE if e.famille not in ("validation", "explication")]


def _construire(entree: Any) -> Any:
    parametres = dict(SPECIAUX.get(entree.id, {}))
    if entree.famille == "selection":
        classification = Probleme.CLASSIFICATION in entree.problemes
        parametres["probleme"] = "classification" if classification else "regression"
    return entree.construire(42, **parametres)


def relire(
    estimateur: Any, nom: str, graine: int = 42, contexte: pd.DataFrame | None = None
) -> tuple[Any, str]:
    """Code d'un estimateur écrit par le sérialiseur, exécuté dans un module neuf ; (objet relu, source)."""
    imports = codegen._Imports()
    ecrivain = codegen._Ecrivain(graine, imports, contexte)
    code = codegen.rendre(ecrivain.estimateur(estimateur))
    annexe = codegen.annexe_des(ecrivain.embarques)
    for instruction in annexe.imports:
        imports.instruction(instruction)
    source = "\n".join(imports.lignes()) + f"\n\nGRAINE = {graine}\n\n{annexe.code}\n\nobjet = {code}\n"
    module = outils_types.ModuleType(nom)
    module.contexte = contexte
    sys.modules[nom] = module
    exec(compile("from __future__ import annotations\n" + source, nom, "exec"), module.__dict__)
    return module.objet, source


@pytest.mark.parametrize("entree", ENTREES_A_ECRIRE, ids=lambda e: e.id)
def test_chaque_estimateur_du_catalogue_se_relit_a_l_identique(entree: Any) -> None:
    voulu = _construire(entree)
    relu, source = relire(voulu, f"relecture_{entree.id.replace('.', '_')}")
    assert aides.differences(voulu, relu) == [], source[-400:]
    assert "mecaml" not in source


def test_le_catalogue_ecrit_compte_soixante_et_un_estimateurs() -> None:
    assert len(ENTREES_A_ECRIRE) == 61


def test_la_graine_du_projet_s_ecrit_graine_et_une_autre_valeur_reste_un_nombre() -> None:
    _, source = relire(
        RIDGE and _construire(next(e for e in REGISTRE if e.id == "regression.ridge")), "graine_a"
    )
    assert "Ridge(random_state=GRAINE)" in source
    foret = next(e for e in REGISTRE if e.id == "regression.foret").construire(7, n_estimators=7)
    _, source = relire(foret, "graine_b", graine=42)
    assert "random_state=7" in source and "n_estimators=7" in source  # 7 n'est pas la graine du projet


def test_les_valeurs_inhabituelles_se_relisent() -> None:
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    cas = [
        OneHotEncoder(dtype=np.float32, handle_unknown="ignore"),
        SimpleImputer(strategy="constant", fill_value=np.inf),
        SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-np.inf),
        SelectKBest(score_func=functools.partial(mutual_info_regression, random_state=42), k=3),
    ]
    for k, estimateur in enumerate(cas):
        relu, source = relire(estimateur, f"inhabituel_{k}")
        assert aides.differences(estimateur, relu) == [], source[-300:]


def test_une_fonction_inconnue_du_script_est_refusee() -> None:
    from sklearn.preprocessing import FunctionTransformer

    with pytest.raises(codegen.ErreurExport, match=r"non importable|non reproductible"):
        relire(FunctionTransformer(func=lambda x: x), "lambda_refusee")
    with pytest.raises(codegen.ErreurExport, match="non reproductible"):
        codegen._Ecrivain(42, codegen._Imports()).valeur(object())


def test_une_table_de_contexte_doit_se_reconstruire_a_l_identique() -> None:
    contexte = pd.DataFrame({"t": [1, 2, 3], "m": ["a", "a", "a"], "x": [0.1, 0.2, 0.3]})
    ecrivain = codegen._Ecrivain(42, codegen._Imports(), contexte_reel=contexte)
    texte = _plat(ecrivain.valeur(contexte[["t", "x"]].reset_index(drop=True)))
    assert texte == 'contexte_de(contexte, ["t", "x"])'
    with pytest.raises(codegen.ErreurExport, match="ne se reconstruit pas"):
        ecrivain.valeur(contexte[["t", "x"]].iloc[::-1].reset_index(drop=True))  # ordre des lignes différent
    with pytest.raises(codegen.ErreurExport, match="ne se reconstruit pas"):
        codegen._Ecrivain(42, codegen._Imports()).valeur(contexte)  # aucun contexte réel


def _plat(doc: Any) -> str:
    return codegen._a_plat(doc)


def test_le_contexte_relu_dans_un_pipeline_est_celui_de_l_application() -> None:
    contexte = jeux.jeu_serie_mesuree(n_groupes=3, longueur=12)[["machine", "cycle", "x1", "x2"]]
    contexte = contexte.iloc[[2, 3, 9, 20]].reset_index(drop=True)
    reel = pd.concat([contexte, contexte.iloc[:0]])
    estimateur = features_ts.fabrique_variables_temporelles(
        colonne_date="cycle", colonne_groupe="machine", colonnes=("x1", "x2"), retards=(1,), contexte=contexte
    )
    relu, _ = relire(estimateur, "contexte_relu", contexte=reel)
    assert aides.differences(estimateur, relu) == []


# ============================================================
# CODE REPRIS DU PAQUET
# ============================================================
def test_l_extraction_emporte_les_dependances_et_laisse_le_reste() -> None:
    extrait = codegen.extraire_du_paquet(features_ts, ["_classe_transformeur"])
    for nom in (
        "class ConfigTemporelle",
        "def construire_variables",
        "def _historique_de",
        "PROFONDEUR_MAX =",
    ):
        assert nom in extrait.code
    assert (
        "def fabrique_variables_temporelles" not in extrait.code
    )  # la fabrique du registre reste dans le paquet
    assert "mecaml" not in extrait.code
    assert "import functools" in extrait.imports and "import numpy as np" in extrait.imports
    ast.parse(extrait.code)


def test_le_code_repris_est_celui_du_paquet_a_la_lettre() -> None:
    extrait = codegen.extraire_du_paquet(types_du_coeur, ["empreinte_dataframe"])
    assert extrait.code == inspect.getsource(types_du_coeur.empreinte_dataframe).rstrip("\n")


def test_les_fonctions_reprises_donnent_les_memes_resultats_que_le_paquet() -> None:
    annexe = codegen.annexe_des(REPRISES_UTILES)
    module = outils_types.ModuleType("annexe_reprise")
    sys.modules["annexe_reprise"] = module
    source = "from __future__ import annotations\n" + "\n".join(annexe.imports) + "\n" + annexe.code
    exec(compile(source, "annexe_reprise", "exec"), module.__dict__)
    df = jeux.jeu_regression(n=30)
    assert module.empreinte_dataframe(df) == types_du_coeur.empreinte_dataframe(df)
    assert module.empreinte_positions([4, 1, 9]) == split.empreinte_positions([4, 1, 9])
    assert module.empreinte_plis([(np.arange(3), np.arange(3, 5))]) == codegen.empreinte_plis(
        [(np.arange(3), np.arange(3, 5))]
    )
    sales = pd.DataFrame(
        {"c": pd.array(["a", None, "b"], dtype="string"), "v": pd.array([1, None, 3], dtype="Int64")}
    )
    from mecaml.core.preprocess import normaliser_manquants

    pd.testing.assert_frame_equal(module.normaliser_manquants(sales), normaliser_manquants(sales))


REPRISES_UTILES = ("empreinte_dataframe", "empreinte_positions", "empreinte_plis", "normaliser_manquants")


def test_un_nom_qui_depend_du_paquet_n_est_pas_repris(tmp_path: Path) -> None:
    module = _module_temporaire(
        tmp_path,
        "dependant",
        "from mecaml.core.guards import decimale\n\ndef formater(x):\n    return decimale(x)\n\ndef "
        "seul(x):\n    return x\n",
    )
    with pytest.raises(codegen.ErreurExport, match="importé du paquet"):
        codegen.extraire_du_paquet(module, ["formater"])
    assert codegen.extraire_du_paquet(module, ["seul"]).code.startswith("def seul")
    with pytest.raises(codegen.ErreurExport, match="introuvable"):
        codegen.extraire_du_paquet(module, ["absent"])


def test_le_renommage_ne_touche_que_des_mots_entiers(tmp_path: Path) -> None:
    module = _module_temporaire(
        tmp_path, "renommable", "def _f():\n    return 1\n\ndef _fx():\n    return _f() + 1\n"
    )
    extrait = codegen.extraire_du_paquet(module, ["_fx"], {"_f": "_g"})
    assert "def _g()" in extrait.code and "def _fx()" in extrait.code and "_g() + 1" in extrait.code


def test_un_code_source_introuvable_est_signale_en_francais(monkeypatch: pytest.MonkeyPatch) -> None:
    def indisponible(_: Any) -> str:
        raise OSError("pas de source")

    codegen._analyser_module.cache_clear()
    monkeypatch.setattr(inspect, "getsource", indisponible)
    with pytest.raises(codegen.ErreurExport, match="code source du module"):
        codegen.extraire_du_paquet(features_ts, ["_classe_transformeur"])
    monkeypatch.undo()
    codegen._analyser_module.cache_clear()
    assert codegen.extraire_du_paquet(features_ts, ["_classe_transformeur"]).code


def _module_temporaire(dossier: Path, nom: str, source: str) -> outils_types.ModuleType:
    import importlib.util

    chemin = dossier / f"{nom}.py"
    chemin.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(nom, chemin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# PROJETS SYNTHÉTIQUES ÉCRITS EN .csv
# ============================================================
def _projet(tmp_path: Path, jeu: pd.DataFrame, fabrique: Any, **options: Any) -> aides.Projet:
    return aides.mener_le_projet(aides.ecrire_csv(jeu, tmp_path / "jeu.csv"), fabrique, **options)


def _spec_regression(table: Table) -> SpecProjet:
    return jeux.spec_regression(modeles=(RIDGE, ARBRE))


def _spec_classification(table: Table) -> SpecProjet:
    return jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS))


# ============================================================
# CEINTURE À L'EXPORT : LA CIBLE PASSÉE
# ============================================================
def _table_groupe(tmp_path: Path, jeu: pd.DataFrame) -> Table:
    return aides.charger(aides.ecrire_csv(jeu, tmp_path / "serie.csv")).table


def test_l_export_refuse_une_cible_passee_non_declaree(tmp_path: Path) -> None:
    table = _table_groupe(tmp_path, jeux.jeu_serie_mesuree())
    spec = jeux.spec_groupe(modeles=(CT("reference.dummy_regression"), CT("reference.persistance")))
    _, coffre = split.sceller(table, spec)
    with pytest.raises(codegen.ErreurExport, match=r"cible_connue=True"):
        codegen.generer_script(spec, table, coffre)
    with pytest.raises(codegen.ErreurExport, match=r"cible_connue=True"):
        codegen.verifier_ceinture(spec, table, coffre)


def test_l_export_refuse_une_cible_qui_decompte_meme_declaree(tmp_path: Path) -> None:
    table = _table_groupe(tmp_path, jeux.jeu_groupe(n_groupes=10, longueur=14))
    spec = jeux.spec_groupe(
        modeles=(CT("reference.dummy_regression"), CT("reference.persistance", cible_connue=True))
    )
    _, coffre = split.sceller(table, spec)
    with pytest.raises(codegen.ErreurExport, match="compte à rebours"):
        codegen.generer_script(spec, table, coffre)


def test_l_export_accepte_une_cible_passee_declaree_sur_une_serie_mesuree(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_serie_mesuree(), aides.spec_persistance)
    texte = codegen.generer_script(projet.spec, projet.table, projet.coffre, decoupage=projet.decoupage)
    assert "cible_connue" in texte  # le projet gardé dans le script porte la déclaration
    assert "Hypothèse déclarée par le projet : la cible du pas précédent est connue au moment de la" in texte
    assert "reference.persistance" in texte and "chronologique.cible_decalee" in texte


def test_un_script_sans_cible_passee_n_ecrit_aucune_hypothese(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    texte = codegen.generer_script(projet.spec, projet.table, projet.coffre)
    assert "Hypothèse déclarée" not in texte


def test_l_export_demande_un_coffre_et_des_modeles(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    with pytest.raises(codegen.ErreurExport, match="sous scellé"):
        codegen.generer_script(projet.spec, projet.table, None)
    with pytest.raises(codegen.ErreurExport, match="Aucun modèle"):
        codegen.generer_script(projet.spec.avec(modeles=()), projet.table, projet.coffre)
    doublon = projet.spec.avec(modeles=(RIDGE, RIDGE))
    with pytest.raises(codegen.ErreurExport, match="même technique"):
        codegen.generer_script(doublon, projet.table, projet.coffre, decoupage=projet.decoupage)


# ============================================================
# DÉCOUPAGE RETROUVÉ DEPUIS LE COFFRE
# ============================================================
def _cas_de_decoupage() -> list[Any]:
    return [
        pytest.param(jeux.jeu_regression(n=90), _spec_regression, None, id="aleatoire"),
        pytest.param(jeux.jeu_classification(n=120), _spec_classification, None, id="stratifie"),
        pytest.param(
            jeux.jeu_groupe(n_groupes=10, longueur=12),
            lambda t: jeux.spec_groupe(modeles=(RIDGE,)),
            None,
            id="unites",
        ),
        pytest.param(aides.jeu_pannes(), aides.spec_classification_groupee, None, id="unites_stratifiees"),
        pytest.param(aides.jeu_chronologique(), aides.spec_chronologique, None, id="chronologique"),
        pytest.param(
            aides.jeu_pannes(), aides.spec_classification_groupee, split.Methode.UNITES, id="unites_imposees"
        ),
    ]


@pytest.mark.parametrize(("jeu", "fabrique", "methode"), _cas_de_decoupage())
def test_le_decoupage_se_retrouve_depuis_le_coffre(tmp_path: Path, jeu, fabrique, methode) -> None:
    chargement = aides.charger(aides.ecrire_csv(jeu, tmp_path / "jeu.csv"))
    spec = fabrique(chargement.table)
    declare, coffre = split.sceller(chargement.table, spec, methode=methode)
    assert codegen.retrouver_decoupage(chargement.table, spec, coffre) == declare


def test_le_decoupage_d_un_test_fourni_a_part_se_retrouve(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=120)
    table = aides.charger(aides.ecrire_csv(jeu.iloc[:100], tmp_path / "app.csv")).table
    test = aides.charger(aides.ecrire_csv(jeu.iloc[100:], tmp_path / "test.csv")).table
    spec = _spec_regression(table)
    declare, coffre = split.sceller_fichier(table, test, spec)
    retrouve = codegen.retrouver_decoupage(table, spec, coffre)
    assert retrouve == declare and retrouve.methode is split.Methode.FICHIER


def test_un_coffre_qui_n_est_pas_celui_de_split_est_refuse(tmp_path: Path) -> None:
    table = aides.charger(aides.ecrire_csv(jeux.jeu_regression(n=80), tmp_path / "jeu.csv")).table
    spec = _spec_regression(table)
    coffre = jeux.coffre_par_pas(table, spec, pas=4)  # une ligne sur quatre : aucune méthode ne l'explique
    with pytest.raises(codegen.ErreurExport, match="aucune méthode de découpage"):
        codegen.retrouver_decoupage(table, spec, coffre)


# ============================================================
# LE TEXTE DU SCRIPT
# ============================================================
def test_le_script_est_du_python_valide_et_se_reproduit_a_l_identique(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    premier = aides.script_du_projet(projet, tmp_path / "a").read_text(encoding="utf-8")
    second = aides.script_du_projet(projet, tmp_path / "b").read_text(encoding="utf-8")
    assert premier == second
    ast.parse(premier)
    assert "\r" not in premier and premier.endswith("\n") and not premier.endswith("\n\n")
    assert premier.rstrip().endswith('if __name__ == "__main__":\n    main()')


def test_le_script_porte_les_empreintes_la_graine_et_le_projet(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    script = aides.script_du_projet(projet, tmp_path / "s")
    module = aides.charger_script(script, "script_porte")
    assert projet.table.empreinte == module.EMPREINTE_DONNEES
    assert projet.decoupage.empreinte_sha256 == module.EMPREINTE_TEST
    assert codegen.empreinte_plis(projet.resultat.plis) == module.EMPREINTE_PLIS
    assert module.GRAINE == projet.spec.graine == 42
    assert [c.id for c in projet.spec.modeles] == module.MODELES
    assert projet.decision.modele_id == module.MODELE_RETENU and module.SEUIL is None
    assert SpecProjet.depuis_dict(module.PROJET) == projet.spec
    assert ast.literal_eval(_affectation(script, "PROJET")) == projet.spec.vers_dict()


def _affectation(script: Path, nom: str) -> str:
    for noeud in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(noeud, ast.Assign) and any(getattr(c, "id", "") == nom for c in noeud.targets):
            return ast.unparse(noeud.value)
    raise AssertionError(nom)


def test_les_options_du_script_sont_documentees_dans_son_en_tete(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    texte = aides.script_du_projet(projet, tmp_path / "s").read_text(encoding="utf-8")
    for option in ("donnees", "sortie", "ignorer-empreintes"):
        assert f"{OPT}{option}" in texte
    assert f"{OPT}test" not in texte.split('"""', 2)[2]  # pas d'option de test hors du mode « fichier »
    assert "Dépendances : numpy, pandas, scikit-learn" in texte
    assert "relancer ce script ne le rend pas neuf" in texte


def test_un_script_sans_decision_valide_sans_ouvrir_le_test(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    texte = codegen.generer_script(projet.spec, projet.table, projet.coffre, decoupage=projet.decoupage)
    assert "MODELE_RETENU = None" in texte
    module = _importer(texte, tmp_path / "sd", "sans_decision")
    module.main([f"{OPT}donnees", str(projet.fichier), f"{OPT}sortie", str(tmp_path / "sd")])
    resultats = aides.lire_resultats(tmp_path / "sd")
    assert "test" not in resultats and "decision" not in resultats
    aides.verifier_validation(projet, resultats)


def _importer(texte: str, dossier: Path, nom: str) -> outils_types.ModuleType:
    dossier.mkdir(parents=True, exist_ok=True)
    return aides.charger_script(codegen.ecrire_script(texte, dossier), nom)


def test_l_ecriture_du_script_reste_dans_le_dossier_choisi(tmp_path: Path) -> None:
    chemin = codegen.ecrire_script("x = 1\n", tmp_path)
    assert chemin == (tmp_path / codegen.NOM_DU_SCRIPT).resolve()
    assert chemin.read_bytes() == b"x = 1\n"
    with pytest.raises(ErreurDonnees, match="sort du dossier"):
        codegen.ecrire_script("x = 1\n", tmp_path, "../dehors.py")
    with pytest.raises(ErreurDonnees, match="n'existe pas"):
        codegen.ecrire_script("x = 1\n", tmp_path / "absent")
    assert not (tmp_path.parent / "dehors.py").exists()


def test_le_script_arrete_si_les_donnees_ne_sont_pas_celles_du_projet(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    script = aides.script_du_projet(projet, tmp_path / "s")
    module = aides.charger_script(script, "script_arret")
    modifie = pd.read_csv(projet.fichier)
    modifie.loc[3, "a"] = 123.0
    fichier = aides.ecrire_csv(modifie, tmp_path / "modifie.csv")
    with pytest.raises(SystemExit, match="ne sont pas celles du projet"):
        module.main([f"{OPT}donnees", str(fichier)])
    with pytest.raises(SystemExit, match="introuvable"):
        module.main([f"{OPT}donnees", str(tmp_path / "absent.csv")])
    module.main(
        [f"{OPT}donnees", str(fichier), f"{OPT}ignorer-empreintes", f"{OPT}sortie", str(tmp_path / "s")]
    )


@pytest.mark.parametrize(
    ("constante", "message"),
    [("EMPREINTE_TEST", "jeu de test du projet"), ("EMPREINTE_PLIS", "plis de validation")],
)
def test_le_script_arrete_si_le_decoupage_ou_les_plis_refaits_ne_sont_pas_ceux_du_projet(
    tmp_path: Path, constante: str, message: str
) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=80), _spec_regression)
    texte = codegen.generer_script(projet.spec, projet.table, projet.coffre, decoupage=projet.decoupage)
    faux = re.sub(rf"^{constante} = .*$", f'{constante} = "{"0" * 64}"', texte, flags=re.MULTILINE)
    assert faux != texte
    module = _importer(faux, tmp_path / constante, f"script_{constante}")
    sortie = str(tmp_path / constante / "sortie")
    with pytest.raises(SystemExit, match=message):
        module.main([f"{OPT}donnees", str(projet.fichier), f"{OPT}sortie", sortie])
    module.main([f"{OPT}donnees", str(projet.fichier), f"{OPT}ignorer-empreintes", f"{OPT}sortie", sortie])


# ============================================================
# MATRICE : CHAQUE MÉTHODE DE DÉCOUPAGE ET DE VALIDATION, REJOUÉE PAR LE SCRIPT
# ============================================================
def _decider_au_seuil(seuil: float):
    return lambda resultat: decision.decider(resultat, "classification.logistique", seuil=seuil)


def _cas_de_matrice() -> list[Any]:
    reg = (jeux.jeu_regression(n=100), _spec_regression, {})
    cls = (jeux.jeu_classification(n=240), _spec_classification, {})
    pannes = (aides.jeu_pannes(), aides.spec_classification_groupee, {})
    return [
        pytest.param(*reg, id="regression_aleatoire_kfold"),
        pytest.param(
            jeux.jeu_regression(n=60),
            lambda t: jeux.spec_regression(modeles=(RIDGE,), validation="validation.exclusion_unique"),
            {},
            id="regression_exclusion_unique",
        ),
        pytest.param(*cls, id="classification_stratifiee"),
        pytest.param(
            jeux.jeu_classification(n=240),
            _spec_classification,
            {"decider": _decider_au_seuil(0.35)},
            id="classification_seuil",
        ),
        pytest.param(
            jeux.jeu_classification(n=240),
            lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS), metrique="cout:fp=1,fn=5"),
            {"decider": _decider_au_seuil(0.4)},
            id="classification_cout_au_seuil",
        ),
        pytest.param(
            jeux.jeu_classification(n=240, classes=3),
            lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS), binaire=False),
            {},
            id="classification_multiclasse",
        ),
        pytest.param(*pannes, id="groupee_unites_stratifiees"),
        pytest.param(
            aides.jeu_pannes(),
            lambda t: aides.spec_classification_groupee(t).avec(validation="validation.groupes_aleatoires"),
            {},
            id="groupee_groupes_aleatoires",
        ),
        pytest.param(
            aides.jeu_pannes(),
            lambda t: aides.spec_classification_groupee(t).avec(validation="validation.plis_groupes"),
            {"methode": split.Methode.UNITES},
            id="groupee_unites_plis_groupes",
        ),
        pytest.param(
            aides.jeu_chronologique(), aides.spec_chronologique, {}, id="chronologique_fenetre_croissante"
        ),
        pytest.param(
            jeux.jeu_regression(n=100),
            lambda t: jeux.spec_regression(
                modeles=(RIDGE,),
                pretraitement=(
                    CT("manquants.simple", strategy="median"),
                    CT("encodage.one_hot"),
                    CT("transformation.ecretage_tukey", k=1.5),
                    CT("transformation.polynomiale", degree=2),
                    CT("transformation.standard"),
                    CT("selection.k_meilleures", k=5),
                ),
            ),
            {},
            id="pretraitement_tukey_polynomial_selection",
        ),
        pytest.param(
            jeux.jeu_regression(n=100),
            lambda t: jeux.spec_regression(
                modeles=(
                    CT("regression.vote"),
                    CT("regression.empilement"),
                    CT("regression.bagging", n_estimators=5),
                )
            ),
            {},
            id="modeles_composites",
        ),
    ]


@pytest.mark.parametrize(("jeu", "fabrique", "options"), _cas_de_matrice())
def test_le_script_rejoue_les_memes_scores_pour_chaque_methode(
    tmp_path: Path, jeu, fabrique, options
) -> None:
    projet = _projet(tmp_path, jeu, fabrique, **options)
    resultats = aides.rejouer_dans_le_processus(projet, tmp_path / "rejeu")
    aides.verifier_validation(projet, resultats)
    aides.verifier_test(projet, resultats)


def test_le_script_rejoue_un_test_fourni_a_part(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=120)
    app = aides.ecrire_csv(jeu.iloc[:100], tmp_path / "app.csv")
    test = aides.ecrire_csv(jeu.iloc[100:], tmp_path / "test.csv")
    projet = aides.mener_le_projet(app, _spec_regression, fichier_test=test)
    assert projet.decoupage.methode is split.Methode.FICHIER
    texte = aides.script_du_projet(projet, tmp_path / "s").read_text(encoding="utf-8")
    assert f"{OPT}test" in texte and "FEUILLE_TEST" in texte
    resultats = aides.rejouer_dans_le_processus(projet, tmp_path / "rejeu")
    aides.verifier_validation(projet, resultats)
    aides.verifier_test(projet, resultats)
    module = aides.charger_script(tmp_path / "s" / codegen.NOM_DU_SCRIPT, "script_fichier")
    with pytest.raises(SystemExit, match="fourni à part"):
        module.main([f"{OPT}donnees", str(app)])  # sans le jeu de test
    with pytest.raises(SystemExit, match="n'est pas celui du projet"):
        module.main([f"{OPT}donnees", str(app), f"{OPT}test", str(app)])  # un autre jeu de test


def test_un_csv_francais_se_relit_dans_le_script(tmp_path: Path) -> None:
    jeu = jeux.jeu_regression(n=90)
    fichier = aides.ecrire_csv(jeu, tmp_path / "fr.csv", sep=";", decimal=",", encoding="utf-8")
    projet = aides.mener_le_projet(fichier, _spec_regression)
    assert (projet.chargement.separateur, projet.chargement.decimal) == (";", ",")
    resultats = aides.rejouer_dans_le_processus(projet, tmp_path / "rejeu")
    aides.verifier_validation(projet, resultats)
    texte = (tmp_path / "rejeu" / codegen.NOM_DU_SCRIPT).read_text(encoding="utf-8")
    assert 'SEPARATEUR = ";"' in texte and 'DECIMALE = ","' in texte


def test_un_csv_sans_son_chargement_se_lit_avec_les_options_de_pandas(tmp_path: Path) -> None:
    projet = _projet(tmp_path, jeux.jeu_regression(n=90), _spec_regression)
    texte = codegen.generer_script(projet.spec, projet.table, projet.coffre, decoupage=projet.decoupage)
    assert "SEPARATEUR" not in texte and "chemin, low_memory=False" in texte
    module = _importer(texte, tmp_path / "sc", "sans_chargement")
    module.main([f"{OPT}donnees", str(projet.fichier), f"{OPT}sortie", str(tmp_path / "sc")])
    aides.verifier_validation(projet, aides.lire_resultats(tmp_path / "sc"))


def test_les_dates_ecrites_en_texte_iso_sont_relues_comme_des_dates(tmp_path: Path) -> None:
    df = aides.jeu_chronologique(n=120)
    df["t"] = pd.date_range("2026-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d")
    projet = _projet(tmp_path, df, aides.spec_chronologique)
    assert pd.api.types.is_datetime64_any_dtype(
        projet.table.donnees["t"]
    )  # lu comme une date par l'application
    resultats = aides.rejouer_dans_le_processus(projet, tmp_path / "rejeu")
    aides.verifier_validation(projet, resultats)
    assert 'COLONNES_DATES = ["t"]' in (tmp_path / "rejeu" / codegen.NOM_DU_SCRIPT).read_text(
        encoding="utf-8"
    )


def test_sans_cible_le_script_n_a_ni_decoupage_ni_test(tmp_path: Path) -> None:
    projet = _projet(
        tmp_path,
        jeux.jeu_sans_cible(n=120),
        lambda t: jeux.spec_sans_cible(
            modeles=(CT("sans_cible.kmeans", n_clusters=2), CT("sans_cible.acp", n_components=2))
        ),
    )
    assert projet.spec.probleme is Probleme.SANS_CIBLE and projet.coffre is None
    texte = aides.script_du_projet(projet, tmp_path / "s").read_text(encoding="utf-8")
    assert (
        "def positions_de_test" not in texte and "EMPREINTE_TEST" not in texte and "def separer" not in texte
    )
    resultats = aides.rejouer_dans_le_processus(projet, tmp_path / "rejeu")
    aides.verifier_validation(projet, resultats)
    aides.verifier_test(projet, resultats)


def test_les_structures_et_les_problemes_se_disent_en_francais(tmp_path: Path) -> None:
    projet = _projet(tmp_path, aides.jeu_chronologique(), aides.spec_chronologique)
    entete = aides.script_du_projet(projet, tmp_path / "s").read_text(encoding="utf-8").split('"""')[1]
    assert "régression de « y »" in entete and "lignes ordonnées dans le temps" in entete
    assert "part de test 0,20" in entete and "3 plis de validation" in entete
    assert Structure.CHRONOLOGIQUE.value not in entete  # pas de valeur d'énumération brute à l'écran


def test_la_perte_de_precision_des_empreintes_est_impossible(tmp_path: Path) -> None:
    """Un octet différent dans une prédiction change l'empreinte ; un pli différent change celle des plis."""
    plis = [(np.arange(5), np.arange(5, 8))]
    autres = [(np.arange(5), np.arange(5, 9))]
    assert codegen.empreinte_plis(plis) != codegen.empreinte_plis(autres)
    assert codegen.empreinte_plis(plis) == codegen.empreinte_plis([(list(range(5)), list(range(5, 8)))])
    assert not math.isnan(len(plis))
