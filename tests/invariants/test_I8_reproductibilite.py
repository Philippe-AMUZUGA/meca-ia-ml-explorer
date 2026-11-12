"""I8 : reproductibilité. Le script exporté, lancé dans un processus neuf sans le paquet, rejoue les scores.

Quatre jeux d'exemple (régression, classification, série chronologique groupée, sans cible) sont
toujours joués ; les jeux canoniques correspondants le sont quand MECAML_JEUX désigne leur dossier.
Quatre jeux synthétiques écrits en `.csv` français (point-virgule, virgule décimale) complètent : une
température dont la cible du pas précédent est déclarée connue, une série dont des cibles manquent, et deux
cibles codées en entiers (0 et 1), l'une en lignes indépendantes, l'autre en série chronologique.

Pour chaque jeu :

- les pipelines du script ont les mêmes classes et les mêmes paramètres que ceux de l'application
  (comparaison récursive : estimateurs, tableaux, tables) ;
- le script, exécuté en sous-processus, retrouve les trois empreintes (données, test, plis) et rend les mêmes
  scores de validation, par pli, et les mêmes scores de test, à moins de 1e-9, avec les mêmes prédictions
  à l'octet ;
- il n'importe pas le paquet.

Les scripts de tous les jeux demandés se lancent en parallèle, pendant que l'application mène les suivants.
"""

from __future__ import annotations

import ast
import dataclasses
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import aides_apprentissage as aides_jeux
import aides_reproductibilite as aides
from mecaml.core import codegen, report, train
from mecaml.core.types import ChoixTechnique, SpecProjet

CT = ChoixTechnique.creer
RACINE = Path(__file__).resolve().parents[2]
EXEMPLES = RACINE / "examples"
JEUX = Path(os.environ["MECAML_JEUX"]) if os.environ.get("MECAML_JEUX") else None
IMPORTS_PERMIS = {
    "__future__",
    "argparse",
    "collections",
    "dataclasses",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "math",
    "numpy",
    "pandas",
    "pathlib",
    "re",
    "sklearn",
    "sys",
    "typing",
    "warnings",
}

CAS_EXEMPLES = {
    "acier": ("MecaIA-Traitement_Acier.xlsx", aides.spec_acier),
    "conformite": ("MecaIA-Controle_Conformite.xlsx", aides.spec_conformite),
    "serie": ("MecaIA-TimeSeries_RUL_extrait.xlsx", aides.spec_serie),
    "pompe": ("MecaIA-Anomalies_Pompe.xlsx", aides.spec_pompe),
}
CAS_CANONIQUES = {
    "RUL": ("MecaIA-RUL.xlsx", aides.spec_rul),
    "Qualite_soudure": ("MecaIA-Qualite_soudure.xlsx", aides.spec_soudure),
    "Anomalies_Pompe": ("MecaIA-Anomalies_Pompe.xlsx", aides.spec_pompe),
    "TimeSeries_RUL": ("MecaIA-TimeSeries_RUL.xlsx", aides.spec_serie_complete),
}
CAS_SYNTHETIQUES = {
    "persistance": (aides_jeux.jeu_serie_mesuree, aides.spec_persistance),
    "lacunaire": (aides.jeu_serie_lacunaire, aides.spec_lacunaire),
    "cible_entiere": (aides.jeu_defauts_entiers, aides.spec_defauts_entiers),
    "cible_entiere_chronologique": (aides.jeu_chronologique_entier, aides.spec_chronologique_entiere),
}
TOUS = (
    [pytest.param(("exemple", n), id=f"exemple_{n}") for n in CAS_EXEMPLES]
    + [pytest.param(("canonique", n), id=f"canonique_{n}") for n in CAS_CANONIQUES]
    + [pytest.param(("synthetique", n), id=f"synthetique_{n}") for n in CAS_SYNTHETIQUES]
)


def _fichier_du_cas(famille: str, nom: str) -> tuple[Path, Callable[..., SpecProjet]]:
    """Fichier et fabrique de spécification d'un cas de fichier, ou saut du test si le fichier manque."""
    fichier, fabrique = (CAS_EXEMPLES if famille == "exemple" else CAS_CANONIQUES)[nom]
    dossier = EXEMPLES if famille == "exemple" else JEUX
    if dossier is None or not (dossier / fichier).is_file():
        pytest.skip(f"{fichier} introuvable (MECAML_JEUX)")
    return dossier / fichier, fabrique


class Rejeux:
    """Projets menés dans l'application, scripts engendrés et lancés en parallèle dans des processus neufs."""

    def __init__(self, racine: Path) -> None:
        self.racine = racine
        self.projets: dict[tuple[str, str], tuple[aides.Projet, Path]] = {}
        self.executions: dict[tuple[str, str], Future] = {}
        self.pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2))

    def _fichier(self, famille: str, nom: str) -> tuple[Path, Callable[..., SpecProjet]]:
        if famille == "synthetique":
            fabrique_jeu, fabrique_spec = CAS_SYNTHETIQUES[nom]
            dossier = self.racine / f"donnees_{nom}"
            dossier.mkdir(exist_ok=True)
            return aides.ecrire_csv(
                fabrique_jeu(), dossier / f"{nom}.csv", sep=";", decimal=","
            ), fabrique_spec
        return _fichier_du_cas(famille, nom)

    def lancer(self, cle: tuple[str, str]) -> None:
        if cle in self.projets:
            return
        chemin, fabrique = self._fichier(*cle)
        projet = aides.mener_le_projet(chemin, fabrique)
        dossier = self.racine / f"{cle[0]}_{cle[1]}"
        dossier.mkdir()
        script = aides.script_du_projet(projet, dossier)
        self.projets[cle] = (projet, script)
        self.executions[cle] = self.pool.submit(
            aides.lancer_script,
            script,
            f"{aides.OPT}donnees",
            str(chemin),
            f"{aides.OPT}sortie",
            str(dossier),
        )

    def obtenir(self, cle: tuple[str, str]) -> tuple[aides.Projet, Path, dict[str, Any]]:
        self.lancer(cle)
        projet, script = self.projets[cle]
        execution = self.executions[cle].result()
        assert execution.returncode == 0, execution.stderr[-2500:]
        return projet, script, aides.lire_resultats(script.parent)


@pytest.fixture(scope="module")
def rejeux(tmp_path_factory: pytest.TempPathFactory) -> Rejeux:
    return Rejeux(tmp_path_factory.mktemp("i8"))


@pytest.fixture
def cas(request: pytest.FixtureRequest, rejeux: Rejeux) -> tuple[aides.Projet, Path, dict[str, Any]]:
    # Les jeux demandés par la session se lancent ensemble : le premier appel les prépare tous.
    for item in request.session.items:
        parametres = getattr(getattr(item, "callspec", None), "params", {})
        if item.module is request.module and "cas" in parametres:
            try:
                rejeux.lancer(parametres["cas"])
            except pytest.skip.Exception:
                continue
    return rejeux.obtenir(request.param)


# ============================================================
# PIPELINES : MÊMES CLASSES, MÊMES PARAMÈTRES
# ============================================================
@pytest.mark.parametrize("cas", TOUS, indirect=True)
def test_I8_les_pipelines_du_script_ont_les_memes_parametres(cas) -> None:
    projet, script, _ = cas
    module = aides.charger_script(script, f"script_{script.parent.name}")
    spec = projet.spec
    contexte = codegen.contexte_d_apprentissage(spec, projet.table, projet.coffre)
    for choix in spec.modeles:
        voulu = train.construire_pipeline(spec, choix, contexte=contexte)
        obtenu = module.construire_pipeline(choix.id, contexte)
        assert aides.differences(voulu, obtenu) == [], choix.id


# ============================================================
# SCORES : DANS UN PROCESSUS NEUF, À 1e-9
# ============================================================
@pytest.mark.parametrize("cas", TOUS, indirect=True)
def test_I8_le_script_retrouve_les_empreintes_et_les_scores_de_validation(cas) -> None:
    projet, _, resultats = cas
    aides.verifier_validation(projet, resultats)


@pytest.mark.parametrize("cas", TOUS, indirect=True)
def test_I8_le_script_retrouve_le_score_de_test_et_les_predictions(cas) -> None:
    projet, _, resultats = cas
    aides.verifier_test(projet, resultats)


@pytest.mark.parametrize("cas", TOUS, indirect=True)
def test_I8_le_script_est_autonome(cas) -> None:
    _, script, _ = cas
    arbre = ast.parse(script.read_text(encoding="utf-8"))
    importes = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
        for n in ast.walk(arbre)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert "mecaml" not in importes
    assert importes <= IMPORTS_PERMIS, importes - IMPORTS_PERMIS


@pytest.mark.parametrize("cas", TOUS, indirect=True)
def test_I8_le_projet_du_script_est_celui_de_la_specification(cas) -> None:
    projet, script, _ = cas
    module = aides.charger_script(script, f"projet_{script.parent.name}")
    assert SpecProjet.depuis_dict(module.PROJET) == projet.spec
    assert projet.spec.graine == module.GRAINE


# ============================================================
# CE QUE LES JEUX SYNTHÉTIQUES PROUVENT EN PLUS
# ============================================================
def test_I8_la_cible_du_pas_precedent_declaree_est_gardee_par_le_script(rejeux: Rejeux) -> None:
    projet, script, resultats = rejeux.obtenir(("synthetique", "persistance"))
    texte = script.read_text(encoding="utf-8")
    assert "cible_connue" in texte  # le projet gardé dans le script porte la déclaration
    assert "Hypothèse déclarée par le projet : la cible du pas précédent est connue au moment de la" in texte
    assert resultats["decision"]["modele"] == projet.decision.modele_id
    assert {c.id for c in projet.spec.modeles} >= {"reference.persistance", "reference.dummy_regression"}


def test_I8_le_contexte_des_lignes_sans_cible_est_reconstruit_a_l_identique(rejeux: Rejeux) -> None:
    projet, script, _ = rejeux.obtenir(("synthetique", "lacunaire"))
    contexte = codegen.contexte_d_apprentissage(projet.spec, projet.table, projet.coffre)
    assert contexte is not None and len(contexte) > 0  # des lignes d'apprentissage sans cible existent
    assert "contexte_de(contexte" in script.read_text(encoding="utf-8")
    assert projet.resultat.n_sans_cible == len(contexte)


@pytest.mark.parametrize(
    ("famille", "nom"),
    [
        pytest.param("exemple", "serie", id="exemple_serie"),
        pytest.param("canonique", "TimeSeries_RUL", id="canonique_TimeSeries_RUL"),
    ],
)
def test_I8_une_duree_de_vie_restante_declaree_connue_n_est_pas_exportee(
    famille: str, nom: str, tmp_path: Path
) -> None:
    """Un compte à rebours ne devient pas une variable connue parce qu'on le déclare : l'export est refusé."""
    chemin, fabrique = _fichier_du_cas(famille, nom)
    projet = aides.mener_le_projet(chemin, fabrique)
    session = aides.mener_la_session(projet)
    declaree = projet.spec.avec(
        modeles=(CT("reference.dummy_regression"), CT("reference.persistance", cible_connue=True))
    )
    with pytest.raises(codegen.ErreurExport, match="compte à rebours"):
        codegen.generer_script(declaree, projet.table, projet.coffre)
    falsifiee = dataclasses.replace(session, spec=declaree)  # une interface qui contournerait les règles
    with pytest.raises(codegen.ErreurExport, match="compte à rebours"):
        report.exporter(falsifiee, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_I8_la_comparaison_recursive_voit_une_difference() -> None:
    from sklearn.linear_model import Ridge

    from mecaml.core.catalog import entree

    ridge = entree("regression.ridge").construire(42)
    assert aides.differences(ridge, Ridge(random_state=42)) == []
    assert aides.differences(ridge, Ridge(random_state=7))
    assert aides.differences(ridge, Ridge(alpha=2.0, random_state=42))
    assert aides.differences({"a": float("nan")}, {"a": float("nan")}) == []


def test_I8_la_comparaison_recursive_voit_une_table_ou_un_tableau_different() -> None:
    """Un contexte (DataFrame) ou un tableau qui change doit se voir : ni `==` ni `repr` ne le disent."""
    import numpy as np
    import pandas as pd

    table = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert aides.differences({"c": table}, {"c": table.copy()}) == []
    assert aides.differences({"c": table}, {"c": table.assign(x=[1.0, 2.0, 4.0])})
    assert aides.differences({"c": table}, {"c": np.array([1.0, 2.0, 3.0])})
    tableau = np.arange(5.0)
    assert aides.differences({"t": tableau}, {"t": tableau.copy()}) == []
    assert aides.differences({"t": tableau}, {"t": tableau + 1e-12})
    assert aides.differences({"t": tableau}, {"t": list(tableau)})
