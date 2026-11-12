"""Aides des tests de reproductibilité : comparaison récursive d'estimateurs, exécution du script exporté.

L'égalité de deux pipelines ne se lit ni par `==` ni par `repr` : un DataFrame de contexte (paramètre d'une
branche chronologique) ne se compare pas ainsi. La comparaison descend dans les estimateurs (classe et
paramètres), les conteneurs, les tableaux et les tables.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

import aides_apprentissage as aides
from mecaml.core import audit, codegen, decision, evaluate, split, train
from mecaml.core.io import Chargement, charger
from mecaml.core.session import (
    Acquitter,
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

RACINE = Path(__file__).resolve().parents[1]
SRC = RACINE / "src"
OPT = "-" * 2  # préfixe des options longues, écrit sans double trait d'union littéral


def _meme_classe(a: Any, b: Any) -> bool:
    """Même classe ; une classe du paquet embarquée dans un script n'est reconnue qu'à son nom."""
    ca, cb = type(a), type(b)
    if ca is cb:
        return True
    return ca.__name__ == cb.__name__ and ca.__module__.startswith("mecaml") != cb.__module__.startswith(
        "mecaml"
    )


def differences(a: Any, b: Any, chemin: str = "") -> list[str]:
    """Liste des écarts entre deux valeurs de paramètres, estimateurs compris (vide : égaux)."""
    if hasattr(a, "get_params") and hasattr(b, "get_params"):
        if not _meme_classe(a, b):
            return [f"{chemin} : classes {type(a).__name__} et {type(b).__name__}"]
        return differences(a.get_params(deep=False), b.get_params(deep=False), f"{chemin}{type(a).__name__}")
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        egaux = isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and a.equals(b)
        return [] if egaux else [f"{chemin} : tables différentes"]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        egaux = (
            isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b, equal_nan=True)
        )
        return [] if egaux else [f"{chemin} : tableaux différents"]
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return [f"{chemin} : clés {sorted(map(str, set(a) ^ set(b)))}"]
        return [e for k in a for e in differences(a[k], b[k], f"{chemin}.{k}")]
    if type(a) is not type(b) and not (callable(a) and callable(b)):
        return [f"{chemin} : types {type(a).__name__} et {type(b).__name__}"]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{chemin} : longueurs {len(a)} et {len(b)}"]
        return [
            e for k, (x, y) in enumerate(zip(a, b, strict=True)) for e in differences(x, y, f"{chemin}[{k}]")
        ]
    if isinstance(a, functools.partial):
        return differences(a.func, b.func, chemin) + differences(a.keywords, b.keywords, chemin)
    if callable(a) and callable(b) and not isinstance(a, type):
        return (
            [] if getattr(a, "__name__", None) == getattr(b, "__name__", None) else [f"{chemin} : fonctions"]
        )
    if isinstance(a, float) and math.isnan(a):
        return [] if math.isnan(b) else [f"{chemin} : nan"]
    return [] if a == b else [f"{chemin} : {a!r} et {b!r}"]


def charger_script(chemin: Path, nom: str = "script_exporte") -> ModuleType:
    """Importe le script exporté comme un module (son programme principal ne se lance pas)."""
    spec = importlib.util.spec_from_file_location(nom, chemin)
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module


def lancer_script(chemin: Path, *arguments: str, dossier: Path | None = None, delai: int = 300):
    """Exécute le script dans un processus neuf, sans le paquet mecaml sur le chemin des modules."""
    environnement = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    environnement["PYTHONIOENCODING"] = "utf-8"
    environnement["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(chemin), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environnement,
        cwd=dossier or chemin.parent,
        timeout=delai,
        check=False,
    )


def lire_resultats(dossier: Path, nom: str = "resultats_script.json") -> dict[str, Any]:
    """Résultats écrits par le script (NaN écrit null, relu comme nan)."""

    def nan(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: nan(x) for k, x in v.items()}
        if isinstance(v, list):
            return [nan(x) for x in v]
        return math.nan if v is None else v

    return nan(json.loads((dossier / nom).read_text(encoding="utf-8")))


def ecart_max(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Plus grand écart absolu entre deux tables de scores de mêmes clés (nan égal à nan)."""
    assert set(a) == set(b), (sorted(a), sorted(b))
    pire = 0.0
    for cle in a:
        x, y = a[cle], b[cle]
        if isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            ecarts = [_ecart(u, v) for u, v in zip(x, y, strict=True)]
            pire = max(pire, *ecarts) if ecarts else pire
        else:
            pire = max(pire, _ecart(x, y))
    return pire


def _ecart(x: float, y: float) -> float:
    if math.isnan(x) and math.isnan(y):
        return 0.0
    if math.isnan(x) or math.isnan(y):
        return math.inf
    return abs(x - y)


# ============================================================
# PROJETS DE BOUT EN BOUT SUR DES FICHIERS (jeux d'exemple et jeux canoniques)
# ============================================================
TOLERANCE = 1e-9


@dataclass
class Projet:
    """Un projet mené jusqu'à la décision : tout ce que l'export reçoit."""

    spec: SpecProjet
    table: Table
    chargement: Chargement
    coffre: CoffreTest | None
    decoupage: split.Decoupage | None
    resultat: train.Resultat
    decision: Decision
    fichier: Path | None = None
    fichier_test: Path | None = None


def mener_le_projet(
    chemin: Path,
    fabrique_spec: Callable[[Table], SpecProjet],
    decider: Callable[[train.Resultat], Decision] | None = None,
    fichier_test: Path | None = None,
    methode: split.Methode | None = None,
) -> Projet:
    """Charge le fichier, découpe, entraîne et décide, comme le ferait le parcours de l'application.

    `decider` choisit le modèle et le seuil (par défaut le meilleur modèle, sans seuil) ; `fichier_test`
    met en place un jeu de test fourni à part ; `methode` impose la méthode de découpage (sinon celle
    de la structure).
    """
    chargement = charger(chemin)
    table = chargement.table
    spec = fabrique_spec(table)
    if not spec.probleme.avec_cible:
        resultat = train.entrainer(spec, table)
        choix = (decider or decision.proposer)(resultat)
        return Projet(resultat.spec, table, chargement, None, None, resultat, choix, chemin)
    if fichier_test is not None:
        decoupage, coffre = split.sceller_fichier(table, charger(fichier_test).table, spec)
    else:
        decoupage, coffre = split.sceller(table, spec, methode=methode)
    resultat = train.entrainer(spec, table, coffre)
    choix = (decider or decision.proposer)(resultat)
    return Projet(resultat.spec, table, chargement, coffre, decoupage, resultat, choix, chemin, fichier_test)


CT = ChoixTechnique.creer
STANDARD = (CT("transformation.standard"),)


def spec_acier(table: Table) -> SpecProjet:
    """Régression de la résistance à la traction de l'acier (150 lignes, une modalité)."""
    roles = (
        ("temperature_traitement_thermique_C", Role.NUMERIQUE),
        ("temps_trempe_min", Role.NUMERIQUE),
        ("teneur_carbone_pourcent", Role.NUMERIQUE),
        ("taux_chrome_pourcent", Role.NUMERIQUE),
        ("type_alliage", Role.CATEGORIELLE),
        ("resistance_traction_MPa", Role.CIBLE),
    )
    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="resistance_traction_MPa",
        roles=roles,
        metrique="rmse",
        pretraitement=(CT("manquants.simple", strategy="median"), CT("encodage.one_hot"), *STANDARD),
        modeles=(
            CT("regression.ridge"),
            CT("regression.arbre", max_depth=3),
            CT("regression.foret", n_estimators=20),
        ),
    )


def spec_conformite(table: Table) -> SpecProjet:
    """Classification binaire d'une pièce conforme ou rebut, F1 sur la classe « rebut »."""
    mesures = [c for c in table.colonnes if c not in ("id_piece", "etat_piece")]
    return SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="etat_piece",
        roles=(
            ("id_piece", Role.IDENTIFIANT),
            *((c, Role.NUMERIQUE) for c in mesures),
            ("etat_piece", Role.CIBLE),
        ),
        metrique="f1",
        classe_positive="rebut",
        pretraitement=STANDARD,
        modeles=(CT("classification.logistique"), CT("classification.arbre", max_depth=3)),
    )


def spec_serie(table: Table) -> SpecProjet:
    """Durée de vie restante de machines suivies : unités entières, variables temporelles par machine."""
    capteurs = [
        c
        for c in table.colonnes
        if c not in ("ID_Machine", "Cycle_Operation_Heures", "Duree_Vie_Restante_Cycles")
    ]
    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="Duree_Vie_Restante_Cycles",
        roles=(
            ("ID_Machine", Role.GROUPE),
            ("Cycle_Operation_Heures", Role.DATE),
            *((c, Role.NUMERIQUE) for c in capteurs),
            ("Duree_Vie_Restante_Cycles", Role.CIBLE),
        ),
        metrique="rmse",
        structure=Structure.GROUPEE_CHRONOLOGIQUE,
        n_plis=3,
        pretraitement=(
            CT("chronologique.variables_temporelles", retards=(1, 5), fenetres=(5,), demi_vies=(3.0,)),
            *STANDARD,
        ),
        modeles=(CT("regression.ridge"), CT("regression.arbre", max_depth=4)),
    )


def spec_pompe(table: Table) -> SpecProjet:
    """Sans cible : régimes de fonctionnement et cycles anormaux d'une pompe."""
    return SpecProjet(
        probleme=Probleme.SANS_CIBLE,
        roles=tuple((c, Role.NUMERIQUE) for c in table.colonnes),
        metrique="silhouette",
        pretraitement=STANDARD,
        modeles=(
            CT("sans_cible.kmeans", n_clusters=3),
            CT("sans_cible.melange_gaussien", n_components=3),
            CT("sans_cible.dbscan", eps=0.6, min_samples=5),
            CT("sans_cible.acp", n_components=2),
            CT("sans_cible.foret_isolement", contamination=0.05),
            CT("sans_cible.facteur_local", contamination=0.05),
        ),
    )


def spec_rul(table: Table) -> SpecProjet:
    """Durée de vie restante d'un équipement (240 lignes), une modalité et quatre mesures."""
    mesures = [c for c in table.colonnes if c not in ("niveau_sollicitation_thermique_cat", "RUL_jours")]
    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="RUL_jours",
        roles=(
            ("niveau_sollicitation_thermique_cat", Role.CATEGORIELLE),
            *((c, Role.NUMERIQUE) for c in mesures),
            ("RUL_jours", Role.CIBLE),
        ),
        metrique="mae",
        pretraitement=(CT("encodage.one_hot"), *STANDARD),
        modeles=(CT("regression.ridge"), CT("regression.gradient_histogramme", max_iter=30)),
    )


def spec_soudure(table: Table) -> SpecProjet:
    """Qualité d'une soudure : classification binaire, valeurs manquantes et booléens dans les variables."""
    from mecaml.core import audit

    proposes = audit.proposer_roles(table, "qualite_soudure")
    roles = (
        *((nom, role) for nom, role in proposes if nom != "controle_visuel_resultat"),
        ("controle_visuel_resultat", Role.IGNOREE),
    )
    roles = tuple(sorted({nom: role for nom, role in roles}.items()))
    return SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="qualite_soudure",
        roles=roles,
        metrique="f1",
        classe_positive="non-conforme",
        pretraitement=(
            CT("manquants.simple", strategy="median"),
            CT(
                "manquants.simple",
                strategy="most_frequent",
                colonnes=tuple(c for c, r in roles if r is Role.CATEGORIELLE),
            ),
            CT("encodage.one_hot"),
            *STANDARD,
        ),
        modeles=(CT("classification.logistique"), CT("classification.gradient_histogramme", max_iter=30)),
    )


def spec_serie_complete(table: Table) -> SpecProjet:
    """Série complète de cent machines : comme `spec_serie`, avec des lignes sans cible en contexte."""
    return spec_serie(table).avec(
        pretraitement=(CT("chronologique.variables_temporelles", retards=(1, 3), fenetres=(5,)), *STANDARD),
        modeles=(CT("regression.ridge"), CT("regression.arbre", max_depth=4)),
    )


# ============================================================
# COMPARAISON DU SCRIPT ET DE L'APPLICATION
# ============================================================
def verifier_validation(projet: Projet, resultats: Mapping[str, Any], tolerance: float = TOLERANCE) -> None:
    """Empreintes et scores de validation, pli par pli, du script contre ceux de l'application."""
    assert resultats["graine"] == projet.spec.graine
    assert resultats["empreintes"]["donnees"] == projet.table.empreinte
    if projet.spec.probleme.avec_cible:
        assert resultats["empreintes"]["test"] == projet.decoupage.empreinte_sha256
        assert resultats["empreintes"]["plis"] == codegen.empreinte_plis(projet.resultat.plis)
    assert set(resultats["validation"]) == {m.id for m in projet.resultat.modeles}
    for modele in projet.resultat.modeles:
        attendus = {nom: list(valeurs) for nom, valeurs in modele.scores_plis.items()}
        assert ecart_max(attendus, resultats["validation"][modele.id]) <= tolerance, modele.id


def verifier_test(projet: Projet, resultats: Mapping[str, Any], tolerance: float = TOLERANCE) -> None:
    """Décision, scores de test et prédictions du script contre l'évaluation de l'application."""
    assert resultats["decision"]["modele"] == projet.decision.modele_id
    if not projet.spec.probleme.avec_cible:
        return
    seuil = resultats["decision"]["seuil"]
    assert (seuil is None or math.isnan(seuil)) == (projet.decision.seuil is None)
    if projet.decision.seuil is not None:
        assert seuil == projet.decision.seuil
    evaluer = evaluate.preparer_test(
        projet.spec, projet.table, projet.coffre, projet.decision, importances=False
    )
    attendu = evaluer(projet.coffre.ouvrir(projet.decision))
    obtenu = resultats["test"]
    assert ecart_max(attendu.scores, obtenu["scores"]) <= tolerance
    assert obtenu["n_lignes"] == attendu.n_lignes
    assert obtenu["n_sans_cible"] == attendu.n_sans_cible
    assert obtenu["empreinte_predictions"] == attendu.empreinte_predictions


def script_du_projet(projet: Projet, dossier: Path) -> Path:
    """Script engendré pour le projet, écrit dans le dossier (créé au besoin)."""
    dossier.mkdir(parents=True, exist_ok=True)
    texte = codegen.generer_script(
        projet.spec,
        projet.table,
        projet.coffre,
        decoupage=projet.decoupage,
        decision=projet.decision,
        chargement=projet.chargement,
    )
    return codegen.ecrire_script(texte, dossier)


def rejouer_dans_le_processus(projet: Projet, dossier: Path, *options: str) -> dict[str, Any]:
    """Le script du projet, importé puis lancé dans ce processus (rapide) ; ses résultats sont lus.

    Les jeux d'exemple et canoniques passent par un processus neuf (test I8) ; cette voie sert la matrice des
    méthodes de découpage et de validation, où l'on ne cherche que l'équivalence des calculs.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    script = script_du_projet(projet, dossier)
    module = charger_script(script, f"script_{dossier.name}")
    arguments = [f"{OPT}donnees", str(projet.fichier), f"{OPT}sortie", str(dossier)]
    if projet.fichier_test is not None:
        arguments += [f"{OPT}test", str(projet.fichier_test)]
    module.main([*arguments, *options])
    return lire_resultats(dossier)


# ============================================================
# JEUX SYNTHÉTIQUES ÉCRITS EN FICHIER
# ============================================================
def ecrire_csv(
    df: pd.DataFrame, chemin: Path, *, sep: str = ",", decimal: str = ".", encoding: str = "utf-8"
) -> Path:
    """Jeu de données écrit en `.csv`, que `charger` relit comme l'application le ferait."""
    df.to_csv(chemin, index=False, sep=sep, decimal=decimal, encoding=encoding)
    return chemin


def jeu_pannes(n_machines: int = 20, longueur: int = 12, graine: int = 0) -> pd.DataFrame:
    """Machines suivies : la moitié tombe en panne à leurs derniers cycles, l'autre marche toujours."""
    df = aides.jeu_groupe(n_groupes=n_machines, longueur=longueur, graine=graine)
    en_panne = df["machine"].isin([f"M{k:02d}" for k in range(n_machines // 2)]) & (df["y"] <= 3)
    df["y"] = np.where(en_panne, "panne", "marche")
    return df


def jeu_chronologique(n: int = 160, graine: int = 0) -> pd.DataFrame:
    """Une seule série : date, deux mesures, cible qui dépend des mesures et d'une dérive lente."""
    rng = np.random.default_rng(graine)
    t = np.arange(n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2, "y": 2.0 * x1 - x2 + 0.05 * t + rng.normal(0, 0.2, n)})


def jeu_serie_lacunaire(graine: int = 0) -> pd.DataFrame:
    """Série mesurée dont quelques cibles manquent en cours de série (contexte des variables temporelles)."""
    df = aides.jeu_serie_mesuree(n_groupes=8, longueur=40, graine=graine)
    df.loc[df.index[[7, 8, 61, 150, 151, 152, 200]], "y"] = np.nan
    return df


def jeu_defauts_entiers(graine: int = 0) -> pd.DataFrame:
    """Mesures de pièces dont le défaut est codé 1 et le conforme 0."""
    df = aides.jeu_classification(n=240, graine=graine)
    df["y"] = (df["y"] == "defaut").astype(int)
    return df


def spec_defauts_entiers(table: Table) -> SpecProjet:
    """Classification binaire dont la cible vaut 0 ou 1 : les prédictions gardent le type entier."""
    return SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="y",
        roles=(
            ("a", Role.NUMERIQUE),
            ("b", Role.NUMERIQUE),
            ("c", Role.CATEGORIELLE),
            ("y", Role.CIBLE),
        ),
        metrique="f1",
        classe_positive=1,
        pretraitement=(CT("manquants.simple", strategy="median"), CT("encodage.one_hot"), *STANDARD),
        modeles=(CT("classification.logistique"), CT("classification.arbre", max_depth=3)),
    )


def jeu_chronologique_entier(n: int = 160, graine: int = 0) -> pd.DataFrame:
    """Une série ordonnée dont la cible vaut 1 quand la grandeur dépasse sa médiane, 0 sinon."""
    df = jeu_chronologique(n, graine)
    df["y"] = (df["y"] > df["y"].median()).astype(int)
    return df


def spec_chronologique_entiere(table: Table) -> SpecProjet:
    """Classification d'une série ordonnée : fenêtre croissante, premières dates jamais validées."""
    return SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="y",
        roles=(
            ("t", Role.DATE),
            ("x1", Role.NUMERIQUE),
            ("x2", Role.NUMERIQUE),
            ("y", Role.CIBLE),
        ),
        metrique="f1",
        classe_positive=1,
        structure=Structure.CHRONOLOGIQUE,
        n_plis=3,
        pretraitement=STANDARD,
        modeles=(CT("classification.logistique"), CT("classification.arbre", max_depth=3)),
    )


def _roles_serie_mesuree() -> tuple[tuple[str, Role], ...]:
    return (
        ("machine", Role.GROUPE),
        ("cycle", Role.DATE),
        ("x1", Role.NUMERIQUE),
        ("x2", Role.NUMERIQUE),
        ("y", Role.CIBLE),
    )


def spec_persistance(table: Table) -> SpecProjet:
    """Température relevée à chaque cycle : la cible du pas précédent est connue, le projet le déclare."""
    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="y",
        roles=_roles_serie_mesuree(),
        metrique="rmse",
        structure=Structure.GROUPEE_CHRONOLOGIQUE,
        n_plis=3,
        pretraitement=(CT("chronologique.cible_decalee", retards=(1, 2), cible_connue=True), *STANDARD),
        modeles=(CT("reference.persistance", cible_connue=True), CT("regression.ridge")),
    )


def spec_lacunaire(table: Table) -> SpecProjet:
    """Série dont des cibles manquent : les lignes sans cible complètent les séries (contexte)."""
    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="y",
        roles=_roles_serie_mesuree(),
        metrique="rmse",
        structure=Structure.GROUPEE_CHRONOLOGIQUE,
        n_plis=3,
        pretraitement=(
            CT("chronologique.variables_temporelles", retards=(1, 2), fenetres=(5,), demi_vies=(3.0,)),
            *STANDARD,
        ),
        modeles=(CT("regression.ridge"), CT("regression.arbre", max_depth=4)),
    )


def spec_classification_groupee(table: Table) -> SpecProjet:
    """Pannes de machines suivies : unités entières, classe rare « panne » à surveiller."""
    return SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="y",
        roles=(
            ("machine", Role.GROUPE),
            ("x1", Role.NUMERIQUE),
            ("x2", Role.NUMERIQUE),
            ("y", Role.CIBLE),
        ),
        metrique="f1",
        classe_positive="panne",
        structure=Structure.GROUPEE,
        n_plis=3,
        pretraitement=STANDARD,
        modeles=(CT("classification.logistique"), CT("classification.arbre", max_depth=3)),
    )


def spec_chronologique(table: Table) -> SpecProjet:
    """Une seule série ordonnée par la date : coupure entre deux dates, fenêtre croissante en validation."""
    return SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="y",
        roles=(
            ("t", Role.DATE),
            ("x1", Role.NUMERIQUE),
            ("x2", Role.NUMERIQUE),
            ("y", Role.CIBLE),
        ),
        metrique="rmse",
        structure=Structure.CHRONOLOGIQUE,
        n_plis=3,
        pretraitement=STANDARD,
        modeles=(CT("regression.ridge"), CT("regression.arbre", max_depth=3)),
    )


# ============================================================
# SESSION MENÉE JUSQU'AU BOUT
# ============================================================
def mener_la_session(
    projet: Projet,
    *,
    auditeur: Callable[..., Any] = audit.auditer,
    importances: bool = False,
    jusqu_au_test: bool = True,
) -> Session:
    """Session menée de S0 à S8 (S7 sans cible) par les actions de l'application, alertes acquittées.

    L'auditeur est l'audit réel par défaut ; un constat bloquant arrête la chaîne comme dans l'application.
    """
    session = Session(auditeur=auditeur).appliquer(Charger(projet.table)).appliquer(Auditer())
    session = session.appliquer(FixerObjectif(projet.spec)).appliquer(
        Decouper(projet.coffre, spec=projet.spec)
    )
    alertes = frozenset(c.code for c in session.alertes_en_attente())
    if alertes:
        session = session.appliquer(Acquitter(alertes))
    session = session.appliquer(Pretraiter(projet.spec))
    session = session.appliquer(Entrainer(projet.resultat, spec=projet.resultat.spec))
    session = session.appliquer(Decider(projet.decision))
    if projet.spec.probleme.avec_cible and jusqu_au_test:
        evaluer = evaluate.preparer_test(
            projet.spec, projet.table, projet.coffre, projet.decision, importances=importances
        )
        session = session.appliquer(Tester(evaluer))
    return session
