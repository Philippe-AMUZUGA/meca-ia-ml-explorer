"""Cas dorés : quatre jeux d'exemple, scores de validation croisée figés dans `attendus.json`.

Les jeux sont ceux de `examples/` (ou du dossier désigné par la variable d'environnement MECAML_JEUX).
Les scores attendus dépendent des versions figées dans `requirements.txt` ; après une montée de version
voulue, les régénérer avec la variable MECAML_REGENERER_DORES=1 puis relire la différence du fichier.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core import train
from mecaml.core.types import ChoixTechnique, CoffreTest, Probleme, Role, SpecProjet, Structure, Table

CT = ChoixTechnique.creer
RACINE = Path(__file__).resolve().parents[2]
ATTENDUS = Path(__file__).with_name("attendus.json")
REGENERER = bool(os.environ.get("MECAML_REGENERER_DORES"))
METRIQUES_RETENUES = {
    "regression": ("rmse", "r2", "mae"),
    "classification": ("f1", "rappel", "precision", "roc_auc", "exactitude"),
    "serie": ("rmse", "r2"),
    "sans_cible": ("silhouette", "taux_anomalies", "variance_expliquee"),
}


def lire(nom: str) -> pd.DataFrame:
    """Classeur d'exemple : `MECAML_JEUX` d'abord, puis `examples/` ; le test est ignoré s'il manque."""
    for dossier in (os.environ.get("MECAML_JEUX"), RACINE / "examples"):
        if dossier and (Path(dossier) / nom).is_file():
            return pd.read_excel(Path(dossier) / nom)
    pytest.skip(f"{nom} introuvable (examples/ ou MECAML_JEUX)")


# ============================================================
# LES QUATRE CAS
# ============================================================
def cas_acier() -> train.Resultat:
    """Régression : résistance à la traction selon le traitement thermique (150 lignes)."""
    df = lire("MecaIA-Traitement_Acier.xlsx")
    roles = (
        ("temperature_traitement_thermique_C", Role.NUMERIQUE),
        ("temps_trempe_min", Role.NUMERIQUE),
        ("teneur_carbone_pourcent", Role.NUMERIQUE),
        ("taux_chrome_pourcent", Role.NUMERIQUE),
        ("type_alliage", Role.CATEGORIELLE),
        ("resistance_traction_MPa", Role.CIBLE),
    )
    spec = SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="resistance_traction_MPa",
        roles=roles,
        metrique="rmse",
        graine=42,
        pretraitement=aides.PRETRAITEMENT_STANDARD,
        modeles=(
            CT("regression.ridge"),
            CT("regression.arbre", max_depth=3),
            CT("regression.foret", n_estimators=50),
            CT("regression.plus_proches_voisins"),
        ),
    )
    table = Table(df, "acier")
    return train.entrainer(spec, table, aides.coffre_par_pas(table, spec))


def cas_conformite() -> train.Resultat:
    """Classification binaire : pièce conforme ou rebut selon 30 descripteurs de défaut (569 lignes)."""
    df = lire("MecaIA-Controle_Conformite.xlsx")
    mesures = [c for c in df.columns if c not in ("id_piece", "etat_piece")]
    roles = (
        ("id_piece", Role.IDENTIFIANT),
        *((c, Role.NUMERIQUE) for c in mesures),
        ("etat_piece", Role.CIBLE),
    )
    spec = SpecProjet(
        probleme=Probleme.CLASSIFICATION,
        cible="etat_piece",
        roles=roles,
        metrique="f1",
        classe_positive="rebut",
        graine=42,
        pretraitement=(CT("transformation.standard"),),
        modeles=(
            CT("classification.logistique"),
            CT("classification.arbre", max_depth=3),
            CT("classification.foret", n_estimators=50),
        ),
    )
    table = Table(df, "conformite")
    return train.entrainer(spec, table, aides.coffre_par_pas(table, spec))


def _spec_serie(df: pd.DataFrame, **changements: object) -> SpecProjet:
    capteurs = [
        c
        for c in df.columns
        if c not in ("ID_Machine", "Cycle_Operation_Heures", "Duree_Vie_Restante_Cycles")
    ]
    roles = (
        ("ID_Machine", Role.GROUPE),
        ("Cycle_Operation_Heures", Role.DATE),
        *((c, Role.NUMERIQUE) for c in capteurs),
        ("Duree_Vie_Restante_Cycles", Role.CIBLE),
    )
    base = {
        "probleme": Probleme.REGRESSION,
        "cible": "Duree_Vie_Restante_Cycles",
        "roles": roles,
        "metrique": "rmse",
        "structure": Structure.GROUPEE_CHRONOLOGIQUE,
        "n_plis": 3,
        "graine": 42,
        "pretraitement": (CT("transformation.standard"),),
        "modeles": (CT("regression.ridge"), CT("regression.arbre", max_depth=6)),
        **changements,
    }
    return SpecProjet(**base)


def _entrainer_serie(**changements: object) -> train.Resultat:
    df = lire("MecaIA-TimeSeries_RUL_extrait.xlsx")
    spec = _spec_serie(df, **changements)
    table = Table(df, "serie")
    positions = np.flatnonzero(df["ID_Machine"].isin([17, 18, 19, 20]).to_numpy())
    return train.entrainer(
        spec, table, CoffreTest.par_positions(table, positions, decoupage=spec.decoupage())
    )


def cas_serie() -> train.Resultat:
    """Série chronologique par machine : durée de vie restante, plis par machines entières (20 machines)."""
    return _entrainer_serie()


def cas_serie_retards() -> train.Resultat:
    """Même série avec retards, fenêtre glissante et moyenne exponentielle construits par machine."""
    temporelles = CT("chronologique.variables_temporelles", retards=(1, 5), fenetres=(5,), demi_vies=(3.0,))
    return _entrainer_serie(pretraitement=(temporelles, CT("transformation.standard")))


def cas_pompe() -> train.Resultat:
    """Sans cible : regimes de fonctionnement et cycles anormaux d'une pompe (276 cycles)."""
    df = lire("MecaIA-Anomalies_Pompe.xlsx")
    roles = tuple((c, Role.NUMERIQUE) for c in df.columns)
    spec = SpecProjet(
        probleme=Probleme.SANS_CIBLE,
        roles=roles,
        metrique="silhouette",
        graine=42,
        pretraitement=(CT("transformation.standard"),),
        modeles=(
            CT("sans_cible.kmeans", n_clusters=3),
            CT("sans_cible.melange_gaussien", n_components=3),
            CT("sans_cible.dbscan", eps=0.6, min_samples=5),
            CT("sans_cible.acp", n_components=2),
            CT("sans_cible.foret_isolement", contamination=0.05),
            CT("sans_cible.facteur_local", contamination=0.05),
        ),
    )
    return train.entrainer(spec, Table(df, "pompe"))


CAS = {
    "acier": (cas_acier, "regression"),
    "conformite": (cas_conformite, "classification"),
    "serie": (cas_serie, "serie"),
    "serie_retards": (cas_serie_retards, "serie"),
    "pompe": (cas_pompe, "sans_cible"),
}


def resume(resultat: train.Resultat, famille: str) -> dict[str, dict[str, float]]:
    """Moyenne par pli des métriques retenues, pour chaque modèle ajusté."""
    sortie: dict[str, dict[str, float]] = {}
    for m in resultat.modeles:
        valeurs = {nom: m.moyenne(nom) for nom in METRIQUES_RETENUES[famille] if nom in m.scores_plis}
        sortie[m.id] = {nom: (None if math.isnan(v) else float(v)) for nom, v in valeurs.items()}
    return sortie


@pytest.fixture(scope="module")
def calculs() -> dict[str, tuple[train.Resultat, dict]]:
    return {}


def _calculer(calculs: dict, nom: str) -> tuple[train.Resultat, dict]:
    if nom not in calculs:
        fabrique, famille = CAS[nom]
        resultat = fabrique()
        calculs[nom] = (resultat, resume(resultat, famille))
    return calculs[nom]


def _attendus() -> dict:
    return json.loads(ATTENDUS.read_text(encoding="utf-8")) if ATTENDUS.is_file() else {}


@pytest.fixture(scope="module", autouse=True)
def _prechauffage() -> None:
    train.prechauffer()


# ============================================================
# COMPARAISON AUX VALEURS FIGÉES
# ============================================================
@pytest.mark.parametrize("nom", sorted(CAS))
def test_scores_figes(nom: str, calculs: dict) -> None:
    resultat, obtenu = _calculer(calculs, nom)
    if REGENERER:
        attendus = _attendus()
        attendus[nom] = obtenu
        ATTENDUS.write_text(
            json.dumps(attendus, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        pytest.skip("valeurs régénérées")
    attendu = _attendus().get(nom)
    assert attendu is not None, f"aucune valeur figée pour {nom} : régénérer avec MECAML_REGENERER_DORES=1"
    assert set(obtenu) == set(attendu), "les modèles comparés ont changé"
    for modele, scores in attendu.items():
        assert set(obtenu[modele]) == set(scores), modele
        for metrique, valeur in scores.items():
            if (
                valeur is None
            ):  # mesure indéfinie (précision d'un modèle qui ne prédit jamais la classe positive)
                assert obtenu[modele][metrique] is None, f"{nom} {modele} {metrique}"
            else:
                assert obtenu[modele][metrique] == pytest.approx(valeur, rel=1e-6, abs=1e-9), (
                    f"{nom} {modele} {metrique}"
                )
    assert all(m.ok for m in resultat.modeles), [m.erreur for m in resultat.modeles if not m.ok]


# ============================================================
# PROPRIÉTÉS QUI NE DÉPENDENT PAS DES VALEURS EXACTES
# ============================================================
def test_acier_les_modeles_battent_la_reference(calculs: dict) -> None:
    resultat, _ = _calculer(calculs, "acier")
    assert resultat.reference().id == "reference.dummy_regression"
    assert resultat.validation == "validation.kfold" and len(resultat.plis) == 5
    for id_modele in (
        "regression.ridge",
        "regression.arbre",
        "regression.foret",
        "regression.plus_proches_voisins",
    ):
        assert resultat.gain(id_modele) > 0, id_modele
    assert resultat.classement()[-1].id == "reference.dummy_regression"


def test_conformite_l_identifiant_n_entre_pas_et_les_modeles_battent_la_reference(calculs: dict) -> None:
    resultat, _ = _calculer(calculs, "conformite")
    assert "id_piece" not in resultat.spec.variables()
    assert resultat.classe_positive == "rebut" and resultat.classe_positive_source == "specification"
    assert resultat.validation == "validation.kfold_stratifie"
    reference = resultat.reference()
    assert reference.moyenne("f1") == 0.0 and reference.moyenne("roc_auc") == pytest.approx(0.5)
    assert (
        math.isnan(reference.moyenne("precision")) and reference.plis_valides("precision") == 0
    )  # jamais positive
    logistique = resultat.modele("classification.logistique")
    assert logistique.moyenne("f1") > 0.9 and logistique.moyenne("roc_auc") > 0.95
    assert resultat.gain("classification.logistique") > 0.9


def test_serie_plis_par_machines_et_variables_temporelles(calculs: dict) -> None:
    resultat, _ = _calculer(calculs, "serie")
    assert resultat.validation == "validation.plis_groupes" and len(resultat.plis) == 3
    ridge = resultat.modele("regression.ridge")
    assert ridge.moyenne("rmse") < resultat.reference().moyenne("rmse")
    # la durée de vie restante décroît d'un cycle par cycle : aucune référence ne lit la cible passée
    assert [m.id for m in resultat.modeles if m.famille == "reference"] == ["reference.dummy_regression"]
    assert resultat.classement()[0].id == "regression.ridge"
    # le test (machines 17 à 20) est resté hors de la validation
    assert resultat.positions.size == 3881 - int(
        (lire("MecaIA-TimeSeries_RUL_extrait.xlsx")["ID_Machine"] >= 17).sum()
    )


def test_serie_les_variables_construites_par_machine_aident(calculs: dict) -> None:
    sans, _ = _calculer(calculs, "serie")
    avec, _ = _calculer(calculs, "serie_retards")
    assert avec.modele("regression.ridge").moyenne("rmse") < sans.modele("regression.ridge").moyenne("rmse")
    assert np.array_equal(avec.positions, sans.positions)  # mêmes lignes d'apprentissage, mêmes plis
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(avec.plis, sans.plis, strict=True))


def test_pompe_regroupement_et_anomalies(calculs: dict) -> None:
    resultat, _ = _calculer(calculs, "pompe")
    assert resultat.reference() is None and resultat.validation == "aucune"
    assert resultat.modele("sans_cible.kmeans").moyenne("silhouette") > 0.2
    acp = resultat.modele("sans_cible.acp")
    assert 0.5 < acp.moyenne("variance_expliquee") <= 1.0
    for id_modele in ("sans_cible.foret_isolement", "sans_cible.facteur_local"):
        assert 0.0 < resultat.modele(id_modele).moyenne("taux_anomalies") < 0.2


# ============================================================
# CLASSE POSITIVE PROPOSÉE SUR LES JEUX CANONIQUES (lus dans MECAML_JEUX)
# ============================================================
@pytest.mark.parametrize(
    ("fichier", "cible", "attendue"),
    [
        ("MecaIA-Qualite_soudure.xlsx", "qualite_soudure", "non-conforme"),
        ("MecaIA-Criticite_Fissures.xlsx", "etat_fissure", "critique"),
        ("MecaIA-Identification_Conformite.xlsx", "presence_defaut", None),
        ("MecaIA-Controle_Conformite.xlsx", "etat_piece", "rebut"),
        ("MecaIA-AI4I_maintenance.xlsx", "Panne", 1),
    ],
)
def test_la_proposition_de_classe_positive_designe_le_defaut_sur_les_jeux_canoniques(
    fichier: str, cible: str, attendue: object
) -> None:
    from mecaml.core import metrics

    y = lire(fichier)[cible].dropna().to_numpy()
    proposee, origine = metrics.proposer_classe_positive(y)
    if attendue is None:  # le jeu dit « défaut présent » contre « pas de défaut » : jamais la classe saine
        assert "pas" not in str(proposee).lower() and origine == "lexique", (proposee, origine)
    else:
        assert str(proposee).lower() == str(attendue).lower(), (proposee, origine)
