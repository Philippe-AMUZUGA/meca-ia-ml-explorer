"""I1 Scellé : aucune ligne de test vue par `fit`, `predict` ou une métrique avant la décision.

Un transformeur espion, placé en tête du pipeline, note chaque ligne qu'il reçoit
pendant la validation croisée sur les plis de `plis_validation` : aucune ligne du
coffre n'y figure, pour chaque structure, jusqu'à l'ouverture après la décision.
Jeux : répliques synthétiques des 100 machines et de la maintenance horodatée, et
les jeux réels si MECAML_JEUX les fournit.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from mecaml.core import features_ts
from mecaml.core.session import Decider, Decouper, Etat, Revenir, Tester
from mecaml.core.split import plis_validation, sceller, sceller_base_test
from mecaml.core.types import CoffreTest, Decision, ErreurScelle, Probleme, Role, SpecProjet, Structure, Table

CAPTEURS = ("Temp_Carter_K", "Vib_RMS_AxeZ_mm_s", "Courant_Moteur_Amp")
MESURES = ("T_palier_C", "Couple_Nm", "Vibration_mm_s")


class Espion(TransformerMixin, BaseEstimator):
    """Note les étiquettes (positions dans la table) des lignes reçues par `fit` et `transform`."""

    vues: ClassVar[set[int]] = set()

    def fit(self, x: pd.DataFrame, y: Any = None) -> Espion:
        Espion.vues.update(int(i) for i in x.index)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        Espion.vues.update(int(i) for i in x.index)
        return x


# ============================================================
# JEUX ET SPÉCIFICATIONS
# ============================================================
def _lire(fichier: str) -> pd.DataFrame:
    dossier = os.environ.get("MECAML_JEUX")
    if not dossier or not (Path(dossier) / fichier).is_file():
        pytest.skip(f"{fichier} absent de MECAML_JEUX")
    return pd.read_excel(Path(dossier) / fichier)


@functools.cache
def _jeu(nom: str, source: str) -> pd.DataFrame:
    if source == "reel":
        return _lire("MecaIA-TimeSeries_RUL.xlsx" if nom == "rul" else "MecaIA-AI4I_maintenance.xlsx")
    rng = np.random.default_rng(5)
    if nom == "rul":
        durees = rng.integers(20, 50, 100)
        cycle = np.concatenate([np.arange(1, d + 1) for d in durees])
        df = pd.DataFrame(
            {"ID_Machine": np.repeat(np.arange(1, 101), durees), "Cycle_Operation_Heures": cycle}
        )
        for c in CAPTEURS:
            df[c] = rng.normal(size=len(df)) + 0.1 * cycle
        df["Duree_Vie_Restante_Cycles"] = np.repeat(durees, durees) - cycle
        return df
    df = pd.DataFrame({"time": np.round(np.arange(2000) / 6, 4)})
    for c in MESURES:
        df[c] = rng.normal(size=2000)
    df["Panne"] = ((df["Couple_Nm"] + rng.normal(0, 0.5, 2000)) > 1.8).astype(int)
    return df


@pytest.fixture(params=["reel", "replique"])
def source(request: pytest.FixtureRequest) -> str:
    return request.param


def _spec(nom: str, structure: Structure) -> SpecProjet:
    if nom == "rul":
        roles = (
            ("ID_Machine", Role.GROUPE),
            ("Cycle_Operation_Heures", Role.DATE),
            ("Duree_Vie_Restante_Cycles", Role.CIBLE),
            *((c, Role.NUMERIQUE) for c in CAPTEURS),
        )
        return SpecProjet(
            Probleme.REGRESSION, "Duree_Vie_Restante_Cycles", roles, "rmse", structure, graine=42
        )
    roles = (("time", Role.DATE), ("Panne", Role.CIBLE), *((c, Role.NUMERIQUE) for c in MESURES))
    return SpecProjet(Probleme.CLASSIFICATION, "Panne", roles, "f1_macro", structure, graine=42)


CAS = [
    ("rul", Structure.GROUPEE),
    ("rul", Structure.GROUPEE_CHRONOLOGIQUE),
    ("rul", Structure.INDEPENDANTE),
    ("ai4i", Structure.CHRONOLOGIQUE),
    ("ai4i", Structure.INDEPENDANTE),
]


def _pipeline(nom: str, structure: Structure) -> Pipeline:
    """Espion en tête ; variables temporelles par unité dès que la structure suit le temps."""
    etapes: list = [Espion()]
    if structure.chronologique:
        groupe = "ID_Machine" if structure.groupee else None
        date = "Cycle_Operation_Heures" if nom == "rul" else "time"
        mesures = CAPTEURS if nom == "rul" else MESURES
        etapes.append(
            features_ts.VariablesTemporelles(
                colonne_date=date, colonne_groupe=groupe, colonnes=mesures, retards=(1, 2), fenetres=(5,)
            )
        )
    etapes += [SimpleImputer(), StandardScaler()]
    etapes.append(Ridge() if nom == "rul" else LogisticRegression(max_iter=1000))
    return make_pipeline(*etapes)


def _entrees(spec: SpecProjet, lignes: pd.DataFrame) -> pd.DataFrame:
    """Variables, plus l'unité et la date quand les variables temporelles les exigent."""
    temps = [c for c in (spec.colonne_groupe, spec.colonne_date) if c] if spec.structure.chronologique else []
    return lignes[temps + list(spec.variables())]


def _positions_test(coffre: CoffreTest, n: int) -> set[int]:
    return set(np.setdiff1d(np.arange(n), coffre.positions_apprentissage()).tolist())


# ============================================================
# ESPION : VALIDATION CROISÉE, PUIS OUVERTURE
# ============================================================
@pytest.mark.parametrize(("nom", "structure"), CAS, ids=lambda v: str(v))
def test_I1_validation_croisee_ne_voit_aucune_ligne_de_test(source, nom, structure) -> None:
    df = _jeu(nom, source)
    spec = _spec(nom, structure)
    _, coffre = sceller(Table(df, nom=nom), spec)
    test = _positions_test(coffre, len(df))
    lignes = coffre.apprentissage()
    x, y = _entrees(spec, lignes), lignes[spec.cible]
    Espion.vues.clear()
    for app, val in plis_validation(lignes, spec):
        modele = _pipeline(nom, structure).fit(x.iloc[app], y.iloc[app])
        modele.predict(x.iloc[val])
    assert Espion.vues == set(lignes.index.tolist())  # tout l'apprentissage, rien d'autre
    assert not Espion.vues & test and coffre.ouvertures == 0

    final = _pipeline(nom, structure).fit(x, y)
    assert not Espion.vues & test
    ouvertes = coffre.ouvrir(Decision("regression.ridge", spec.metrique, 0.5))
    final.predict(_entrees(spec, ouvertes))
    assert test <= Espion.vues and coffre.ouvertures == 1


def test_I1_memoire_des_variables_temporelles_issue_de_l_apprentissage() -> None:
    df = _jeu("ai4i", "replique")
    spec = _spec("ai4i", Structure.CHRONOLOGIQUE)
    _, coffre = sceller(Table(df, nom="ai4i"), spec)
    lignes = coffre.apprentissage()
    tr = features_ts.VariablesTemporelles(colonne_date="time", colonnes=MESURES, fenetres=(20,)).fit(lignes)
    dates_test = set(df["time"].to_numpy()[sorted(_positions_test(coffre, len(df)))].tolist())
    assert len(tr.historique_) == 19
    assert set(tr.historique_["date"].tolist()) <= set(lignes["time"].tolist())
    assert not set(tr.historique_["date"].tolist()) & dates_test


# ============================================================
# SCELLÉ ET OUVERTURE
# ============================================================
def test_I1_sceller_n_ouvre_rien(table_petite, spec_reg) -> None:
    decoupage, coffre = sceller(table_petite, spec_reg)
    _, _, relu = sceller_base_test(table_petite, spec_reg, decoupage.vers_base_test())
    assert coffre.ouvertures == relu.ouvertures == 0
    assert set(vars(decoupage)) == {
        "methode",
        "test_size",
        "random_state",
        "colonne_groupe",
        "lignes_test",
        "empreinte_sha256",
    }  # aucune valeur de ligne dans le découpage déclaré


@pytest.mark.parametrize("jeton", [None, "décision", {"modele_id": "x"}])
def test_I1_ouverture_sans_decision_refusee(table_petite, spec_reg, jeton) -> None:
    _, coffre = sceller(table_petite, spec_reg)
    with pytest.raises(ErreurScelle, match="scellé"):
        coffre.ouvrir(jeton)
    assert coffre.ouvertures == 0


def test_I1_seconde_ouverture_comptee_et_copie_rendue(table_petite, spec_reg) -> None:
    _, coffre = sceller(table_petite, spec_reg)
    decision = Decision("regression.ridge", "rmse", 1.0)
    lignes = coffre.ouvrir(decision)
    lignes.iloc[:, 3] = -1.0
    assert table_petite.est_intacte() and not coffre.score_optimiste
    coffre.ouvrir(decision)
    assert coffre.ouvertures == 2 and coffre.score_optimiste


def test_I1_session_n_ouvre_qu_apres_la_decision(parcours) -> None:
    s3 = parcours.jusqua(Etat.OBJECTIF)
    _, coffre = sceller(s3.table, s3.spec)
    s = s3.appliquer(Decouper(coffre))
    evaluer = Tester(lambda lignes: len(lignes))
    for etat in (Etat.DECOUPE, Etat.PRETRAITEMENT, Etat.MODELES):
        s = parcours.jusqua(etat, depart=s)
        assert not s.peut(evaluer) and coffre.ouvertures == 0
    s7 = parcours.jusqua(Etat.DIAGNOSTIC, depart=s)
    s8 = s7.appliquer(evaluer)
    assert s8.resultat_test == coffre.n_test and coffre.ouvertures == 1 and not s8.score_test_optimiste
    rejoue = s8.appliquer(Revenir(Etat.MODELES))
    rejoue = rejoue.appliquer(Decider(s7.decision)).appliquer(evaluer)
    assert coffre.ouvertures == 2 and rejoue.score_test_optimiste
