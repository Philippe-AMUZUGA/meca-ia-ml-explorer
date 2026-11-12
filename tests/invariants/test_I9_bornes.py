"""I9, bornes : 200 000 lignes et 500 colonnes ; classeur de plus de 50 000 lignes lu dans un fil, avec
progression ; SVC, SVR, OneClassSVM et LeaveOneOut coupés au delà de 20 000 lignes ; messages actionnables.

Les grilles et les arbres bornés relèvent de l'apprentissage. Les mesures aux bornes réelles (200 000
lignes, 500 colonnes) sont marquées `lent` : `python -m pytest -m lent tests/invariants/test_I9_bornes.py -s`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from mecaml.core import guards, io
from mecaml.core.audit import auditer, proposer_roles, proposer_structure
from mecaml.core.guards import (
    COLONNES_MAX,
    LIGNES_LECTURE_EN_FIL,
    LIGNES_MAX,
    LIGNES_MAX_LENTES,
    ErreurAnnulee,
    ErreurDonnees,
    lecture_en_fil_requise,
    technique_coupee,
)
from mecaml.core.io import charger
from mecaml.core.types import Probleme, SpecProjet


# ============================================================
# OUTILS
# ============================================================
def ecrire_csv(chemin: Path, n_lignes: int, n_colonnes: int) -> Path:
    """Fichier .csv de valeurs entières : écriture rapide pour des millions de cellules."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.integers(0, 1000, size=(n_lignes, n_colonnes)), columns=[f"x{i}" for i in range(n_colonnes)]
    )
    df.to_csv(chemin, index=False)
    return chemin


def ecrire_xlsx_long(chemin: Path, n_lignes: int, n_colonnes: int = 3) -> Path:
    """Classeur écrit en flux : rapide même pour des dizaines de milliers de lignes."""
    classeur = Workbook(write_only=True)
    ws = classeur.create_sheet("donnees")
    ws.append([f"x{i}" for i in range(n_colonnes)])
    rng = np.random.default_rng(0)
    for ligne in rng.integers(0, 1000, size=(n_lignes, n_colonnes)).tolist():
        ws.append(ligne)
    classeur.save(chemin)
    return chemin


def message_actionnable(message: str, limite: str) -> bool:
    return limite in message and any(
        verbe in message for verbe in ("Échantillonner", "Retirer", "Réduire", "Choisir")
    )


# ============================================================
# BORNES SUR LE NOMBRE DE LIGNES ET DE COLONNES
# ============================================================
def test_les_bornes_sont_celles_de_la_feuille_de_route() -> None:
    assert (LIGNES_MAX, COLONNES_MAX, LIGNES_LECTURE_EN_FIL, LIGNES_MAX_LENTES) == (
        200_000,
        500,
        50_000,
        20_000,
    )


def test_csv_a_la_borne_des_lignes_accepte(tmp_path: Path) -> None:
    r = charger(ecrire_csv(tmp_path / "t.csv", LIGNES_MAX, 3))
    assert r.table.n_lignes == LIGNES_MAX


def test_csv_au_dela_de_la_borne_des_lignes_refuse_avec_conseil(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees) as e:
        charger(ecrire_csv(tmp_path / "t.csv", LIGNES_MAX + 1, 3))
    assert message_actionnable(str(e.value), "200 000") and "200 001 lignes" in str(e.value)


def test_csv_demesure_refuse_avant_d_etre_lu_en_entier(tmp_path: Path) -> None:
    chemin = ecrire_csv(tmp_path / "t.csv", 2 * LIGNES_MAX + 10, 2)
    debut = time.perf_counter()
    with pytest.raises(ErreurDonnees, match="lignes"):
        charger(chemin)
    assert time.perf_counter() - debut < 5.0


def test_csv_a_la_borne_des_colonnes_accepte(tmp_path: Path) -> None:
    assert charger(ecrire_csv(tmp_path / "t.csv", 5, COLONNES_MAX)).table.n_colonnes == COLONNES_MAX


def test_csv_au_dela_de_la_borne_des_colonnes_refuse_avec_conseil(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees) as e:
        charger(ecrire_csv(tmp_path / "t.csv", 5, COLONNES_MAX + 1))
    assert message_actionnable(str(e.value), "501 colonnes") and "limite de 500" in str(e.value)


def test_classeur_a_la_borne_des_colonnes_et_au_dela(tmp_path: Path) -> None:
    assert charger(ecrire_xlsx_long(tmp_path / "ok.xlsx", 5, COLONNES_MAX)).table.n_colonnes == COLONNES_MAX
    with pytest.raises(ErreurDonnees, match="501 colonnes"):
        charger(ecrire_xlsx_long(tmp_path / "trop.xlsx", 5, COLONNES_MAX + 1))


@pytest.fixture
def bornes_reduites(monkeypatch: pytest.MonkeyPatch) -> int:
    """Même code, bornes ramenées à 300 lignes : le chemin du classeur se teste sans écrire 200 000 lignes."""
    monkeypatch.setattr(guards, "LIGNES_MAX", 300)
    monkeypatch.setattr(io, "LIGNES_MAX", 300)
    return 300


def test_classeur_au_dela_des_lignes_refuse_par_l_en_tete(tmp_path: Path, bornes_reduites: int) -> None:
    chemin = ecrire_xlsx_long(tmp_path / "t.xlsx", bornes_reduites + 1)
    with pytest.raises(ErreurDonnees, match="au delà de la limite de 300"):
        charger(chemin)


def test_classeur_au_dela_des_lignes_refuse_en_cours_de_lecture(
    tmp_path: Path, bornes_reduites: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(io, "_lignes_declarees", lambda *_: None)  # en-tête muet : seul le flux borne
    chemin = ecrire_xlsx_long(tmp_path / "t.xlsx", bornes_reduites + 1)
    with pytest.raises(ErreurDonnees, match="au delà de la limite de 300"):
        charger(chemin)
    assert charger(ecrire_xlsx_long(tmp_path / "ok.xlsx", bornes_reduites)).table.n_lignes == bornes_reduites


# ============================================================
# CLASSEUR DE PLUS DE 50 000 LIGNES : DANS UN FIL, AVEC PROGRESSION
# ============================================================
def test_classeur_de_plus_de_50000_lignes_lu_dans_un_fil_avec_progression(tmp_path: Path) -> None:
    n = LIGNES_LECTURE_EN_FIL + 5_000
    assert lecture_en_fil_requise(n)
    chemin = ecrire_xlsx_long(tmp_path / "gros.xlsx", n)
    appels: list[tuple[float, int]] = []
    resultat: dict[str, object] = {}

    def travail() -> None:
        resultat["chargement"] = charger(
            chemin, progression=lambda f, _m: appels.append((f, threading.get_ident()))
        )

    fil = threading.Thread(target=travail)
    fil.start()
    fil.join(timeout=120)
    assert not fil.is_alive() and resultat["chargement"].table.n_lignes == n
    fractions = [f for f, _ in appels]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and len(appels) >= 10
    assert {ident for _, ident in appels} == {fil.ident} != {threading.main_thread().ident}


def test_lecture_annulee_depuis_un_autre_fil(tmp_path: Path) -> None:
    chemin = ecrire_xlsx_long(tmp_path / "gros.xlsx", 40_000)
    arret = threading.Event()
    sortie: dict[str, BaseException] = {}

    def travail() -> None:
        try:
            charger(chemin, annuler=arret.is_set, progression=lambda f, _m: arret.set() if f > 0.2 else None)
        except BaseException as erreur:
            sortie["erreur"] = erreur

    fil = threading.Thread(target=travail)
    fil.start()
    fil.join(timeout=60)
    assert not fil.is_alive() and isinstance(sortie.get("erreur"), ErreurAnnulee)


def test_la_lecture_ne_garde_aucun_etat_global_deux_lectures_en_parallele(tmp_path: Path) -> None:
    a = ecrire_xlsx_long(tmp_path / "a.xlsx", 3_000, 2)
    b = ecrire_xlsx_long(tmp_path / "b.xlsx", 5_000, 4)
    formes: dict[str, tuple[int, int]] = {}
    fils = [
        threading.Thread(target=lambda c=c: formes.__setitem__(c.name, charger(c).table.forme))
        for c in (a, b)
    ]
    for fil in fils:
        fil.start()
    for fil in fils:
        fil.join(timeout=60)
    assert formes == {"a.xlsx": (3_000, 2), "b.xlsx": (5_000, 4)}


# ============================================================
# TECHNIQUES COUPÉES AU DELÀ DE 20 000 LIGNES
# ============================================================
@pytest.mark.parametrize(
    "id_technique",
    [
        "classification.svc",
        "regression.svr",
        "sans_cible.svm_une_classe",
        "validation.exclusion_unique",
    ],
)
def test_technique_coupee_au_dela_de_20000_lignes_avec_conseil(id_technique: str) -> None:
    assert technique_coupee(id_technique, LIGNES_MAX_LENTES) is None
    message = technique_coupee(id_technique, LIGNES_MAX_LENTES + 1)
    assert message is not None and message_actionnable(message, "20 000") and id_technique in message


# ============================================================
# BUDGET DE L'AUDIT
# ============================================================
def test_audit_d_un_tableau_de_20000_lignes_en_moins_de_2_s() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20_000, 20)), columns=[f"x{i:02d}" for i in range(20)])
    df["cible"] = df["x00"] + rng.normal(size=len(df))
    from mecaml.core.types import Table

    t = Table(df, nom="essai.csv")
    roles = proposer_roles(t, "cible")
    spec = SpecProjet(Probleme.REGRESSION, "cible", roles, "rmse", proposer_structure(roles))
    debut = time.perf_counter()
    auditer(t, None, None)
    auditer(t, spec, None)
    assert time.perf_counter() - debut < 2.0


# ============================================================
# MESURES AUX BORNES RÉELLES
# ============================================================
@pytest.mark.lent
def test_classeur_a_200001_lignes_refuse(tmp_path: Path) -> None:
    chemin = ecrire_xlsx_long(tmp_path / "trop.xlsx", LIGNES_MAX + 1, 2)
    debut = time.perf_counter()
    with pytest.raises(ErreurDonnees) as e:
        charger(chemin)
    print(f"\n[I9] classeur de {LIGNES_MAX + 1} lignes refusé en {time.perf_counter() - debut:.2f} s")
    assert message_actionnable(str(e.value), "200 000")


@pytest.mark.lent
def test_fichier_aux_deux_bornes_200000_lignes_et_500_colonnes(tmp_path: Path) -> None:
    from mecaml.core.types import Table

    debut = time.perf_counter()
    chemin = ecrire_csv(tmp_path / "borne.csv", LIGNES_MAX, COLONNES_MAX)
    ecriture = time.perf_counter() - debut
    debut = time.perf_counter()
    r = charger(chemin)
    lecture = time.perf_counter() - debut
    assert r.table.forme == (LIGNES_MAX, COLONNES_MAX)
    debut = time.perf_counter()
    auditer(Table(r.table.donnees, nom="borne.csv"), None, None)
    audit = time.perf_counter() - debut
    print(
        f"\n[I9] 200 000 x 500 : écriture {ecriture:.1f} s, lecture {lecture:.1f} s, audit S2 {audit:.1f} s"
    )
