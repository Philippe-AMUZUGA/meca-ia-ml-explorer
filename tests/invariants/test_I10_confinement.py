"""I10, confinement : aucun accès réseau ; écriture seulement dans le dossier choisi ; aucun objet sérialisé
chargé ; formules de classeur lues en valeurs, jamais évaluées.

Le chargement et l'audit sont rejoués réseau coupé, écriture surveillée et dossier temporaire surveillé.
L'analyse statique du code, elle, est unique : `tests/unit/test_architecture.py` la fait pour tous les
modules du cœur (désérialisation, import dynamique, processus, réseau, écritures).
"""

from __future__ import annotations

import builtins
import io as bibliotheque_io
import pickle
import shutil
import socket
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from mecaml.core import io
from mecaml.core.audit import auditer, proposer_roles, proposer_structure
from mecaml.core.guards import ErreurDonnees, chemin_dans_dossier
from mecaml.core.io import charger
from mecaml.core.session import Auditer, Charger, Decouper, FixerObjectif, Session
from mecaml.core.types import CoffreTest, Probleme, SpecProjet

EXEMPLES = Path(__file__).resolve().parents[2] / "examples"


# ============================================================
# AUCUN ACCÈS RÉSEAU
# ============================================================
@pytest.fixture
def reseau_coupe(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Toute tentative de connexion ou de résolution de nom échoue et est consignée."""
    tentatives: list[str] = []

    def refuser(nom: str):
        def _refus(*_args, **_kw):
            tentatives.append(nom)
            raise OSError(f"réseau coupé pour le test ({nom})")

        return _refus

    for nom in ("connect", "connect_ex", "bind", "sendto"):
        monkeypatch.setattr(socket.socket, nom, refuser(f"socket.{nom}"))
    for nom in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "create_connection", "socketpair"):
        monkeypatch.setattr(socket, nom, refuser(f"socket.{nom}"))
    return tentatives


def test_chargement_audit_et_session_reseau_coupe(reseau_coupe: list[str]) -> None:
    for fichier in sorted(EXEMPLES.glob("*.xlsx")):
        table = charger(fichier).table
        auditer(table, None, None)
    table = charger(EXEMPLES / "MecaIA-Controle_Conformite.xlsx").table
    roles = proposer_roles(table, "etat_piece")
    spec = SpecProjet(
        Probleme.CLASSIFICATION, "etat_piece", roles, "f1", proposer_structure(roles), classe_positive="rebut"
    )
    coffre = CoffreTest.par_positions(table, np.arange(0, table.n_lignes, 5), decoupage=spec.decoupage())
    s = (
        Session(auditeur=auditer)
        .appliquer(Charger(table))
        .appliquer(Auditer())
        .appliquer(FixerObjectif(spec))
    )
    s = s.appliquer(Decouper(coffre))
    assert s.etat == 4 and reseau_coupe == []


def test_csv_et_classeur_reseau_coupe(tmp_path: Path, reseau_coupe: list[str]) -> None:
    (tmp_path / "t.csv").write_text("a;b\n1,5;2\n3,5;4\n", encoding="utf-8")
    charger(tmp_path / "t.csv")
    charger(EXEMPLES / "MecaIA-Traitement_Acier.xlsx")
    assert reseau_coupe == []


# ============================================================
# ÉCRITURE SEULEMENT DANS LE DOSSIER CHOISI : AUCUNE PAR LE CHARGEMENT NI L'AUDIT
# ============================================================
def instantane(dossier: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(dossier)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in dossier.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def ecritures_surveillees(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Toute ouverture en écriture est consignée ; le dossier temporaire est neuf et doit rester vide."""
    ouvertures: list[str] = []
    temporaire = tmp_path / "temporaire"
    temporaire.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporaire))
    vrai_open = builtins.open

    def surveille(fichier, mode="r", *args, **kw):
        if isinstance(mode, str) and any(m in mode for m in "wax+"):
            ouvertures.append(f"{fichier} ({mode})")
        return vrai_open(fichier, mode, *args, **kw)

    monkeypatch.setattr(builtins, "open", surveille)
    monkeypatch.setattr(bibliotheque_io, "open", surveille)
    return ouvertures


def test_chargement_et_audit_n_ecrivent_rien(tmp_path: Path, ecritures_surveillees: list[str]) -> None:
    travail = tmp_path / "travail"
    shutil.copytree(EXEMPLES, travail)
    avant = instantane(travail)
    for fichier in sorted(travail.glob("*.xlsx")):
        auditer(charger(fichier).table, None, None)
    table = charger(travail / "MecaIA-TimeSeries_RUL_extrait.xlsx").table
    roles = proposer_roles(table, "Duree_Vie_Restante_Cycles")
    spec = SpecProjet(
        Probleme.REGRESSION, "Duree_Vie_Restante_Cycles", roles, "rmse", proposer_structure(roles)
    )
    auditer(table, spec, None)
    assert instantane(travail) == avant
    assert ecritures_surveillees == []
    assert list((tmp_path / "temporaire").iterdir()) == []


def test_le_fichier_source_n_est_jamais_modifie(tmp_path: Path) -> None:
    copie = shutil.copyfile(EXEMPLES / "MecaIA-Controle_Conformite.xlsx", tmp_path / "jeu.xlsx")
    avant = copie.read_bytes()
    table = charger(copie).table
    auditer(table, None, None)
    assert copie.read_bytes() == avant and table.est_intacte()


@pytest.mark.parametrize(
    "nom",
    [
        "../hors.html",
        "a/../../hors.html",
        "..\\hors.html",
        "/tmp/x",
        "C:\\Windows\\x.txt",
        "\\\\hote\\partage\\x",
        "x.txt:flux",
        "NUL",
    ],
)
def test_une_ecriture_ne_sort_jamais_du_dossier_choisi(tmp_path: Path, nom: str) -> None:
    dossier = tmp_path / "sortie"
    dossier.mkdir()
    with pytest.raises(ErreurDonnees):
        chemin_dans_dossier(dossier, nom)
    assert chemin_dans_dossier(dossier, "rapport.html").parent == dossier.resolve()


# ============================================================
# AUCUN OBJET SÉRIALISÉ CHARGÉ, AUCUN CONTENU EXÉCUTÉ
# ============================================================
class Piege:
    """Objet sérialisé qui crée un fichier-témoin s'il est jamais rechargé."""

    def __init__(self, temoin: str) -> None:
        self.temoin = temoin

    def __reduce__(self):
        return (Path(self.temoin).write_text, ("charge", "utf-8"))


@pytest.mark.parametrize("nom", ["modele.pkl", "modele.pickle", "modele.joblib", "modele.npy", "modele.npz"])
def test_un_objet_serialise_n_est_jamais_charge(tmp_path: Path, nom: str) -> None:
    temoin = tmp_path / "temoin.txt"
    chemin = tmp_path / nom
    chemin.write_bytes(pickle.dumps(Piege(str(temoin))))
    with pytest.raises(ErreurDonnees, match=r"\.xlsx ou \.csv"):
        charger(chemin)
    assert not temoin.exists()


def test_un_classeur_a_macros_n_execute_rien(tmp_path: Path) -> None:
    base = tmp_path / "ok.xlsx"
    classeur = Workbook()
    classeur.active.append(["a", "b"])
    classeur.active.append([1, 2])
    classeur.save(base)
    avec_macros = tmp_path / "macros.xlsx"
    with zipfile.ZipFile(base) as lu, zipfile.ZipFile(avec_macros, "w") as ecrit:
        for info in lu.infolist():
            ecrit.writestr(info.filename, lu.read(info.filename))
        ecrit.writestr("xl/vbaProject.bin", b'Sub Auto_Open(): Shell("calc"): End Sub')
    with pytest.raises(ErreurDonnees, match="macros"):
        charger(avec_macros)


def test_formules_lues_en_valeurs_jamais_evaluees(tmp_path: Path) -> None:
    classeur = Workbook()
    ws = classeur.active
    ws.append(["a", "b", "c"])
    ws.append([1, 2, "=A2+B2"])
    ws.append([3, 4, '=HYPERLINK("http://exemple.invalide/x","clic")'])
    ws.append([5, 6, '=IMPORTXML("http://exemple.invalide","//a")'])
    source = tmp_path / "f.xlsx"
    classeur.save(source)
    with zipfile.ZipFile(source) as lu, zipfile.ZipFile(tmp_path / "cache.xlsx", "w") as ecrit:
        for info in lu.infolist():
            contenu = lu.read(info.filename)
            if info.filename == "xl/worksheets/sheet1.xml":
                contenu = contenu.replace(b"<v></v>", b"<v>99</v>")
            ecrit.writestr(info.filename, contenu)
    r = charger(tmp_path / "cache.xlsx")
    assert r.formules == 3 and r.formules_sans_valeur == 0
    assert r.table.donnees["c"].tolist() == [99, 99, 99]  # valeur en cache du tableur, jamais A2+B2 = 3
    sans_cache = charger(source)
    assert sans_cache.formules_sans_valeur == 3 and sans_cache.table.donnees["c"].isna().all()


def test_les_imports_du_noyau_sont_les_memes_en_processus_neuf() -> None:
    """Importer io, audit et guards ne charge ni scikit-learn ni interface."""
    import os
    import subprocess
    import sys

    programme = (
        "import sys\n"
        "import mecaml.core.io, mecaml.core.audit, mecaml.core.guards\n"
        "print(','.join(sorted(m for m in ('sklearn', 'tkinter', 'customtkinter') if m in sys.modules)))\n"
    )
    src = Path(io.__file__).resolve().parents[2]
    sortie = subprocess.run(
        [sys.executable, "-c", programme],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        check=True,
        timeout=120,
    )
    chargés = set(filter(None, sortie.stdout.strip().split(",")))
    assert chargés == set(), chargés
