"""Empaquetage : jeux d'exemple, lanceur, point d'entrée `python -m mecaml`, spécification de l'exécutable."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
import tkinter as tk
import tomllib
import types
from pathlib import Path

import pytest

import mecaml
from mecaml import __main__ as principal_module
from mecaml import cli

RACINE = Path(__file__).resolve().parents[2]
EXEMPLES = RACINE / "examples"
EMPREINTES = json.loads((EXEMPLES / "empreintes.json").read_text(encoding="utf-8"))
JEUX = EMPREINTES["jeux"]
LIMITE_JEUX_OCTETS = 1_000_000


def _sha256(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


# ============================================================
# JEUX D'EXEMPLE
# ============================================================


def test_quatre_jeux_d_exemple():
    noms = sorted(p.name for p in EXEMPLES.glob("*.xlsx"))
    assert noms == sorted(jeu["fichier"] for jeu in JEUX)
    assert len(noms) == 4


@pytest.mark.parametrize("jeu", JEUX, ids=lambda jeu: jeu["fichier"])
def test_empreinte_de_chaque_jeu(jeu):
    chemin = EXEMPLES / jeu["fichier"]
    assert _sha256(chemin) == jeu["sha256"]
    assert chemin.stat().st_size == jeu["octets"]


def test_les_quatre_jeux_pesent_moins_d_un_mega_octet():
    assert sum(p.stat().st_size for p in EXEMPLES.glob("*.xlsx")) < LIMITE_JEUX_OCTETS
    assert sum(p.stat().st_size for p in EXEMPLES.rglob("*") if p.is_file()) < LIMITE_JEUX_OCTETS


@pytest.mark.parametrize("jeu", [j for j in JEUX if j["extrait"] is None], ids=lambda jeu: jeu["fichier"])
def test_jeu_identique_a_sa_source_publiee(jeu):
    """Quand le dossier des jeux du livre est désigné, la copie en est octet pour octet la même."""
    dossier = os.environ.get("MECAML_JEUX")
    if not dossier or not (Path(dossier) / Path(jeu["source"]).name).is_file():
        pytest.skip("dossier des jeux du livre non désigné (MECAML_JEUX)")
    assert _sha256(Path(dossier) / Path(jeu["source"]).name) == jeu["sha256_source"] == jeu["sha256"]


def test_le_readme_des_exemples_nomme_chaque_fichier():
    texte = (EXEMPLES / "README.md").read_text(encoding="utf-8")
    for jeu in JEUX:
        assert jeu["fichier"] in texte
    assert "projet_exemple" in texte


def test_le_projet_exporte_d_exemple_se_rejoue_a_l_identique(capsys):
    dossier = EXEMPLES / "projet_exemple"
    assert {p.name for p in dossier.iterdir()} == set(EMPREINTES["projet_exemple"]["fichiers"])
    code = cli.principal(
        [
            "rejouer",
            str(dossier / "projet.json"),
            "-d",
            str(EXEMPLES / EMPREINTES["projet_exemple"]["jeu"]),
            "-v",
        ]
    )
    assert code == 0
    assert "reproduit les scores attendus" in capsys.readouterr().out


def test_ressources_trouvent_les_exemples_et_les_icones():
    from mecaml.ui.ressources import chemin_asset, exemples

    assert {p.name for p in exemples()} >= {jeu["fichier"] for jeu in JEUX}
    assert chemin_asset("mecaml_icone.png") is not None
    assert chemin_asset("mecaml_icone.ico") is not None


# ============================================================
# POINT D'ENTREE
# ============================================================


def test_python_m_mecaml_donne_la_version():
    env = {**os.environ, "PYTHONPATH": str(RACINE / "src"), "PYTHONUTF8": "1"}
    sortie = subprocess.run(
        [sys.executable, "-m", "mecaml", "-V"], capture_output=True, text=True, encoding="utf-8", env=env
    )
    assert sortie.returncode == 0
    assert f"MécaML Explorer {mecaml.__version__}" in sortie.stdout


def test_le_poste_d_essai_est_pret():
    assert principal_module.verifier_poste() is None


def test_motif_clair_quand_une_bibliotheque_manque(monkeypatch):
    vrai = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda nom, *a: None if nom == "scipy" else vrai(nom, *a)
    )
    motif = principal_module.verifier_poste()
    assert "scipy" in motif
    assert "pip install -r requirements.txt" in motif


def test_motif_clair_quand_tk_manque(monkeypatch):
    vrai = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda nom, *a: None if nom == "tkinter" else vrai(nom, *a)
    )
    assert "tkinter" in principal_module.verifier_poste()


def test_motif_clair_quand_python_est_trop_ancien(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
    assert "Python 3.12" in principal_module.verifier_poste()


def test_un_poste_sans_motif_ne_bloque_pas_le_lancement(monkeypatch):
    lance = []
    monkeypatch.setattr(principal_module, "verifier_poste", lambda: "motif de test")
    monkeypatch.setattr("mecaml.ui.app.lancer", lambda: lance.append(True))
    assert principal_module.principal([]) == 1
    assert not lance
    monkeypatch.setattr(principal_module, "verifier_poste", lambda: None)
    assert principal_module.principal([]) == 0
    assert lance == [True]


def test_essai_automatique_mene_un_jeu_d_exemple_jusqu_a_l_export(tmp_path):
    try:
        tk.Tk().destroy()
    except tk.TclError as erreur:
        pytest.skip(f"Tk indisponible : {erreur}")
    assert principal_module.essai_automatique(str(tmp_path)) == 0
    gc.collect()
    ecrits = {p.name for p in tmp_path.iterdir()}
    assert {"projet.json", "rapport.html", "rapport.xlsx", "modele.joblib"} <= ecrits


def test_la_commande_mecaml_et_les_donnees_de_la_roue_sont_declarees():
    projet = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    module, _, fonction = projet["project"]["scripts"]["mecaml"].partition(":")
    assert callable(getattr(importlib.import_module(module), fonction))
    donnees = projet["tool"]["setuptools"]["package-data"]
    assert donnees["mecaml.textes"] == ["*.toml"]
    assert "assets/*.ico" in donnees["mecaml"]
    assert projet["project"]["license"] == "MIT"


# ============================================================
# LANCEUR WINDOWS
# ============================================================


def test_le_lanceur_cherche_python_312_et_dit_quoi_installer():
    brut = (RACINE / "Lancer_MecaML.bat").read_bytes()
    assert b"\r\n" in brut and b"\n" not in brut.replace(b"\r\n", b"")  # fins de ligne CRLF
    texte = brut.decode("utf-8")
    assert "py -3.12" in texte
    assert "(3, 12)" in texte
    assert "python.org" in texte
    assert "requirements.txt" in texte
    assert "-m mecaml %*" in texte
    assert "PYTHONPATH" in texte  # sans installation du paquet


# ============================================================
# SPECIFICATION PYINSTALLER
# ============================================================


@pytest.fixture(scope="module")
def specification() -> dict:
    """Exécute `mecaml.spec` avec un faux PyInstaller qui garde les arguments de l'analyse et de l'EXE."""
    saisies: dict = {}

    def analysis(scripts, **options):
        saisies["analysis"] = {"scripts": scripts, **options}
        return types.SimpleNamespace(pure=[], scripts=[], binaries=[], datas=options["datas"])

    def exe(*arguments, **options):
        saisies["exe"] = options

    def sous_modules(nom):
        paquet = importlib.import_module(nom)
        return [m.name for m in pkgutil.walk_packages(paquet.__path__, nom + ".")]

    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_submodules = sous_modules
    anciens = {n: sys.modules.get(n) for n in ("PyInstaller", "PyInstaller.utils", "PyInstaller.utils.hooks")}
    sys.modules.update(
        {"PyInstaller": types.ModuleType("PyInstaller"), "PyInstaller.utils": types.ModuleType("u")}
    )
    sys.modules["PyInstaller.utils.hooks"] = hooks
    try:
        source = (RACINE / "build" / "mecaml.spec").read_text(encoding="utf-8")
        espace = {
            "SPECPATH": str(RACINE / "build"),
            "Analysis": analysis,
            "PYZ": lambda *a, **k: None,
            "EXE": exe,
        }
        exec(compile(source, "mecaml.spec", "exec"), espace)
    finally:
        for nom, ancien in anciens.items():
            if ancien is None:
                sys.modules.pop(nom, None)
            else:
                sys.modules[nom] = ancien
    return saisies


def _fichiers_de(datas: list, destination: str) -> list[Path]:
    trouves: list[Path] = []
    for source, dest in datas:
        if dest == destination:
            motif = Path(source)
            trouves += [p for p in motif.parent.glob(motif.name) if p.is_file()]
    return trouves


def test_la_specification_emporte_textes_icones_et_exemples(specification):
    datas = specification["analysis"]["datas"]
    textes = {p.name for p in _fichiers_de(datas, "mecaml/textes")}
    assert textes == {"interface_fr.toml", "etapes_fr.toml"}
    assert {p.name for p in _fichiers_de(datas, "mecaml/assets")} == {"mecaml_icone.png", "mecaml_icone.ico"}
    exemples = {p.name for p in _fichiers_de(datas, "examples")}
    assert {jeu["fichier"] for jeu in JEUX} <= exemples
    assert {p.name for p in _fichiers_de(datas, "examples/projet_exemple")} == {"projet.json", "rapport.xlsx"}


def test_la_specification_part_du_point_d_entree_du_paquet(specification):
    assert Path(specification["analysis"]["scripts"][0]) == RACINE / "src" / "mecaml" / "__main__.py"
    assert specification["exe"]["name"] == "MecaML"
    assert specification["exe"]["console"] is False
    assert Path(specification["exe"]["icon"]).is_file()


def test_les_modules_caches_de_la_specification_existent(specification):
    caches = specification["analysis"]["hiddenimports"]
    assert "mecaml.ui.app" in caches
    assert "matplotlib.backends.backend_tkagg" in caches
    inconnus = [nom for nom in caches if importlib.util.find_spec(nom) is None]
    assert not inconnus


def test_la_specification_exclut_les_outils_de_developpement(specification):
    exclus = set(specification["analysis"]["excludes"])
    assert {"pytest", "IPython", "tensorflow", "torch"} <= exclus
    assert not exclus & {"numpy", "pandas", "sklearn", "scipy", "matplotlib", "openpyxl", "joblib", "tkinter"}
