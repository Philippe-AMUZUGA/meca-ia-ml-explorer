"""Architecture et charte du dépôt.

Le cœur n'importe aucune interface et scikit-learn n'y est importé qu'à l'usage
(analyse `ast` de chaque module de `src/mecaml/core`, puis contrôle dans un
processus neuf) ; le registre accepte un module de technique importé en tête
sans import circulaire ; le format ruff et la charte des tirets sont tenus.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
SRC = RACINE / "src"
CORE = SRC / "mecaml" / "core"
VERIFIER_TIRETS = RACINE / "tests" / "outils" / "verifier_tirets.py"
# Option de vérification de `ruff format`, écrite sans trait d'union doublé littéral (charte du dépôt).
OPTION_VERIFICATION = "-" * 2 + "check"
RUFF_PRESENT = importlib.util.find_spec("ruff") is not None

# Imports interdits partout dans le cœur (préfixes de modules).
INTERDITS = (
    "tkinter",
    "_tkinter",
    "customtkinter",
    "mecaml.ui",
    "matplotlib.pyplot",  # choisit un moteur graphique d'interface ; utiliser matplotlib.figure
    "pkgutil",  # registre explicite : aucune découverte automatique
    "socket",  # aucun accès réseau
    "urllib",
    "http",
    "requests",
    "ftplib",
    "smtplib",
)
# Imports permis seulement dans les fonctions (import paresseux).
PARESSEUX = ("sklearn",)
# Appels interdits : aucun contenu exécuté, aucun objet sérialisé chargé.
APPELS_INTERDITS = {("joblib", "load"), ("pickle", "load"), ("pickle", "loads"), ("pd", "read_pickle")}
NOMS_INTERDITS = {"eval", "exec"}


@dataclass(frozen=True)
class Violation:
    fichier: str
    ligne: int
    motif: str


def _module_de(chemin: Path, racine: Path) -> str:
    parties = list(chemin.relative_to(racine).with_suffix("").parts)
    if parties[-1] == "__init__":
        parties.pop()
    return ".".join(parties)


def _absolu(module: str | None, niveau: int, paquet: str) -> str:
    if niveau == 0:
        return module or ""
    base = paquet.split(".")[: len(paquet.split(".")) - niveau + 1]
    return ".".join([*base, module] if module else base)


def _commence_par(nom: str, prefixes: tuple[str, ...]) -> str | None:
    for p in prefixes:
        if nom == p or nom.startswith(p + "."):
            return p
    return None


def _sous_type_checking(noeud: ast.If) -> bool:
    test = noeud.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def analyser(source: str, nom_fichier: str, module: str, est_paquet: bool) -> list[Violation]:
    """Violations d'architecture d'un module source."""
    arbre = ast.parse(source, filename=nom_fichier)
    paquet = module if est_paquet else module.rpartition(".")[0]
    violations: list[Violation] = []

    def importes(noeud: ast.Import | ast.ImportFrom) -> list[str]:
        if isinstance(noeud, ast.Import):
            return [alias.name for alias in noeud.names]
        base = _absolu(noeud.module, noeud.level, paquet)
        return [base, *(f"{base}.{alias.name}" for alias in noeud.names)]

    def visiter(noeud: ast.AST, niveau_module: bool) -> None:
        for enfant in ast.iter_child_nodes(noeud):
            if isinstance(enfant, (ast.Import, ast.ImportFrom)):
                for nom in importes(enfant):
                    if p := _commence_par(nom, INTERDITS):
                        violations.append(Violation(nom_fichier, enfant.lineno, f"import interdit {p}"))
                    if niveau_module and (p := _commence_par(nom, PARESSEUX)):
                        violations.append(Violation(nom_fichier, enfant.lineno, f"import en tête {p}"))
            elif isinstance(enfant, ast.Call):
                f = enfant.func
                if isinstance(f, ast.Name) and f.id in NOMS_INTERDITS:
                    violations.append(Violation(nom_fichier, enfant.lineno, f"appel interdit {f.id}"))
                if (
                    isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name)
                    and (f.value.id, f.attr) in APPELS_INTERDITS
                ):
                    motif = f"appel interdit {f.value.id}.{f.attr}"
                    violations.append(Violation(nom_fichier, enfant.lineno, motif))
            dans_fonction = isinstance(enfant, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            reserve_au_typage = isinstance(enfant, ast.If) and _sous_type_checking(enfant)
            visiter(enfant, niveau_module and not (dans_fonction or reserve_au_typage))

    visiter(arbre, niveau_module=True)
    return violations


def modules_du_coeur() -> list[Path]:
    return sorted(CORE.rglob("*.py"))


def test_le_coeur_existe() -> None:
    noms = {p.name for p in modules_du_coeur()}
    assert {"__init__.py", "types.py", "session.py"} <= noms


@pytest.mark.parametrize("chemin", modules_du_coeur(), ids=lambda p: p.relative_to(CORE).as_posix())
def test_architecture_du_coeur(chemin: Path) -> None:
    source = chemin.read_text(encoding="utf-8")
    module = _module_de(chemin, SRC)
    violations = analyser(source, chemin.name, module, chemin.name == "__init__.py")
    assert violations == [], "\n".join(f"{v.fichier}:{v.ligne} {v.motif}" for v in violations)


@pytest.mark.parametrize(
    ("source", "motif"),
    [
        ("import tkinter as tk\n", "tkinter"),
        ("from tkinter import ttk\n", "tkinter"),
        ("from ..ui import app\n", "mecaml.ui"),
        ("from mecaml.ui.screens import s1\n", "mecaml.ui"),
        ("import matplotlib.pyplot as plt\n", "matplotlib.pyplot"),
        ("import pkgutil\n", "pkgutil"),
        ("from urllib.request import urlopen\n", "urllib"),
        ("from sklearn.linear_model import Ridge\n", "import en tête sklearn"),
        ("import joblib\nm = joblib.load('m.joblib')\n", "joblib.load"),
        ("x = eval('1 + 1')\n", "eval"),
    ],
)
def test_l_analyse_detecte_les_violations(source: str, motif: str) -> None:
    violations = analyser(source, "exemple.py", "mecaml.core.exemple", est_paquet=False)
    assert any(motif in v.motif for v in violations), violations


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    from sklearn.dummy import DummyRegressor\n    return DummyRegressor()\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from sklearn.base import BaseEstimator\n",
        "import re\npattern = re.compile('a')\n",
        "from matplotlib.figure import Figure\n",
        "from .types import Table\n",
    ],
    ids=["sklearn_dans_fonction", "type_checking", "re_compile", "figure", "relatif_interne"],
)
def test_l_analyse_laisse_passer_le_permis(source: str) -> None:
    assert analyser(source, "exemple.py", "mecaml.core.exemple", est_paquet=False) == []


# ============================================================
# CONFINEMENT (I10) : TOUT LE CŒUR, IMPORTS DYNAMIQUES ET ÉCRITURES
# ============================================================
# Seuls modules du cœur autorisés à écrire (sorties de S9), et seulement dans le dossier
# choisi, par `guards.chemin_dans_dossier`.
ECRIVAINS = frozenset({"report.py", "codegen.py"})
# Modules interdits partout : désérialisation, import dynamique, processus, réseau, fichiers temporaires.
IMPORTS_CONFINES = frozenset(
    {
        "pickle",
        "cPickle",
        "_pickle",
        "shelve",
        "marshal",
        "dill",
        "cloudpickle",
        "importlib",
        "ctypes",
        "subprocess",
        "multiprocessing",
        "webbrowser",
        "telnetlib",
        "tempfile",
        "shutil",
    }
)
IMPORTS_ECRIVAINS = frozenset({"joblib"})  # export du modèle (dump) : écrivains seulement
NOMS_CONFINES = frozenset({"eval", "exec", "compile", "__import__"})  # fonctions intégrées
METHODES_CONFINEES = frozenset(
    {"read_pickle", "load", "loads", "import_module", "system", "popen", "startfile", "spawnv", "spawnl"}
)
METHODES_ECRITURE = frozenset(
    {
        "to_csv",
        "to_excel",
        "to_json",
        "to_parquet",
        "to_hdf",
        "to_pickle",
        "to_feather",
        "to_stata",
        "to_sql",
        "write_text",
        "write_bytes",
        "save",
        "savefig",
        "savez",
        "savez_compressed",
        "savetxt",
        "tofile",
        "dump",
        "unlink",
        "rmdir",
        "mkdir",
        "makedirs",
        "rename",
        "remove",
        "removedirs",
        "touch",
        "extractall",
        "extract",
    }
)
NOMS_ECRITURE = frozenset({"ExcelWriter", "ZipFile"})


def _mode_ecriture(noeud: ast.Call, position: int) -> str | None:
    """Mode d'ouverture s'il écrit, « non littéral » s'il ne peut être lu, None s'il lit."""
    mode: ast.expr | None = noeud.args[position] if len(noeud.args) > position else None
    for k in noeud.keywords:
        if k.arg == "mode":
            mode = k.value
    if mode is None:
        return None
    if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str):
        return "non littéral"
    return mode.value if any(m in mode.value for m in "wax+") else None


def _mode_methode_open(noeud: ast.Call) -> str | None:
    """`objet.open(...)` : `Path.open(mode)` ou `ZipFile.open(nom, mode)`.

    Le premier argument n'est lu comme un mode que s'il en a la forme (« w », « rb »...).
    """
    premier = noeud.args[0] if noeud.args else None
    if isinstance(premier, ast.Constant) and isinstance(premier.value, str) and _RE_MODE.match(premier.value):
        return _mode_ecriture(noeud, 0)
    return _mode_ecriture(noeud, 1)


_RE_MODE = re.compile(r"^[rwxabtU+]{1,4}$")


def analyser_confinement(source: str, nom_fichier: str, ecrivain: bool = False) -> list[Violation]:
    """Désérialisation, import dynamique, processus, réseau et écritures d'un module du cœur."""
    violations: list[Violation] = []
    ecritures: list[int] = []
    passe_par_le_garde = False

    def signaler(noeud: ast.AST, motif: str) -> None:
        violations.append(Violation(nom_fichier, getattr(noeud, "lineno", 0), motif))

    for noeud in ast.walk(ast.parse(source, filename=nom_fichier)):
        if isinstance(noeud, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in noeud.names] if isinstance(noeud, ast.Import) else [noeud.module or ""]
            for m in modules:
                racine = m.split(".")[0]
                if racine in IMPORTS_CONFINES or (racine in IMPORTS_ECRIVAINS and not ecrivain):
                    signaler(noeud, f"import confiné {m}")
            if isinstance(noeud, ast.ImportFrom):
                for alias in noeud.names:
                    if alias.name in METHODES_CONFINEES:
                        signaler(noeud, f"import confiné {alias.name}")
            continue
        if not isinstance(noeud, ast.Call):
            continue
        f = noeud.func
        nom = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if nom == "chemin_dans_dossier":
            passe_par_le_garde = True
        if isinstance(f, ast.Name) and nom in NOMS_CONFINES:
            signaler(noeud, f"appel confiné {nom}")
        if isinstance(f, ast.Attribute) and nom in METHODES_CONFINEES:
            signaler(noeud, f"appel confiné {nom}")
        mode = None
        if isinstance(f, ast.Name) and nom == "open":
            mode = _mode_ecriture(noeud, 1)
        elif isinstance(f, ast.Attribute) and nom == "open":
            mode = _mode_methode_open(noeud)
        elif nom == "ZipFile":
            mode = _mode_ecriture(noeud, 1)
        if mode is not None:
            ecritures.append(noeud.lineno)
            if not ecrivain:
                signaler(noeud, f"écriture : open en mode {mode}")
        elif (isinstance(f, ast.Attribute) and nom in METHODES_ECRITURE) or (
            nom in NOMS_ECRITURE and nom != "ZipFile"
        ):
            ecritures.append(noeud.lineno)
            if not ecrivain:
                signaler(noeud, f"écriture : {nom}")
        if (
            isinstance(f, ast.Attribute)
            and nom == "replace"
            and isinstance(f.value, ast.Name)
            and f.value.id == "os"
        ):
            ecritures.append(noeud.lineno)
            if not ecrivain:
                signaler(noeud, "écriture : os.replace")
    if ecrivain and ecritures and not passe_par_le_garde:
        violations.append(Violation(nom_fichier, ecritures[0], "écriture sans guards.chemin_dans_dossier"))
    return violations


@pytest.mark.parametrize("chemin", modules_du_coeur(), ids=lambda p: p.relative_to(CORE).as_posix())
def test_I10_confinement_du_coeur(chemin: Path) -> None:
    ecrivain = chemin.parent == CORE and chemin.name in ECRIVAINS
    violations = analyser_confinement(chemin.read_text(encoding="utf-8"), chemin.name, ecrivain)
    assert violations == [], "\n".join(f"{v.fichier}:{v.ligne} {v.motif}" for v in violations)


@pytest.mark.parametrize(
    ("source", "motif"),
    [
        ("import pickle\n", "import confiné pickle"),
        ("import importlib\nm = importlib.import_module('x')\n", "import confiné importlib"),
        ("from importlib import import_module\n", "import confiné import_module"),
        ("m = __import__('os')\n", "appel confiné __import__"),
        ("import pandas\ndf = pandas.read_pickle('m')\n", "appel confiné read_pickle"),
        ("from pandas import read_pickle\n", "import confiné read_pickle"),
        ("import json\nx = json.loads('1')\n", "appel confiné loads"),
        ("import os\nos.system('dir')\n", "appel confiné system"),
        ("import subprocess\n", "import confiné subprocess"),
        ("import tempfile\n", "import confiné tempfile"),
        ("import joblib\n", "import confiné joblib"),
        ("df.to_csv('x.csv')\n", "écriture : to_csv"),
        ("from pathlib import Path\nPath('x').write_text('a')\n", "écriture : write_text"),
        ("with open('x', 'w') as f:\n    pass\n", "open en mode w"),
        ("with open('x', mode='ab') as f:\n    pass\n", "open en mode ab"),
        ("m = 'w'\nwith open('x', m) as f:\n    pass\n", "open en mode non littéral"),
        ("from pathlib import Path\nPath('x').open('w')\n", "open en mode w"),
        ("import zipfile\nzipfile.ZipFile('a.zip', 'w')\n", "open en mode w"),
        ("flux = archive.open('partie.xml', 'w')\n", "open en mode w"),
        ("figure.savefig('f.png')\n", "écriture : savefig"),
        ("import os\nos.replace('a', 'b')\n", "écriture : os.replace"),
        ("x = eval('1 + 1')\n", "appel confiné eval"),
    ],
)
def test_I10_l_analyse_detecte_les_fautes_plantees(source: str, motif: str) -> None:
    violations = analyser_confinement(source, "exemple.py")
    assert any(motif in v.motif for v in violations), violations


@pytest.mark.parametrize(
    "source",
    [
        "with open('x', encoding='utf-8') as f:\n    texte = f.read()\n",
        "from pathlib import Path\ntexte = Path('x').open().read()\n",
        "import zipfile\nz = zipfile.ZipFile('a.xlsx')\nwith z.open(nom) as flux:\n    flux.read()\n",
        "import re\nmotif = re.compile('a')\n",
        "texte = 'a-b'.replace('-', ' ')\n",
        "import warnings\nwarnings.warn('a')\n",
    ],
    ids=["lecture", "path_lecture", "zip_lecture", "re_compile", "str_replace", "warnings"],
)
def test_I10_l_analyse_laisse_passer_la_lecture(source: str) -> None:
    assert analyser_confinement(source, "exemple.py") == []


def test_I10_ecrivains_passent_par_le_garde() -> None:
    sans_garde = "import joblib\ndef f(m, d):\n    joblib.dump(m, d / 'm.joblib')\n"
    avec_garde = (
        "import joblib\nfrom mecaml.core.guards import chemin_dans_dossier\n"
        "def f(m, d):\n    joblib.dump(m, chemin_dans_dossier(d, 'm.joblib'))\n"
    )
    assert [v.motif for v in analyser_confinement(sans_garde, "report.py", ecrivain=True)] == [
        "écriture sans guards.chemin_dans_dossier"
    ]
    assert analyser_confinement(avec_garde, "report.py", ecrivain=True) == []
    assert any("écriture" in v.motif for v in analyser_confinement(avec_garde, "train.py"))


def test_import_du_coeur_sans_interface_ni_sklearn() -> None:
    """Dans un processus neuf, importer le cœur ne charge ni tkinter ni scikit-learn."""
    programme = (
        "import sys\n"
        "import mecaml, mecaml.core, mecaml.core.types, mecaml.core.session, mecaml.core.catalog\n"
        "charges = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'tkinter', '_tkinter', 'customtkinter', 'sklearn', 'mecaml.ui'} or m.startswith('mecaml.ui'))\n"
        "print(','.join(charges))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    sortie = subprocess.run(
        [sys.executable, "-c", programme], capture_output=True, text=True, env=env, check=True, timeout=120
    )
    assert sortie.stdout.strip() == ""


# ============================================================
# REGISTRE : AJOUT D'UNE TECHNIQUE PAR LA PROCÉDURE DOCUMENTÉE
# ============================================================
MODULE_ESSAI = '''"""Technique d'essai ajoutée selon la procédure du registre."""

from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme


def _fabrique(graine: int, **parametres: object) -> object:
    from sklearn.dummy import DummyRegressor

    return DummyRegressor(**parametres)


ENTREES = (
    EntreeCatalogue(
        id="reference.essai",
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre="C02",
        texte_id="reference.essai",
    ),
)
'''


def _copie_avec_technique(tmp_path: Path) -> Path:
    """Copie du dépôt (code et configuration) où une technique est ajoutée comme documenté."""
    copie = tmp_path / "depot"
    shutil.copytree(SRC, copie / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(RACINE / "pyproject.toml", copie / "pyproject.toml")
    catalogue = copie / "src" / "mecaml" / "core" / "catalog"
    (catalogue / "essai.py").write_text(MODULE_ESSAI, encoding="utf-8")
    init = catalogue / "__init__.py"
    texte = init.read_text(encoding="utf-8")
    # Procédure documentée : `essai` s'ajoute au bloc d'imports (ordre alphabétique) et à `_MODULES`.
    ancre_import = "    exclusion_unique,\n"
    ancre_modules = "_MODULES: tuple[ModuleType, ...] = (\n"
    assert ancre_import in texte and ancre_modules in texte, "procédure du registre à mettre à jour"
    texte = texte.replace(ancre_import, "    essai,\n" + ancre_import, 1)
    texte = texte.replace(ancre_modules, ancre_modules + "    essai,\n", 1)
    init.write_text(texte, encoding="utf-8")
    return copie


def test_registre_ajout_d_une_technique_sans_import_circulaire(tmp_path: Path) -> None:
    copie = _copie_avec_technique(tmp_path)
    programme = "import mecaml.core.catalog as c\nprint(','.join(e.id for e in c.REGISTRE))\n"
    env = {**os.environ, "PYTHONPATH": str(copie / "src")}
    sortie = subprocess.run(
        [sys.executable, "-c", programme], capture_output=True, text=True, env=env, timeout=120
    )
    assert sortie.returncode == 0, sortie.stderr
    identifiants = sortie.stdout.strip().split(",")
    assert identifiants[0] == "reference.essai"
    assert len(identifiants) > 1  # la technique s'ajoute au catalogue existant


@pytest.mark.skipif(not RUFF_PRESENT, reason="ruff non installé")
def test_registre_ajout_d_une_technique_reste_conforme_a_ruff(tmp_path: Path) -> None:
    copie = _copie_avec_technique(tmp_path)
    catalogue = copie / "src" / "mecaml" / "core" / "catalog"
    sortie = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "-n", str(catalogue)],
        capture_output=True,
        text=True,
        cwd=copie,
        timeout=120,
    )
    assert sortie.returncode == 0, sortie.stdout + sortie.stderr


# ============================================================
# CHARTE ET FORMAT DU DÉPÔT
# ============================================================
def _verificateur_tirets():
    spec = importlib.util.spec_from_file_location("verifier_tirets", VERIFIER_TIRETS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_charte_aucun_tiret_ni_trait_d_union_double(capsys: pytest.CaptureFixture[str]) -> None:
    total = _verificateur_tirets().verifier(RACINE)
    assert total == 0, capsys.readouterr().out


def test_charte_le_verificateur_detecte(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.py").write_text("x = 1  # filet " + "-" * 2 + "\n", encoding="utf-8")
    (tmp_path / "b.md").write_text(f"Titre {chr(0x2014)} sous-titre\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("texte" + "-" * 2 + "légal\n", encoding="utf-8")
    assert _verificateur_tirets().verifier(tmp_path) == 2
    sortie = capsys.readouterr().out
    assert "a.py:1" in sortie and "b.md:1" in sortie and "exempté" in sortie


@pytest.mark.skipif(not RUFF_PRESENT, reason="ruff non installé")
def test_format_ruff() -> None:
    sortie = subprocess.run(
        [sys.executable, "-m", "ruff", "format", OPTION_VERIFICATION, "-n", str(RACINE)],
        capture_output=True,
        text=True,
        cwd=RACINE,
        timeout=120,
    )
    assert sortie.returncode == 0, sortie.stdout + sortie.stderr


# Traces de fabrication : écrites par morceaux pour que ce fichier ne les contienne pas lui-même.
TRACES_DE_FABRICATION = (
    "_BUILD" + "_",
    "_mySubA" + "gent",
    "jal" + "on",
    "l'application " + "C2",
    "pilo" + "tage",
    "chan" + "tier",
    "dialec" + "tique",
)
DOSSIERS_IGNORES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def _fichiers_texte(racine: Path):
    for chemin in sorted(racine.rglob("*")):
        if not chemin.is_file() or DOSSIERS_IGNORES & set(chemin.relative_to(racine).parts):
            continue
        try:
            yield chemin, chemin.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binaire : image, classeur


def test_aucune_trace_de_fabrication_dans_le_depot() -> None:
    trouvees = [
        f"{chemin.relative_to(RACINE)} : {trace}"
        for chemin, texte in _fichiers_texte(RACINE)
        for trace in TRACES_DE_FABRICATION
        if trace.lower() in texte.lower()
    ]
    assert not trouvees, "\n".join(trouvees)


def test_anti_trace_le_parcours_detecte(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("Régénéré à chaque " + TRACES_DE_FABRICATION[2] + "\n", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\xff\xfe\x00")
    textes = dict(_fichiers_texte(tmp_path))
    assert list(textes) == [tmp_path / "a.md"]
    assert any(t.lower() in textes[tmp_path / "a.md"].lower() for t in TRACES_DE_FABRICATION)
