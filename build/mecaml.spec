# -*- mode: python ; coding: utf-8 -*-
"""Spécification PyInstaller de MécaML Explorer : un exécutable Windows autonome (un seul fichier).

Construction : python build/construire.py (voir build/README.md). Cette spécification emporte :
  les textes de l'interface et de la pédagogie (mecaml/textes/*.toml), lus à l'exécution par chemin ;
  les icônes (mecaml/assets/), les jeux d'exemple et le projet d'exemple (examples/) ;
  les sous-modules de mecaml et le pilote Tk de matplotlib, que l'analyse statique ne voit pas toujours.
L'exécutable ne contient ni les tests ni les outils de développement.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

RACINE = Path(SPECPATH).resolve().parent  # noqa: F821 (SPECPATH est fourni par PyInstaller)
SRC = RACINE / "src"
PAQUET = SRC / "mecaml"
sys.path.insert(0, str(SRC))  # le paquet doit être importable ici, sans installation, pour en lister les modules

donnees = [
    (str(PAQUET / "textes" / "*.toml"), "mecaml/textes"),
    (str(PAQUET / "assets" / "*.png"), "mecaml/assets"),
    (str(PAQUET / "assets" / "*.ico"), "mecaml/assets"),
    (str(RACINE / "examples" / "*.xlsx"), "examples"),
    (str(RACINE / "examples" / "*.md"), "examples"),
    (str(RACINE / "examples" / "*.json"), "examples"),
    (str(RACINE / "examples" / "projet_exemple" / "*"), "examples/projet_exemple"),
    (str(RACINE / "LICENSE"), "."),
]

modules_caches = collect_submodules("mecaml") + [
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
    "openpyxl.cell._writer",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
]

# Ce que l'application n'emploie pas et qui alourdirait l'exécutable.
exclus = [
    "pytest",
    "IPython",
    "jupyter",
    "notebook",
    "sphinx",
    "torch",
    "tensorflow",
    "keras",
    "numba",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
]

analyse = Analysis(  # noqa: F821
    [str(PAQUET / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=donnees,
    hiddenimports=modules_caches,
    hookspath=[],
    runtime_hooks=[],
    excludes=exclus,
    noarchive=False,
)
archive = PYZ(analyse.pure)  # noqa: F821

executable = EXE(  # noqa: F821
    archive,
    analyse.scripts,
    analyse.binaries,
    analyse.datas,
    [],
    name="MecaML",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon=str(PAQUET / "assets" / "mecaml_icone.ico"),
)
