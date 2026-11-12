"""Construit l'exécutable Windows autonome `dist/MecaML.exe` avec PyInstaller, puis le met à l'essai.

Prérequis : Python 3.12 avec les bibliothèques de `requirements.txt` et PyInstaller
(`python -m pip install pyinstaller`, seule étape qui demande le réseau).

Usage : python build/construire.py
Sortie : taille de l'exécutable et résultat de l'essai automatique (`MecaML.exe -t`).
Code de retour : 0 si l'exécutable est construit et que son essai réussit, 1 sinon.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SPEC = RACINE / "build" / "mecaml.spec"
DIST = RACINE / "dist"
TRAVAIL = RACINE / "build" / "travail"
OPTION = "-" * 2  # préfixe des options longues de PyInstaller, écrit sans double trait d'union littéral
LIMITE_DEPOT_OCTETS = 5_000_000  # au delà, l'exécutable va dans une archive de livraison, pas dans le dépôt


def construire() -> Path:
    """Lance PyInstaller sur la spécification ; rend le chemin de l'exécutable."""
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit(
            "PyInstaller est absent : python -m pip install pyinstaller (réseau requis), puis relancer."
        )
    commande = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "-y",
        f"{OPTION}clean",
        f"{OPTION}distpath",
        str(DIST),
        f"{OPTION}workpath",
        str(TRAVAIL),
    ]
    # `collect_submodules("mecaml")` de la spécification importe le paquet : visible sans installation.
    chemins = [str(RACINE / "src"), os.environ.get("PYTHONPATH", "")]
    environnement = {**os.environ, "PYTHONPATH": os.pathsep.join(c for c in chemins if c)}
    if subprocess.run(commande, cwd=RACINE, env=environnement, check=False).returncode:
        raise SystemExit("La construction a échoué : lire le journal de PyInstaller ci-dessus.")
    executable = DIST / "MecaML.exe"
    if not executable.is_file():
        raise SystemExit(f"Construction terminée sans {executable}.")
    return executable


def principal() -> int:
    executable = construire()
    octets = executable.stat().st_size
    print(f"{executable.name} : {octets / 1e6:.1f} Mo")
    if octets > LIMITE_DEPOT_OCTETS:
        print("Plus de 5 Mo : à ranger dans une archive de livraison, pas dans le dépôt.")
    # Sans console, l'exécutable n'écrit rien : le code de retour et l'export disent si l'essai a réussi.
    with tempfile.TemporaryDirectory(prefix="mecaml_essai_exe_") as dossier:
        essai = subprocess.run([str(executable), "-t", "-s", dossier], check=False)
        ecrits = sorted(p.name for p in Path(dossier).iterdir())
    print(f"Essai automatique : code de retour {essai.returncode}, fichiers exportés : {', '.join(ecrits)}")
    return 0 if essai.returncode == 0 and "projet.json" in ecrits else 1


if __name__ == "__main__":
    sys.exit(principal())
