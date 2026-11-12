"""Contrôle de charte : ni tiret demi-cadratin, ni tiret cadratin, ni trait d'union doublé.

Parcourt tous les fichiers texte du dépôt (code, commentaires, documentation,
configuration) et signale chaque occurrence. Le fichier LICENSE est exempté : c'est
le texte de la licence MIT, texte légal reproduit à l'identique.
Code de retour : 0 si aucune occurrence, 1 sinon.

Usage : python tests/outils/verifier_tirets.py [dossier]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Demi-cadratin, cadratin, ou au moins deux traits d'union consécutifs.
DEMI_CADRATIN, CADRATIN = chr(0x2013), chr(0x2014)  # par leur code : aucun tiret dans ce fichier
MOTIF = re.compile(f"[{DEMI_CADRATIN}{CADRATIN}]|-{{2,}}")
EXTENSIONS = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
    ".cfg",
    ".ini",
    ".json",
    ".csv",
    ".bat",
    ".spec",
}
NOMS = {".gitignore", "README", "CHANGELOG"}
EXEMPTES = {"LICENSE"}  # texte légal reproduit à l'identique
IGNORES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist"}


def _ignore(parties: tuple[str, ...]) -> bool:
    """Caches, environnements, et sorties de construction (sous-dossiers de `build/`)."""
    return any(p in IGNORES for p in parties[:-1]) or (parties[0] == "build" and len(parties) > 2)


def fichiers_texte(racine: Path) -> list[Path]:
    """Fichiers texte du dépôt, hors caches et environnements."""
    retenus = []
    for chemin in sorted(racine.rglob("*")):
        if not chemin.is_file() or _ignore(chemin.relative_to(racine).parts):
            continue
        if chemin.suffix.lower() in EXTENSIONS or chemin.name in NOMS or chemin.name in EXEMPTES:
            retenus.append(chemin)
    return retenus


def occurrences(chemin: Path) -> list[tuple[int, str]]:
    """(ligne, extrait) de chaque occurrence interdite."""
    trouvees = []
    for numero, ligne in enumerate(chemin.read_text(encoding="utf-8").splitlines(), 1):
        if MOTIF.search(ligne):
            trouvees.append((numero, ligne.strip()[:100]))
    return trouvees


def verifier(racine: Path) -> int:
    """Affiche le bilan et rend le nombre d'occurrences hors fichiers exemptés."""
    total = 0
    fichiers = fichiers_texte(racine)
    for chemin in fichiers:
        relatif = chemin.relative_to(racine).as_posix()
        trouvees = occurrences(chemin)
        if chemin.name in EXEMPTES:
            if trouvees:
                print(f"{relatif} : {len(trouvees)} occurrence(s), exempté (texte légal à l'identique)")
            continue
        for numero, extrait in trouvees:
            print(f"{relatif}:{numero}: {extrait}")
        total += len(trouvees)
    print(f"{len(fichiers)} fichier(s) contrôlé(s) : {total} occurrence(s) interdite(s).")
    return total


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    dossier = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    sys.exit(1 if verifier(dossier.resolve()) else 0)
