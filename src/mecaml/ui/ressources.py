"""MécaML : chemins des ressources de l'interface (icônes, jeux d'exemple), en source comme en exécutable."""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["chemin_asset", "dossier_exemples", "exemples"]


def _racines() -> list[Path]:
    """Dossiers où chercher : l'exécutable autonome d'abord, puis le dépôt, puis le dossier courant."""
    paquet = Path(__file__).resolve().parents[1]
    candidats = []
    emballe = getattr(sys, "_MEIPASS", None)
    if emballe:
        candidats.append(Path(emballe))
    candidats += [paquet, paquet.parents[1], Path.cwd()]
    return candidats


def chemin_asset(nom: str) -> Path | None:
    """Chemin d'un fichier du dossier `assets`, ou None s'il manque."""
    for racine in _racines():
        for base in (racine / "assets", racine / "mecaml" / "assets"):
            if (base / nom).is_file():
                return base / nom
    return None


def dossier_exemples() -> Path | None:
    """Dossier `examples` des jeux d'exemple, ou None s'il manque."""
    for racine in _racines():
        if (racine / "examples").is_dir():
            return racine / "examples"
    return None


def exemples() -> list[Path]:
    """Classeurs d'exemple, par nom."""
    dossier = dossier_exemples()
    return sorted(dossier.glob("*.xlsx")) if dossier else []
