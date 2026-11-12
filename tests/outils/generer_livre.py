"""Régénère `src/mecaml/core/livre.py` depuis la liste des techniques du livre.

La liste est le tableau de `tests/data/techniques_livre.md` (identifiants `T<cc>-<s>-<k>`, chapitre,
section, libellé). Après une mise à jour de ce fichier :

    python tests/outils/generer_livre.py

Le test `test_livre.py` compare le module généré au tableau et signale tout écart.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
SOURCE = RACINE / "tests" / "data" / "techniques_livre.md"
CIBLE = RACINE / "src" / "mecaml" / "core" / "livre.py"
LIGNE = re.compile(r"^\|\s*(T\d{2}-\d+-\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(.+?)\s*\|\s*$")
HORS = "[hors application]"


def lire_tableau(texte: str) -> list[tuple[str, int, str, str, bool]]:
    """Lignes du tableau : (identifiant, chapitre, section, libellé, hors application)."""
    lignes = []
    for ligne in texte.splitlines():
        trouve = LIGNE.match(ligne)
        if trouve:
            identifiant, chapitre, section, libelle = trouve.groups()
            hors = libelle.endswith(HORS)
            lignes.append((identifiant, int(chapitre), section, libelle.removesuffix(HORS).strip(), hors))
    return lignes


def _litteral(libelle: str) -> str:
    """Libellé sous forme d'une ou plusieurs chaînes accolées, sans ligne de plus de 100 caractères."""
    morceaux = textwrap.wrap(libelle, 70, break_long_words=False, drop_whitespace=False)
    return "\n".join(f"            {morceau!r}" for morceau in morceaux)


def rendre(lignes: list[tuple[str, int, str, str, bool]]) -> str:
    corps = "".join(
        f"        {identifiant!r}: TechniqueLivre(\n            {identifiant!r},\n            {chapitre},\n"
        f"            {section!r},\n{_litteral(libelle)},\n            {hors},\n        ),\n"
        for identifiant, chapitre, section, libelle, hors in lignes
    )
    return (
        '"""\n'
        "MécaML : techniques du livre Méca-IA, avec leur identifiant stable `T<cc>-<s>-<k>`.\n\n"
        "Chaque renvoi au livre de l'application (techniques du catalogue, métriques, constats) est un\n"
        "identifiant de cette liste : chapitre, section, rang dans la section. Module généré par\n"
        "`tests/outils/generer_livre.py` depuis `tests/data/techniques_livre.md` ;\n"
        "ne pas le modifier à la main.\n"
        "Les techniques marquées hors application (images, séquences, réseaux Keras) sont exclues de\n"
        "l'application : données non tabulaires ou dépendance à TensorFlow.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import re\n"
        "from collections.abc import Mapping\n"
        "from dataclasses import dataclass\n"
        "from types import MappingProxyType\n\n"
        '__all__ = ["TECHNIQUES_LIVRE", "TechniqueLivre", "est_identifiant", "technique_du_livre"]\n\n'
        'RE_IDENTIFIANT = re.compile(r"^T\\d{2}-\\d+-\\d+$")\n\n\n'
        "@dataclass(frozen=True)\n"
        "class TechniqueLivre:\n"
        '    """Une technique du livre : où le lecteur la trouve."""\n\n'
        "    id: str\n"
        "    chapitre: int\n"
        "    section: str\n"
        "    libelle: str\n"
        "    hors_application: bool\n\n\n"
        "TECHNIQUES_LIVRE: Mapping[str, TechniqueLivre] = MappingProxyType(\n"
        "    {\n" + corps + "    }\n)\n\n\n"
        "def est_identifiant(texte: object) -> bool:\n"
        '    """Vrai si le texte est un identifiant du livre qui existe dans la liste."""\n'
        "    return isinstance(texte, str) and texte in TECHNIQUES_LIVRE\n\n\n"
        "def technique_du_livre(identifiant: str) -> TechniqueLivre:\n"
        '    """Technique d\'identifiant donné ; `KeyError` explicite sinon."""\n'
        "    try:\n"
        "        return TECHNIQUES_LIVRE[identifiant]\n"
        "    except KeyError:\n"
        '        raise KeyError(f"Renvoi au livre inconnu : {identifiant!r}.") from None\n'
    )


def main() -> int:
    lignes = lire_tableau(SOURCE.read_text(encoding="utf-8"))
    if not lignes:
        print("Aucune technique lue dans", SOURCE)
        return 1
    CIBLE.write_text(rendre(lignes), encoding="utf-8")
    print(f"{len(lignes)} techniques écrites dans {CIBLE.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
