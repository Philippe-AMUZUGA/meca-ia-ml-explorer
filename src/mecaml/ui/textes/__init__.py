"""
MécaML : textes de l'interface, rangés par clés dans `interface_fr.toml`.

L'interface n'écrit aucune phrase en dur : elle demande un texte par sa clé (`t("ecran.s3.titre")`). Un texte
peut contenir des champs `{nom}` remplis par les arguments de `t`. L'espace insécable est posée à la lecture
avant « : ; ! ? » et à l'intérieur des guillemets français, pour que le fichier reste simple à relire.

Les titres des techniques (clés `technique.<identifiant>`) viennent du cœur (`mecaml.core.libelles`), qui
les emploie aussi dans le rapport et les figures.

Les textes pédagogiques des techniques (ce qu'elles font, ce qu'il faut regarder) sont ceux de
`mecaml/textes/etapes_fr.toml` quand ce fichier existe ; `pedagogie(texte_id)` les rend, ou `None`.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from mecaml.core import catalog, libelles
from mecaml.core.catalog.reseau_dense import TEXTES as TEXTES_OPTIONNELS

__all__ = ["cles", "existe", "pedagogie", "t", "titre_technique"]

_INSECABLE = " "
_DOSSIER = Path(__file__).resolve().parents[2] / "textes"
_AVANT_PONCTUATION = re.compile(r" ([:;!?»])")


def _normaliser(texte: str) -> str:
    """Espaces insécables de la typographie française."""
    texte = _AVANT_PONCTUATION.sub(_INSECABLE + r"\1", texte)
    return texte.replace("« ", "«" + _INSECABLE)


def _aplatir(table: dict[str, Any], prefixe: str = "") -> dict[str, str]:
    plat: dict[str, str] = {}
    for nom, valeur in table.items():
        cle = f"{prefixe}{nom}"
        if isinstance(valeur, dict):
            plat.update(_aplatir(valeur, f"{cle}."))
        else:
            plat[cle] = _normaliser(str(valeur))
    return plat


@lru_cache(maxsize=1)
def _textes() -> dict[str, str]:
    source = (_DOSSIER / "interface_fr.toml").read_text(encoding="utf-8")
    textes = _aplatir(tomllib.loads(source))
    textes.update({f"technique.{identifiant}": titre for identifiant, titre in libelles.TITRES.items()})
    for entree in catalog.optionnelles_disponibles():
        textes[f"technique.{entree.id}"] = libelles.technique(entree.id)
    return textes


def cles() -> frozenset[str]:
    """Toutes les clés de texte de l'interface."""
    return frozenset(_textes())


def existe(cle: str) -> bool:
    """Vrai si la clé a un texte."""
    return cle in _textes()


def t(cle: str, **valeurs: Any) -> str:
    """Texte de la clé, champs remplis ; `KeyError` explicite si la clé manque."""
    try:
        texte = _textes()[cle]
    except KeyError:
        raise KeyError(f"Clé de texte absente de interface_fr.toml : {cle!r}.") from None
    return texte.format(**valeurs) if valeurs else texte


def titre_technique(id_technique: str) -> str:
    """Titre court d'une technique du catalogue ; son identifiant à défaut."""
    return _textes().get(f"technique.{id_technique}", id_technique)


@lru_cache(maxsize=1)
def _pedagogie() -> dict[str, dict[str, str]]:
    try:
        source = (_DOSSIER / "etapes_fr.toml").read_text(encoding="utf-8")
    except OSError:
        return {}
    textes = {
        cle: {k: _normaliser(str(v)) for k, v in valeur.items()}
        for cle, valeur in tomllib.loads(source).items()
        if isinstance(valeur, dict)
    }
    for entree in catalog.optionnelles_disponibles():
        textes[entree.texte_id] = {k: _normaliser(v) for k, v in TEXTES_OPTIONNELS[entree.texte_id].items()}
    return textes


def pedagogie(texte_id: str) -> dict[str, str] | None:
    """Textes pédagogiques d'une technique ou métrique (clés `quoi` et `regarder`)."""
    return _pedagogie().get(texte_id)
