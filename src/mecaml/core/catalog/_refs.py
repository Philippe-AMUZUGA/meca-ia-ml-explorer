"""
MécaML : accès du catalogue à la table unique des renvois au livre.

Chaque module de technique lit son renvoi par `ref(<identifiant>)`. La table, les extensions (techniques
que le livre ne nomme pas) et le contrôle des identifiants sont dans `mecaml.core.renvois` : ce module
n'écrit aucun identifiant du livre.
"""

from __future__ import annotations

from mecaml.core.renvois import technique

__all__ = ["ref"]


def ref(id_technique: str) -> str:
    """Identifiant du livre pour une technique ; `KeyError` explicite si elle n'est pas recensée."""
    return technique(id_technique)
