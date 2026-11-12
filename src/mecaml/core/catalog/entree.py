"""
MécaML : forme d'une entrée du registre des techniques.

Les modules de techniques importent d'ici (`from mecaml.core.catalog.entree import
EntreeCatalogue, Cout`), jamais du paquet `mecaml.core.catalog` lui-même : le
registre peut ainsi les importer en tête de fichier sans import circulaire.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from math import prod
from typing import Any, Protocol

from mecaml.core.types import GRAINE_CONTRAT, LIGNES_MAX_LENTES, Probleme

__all__ = ["FAMILLES", "GRILLE_MAX_COMBINAISONS", "LIGNES_MAX_LENTES", "Cout", "EntreeCatalogue", "Fabrique"]

# Familles reconnues : premier segment de l'identifiant « famille.nom ».
FAMILLES: frozenset[str] = frozenset(
    {
        "manquants",
        "encodage",
        "transformation",
        "selection",
        "chronologique",
        "regression",
        "classification",
        "reference",
        "validation",
        "sans_cible",
        "explication",
    }
)

# Borne du nombre de combinaisons d'une grille de recherche (garde-fou de calcul, provisoire).
GRILLE_MAX_COMBINAISONS = 50

_RE_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


class Cout(StrEnum):
    """Coût de calcul relatif ; LOURD : technique coupée au delà de `LIGNES_MAX_LENTES` lignes.

    C'est le seul critère de coupure : `EntreeCatalogue.coupee(n_lignes)`.
    """

    LEGER = "leger"
    MOYEN = "moyen"
    LOURD = "lourd"


class Fabrique(Protocol):
    """Construit l'objet scikit-learn de la technique (import paresseux à l'intérieur)."""

    def __call__(self, graine: int, **parametres: Any) -> Any: ...


@dataclass(frozen=True)
class EntreeCatalogue:
    """Technique du registre, avec son renvoi au livre et sa clé de texte pédagogique.

    `grille_bornee` : paires (paramètre, valeurs candidates) pour la recherche d'hyperparamètres.
    """

    id: str
    problemes: frozenset[Probleme]
    fabrique: Fabrique
    grille_bornee: tuple[tuple[str, tuple[Any, ...]], ...]
    cout: Cout
    ref_livre: str
    texte_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _RE_ID.match(self.id):
            raise ValueError(f"EntreeCatalogue : identifiant invalide {self.id!r} (forme « famille.nom »).")
        if self.famille not in FAMILLES:
            raise ValueError(f"EntreeCatalogue {self.id} : famille inconnue {self.famille!r}.")
        if not isinstance(self.problemes, frozenset) or not self.problemes:
            raise ValueError(f"EntreeCatalogue {self.id} : `problemes` doit être un frozenset non vide.")
        if any(not isinstance(p, Probleme) for p in self.problemes):
            raise TypeError(f"EntreeCatalogue {self.id} : `problemes` ne contient que des `Probleme`.")
        if not callable(self.fabrique):
            raise TypeError(f"EntreeCatalogue {self.id} : `fabrique` doit être appelable.")
        if not isinstance(self.cout, Cout):
            raise TypeError(f"EntreeCatalogue {self.id} : `cout` doit être un `Cout`.")
        for nom in ("ref_livre", "texte_id"):
            valeur = getattr(self, nom)
            if not isinstance(valeur, str) or not valeur.strip():
                raise ValueError(f"EntreeCatalogue {self.id} : `{nom}` est vide.")
        self._verifier_grille()

    def _verifier_grille(self) -> None:
        noms = []
        for paire in self.grille_bornee:
            if not isinstance(paire, tuple) or len(paire) != 2 or not isinstance(paire[0], str):
                raise TypeError(f"EntreeCatalogue {self.id} : grille invalide {paire!r}.")
            if not isinstance(paire[1], tuple) or not paire[1]:
                raise ValueError(f"EntreeCatalogue {self.id} : valeurs vides pour {paire[0]!r}.")
            noms.append(paire[0])
        if len(set(noms)) != len(noms):
            raise ValueError(f"EntreeCatalogue {self.id} : paramètre en double dans la grille.")
        if self.combinaisons > GRILLE_MAX_COMBINAISONS:
            raise ValueError(
                f"EntreeCatalogue {self.id} : grille de {self.combinaisons} combinaisons, "
                f"borne {GRILLE_MAX_COMBINAISONS} ; réduire les valeurs candidates."
            )

    @property
    def famille(self) -> str:
        """Premier segment de l'identifiant."""
        return self.id.split(".", 1)[0]

    @property
    def combinaisons(self) -> int:
        """Nombre de combinaisons de la grille (1 si la grille est vide)."""
        return prod(len(valeurs) for _, valeurs in self.grille_bornee)

    def grille(self) -> dict[str, list[Any]]:
        """Grille au format attendu par GridSearchCV (copie)."""
        return {nom: list(valeurs) for nom, valeurs in self.grille_bornee}

    def coupee(self, n_lignes: int) -> bool:
        """Vrai si la technique est coupée pour une table de `n_lignes` (coût LOURD, au delà de la borne)."""
        return self.cout is Cout.LOURD and n_lignes > LIGNES_MAX_LENTES

    def construire(self, graine: int = GRAINE_CONTRAT, **parametres: Any) -> Any:
        """Objet scikit-learn neuf, non ajusté (graine du contrat par défaut)."""
        return self.fabrique(graine, **parametres)
