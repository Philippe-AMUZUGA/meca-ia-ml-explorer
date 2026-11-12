"""Référence par persistance : la prévision est la dernière valeur connue de la cible de la même unité.

« Demain ressemble à hier » : le modèle lit la cible décalée d'un pas, construite par `preprocess` (jamais la
valeur courante) ; la première ligne d'une unité, sans passé, reçoit la médiane de l'apprentissage. Elle ne
vaut comme référence que si la cible du pas précédent est connue au moment de la prévision : le choix doit le
déclarer par `cible_connue=True`, sinon il est refusé (`mecaml.core.cible_passee`) ; une cible qui décompte,
comme une durée de vie restante, est refusée même déclarée. Rien ne l'ajoute d'office à une comparaison. La
classe est créée au premier usage, pour que scikit-learn ne soit importé qu'à ce moment.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.cible_passee import PARAMETRE, PERSISTANCE, exiger_declaration
from mecaml.core.types import Probleme

REFERENCE_PERSISTANCE = PERSISTANCE


@functools.cache
def _classe_modele() -> type:
    """Classe `Persistance`, créée au premier usage."""
    from sklearn.base import BaseEstimator, RegressorMixin

    class Persistance(RegressorMixin, BaseEstimator):
        """Prédit la première colonne reçue : la cible décalée d'un pas."""

        def fit(self, x: Any, y: Any = None) -> Persistance:
            """Ne retient rien ; vérifie qu'une colonne au moins est reçue."""
            valeurs = np.asarray(x, dtype=float)
            if valeurs.ndim != 2 or valeurs.shape[1] < 1:
                raise ValueError("Persistance : la cible décalée doit être la première colonne reçue.")
            self.n_features_in_ = valeurs.shape[1]
            return self

        def predict(self, x: Any) -> np.ndarray:
            """La valeur passée de la cible, telle quelle."""
            return np.asarray(x, dtype=float)[:, 0].copy()

    Persistance.__module__ = __name__
    Persistance.__qualname__ = "Persistance"
    return Persistance


def __getattr__(nom: str) -> Any:
    """Accès paresseux à `Persistance` (et relecture par pickle ou joblib)."""
    if nom == "Persistance":
        return _classe_modele()
    raise AttributeError(f"module {__name__!r} has no attribute {nom!r}")


def _fabrique(graine: int, **parametres: Any) -> Any:
    exiger_declaration(REFERENCE_PERSISTANCE, parametres)
    if set(parametres) - {PARAMETRE}:
        raise ValueError(f"{REFERENCE_PERSISTANCE} : aucun paramètre autre que `{PARAMETRE}`.")
    return _classe_modele()()


ENTREES = (
    EntreeCatalogue(
        id=REFERENCE_PERSISTANCE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(),
        cout=Cout.LEGER,
        ref_livre=ref(REFERENCE_PERSISTANCE),
        texte_id=REFERENCE_PERSISTANCE,
    ),
)
