"""Écrêtage de Tukey : les valeurs hors de [Q1 - k IQR, Q3 + k IQR] sont ramenées aux bornes.

La règle de Tukey sert aussi à repérer les valeurs aberrantes d'une colonne (`valeurs_aberrantes`). Comme tout
prétraitement, l'écrêtage apprend ses bornes sur l'apprentissage du pli, jamais sur la validation
ni sur le test.
La classe du transformeur est créée au premier usage, pour que scikit-learn ne soit importé qu'à ce moment.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

TRANSFORMATION_ECRETAGE_TUKEY = "transformation.ecretage_tukey"
K_MAX = 10.0


def bornes_de_tukey(valeurs: Any, k: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Bornes basse et haute de la règle de Tukey, colonne par colonne (valeurs manquantes ignorées).

    Une colonne sans aucune valeur reçoit des bornes infinies : rien n'y est jugé aberrant.
    """
    matrice = np.asarray(valeurs, dtype=float)
    if matrice.ndim == 1:
        matrice = matrice[:, None]
    basses = np.full(matrice.shape[1], -np.inf)
    hautes = np.full(matrice.shape[1], np.inf)
    for j in range(matrice.shape[1]):
        colonne = matrice[:, j]
        colonne = colonne[~np.isnan(colonne)]
        if colonne.size:
            q1, q3 = np.percentile(colonne, [25, 75])
            basses[j], hautes[j] = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return basses, hautes


def valeurs_aberrantes(valeurs: Any, k: float = 1.5) -> np.ndarray:
    """Masque des valeurs hors des bornes de Tukey (même forme que l'entrée ; manquant : faux)."""
    matrice = np.asarray(valeurs, dtype=float)
    basses, hautes = bornes_de_tukey(matrice, k)
    tableau = matrice if matrice.ndim == 2 else matrice[:, None]
    masque = (tableau < basses) | (tableau > hautes)
    return masque if matrice.ndim == 2 else masque[:, 0]


def _matrice(x: Any) -> np.ndarray:
    matrice = np.asarray(x, dtype=float)
    if matrice.ndim != 2:
        raise ValueError("Écrêtage de Tukey : un tableau à deux dimensions est attendu.")
    return matrice


@functools.cache
def _classe_transformeur() -> type:
    """Classe `EcretageTukey`, créée au premier usage."""
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted

    class EcretageTukey(TransformerMixin, BaseEstimator):
        """Ramène chaque valeur dans [Q1 - k IQR, Q3 + k IQR], bornes apprises à l'ajustement."""

        def __init__(self, k: float = 1.5) -> None:
            self.k = k

        def fit(self, x: Any, y: Any = None) -> EcretageTukey:
            """Apprend les bornes de chaque colonne."""
            valeurs = _matrice(x)
            self.bornes_basses_, self.bornes_hautes_ = bornes_de_tukey(valeurs, self.k)
            self.n_features_in_ = valeurs.shape[1]
            if hasattr(x, "columns"):
                self.feature_names_in_ = np.asarray(x.columns, dtype=object)
            return self

        def transform(self, x: Any) -> np.ndarray:
            """Valeurs ramenées aux bornes apprises ; les manquantes restent manquantes."""
            check_is_fitted(self, "bornes_basses_")
            valeurs = _matrice(x)
            if valeurs.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Écrêtage de Tukey : {valeurs.shape[1]} colonnes reçues, "
                    f"{self.n_features_in_} à l'ajustement."
                )
            return np.clip(valeurs, self.bornes_basses_, self.bornes_hautes_)

        def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
            """Mêmes noms en sortie qu'en entrée."""
            if input_features is not None:
                return np.asarray(input_features, dtype=object)
            if hasattr(self, "feature_names_in_"):
                return self.feature_names_in_
            return np.asarray([f"x{j}" for j in range(self.n_features_in_)], dtype=object)

        def __sklearn_tags__(self) -> Any:
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = True
            return tags

    EcretageTukey.__module__ = __name__
    EcretageTukey.__qualname__ = "EcretageTukey"
    return EcretageTukey


def __getattr__(nom: str) -> Any:
    """Accès paresseux à `EcretageTukey` (et relecture par pickle ou joblib)."""
    if nom == "EcretageTukey":
        return _classe_transformeur()
    raise AttributeError(f"module {__name__!r} has no attribute {nom!r}")


def _fabrique(graine: int, **parametres: Any) -> Any:
    k = float(preparer(TRANSFORMATION_ECRETAGE_TUKEY, {"k": 1.5}, parametres)["k"])
    if not 0.0 < k <= K_MAX:
        raise ValueError(f"{TRANSFORMATION_ECRETAGE_TUKEY} : `k` = {k} hors de ]0 ; {K_MAX:g}].")
    return _classe_transformeur()(k=k)


ENTREES = (
    EntreeCatalogue(
        id=TRANSFORMATION_ECRETAGE_TUKEY,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("k", (1.0, 1.5, 2.0, 3.0)),),
        cout=Cout.LEGER,
        ref_livre=ref(TRANSFORMATION_ECRETAGE_TUKEY),
        texte_id=TRANSFORMATION_ECRETAGE_TUKEY,
    ),
)
