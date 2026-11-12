"""
Réseau dense Keras, en régression et en classification : technique optionnelle.

Keras n'est jamais requis. Ces deux entrées n'entrent dans les listes de l'application que si le paquet
`keras` est installé (`keras_present`) ; sans lui, elles restent inconnues du registre et un projet qui les
cite est refusé avec un message qui le dit. Elles sont hors du registre de base `REGISTRE`.

Le réseau est un empilement de couches denses avec abandon facultatif, entraîné par Adam, avec arrêt précoce
facultatif sur 10 % des lignes. Il se comporte comme un estimateur scikit-learn (`fit`, `predict`,
`predict_proba`), donc se compare aux autres modèles sur les mêmes plis. Pour la reproductibilité, la
graine du
projet fixe les poids de départ et l'ordre des lots, et les opérations TensorFlow sont rendues déterministes
quand ce moteur est celui de Keras.

L'export d'un script Python autonome n'existe pas pour ce modèle : le script ne dépend que de scikit-learn.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

__all__ = [
    "CLASSIFICATION_RESEAU_DENSE",
    "ENTREES",
    "REGRESSION_RESEAU_DENSE",
    "TEXTES",
    "keras_present",
]

REGRESSION_RESEAU_DENSE = "regression.reseau_dense"
CLASSIFICATION_RESEAU_DENSE = "classification.reseau_dense"

TEXTES = {
    REGRESSION_RESEAU_DENSE: {
        "quoi": "Couches denses entraînées par Adam ; la cible est standardisée en interne.",
        "regarder": "Standardiser les entrées ; l'export du script est indisponible avec ce modèle.",
    },
    CLASSIFICATION_RESEAU_DENSE: {
        "quoi": "Couches denses entraînées par Adam, une probabilité par classe en sortie.",
        "regarder": "Standardiser les entrées ; l'export du script est indisponible avec ce modèle.",
    },
}

DEFAUTS: dict[str, Any] = {
    "couches": (32, 32),
    "epochs": 50,
    "taux_apprentissage": 1e-3,
    "taille_lot": 32,
    "abandon": 0.0,
    "activation": "relu",
    "arret_precoce": True,
}
LIGNES_MIN_ARRET_PRECOCE = 30  # en dessous, 10 % de lignes ne valident plus rien
PATIENCE = 10


def keras_present() -> bool:
    """Vrai si le paquet `keras` est installé ; il n'est pas importé ici, la présence se lit sur le chemin."""
    if "keras" in sys.modules:
        return True
    return any((Path(dossier) / "keras" / "__init__.py").is_file() for dossier in sys.path if dossier)


@functools.cache
def _classes() -> tuple[type, type]:
    """(régresseur, classifieur), définis à la première demande : scikit-learn et Keras restent paresseux."""
    import numpy as np
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.validation import check_array, check_X_y

    class _ReseauDense(BaseEstimator):
        def __init__(
            self,
            couches: tuple[int, ...] = DEFAUTS["couches"],
            epochs: int = DEFAUTS["epochs"],
            taux_apprentissage: float = DEFAUTS["taux_apprentissage"],
            taille_lot: int = DEFAUTS["taille_lot"],
            abandon: float = DEFAUTS["abandon"],
            activation: str = DEFAUTS["activation"],
            arret_precoce: bool = DEFAUTS["arret_precoce"],
            random_state: int = 42,
        ) -> None:
            self.couches = couches
            self.epochs = epochs
            self.taux_apprentissage = taux_apprentissage
            self.taille_lot = taille_lot
            self.abandon = abandon
            self.activation = activation
            self.arret_precoce = arret_precoce
            self.random_state = random_state

        # Construction et entraînement
        def _moteur(self) -> Any:
            """Importe Keras ; la graine et les opérations déterministes sont posées avant tout calcul."""
            import keras

            keras.utils.set_random_seed(int(self.random_state))
            try:  # propre à TensorFlow ; sans effet sur un autre moteur
                import tensorflow as tf

                tf.config.experimental.enable_op_determinism()
            except (ImportError, RuntimeError, ValueError):
                pass
            return keras

        def _construire(self, keras: Any, n_entrees: int, n_sorties: int, classement: bool) -> Any:
            modele = keras.Sequential()
            modele.add(keras.Input(shape=(n_entrees,)))
            for unites in self.couches:
                modele.add(keras.layers.Dense(int(unites), activation=self.activation))
                if self.abandon > 0:
                    modele.add(keras.layers.Dropout(float(self.abandon)))
            modele.add(keras.layers.Dense(n_sorties, activation="softmax" if classement else None))
            modele.compile(
                optimizer=keras.optimizers.Adam(learning_rate=float(self.taux_apprentissage)),
                loss="sparse_categorical_crossentropy" if classement else "mse",
            )
            return modele

        def _ajuster(self, x: Any, y: Any, n_sorties: int, classement: bool) -> None:
            keras = self._moteur()
            modele = self._construire(keras, x.shape[1], n_sorties, classement)
            arret = self.arret_precoce and len(x) >= LIGNES_MIN_ARRET_PRECOCE
            historique = modele.fit(
                x,
                y,
                epochs=int(self.epochs),
                batch_size=int(self.taille_lot),
                validation_split=0.1 if arret else 0.0,
                callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]
                if arret
                else [],
                shuffle=True,
                verbose=0,
            )
            self._modele_ = modele
            self._poids_ = [np.asarray(w) for w in modele.get_weights()]
            self.epochs_faites_ = len(historique.history["loss"])
            self.n_features_in_ = x.shape[1]

        def _lire(self) -> Any:
            if getattr(self, "_modele_", None) is None:
                keras = self._moteur()
                modele = self._construire(keras, self.n_features_in_, self._n_sorties_, self._classement_)
                modele.set_weights(self._poids_)
                self._modele_ = modele
            return self._modele_

        def _entrees(self, x: Any) -> Any:
            x = check_array(x, dtype=np.float32, ensure_all_finite=True)
            if x.shape[1] != self.n_features_in_:
                raise ValueError(f"{x.shape[1]} variables reçues, {self.n_features_in_} attendues.")
            return x

        def __getstate__(self) -> dict[str, Any]:
            etat = dict(self.__dict__)
            etat["_modele_"] = None  # reconstruit depuis les poids à la demande
            return etat

    class Regresseur(RegressorMixin, _ReseauDense):
        def fit(self, X: Any, y: Any) -> Regresseur:  # noqa: N803
            x, y = check_X_y(X, y, dtype=np.float32, y_numeric=True, ensure_all_finite=True)
            self._moyenne_, self._echelle_ = float(y.mean()), float(y.std()) or 1.0
            self._classement_, self._n_sorties_ = False, 1
            self._ajuster(x, (y - self._moyenne_) / self._echelle_, 1, False)
            return self

        def predict(self, X: Any) -> Any:  # noqa: N803
            sortie = self._lire().predict(self._entrees(X), verbose=0)[:, 0]
            return sortie.astype("float64") * self._echelle_ + self._moyenne_

    class Classifieur(ClassifierMixin, _ReseauDense):
        def fit(self, X: Any, y: Any) -> Classifieur:  # noqa: N803
            x, y = check_X_y(X, y, dtype=np.float32, ensure_all_finite=True)
            self.classes_ = np.unique(y)
            if len(self.classes_) < 2:
                raise ValueError("Une classification demande au moins deux classes dans l'apprentissage.")
            self._classement_, self._n_sorties_ = True, len(self.classes_)
            self._ajuster(x, np.searchsorted(self.classes_, y), self._n_sorties_, True)
            return self

        def predict_proba(self, X: Any) -> Any:  # noqa: N803
            return self._lire().predict(self._entrees(X), verbose=0).astype("float64")

        def predict(self, X: Any) -> Any:  # noqa: N803
            return self.classes_[self.predict_proba(X).argmax(axis=1)]

    return Regresseur, Classifieur


def _fabrique_regression(graine: int, **parametres: Any) -> Any:
    regresseur, _ = _classes()
    return regresseur(**preparer(REGRESSION_RESEAU_DENSE, DEFAUTS, parametres, graine))


def _fabrique_classification(graine: int, **parametres: Any) -> Any:
    _, classifieur = _classes()
    return classifieur(**preparer(CLASSIFICATION_RESEAU_DENSE, DEFAUTS, parametres, graine))


_GRILLE = (("epochs", (25, 50, 100)), ("taux_apprentissage", (0.001, 0.01)))

ENTREES = (
    EntreeCatalogue(
        id=REGRESSION_RESEAU_DENSE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique_regression,
        grille_bornee=_GRILLE,
        cout=Cout.LOURD,
        ref_livre=ref(REGRESSION_RESEAU_DENSE),
        texte_id=REGRESSION_RESEAU_DENSE,
    ),
    EntreeCatalogue(
        id=CLASSIFICATION_RESEAU_DENSE,
        problemes=frozenset({Probleme.CLASSIFICATION}),
        fabrique=_fabrique_classification,
        grille_bornee=_GRILLE,
        cout=Cout.LOURD,
        ref_livre=ref(CLASSIFICATION_RESEAU_DENSE),
        texte_id=CLASSIFICATION_RESEAU_DENSE,
    ),
)
