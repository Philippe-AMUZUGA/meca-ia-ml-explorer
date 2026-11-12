"""
MécaML : variables construites sur des séries temporelles, sans fuite du futur.

Chaque variable construite à l'instant t d'une unité ne dépend que des valeurs de
cette unité aux instants inférieurs ou égaux à t (invariant I5) : retards,
différences, fenêtres glissantes (moyenne, écart type), moyenne exponentielle et
descripteurs de fréquence d'une fenêtre (fréquence dominante, centroïde
spectral). Les lignes sont rangées par unité puis par date pour le calcul et
rendues dans leur ordre d'arrivée. Le pas de temps est supposé régulier dans
chaque unité : les fenêtres comptent des lignes, non des durées.

Le transformeur `VariablesTemporelles` suit l'interface de scikit-learn et
s'ajuste donc dans le pipeline (invariant I2). À l'ajustement, il garde les
dernières lignes de chaque unité ; appliqué ensuite à des lignes postérieures
(pli de validation, test), il les fait précéder de cet historique : les premiers
retards du test viennent de l'apprentissage, jamais l'inverse. Les lignes
d'apprentissage sans cible, retirées avant le modèle, lui sont données en
`contexte` : elles complètent les séries (le retard d'une ligne est la valeur de
la ligne qui la précède dans le temps, notée ou non) sans être rendues. La classe
est créée au premier usage, pour que scikit-learn ne soit importé qu'à ce moment.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# `VariablesTemporelles` s'obtient aussi par attribut du module (création paresseuse).
__all__ = [
    "DEMI_VIE_MAX",
    "PROFONDEUR_MAX",
    "VARIABLES_MAX",
    "ConfigTemporelle",
    "construire_variables",
    "fabrique_variables_temporelles",
]

PROFONDEUR_MAX = 1000  # retard ou fenêtre la plus longue, en lignes
DEMI_VIE_MAX = 100.0  # en lignes ; l'historique gardé couvre dix demi-vies
VARIABLES_MAX = 500  # borne du nombre de colonnes produites (I9)
_SEUIL_SPECTRE = 1e-12  # énergie relative sous laquelle une fenêtre est dite constante


# ============================================================
# CONFIGURATION
# ============================================================
def _entiers(nom: str, valeurs: Any, minimum: int) -> tuple[int, ...]:
    """Tuple trié d'entiers distincts compris entre `minimum` et PROFONDEUR_MAX."""
    if isinstance(valeurs, (str, bytes)) or not hasattr(valeurs, "__iter__"):
        raise TypeError(f"Variables temporelles : `{nom}` attend une liste d'entiers.")
    propres = []
    for v in valeurs:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"Variables temporelles : `{nom}` attend des entiers, reçu {v!r}.")
        if not minimum <= int(v) <= PROFONDEUR_MAX:
            raise ValueError(
                f"Variables temporelles : `{nom}` = {v} hors de [{minimum}, {PROFONDEUR_MAX}] lignes."
            )
        propres.append(int(v))
    return tuple(sorted(set(propres)))


def _demi_vies(valeurs: Any) -> tuple[float, ...]:
    if isinstance(valeurs, (str, bytes)) or not hasattr(valeurs, "__iter__"):
        raise TypeError("Variables temporelles : `demi_vies` attend une liste de nombres.")
    propres = []
    for v in valeurs:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError(f"Variables temporelles : demi-vie {v!r} non numérique.")
        if not 0.0 < float(v) <= DEMI_VIE_MAX:
            raise ValueError(f"Variables temporelles : demi-vie {v} hors de ]0, {DEMI_VIE_MAX:g}] lignes.")
        propres.append(float(v))
    return tuple(sorted(set(propres)))


@dataclass(frozen=True)
class ConfigTemporelle:
    """Ce que l'on construit, sur quelles colonnes, dans quelles unités, selon quelle date.

    `retards` et `differences` : décalages en lignes (au moins 1) ; `fenetres` :
    largeurs des moyennes et écarts types glissants (au moins 2) ; `demi_vies` :
    moyennes exponentielles ; `fenetres_frequence` : largeurs des fenêtres
    analysées par transformée de Fourier (au moins 4).
    """

    colonne_date: str
    colonnes: tuple[str, ...]
    colonne_groupe: str | None = None
    retards: tuple[int, ...] = (1,)
    fenetres: tuple[int, ...] = ()
    differences: tuple[int, ...] = ()
    demi_vies: tuple[float, ...] = ()
    fenetres_frequence: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self._verifier_colonnes()
        object.__setattr__(self, "retards", _entiers("retards", self.retards, 1))
        object.__setattr__(self, "differences", _entiers("differences", self.differences, 1))
        object.__setattr__(self, "fenetres", _entiers("fenetres", self.fenetres, 2))
        object.__setattr__(
            self, "fenetres_frequence", _entiers("fenetres_frequence", self.fenetres_frequence, 4)
        )
        object.__setattr__(self, "demi_vies", _demi_vies(self.demi_vies))
        n = len(self.noms())
        if n == 0:
            raise ValueError("Variables temporelles : aucune variable demandée (retards, fenêtres...).")
        if n > VARIABLES_MAX:
            raise ValueError(
                f"Variables temporelles : {n} colonnes demandées, au plus {VARIABLES_MAX} ; réduire "
                "le nombre de colonnes, de retards ou de fenêtres."
            )

    def _verifier_colonnes(self) -> None:
        if not isinstance(self.colonne_date, str) or not self.colonne_date:
            raise ValueError("Variables temporelles : `colonne_date` est obligatoire.")
        if self.colonne_groupe is not None and (
            not isinstance(self.colonne_groupe, str) or not self.colonne_groupe
        ):
            raise ValueError("Variables temporelles : `colonne_groupe` doit être un nom de colonne ou None.")
        if isinstance(self.colonnes, str) or any(not isinstance(c, str) for c in self.colonnes):
            raise TypeError("Variables temporelles : `colonnes` attend des noms de colonnes.")
        colonnes = tuple(self.colonnes)
        if not colonnes or len(set(colonnes)) != len(colonnes):
            raise ValueError("Variables temporelles : `colonnes` vide ou avec doublons.")
        reservees = {self.colonne_date, self.colonne_groupe} & set(colonnes)
        if reservees:
            raise ValueError(
                f"Variables temporelles : la date et l'unité ne se transforment pas ({sorted(reservees)})."
            )
        object.__setattr__(self, "colonnes", colonnes)

    @property
    def profondeur(self) -> int:
        """Lignes d'historique utiles à l'instant t (dix demi-vies pour la moyenne exponentielle)."""
        besoins = [*self.retards, *self.differences, *(w - 1 for w in self.fenetres)]
        besoins += [w - 1 for w in self.fenetres_frequence]
        besoins += [math.ceil(10 * h) for h in self.demi_vies]
        return max(besoins, default=0)

    def noms(self) -> tuple[str, ...]:
        """Noms des colonnes produites, colonne par colonne, dans un ordre fixe."""
        noms: list[str] = []
        for c in self.colonnes:
            noms += [f"{c}_retard_{k}" for k in self.retards]
            noms += [f"{c}_difference_{k}" for k in self.differences]
            for w in self.fenetres:
                noms += [f"{c}_moyenne_{w}", f"{c}_ecart_type_{w}"]
            noms += [f"{c}_moyenne_exp_{h:g}" for h in self.demi_vies]
            for w in self.fenetres_frequence:
                noms += [f"{c}_frequence_dominante_{w}", f"{c}_centroide_spectral_{w}"]
        return tuple(noms)


# ============================================================
# CALCUL
# ============================================================
def _extraire(donnees: pd.DataFrame, config: ConfigTemporelle, ordre: np.ndarray) -> pd.DataFrame:
    """Table de travail : unité, date, ordre d'arrivée, valeurs en flottants (noms internes)."""
    if not isinstance(donnees, pd.DataFrame):
        raise TypeError("Variables temporelles : un DataFrame aux colonnes nommées est attendu.")
    requises = [
        config.colonne_date,
        *([config.colonne_groupe] if config.colonne_groupe else []),
        *config.colonnes,
    ]
    absentes = [c for c in requises if c not in donnees.columns]
    if absentes:
        raise ValueError(f"Variables temporelles : colonnes absentes {absentes[:5]}.")
    travail = {"unite": donnees[config.colonne_groupe].to_numpy() if config.colonne_groupe else 0}
    travail["date"] = donnees[config.colonne_date].to_numpy()
    travail["ordre"] = ordre
    for i, c in enumerate(config.colonnes):
        travail[f"v{i}"] = _flottants(donnees[c])
    extrait = pd.DataFrame(travail)
    for cle, quoi in (("unite", "l'unité"), ("date", "la date")):
        if extrait[cle].isna().any():
            raise ValueError(f"Variables temporelles : valeurs manquantes dans la colonne de {quoi}.")
    return extrait


def _flottants(serie: pd.Series) -> np.ndarray:
    if not (pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_bool_dtype(serie)):
        raise ValueError(
            f"Variables temporelles : la colonne « {serie.name} » n'est pas numérique ; la retirer des "
            "colonnes transformées ou l'encoder d'abord."
        )
    return serie.to_numpy(dtype="float64", na_value=np.nan)


def _connues(historique: pd.DataFrame, extrait: pd.DataFrame) -> pd.DataFrame:
    """Lignes connues des unités reçues, hors instants déjà reçus (la ligne reçue l'emporte).

    Une ligne connue postérieure à une ligne reçue n'en change aucune variable (I5) ;
    seules les antérieures servent.
    """
    garde = historique[historique["unite"].isin(extrait["unite"].unique())]
    recus = pd.MultiIndex.from_frame(extrait[["unite", "date"]])
    return garde[~pd.MultiIndex.from_frame(garde[["unite", "date"]]).isin(recus)]


def _preparer(extrait: pd.DataFrame, historique: pd.DataFrame | None) -> pd.DataFrame:
    """Lignes connues et lignes reçues, rangées par unité et par date ; dates uniques par unité."""
    blocs = [extrait] if historique is None or historique.empty else [_connues(historique, extrait), extrait]
    travail = pd.concat(blocs, ignore_index=True) if len(blocs) > 1 else extrait
    travail = travail.sort_values(["unite", "date"], kind="mergesort", ignore_index=True)
    doubles = travail.duplicated(["unite", "date"])
    if doubles.any():
        ligne = travail.loc[doubles.idxmax()]
        raise ValueError(
            f"Variables temporelles : date {ligne['date']} en double dans l'unité {ligne['unite']} ; "
            "l'ordre des lignes serait ambigu. Agréger les mesures simultanées, ou désigner la colonne "
            "d'unité."
        )
    return travail


def _bornes(unites: pd.Series) -> np.ndarray:
    """Débuts de chaque unité dans la table rangée, puis la longueur totale."""
    codes = pd.factorize(unites)[0]
    debuts = np.flatnonzero(np.diff(codes) != 0) + 1
    return np.concatenate(([0], debuts, [len(codes)]))


def _frequences(valeurs: np.ndarray, bornes: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Fréquence dominante et centroïde spectral (cycles par ligne) de la fenêtre finissant en t."""
    dominante = np.full(valeurs.size, np.nan)
    centroide = np.full(valeurs.size, np.nan)
    frequences = np.arange(1, w // 2 + 1) / w
    for debut, fin in itertools.pairwise(bornes):
        if fin - debut < w:
            continue
        brutes = np.lib.stride_tricks.sliding_window_view(valeurs[debut:fin], w)
        finies = np.isfinite(brutes).all(axis=1)
        fenetres = np.where(finies[:, None], brutes, 0.0)
        ecarts = fenetres - fenetres.mean(axis=1, keepdims=True)
        puissance = np.abs(np.fft.rfft(ecarts, axis=1)[:, 1:]) ** 2
        total = puissance.sum(axis=1)
        valides = finies & (total > _SEUIL_SPECTRE * w * (fenetres**2).sum(axis=1)) & (total > 0)
        dominante[debut + w - 1 : fin] = np.where(valides, frequences[np.argmax(puissance, axis=1)], np.nan)
        rapport = np.divide(puissance @ frequences, total, out=np.full(total.size, np.nan), where=valides)
        centroide[debut + w - 1 : fin] = rapport
    return dominante, centroide


def _variables_colonne(
    serie: pd.Series, unites: pd.Series, bornes: np.ndarray, config: ConfigTemporelle
) -> list:
    """Variables d'une colonne, dans l'ordre de `ConfigTemporelle.noms`, alignées sur la table rangée."""
    groupes = serie.groupby(unites, sort=False)
    sorties: list = [groupes.shift(k) for k in config.retards]
    sorties += [serie - groupes.shift(k) for k in config.differences]
    for w in config.fenetres:
        glissant = groupes.rolling(w, min_periods=w)
        plate = (glissant.max() - glissant.min()).droplevel(0) == 0  # écart type exactement nul
        sorties += [glissant.mean().droplevel(0), glissant.std().droplevel(0).mask(plate, 0.0)]
    sorties += [groupes.ewm(halflife=h).mean().droplevel(0) for h in config.demi_vies]
    for w in config.fenetres_frequence:
        sorties += list(_frequences(serie.to_numpy(), bornes, w))
    return [s.reindex(serie.index).to_numpy() if isinstance(s, pd.Series) else s for s in sorties]


def construire_variables(
    donnees: pd.DataFrame, config: ConfigTemporelle, historique: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Variables construites, une ligne par ligne reçue, dans l'ordre et sous l'index reçus.

    `historique` : lignes connues gardées à l'ajustement (forme interne : fin de chaque
    unité, lignes de contexte), rangées avec les lignes reçues de la même unité ; elles
    servent au calcul et ne sont pas rendues.
    """
    extrait = _extraire(donnees, config, np.arange(len(donnees)))
    travail = _preparer(extrait, historique)
    bornes = _bornes(travail["unite"])
    colonnes: list[np.ndarray] = []
    for i in range(len(config.colonnes)):
        colonnes += _variables_colonne(travail[f"v{i}"], travail["unite"], bornes, config)
    matrice = np.column_stack(colonnes) if colonnes else np.empty((len(travail), 0))
    recues = travail["ordre"].to_numpy()
    garde = np.flatnonzero(recues >= 0)
    rangees = garde[np.argsort(recues[garde], kind="stable")]
    return pd.DataFrame(matrice[rangees], index=donnees.index, columns=list(config.noms()))


def _forme_connue(donnees: pd.DataFrame, config: ConfigTemporelle) -> pd.DataFrame:
    """Forme interne de lignes connues, marquées comme non rendues (ordre -1)."""
    return _extraire(donnees, config, np.full(len(donnees), -1))


def _historique_de(
    donnees: pd.DataFrame, config: ConfigTemporelle, contexte: pd.DataFrame | None, memoire: bool
) -> pd.DataFrame | None:
    """Fin de chaque unité (si `memoire`) et toutes les lignes de contexte, en forme interne.

    La fin d'une unité compte les `profondeur` dernières lignes connues jusqu'à sa
    dernière date ajustée, contexte compris. Lignes ajustées et contexte réunis
    doivent avoir des instants distincts par unité.
    """
    contexte_connu = None
    if contexte is not None:
        if not isinstance(contexte, pd.DataFrame):
            raise TypeError("VariablesTemporelles : `contexte` attend un DataFrame ou None.")
        contexte_connu = _forme_connue(contexte, config) if len(contexte) else None
    ajustees = _forme_connue(donnees, config)
    tout = _preparer(
        pd.concat([b for b in (ajustees, contexte_connu) if b is not None], ignore_index=True), None
    )
    if not memoire:
        return contexte_connu
    derniere = ajustees.groupby("unite", sort=False)["date"].max()
    jusqu_a_la_fin = tout[(tout["date"] <= tout["unite"].map(derniere)).to_numpy(dtype=bool, na_value=False)]
    fin = jusqu_a_la_fin.groupby("unite", sort=False).tail(max(config.profondeur, 1))
    retenues = fin if contexte_connu is None else pd.concat([fin, contexte_connu], ignore_index=True)
    return retenues.drop_duplicates(["unite", "date"]).reset_index(drop=True)


# ============================================================
# TRANSFORMEUR COMPATIBLE SCIKIT-LEARN
# ============================================================
def _config_depuis(estimateur: Any, x: pd.DataFrame) -> ConfigTemporelle:
    reservees = {estimateur.colonne_date, estimateur.colonne_groupe}
    colonnes = tuple(estimateur.colonnes) or tuple(c for c in x.columns if c not in reservees)
    return ConfigTemporelle(
        colonne_date=estimateur.colonne_date,
        colonnes=colonnes,
        colonne_groupe=estimateur.colonne_groupe,
        retards=estimateur.retards,
        fenetres=estimateur.fenetres,
        differences=estimateur.differences,
        demi_vies=estimateur.demi_vies,
        fenetres_frequence=estimateur.fenetres_frequence,
    )


def _ajuster(estimateur: Any, x: Any) -> Any:
    if not isinstance(x, pd.DataFrame):
        raise TypeError("VariablesTemporelles : un DataFrame aux colonnes nommées est attendu.")
    config = _config_depuis(estimateur, x)
    estimateur.historique_ = _historique_de(x, config, estimateur.contexte, estimateur.memoire)
    estimateur.config_ = config
    estimateur.n_features_in_ = x.shape[1]
    estimateur.feature_names_in_ = np.asarray(x.columns, dtype=object)
    return estimateur


def _transformer(estimateur: Any, x: Any) -> np.ndarray:
    from sklearn.utils.validation import check_is_fitted

    check_is_fitted(estimateur, "config_")
    if not isinstance(x, pd.DataFrame):
        raise TypeError("VariablesTemporelles : un DataFrame aux colonnes nommées est attendu.")
    return construire_variables(x, estimateur.config_, estimateur.historique_).to_numpy(dtype="float64")


@functools.cache
def _classe_transformeur() -> type:
    """Classe `VariablesTemporelles`, créée au premier usage (import paresseux de scikit-learn)."""
    from sklearn.base import BaseEstimator, TransformerMixin

    class VariablesTemporelles(TransformerMixin, BaseEstimator):
        """Retards, différences, fenêtres, moyennes exponentielles, fréquences, par unité et sans fuite.

        `contexte` : lignes d'apprentissage retirées avant le modèle (cible manquante),
        avec la date, l'unité et les colonnes transformées ; elles complètent les séries
        au calcul et ne sont jamais rendues. `memoire` : garder la fin de chaque unité.
        """

        def __init__(
            self,
            *,
            colonne_date: str,
            colonne_groupe: str | None = None,
            colonnes: tuple[str, ...] = (),
            retards: tuple[int, ...] = (1,),
            fenetres: tuple[int, ...] = (),
            differences: tuple[int, ...] = (),
            demi_vies: tuple[float, ...] = (),
            fenetres_frequence: tuple[int, ...] = (),
            memoire: bool = True,
            contexte: pd.DataFrame | None = None,
        ) -> None:
            self.colonne_date = colonne_date
            self.colonne_groupe = colonne_groupe
            self.colonnes = colonnes
            self.retards = retards
            self.fenetres = fenetres
            self.differences = differences
            self.demi_vies = demi_vies
            self.fenetres_frequence = fenetres_frequence
            self.memoire = memoire
            self.contexte = contexte

        def fit(self, x: pd.DataFrame, y: Any = None) -> VariablesTemporelles:
            """Vérifie la configuration ; garde la fin de chaque unité et le contexte."""
            return _ajuster(self, x)

        def transform(self, x: pd.DataFrame) -> np.ndarray:
            """Variables construites des lignes reçues, séries complétées par les lignes connues."""
            return _transformer(self, x)

        def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
            """Noms des colonnes produites."""
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(self, "config_")
            return np.asarray(self.config_.noms(), dtype=object)

        def __sklearn_tags__(self) -> Any:
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = True
            return tags

    VariablesTemporelles.__module__ = __name__
    VariablesTemporelles.__qualname__ = "VariablesTemporelles"
    return VariablesTemporelles


def __getattr__(nom: str) -> Any:
    """Accès paresseux à `VariablesTemporelles` (et relecture par pickle ou joblib)."""
    if nom == "VariablesTemporelles":
        return _classe_transformeur()
    raise AttributeError(f"module {__name__!r} has no attribute {nom!r}")


def fabrique_variables_temporelles(graine: int = 0, **parametres: Any) -> Any:
    """Fabrique pour le registre des techniques ; la graine est sans objet (calcul déterministe)."""
    return _classe_transformeur()(**parametres)
