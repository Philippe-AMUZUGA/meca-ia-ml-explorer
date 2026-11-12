"""
MécaML : hyperparamètres réglables des modèles du catalogue.

Chaque modèle déclare ici ses réglages principaux (3 à 6 quand scikit-learn en offre autant) et, repliés
sous « avancé », quelques réglages de plus : nom, genre, valeur par défaut, bornes. Le libellé français et
l'aide d'une ligne viennent de `textes/hyperparametres_fr.toml`. Le cœur refuse toute valeur hors bornes,
quelle que soit sa voie d'entrée (fenêtre de réglage, projet rejoué, ligne de commande), avec un message
qui nomme le réglage, la valeur et la plage permise.

Les bornes de calcul de `catalog/_outils.py` (I9) restent le garde-fou de dernier recours : les bornes
d'ici leur sont égales ou plus strictes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mecaml.core.hyperparametres_textes import MOTS, TEXTES

__all__ = [
    "COUCHES_MAX",
    "SCHEMAS",
    "UNITES_MAX",
    "ErreurHyperparametre",
    "Genre",
    "Hyperparametre",
    "aide",
    "avances",
    "controler",
    "convertir",
    "libelle",
    "principaux",
    "schema",
    "texte_bornes",
    "texte_valeur",
    "valider",
]

COUCHES_MAX = 3  # couches cachées d'un réseau
UNITES_MAX = 256  # unités par couche
BASE_MIN, BASE_MAX = 2, 6  # modèles d'un vote ou d'un empilement


class Genre(StrEnum):
    """Forme d'un réglage : entier, réel, choix, oui ou non, couches ou liste de modèles."""

    ENTIER = "entier"
    REEL = "reel"
    PART = "part"  # entier de `minimum` à `maximum`, ou part entre 0 et 1 exclus (variance conservée)
    CHOIX = "choix"
    BOOLEEN = "booleen"
    COUCHES = "couches"
    LISTE = "liste"


class ErreurHyperparametre(ValueError):
    """Valeur de réglage refusée ; le message dit le réglage, la valeur et ce qui est permis."""


@dataclass(frozen=True)
class Hyperparametre:
    """Un réglage : `defaut` est la valeur effective du modèle, `speciaux` les valeurs non numériques admises.

    `speciaux` : paires (valeur, clé du texte affiché), par exemple `((None, "illimite"),)`.
    `minimum` et `maximum` (inclus) bornent un nombre, la taille d'une couche ou la longueur d'une liste.
    """

    nom: str
    genre: Genre
    defaut: Any
    minimum: float | None = None
    maximum: float | None = None
    choix: tuple[Any, ...] = ()
    speciaux: tuple[tuple[Any, str], ...] = ()
    avance: bool = False


# ============================================================
# CONSTRUCTEURS COURTS
# ============================================================
ILLIMITE = ((None, "illimite"),)
AUCUN = ((None, "aucun"),)
AUTO = (("auto", "auto"),)
TOUTES = ((None, "toutes"),)
PAR_DEFAUT = ((None, "defaut"),)


def _e(nom: str, defaut: Any, mini: int, maxi: int, *, sp: tuple = (), av: bool = False) -> Hyperparametre:
    return Hyperparametre(nom, Genre.ENTIER, defaut, mini, maxi, speciaux=sp, avance=av)


def _r(
    nom: str, defaut: Any, mini: float, maxi: float, *, sp: tuple = (), av: bool = False
) -> Hyperparametre:
    return Hyperparametre(nom, Genre.REEL, defaut, mini, maxi, speciaux=sp, avance=av)


def _p(nom: str, defaut: Any, mini: int, maxi: int, *, sp: tuple = (), av: bool = False) -> Hyperparametre:
    return Hyperparametre(nom, Genre.PART, defaut, mini, maxi, speciaux=sp, avance=av)


def _c(nom: str, defaut: Any, *choix: Any, sp: tuple = (), av: bool = False) -> Hyperparametre:
    tous = (defaut, *choix) if defaut is not None else choix
    return Hyperparametre(nom, Genre.CHOIX, defaut, choix=tous, speciaux=sp, avance=av)


def _b(nom: str, defaut: bool, *, av: bool = False) -> Hyperparametre:
    return Hyperparametre(nom, Genre.BOOLEEN, defaut, avance=av)


def _k(nom: str, defaut: tuple[int, ...], *, av: bool = False) -> Hyperparametre:
    return Hyperparametre(nom, Genre.COUCHES, defaut, 1, UNITES_MAX, avance=av)


def _l(nom: str, defaut: tuple[str, ...]) -> Hyperparametre:
    return Hyperparametre(nom, Genre.LISTE, defaut, BASE_MIN, BASE_MAX)


def _poids(*plus: str) -> Hyperparametre:
    return _c("class_weight", None, "balanced", *plus, sp=AUCUN)


_TOL = _r("tol", 1e-4, 1e-10, 1, av=True)


# ============================================================
# SCHÉMAS PAR MODÈLE
# ============================================================
def _arbre(defaut_profondeur: int, critere: tuple, poids: bool) -> tuple[Hyperparametre, ...]:
    base = (
        _e("max_depth", defaut_profondeur, 1, 50, sp=ILLIMITE),
        _e("min_samples_leaf", 1, 1, 1000),
        _e("min_samples_split", 2, 2, 1000),
        _c("criterion", critere[0], *critere[1:]),
    )
    plus = (_poids(),) if poids else ()
    return (
        *base,
        *plus,
        _r("ccp_alpha", 0.0, 0, 1000, av=True),
        _e("max_leaf_nodes", None, 2, 10_000, sp=ILLIMITE, av=True),
    )


def _foret(bootstrap: bool, poids: bool) -> tuple[Hyperparametre, ...]:
    return (
        _e("n_estimators", 100, 1, 500),
        _e("max_depth", None, 1, 50, sp=ILLIMITE),
        _e("min_samples_leaf", 1, 1, 1000),
        _e("min_samples_split", 2, 2, 1000),
        *((_poids("balanced_subsample"),) if poids else ()),
        _b("bootstrap", bootstrap, av=True),
        _r("ccp_alpha", 0.0, 0, 1000, av=True),
    )


def _voisins() -> tuple[Hyperparametre, ...]:
    return (
        _e("n_neighbors", 5, 1, 500),
        _c("weights", "uniform", "distance"),
        _e("p", 2, 1, 10),
        _c("algorithm", "auto", "ball_tree", "kd_tree", "brute", av=True),
        _e("leaf_size", 30, 1, 200, av=True),
    )


def _bagging() -> tuple[Hyperparametre, ...]:
    return (
        _e("n_estimators", 20, 1, 500),
        _r("max_samples", 1.0, 0.05, 1),
        _r("max_features", 1.0, 0.05, 1),
        _b("bootstrap", True),
        _b("bootstrap_features", False, av=True),
    )


def _histogramme(poids: bool) -> tuple[Hyperparametre, ...]:
    return (
        _r("learning_rate", 0.1, 0.001, 1),
        _e("max_iter", 100, 1, 2000),
        _e("max_depth", None, 1, 50, sp=ILLIMITE),
        _e("max_leaf_nodes", 31, 2, 1000),
        _e("min_samples_leaf", 20, 1, 10_000),
        *((_poids(),) if poids else ()),
        _r("l2_regularization", 0.0, 0, 1000, av=True),
        _e("max_bins", 255, 2, 255, av=True),
    )


def _boosting(pertes: tuple) -> tuple[Hyperparametre, ...]:
    return (
        _r("learning_rate", 0.1, 0.001, 1),
        _e("n_estimators", 100, 1, 500),
        _e("max_depth", 3, 1, 50),
        _r("subsample", 1.0, 0.1, 1),
        _e("min_samples_leaf", 1, 1, 1000),
        _e("min_samples_split", 2, 2, 1000, av=True),
        _r("ccp_alpha", 0.0, 0, 1000, av=True),
        _c("loss", pertes[0], *pertes[1:], av=True),
    )


def _noyau(poids: bool, epsilon: bool) -> tuple[Hyperparametre, ...]:
    return (
        _r("C", 1.0, 0.001, 10_000),
        *((_r("epsilon", 0.1, 0, 1000),) if epsilon else ()),
        _c("kernel", "rbf", "linear", "poly", "sigmoid"),
        _c("gamma", "scale", "auto"),
        *((_poids(),) if poids else ()),
        _e("degree", 3, 1, 4, av=True),
        _r("coef0", 0.0, -1000, 1000, av=True),
        _r("tol", 1e-3, 1e-10, 1, av=True),
    )


def _perceptron() -> tuple[Hyperparametre, ...]:
    return (
        _k("hidden_layer_sizes", (50,)),
        _c("activation", "relu", "tanh", "logistic", "identity"),
        _r("alpha", 1e-4, 0, 1000),
        _r("learning_rate_init", 1e-3, 1e-6, 1),
        _e("max_iter", 500, 1, 10_000),
        _b("early_stopping", False),
        _c("solver", "adam", "lbfgs", "sgd", av=True),
        _r("tol", 1e-4, 1e-10, 1, av=True),
        _e("n_iter_no_change", 10, 1, 100, av=True),
    )


def _dense() -> tuple[Hyperparametre, ...]:
    """Réseau dense Keras (technique facultative)."""
    return (
        _k("couches", (32, 32)),
        _e("epochs", 50, 1, 200),
        _r("taux_apprentissage", 1e-3, 1e-5, 1),
        _e("taille_lot", 32, 1, 4096),
        _r("abandon", 0.0, 0, 0.9),
        _c("activation", "relu", "tanh", "elu", "selu", av=True),
        _b("arret_precoce", True, av=True),
    )


def _anomalie_contamination() -> Hyperparametre:
    return _r("contamination", "auto", 0.001, 0.5, sp=AUTO)


SCHEMAS: dict[str, tuple[Hyperparametre, ...]] = {
    "regression.lineaire": (
        _b("fit_intercept", True),
        _b("positive", False),
        _r("tol", 1e-6, 1e-12, 1),
    ),
    "regression.ridge": (
        _r("alpha", 1.0, 0, 1_000_000),
        _b("fit_intercept", True),
        _b("positive", False),
        _c("solver", "auto", "svd", "cholesky", "lsqr", "sparse_cg", "lbfgs", av=True),
        _e("max_iter", None, 1, 10_000, sp=PAR_DEFAUT, av=True),
        _TOL,
    ),
    "regression.lasso": (
        _r("alpha", 0.1, 1e-6, 10_000),
        _b("fit_intercept", True),
        _b("positive", False),
        _e("max_iter", 1000, 1, 10_000),
        _c("selection", "cyclic", "random", av=True),
        _TOL,
    ),
    "regression.elastic_net": (
        _r("alpha", 0.1, 1e-6, 10_000),
        _r("l1_ratio", 0.5, 0, 1),
        _b("fit_intercept", True),
        _e("max_iter", 1000, 1, 10_000),
        _b("positive", False, av=True),
        _c("selection", "cyclic", "random", av=True),
        _TOL,
    ),
    "regression.quantile": (
        _r("quantile", 0.5, 0.01, 0.99),
        _r("alpha", 0.0, 0, 1000),
        _b("fit_intercept", True),
        _c("solver", "highs", "highs-ds", "highs-ipm", av=True),
    ),
    "regression.plus_proches_voisins": _voisins(),
    "classification.plus_proches_voisins": _voisins(),
    "regression.arbre": _arbre(5, ("squared_error", "absolute_error"), False),
    "classification.arbre": _arbre(5, ("gini", "entropy", "log_loss"), True),
    "regression.foret": _foret(True, False),
    "classification.foret": _foret(True, True),
    "regression.arbres_extremes": _foret(False, False),
    "classification.arbres_extremes": _foret(False, True),
    "regression.bagging": _bagging(),
    "classification.bagging": _bagging(),
    "regression.gradient_histogramme": _histogramme(False),
    "classification.gradient_histogramme": _histogramme(True),
    "regression.gradient_boosting": _boosting(("squared_error", "absolute_error", "huber")),
    "classification.gradient_boosting": _boosting(("log_loss", "exponential")),
    "regression.svr": _noyau(False, True),
    "classification.svc": _noyau(True, False),
    "regression.reseau_neurones": _perceptron(),
    "classification.reseau_neurones": _perceptron(),
    "classification.logistique": (
        _r("C", 1.0, 1e-4, 10_000),
        _e("max_iter", 1000, 1, 10_000),
        _poids(),
        _b("fit_intercept", True),
        _c("solver", "lbfgs", "newton-cg", "newton-cholesky", "sag", av=True),
        _r("tol", 1e-4, 1e-10, 1, av=True),
    ),
    "classification.bayes_naif": (_r("var_smoothing", 1e-9, 0, 1),),
    "classification.analyse_discriminante": (
        _c("solver", "svd", "lsqr", "eigen"),
        _c("shrinkage", None, "auto", sp=AUCUN),
        _r("tol", 1e-4, 1e-10, 1),
    ),
    "regression.adaboost": (
        _e("n_estimators", 50, 1, 500),
        _r("learning_rate", 1.0, 0.001, 10),
        _c("loss", "linear", "square", "exponential"),
    ),
    "classification.adaboost": (_e("n_estimators", 50, 1, 500), _r("learning_rate", 1.0, 0.001, 10)),
    "regression.vote": (
        _l("base", ("regression.ridge", "regression.arbre", "regression.plus_proches_voisins")),
    ),
    "classification.vote": (
        _l(
            "base",
            ("classification.logistique", "classification.arbre", "classification.plus_proches_voisins"),
        ),
        _c("voting", "soft", "hard"),
    ),
    "regression.empilement": (
        _l("base", ("regression.ridge", "regression.arbre", "regression.plus_proches_voisins")),
        _b("passthrough", False),
        _e("cv", 3, 2, 10),
    ),
    "classification.empilement": (
        _l(
            "base",
            ("classification.logistique", "classification.arbre", "classification.plus_proches_voisins"),
        ),
        _b("passthrough", False),
        _e("cv", 3, 2, 10),
    ),
    "regression.reseau_dense": _dense(),
    "classification.reseau_dense": _dense(),
    "reference.dummy_regression": (_c("strategy", "mean", "median"),),
    "reference.dummy_classification": (_c("strategy", "prior", "most_frequent", "stratified", "uniform"),),
    "sans_cible.acp": (
        _p("n_components", None, 1, 100, sp=TOUTES),
        _b("whiten", False),
        _c("svd_solver", "auto", "full", "covariance_eigh", "arpack", "randomized"),
    ),
    "sans_cible.kmeans": (
        _e("n_clusters", 3, 1, 50),
        _e("n_init", 10, 1, 100),
        _e("max_iter", 300, 1, 10_000),
        _c("init", "k-means++", "random"),
        _r("tol", 1e-4, 1e-10, 1, av=True),
        _c("algorithm", "lloyd", "elkan", av=True),
    ),
    "sans_cible.melange_gaussien": (
        _e("n_components", 3, 1, 50),
        _c("covariance_type", "full", "tied", "diag", "spherical"),
        _e("max_iter", 100, 1, 10_000),
        _e("n_init", 1, 1, 100),
        _r("reg_covar", 1e-6, 0, 1),
        _r("tol", 1e-3, 1e-10, 1, av=True),
        _c("init_params", "kmeans", "k-means++", "random", "random_from_data", av=True),
    ),
    "sans_cible.classification_hierarchique": (
        _e("n_clusters", 3, 1, 50),
        _c("linkage", "ward", "complete", "average", "single"),
        _c("metric", "euclidean", "manhattan", "cosine"),
    ),
    "sans_cible.dbscan": (
        _r("eps", 0.5, 1e-6, 1000),
        _e("min_samples", 5, 1, 1000),
        _c("metric", "euclidean", "manhattan", "chebyshev"),
        _c("algorithm", "auto", "ball_tree", "kd_tree", "brute", av=True),
        _e("leaf_size", 30, 1, 200, av=True),
    ),
    "sans_cible.foret_isolement": (
        _e("n_estimators", 100, 1, 500),
        _anomalie_contamination(),
        _r("max_features", 1.0, 0.05, 1),
        _b("bootstrap", False, av=True),
    ),
    "sans_cible.facteur_local": (
        _e("n_neighbors", 20, 1, 500),
        _anomalie_contamination(),
        _c("metric", "minkowski", "euclidean", "manhattan", "chebyshev"),
        _c("algorithm", "auto", "ball_tree", "kd_tree", "brute", av=True),
        _e("leaf_size", 30, 1, 200, av=True),
    ),
    "sans_cible.svm_une_classe": (
        _r("nu", 0.05, 0.001, 1),
        _c("kernel", "rbf", "linear", "poly", "sigmoid"),
        _c("gamma", "scale", "auto"),
        _e("degree", 3, 1, 4, av=True),
        _r("coef0", 0.0, -1000, 1000, av=True),
        _r("tol", 1e-3, 1e-10, 1, av=True),
    ),
}


def schema(identifiant: str) -> tuple[Hyperparametre, ...]:
    """Réglages d'un modèle, principaux d'abord, avancés ensuite ; vide si le modèle n'en déclare pas."""
    return SCHEMAS.get(identifiant, ())


def principaux(identifiant: str) -> tuple[Hyperparametre, ...]:
    """Réglages affichés d'emblée."""
    return tuple(h for h in schema(identifiant) if not h.avance)


def avances(identifiant: str) -> tuple[Hyperparametre, ...]:
    """Réglages repliés sous « avancé »."""
    return tuple(h for h in schema(identifiant) if h.avance)


def _reglage(identifiant: str, nom: str) -> Hyperparametre:
    for h in schema(identifiant):
        if h.nom == nom:
            return h
    raise KeyError(f"{identifiant} n'a pas de réglage « {nom} ».")


# ============================================================
# TEXTES
# ============================================================
def _champ(identifiant: str, nom: str, rang: int) -> str:
    for cle in (f"{identifiant}.{nom}", nom):
        if cle in TEXTES:
            return TEXTES[cle][rang]
    return nom if rang == 0 else ""


def libelle(identifiant: str, nom: str) -> str:
    """Libellé français du réglage (le nom scikit-learn à défaut)."""
    return _champ(identifiant, nom, 0)


def aide(identifiant: str, nom: str) -> str:
    """Aide d'une ligne."""
    return _champ(identifiant, nom, 1)


def _mot(cle: str) -> str:
    return MOTS[cle]


def _nombre(x: float) -> str:
    """Nombre à la française : virgule décimale, espace insécable entre milliers, exposant sans zéro."""
    if isinstance(x, int) or (float(x).is_integer() and abs(x) < 1e15):
        entier = int(x)
        return f"{entier:,}".replace(",", " ")
    texte = f"{x:.10g}"
    texte = re.sub(r"e([+-])0*(\d+)", lambda m: "e" + ("-" if m[1] == "-" else "") + m[2], texte)
    return texte.replace(".", ",")


def texte_valeur(h: Hyperparametre, valeur: Any) -> str:
    """Valeur écrite comme l'utilisateur la lit et la saisit."""
    for brute, cle in h.speciaux:
        if valeur == brute and type(valeur) is type(brute):
            return _mot(cle)
    if isinstance(valeur, bool):
        return _mot("oui" if valeur else "non")
    if isinstance(valeur, (int, float)):
        return _nombre(valeur)
    if isinstance(valeur, (tuple, list)):
        return ", ".join(_nombre(v) if isinstance(v, (int, float)) else str(v) for v in valeur)
    return str(valeur)


def texte_bornes(h: Hyperparametre) -> str:
    """Plage permise, lisible : « 1 à 500, aucune limite », « arbre, forêt » ou « oui, non »."""
    if h.genre in (Genre.ENTIER, Genre.REEL):
        base = f"{_nombre(h.minimum)} {_mot('a')} {_nombre(h.maximum)}"
    elif h.genre is Genre.PART:
        base = _mot("part").format(mini=_nombre(h.minimum), maxi=_nombre(h.maximum))
    elif h.genre is Genre.COUCHES:
        base = _mot("couches").format(n=COUCHES_MAX, mini=_nombre(h.minimum), maxi=_nombre(h.maximum))
    elif h.genre is Genre.LISTE:
        base = _mot("liste").format(mini=h.minimum, maxi=h.maximum)
    elif h.genre is Genre.CHOIX:
        base = ", ".join(str(c) for c in h.choix)
    else:
        base = f"{_mot('oui')}, {_mot('non')}"
    return ", ".join([base, *(_mot(cle) for _, cle in h.speciaux)])


# ============================================================
# CONVERSION ET CONTRÔLE
# ============================================================
def _refus(identifiant: str, h: Hyperparametre, valeur: Any, raison: str) -> ErreurHyperparametre:
    brut = texte_valeur(h, valeur) if not isinstance(valeur, str) else valeur
    return ErreurHyperparametre(f"{libelle(identifiant, h.nom)} : « {brut} » {raison}")


def _propre(texte: str) -> str:
    """Texte d'un nombre : virgule décimale permise, espaces (même insécables) ignorés."""
    return re.sub(r"[\s  ]", "", texte).replace(",", ".")


def _nombre_lu(identifiant: str, h: Hyperparametre, brut: Any) -> float:
    if isinstance(brut, bool) or not isinstance(brut, (int, float, str)):
        raise _refus(identifiant, h, brut, _mot("pas_nombre"))
    try:
        x = float(_propre(brut)) if isinstance(brut, str) else float(brut)
    except ValueError:
        raise _refus(identifiant, h, brut, _mot("pas_nombre")) from None
    if not math.isfinite(x):
        raise _refus(identifiant, h, brut, _mot("pas_fini"))
    return x


def _dans_les_bornes(identifiant: str, h: Hyperparametre, brut: Any, x: float) -> None:
    if (h.minimum is not None and x < h.minimum) or (h.maximum is not None and x > h.maximum):
        raise _refus(identifiant, h, brut, _mot("hors_bornes").format(bornes=texte_bornes(h)))


def _special(h: Hyperparametre, brut: Any) -> tuple[bool, Any]:
    for valeur, cle in h.speciaux:
        if isinstance(brut, str):
            if brut.strip().casefold() in (_mot(cle).casefold(), str(valeur).casefold()):
                return True, valeur
        elif brut == valeur and type(brut) is type(valeur):
            return True, valeur
    return False, None


def _couches(identifiant: str, h: Hyperparametre, brut: Any) -> tuple[int, ...]:
    if isinstance(brut, str):
        morceaux = [m for m in re.split(r"[\s,;()\[\]]+", brut) if m]
    elif isinstance(brut, (int, float)) and not isinstance(brut, bool):
        morceaux = [brut]
    else:
        morceaux = list(brut)
    unites: list[int] = []
    for m in morceaux:
        x = _nombre_lu(identifiant, h, m)
        if not x.is_integer():
            raise _refus(identifiant, h, brut, _mot("pas_entier"))
        unites.append(int(x))
    if not unites or len(unites) > COUCHES_MAX or any(not h.minimum <= u <= h.maximum for u in unites):
        raise _refus(identifiant, h, brut, _mot("hors_bornes").format(bornes=texte_bornes(h)))
    return tuple(unites)


def _liste(identifiant: str, h: Hyperparametre, brut: Any) -> tuple[str, ...]:
    morceaux = [m for m in re.split(r"[\s,;]+", brut) if m] if isinstance(brut, str) else list(brut)
    ids = tuple(str(m) for m in morceaux)
    if not h.minimum <= len(ids) <= h.maximum or len(set(ids)) != len(ids):
        raise _refus(identifiant, h, ", ".join(ids), _mot("hors_bornes").format(bornes=texte_bornes(h)))
    return ids


def _booleen(identifiant: str, h: Hyperparametre, brut: Any) -> bool:
    if isinstance(brut, bool):
        return brut
    if isinstance(brut, str):
        mot = brut.strip().casefold()
        if mot in (_mot("oui").casefold(), "true", "1", "vrai"):
            return True
        if mot in (_mot("non").casefold(), "false", "0", "faux"):
            return False
    raise _refus(identifiant, h, brut, _mot("pas_booleen"))


def convertir(identifiant: str, nom: str, brut: Any) -> Any:
    """Valeur typée d'un réglage (texte saisi ou valeur d'un projet) ; refus explicite hors bornes."""
    h = _reglage(identifiant, nom)
    trouve, valeur = _special(h, brut)
    if trouve:
        return valeur
    match h.genre:
        case Genre.ENTIER:
            x = _nombre_lu(identifiant, h, brut)
            if not x.is_integer():
                raise _refus(identifiant, h, brut, _mot("pas_entier"))
            _dans_les_bornes(identifiant, h, brut, x)
            return int(x)
        case Genre.REEL:
            x = _nombre_lu(identifiant, h, brut)
            _dans_les_bornes(identifiant, h, brut, x)
            return float(x)
        case Genre.PART:
            x = _nombre_lu(identifiant, h, brut)
            if 0 < x < 1:
                return float(x)
            if x.is_integer():
                _dans_les_bornes(identifiant, h, brut, x)
                return int(x)
            raise _refus(identifiant, h, brut, _mot("hors_bornes").format(bornes=texte_bornes(h)))
        case Genre.CHOIX:
            for c in h.choix:
                if str(brut).strip().casefold() == str(c).casefold():
                    return c
            raise _refus(identifiant, h, brut, _mot("hors_liste").format(choix=texte_bornes(h)))
        case Genre.BOOLEEN:
            return _booleen(identifiant, h, brut)
        case Genre.COUCHES:
            return _couches(identifiant, h, brut)
        case _:
            return _liste(identifiant, h, brut)


def valider(identifiant: str, saisies: Mapping[str, Any]) -> dict[str, Any]:
    """Valeurs saisies, converties et contrôlées ; une saisie vide ou égale au défaut n'est pas retenue.

    Une saisie ne doit nommer que des réglages du modèle : un nom inconnu est refusé, car il passerait
    sans effet ou ferait échouer la construction plus loin.
    """
    retenues: dict[str, Any] = {}
    for nom, brut in saisies.items():
        if isinstance(brut, str) and not brut.strip():
            continue
        valeur = convertir(identifiant, nom, brut)
        if valeur != _reglage(identifiant, nom).defaut or type(valeur) is not type(
            _reglage(identifiant, nom).defaut
        ):
            retenues[nom] = valeur
    return retenues


def controler(identifiant: str, parametres: Mapping[str, Any]) -> None:
    """Refuse toute valeur hors bornes parmi les réglages déclarés ; les autres paramètres passent.

    Une liste de modèles de base est contrôlée à sa construction, avec les règles du registre.
    """
    declares = {h.nom for h in schema(identifiant) if h.genre is not Genre.LISTE}
    for nom, valeur in parametres.items():
        if nom in declares:
            convertir(identifiant, nom, valeur)
