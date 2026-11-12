"""
MécaML : métriques de régression, de classification et sans cible.

Un registre de métriques (`METRIQUES`) porte, pour chacune, son libellé, les problèmes
où elle s'applique, son sens (plus grand ou plus petit est meilleur), son renvoi au livre
et la clé de son texte pédagogique. La métrique du projet est choisie en S3 sous forme de
texte : « f1 », ou « cout:fp=1,fn=10 » pour fixer les coûts d'une décision.

Règles communes :

- toutes les fonctions reçoivent leurs données par un objet `Entrees` ; une donnée absente
  ou une métrique indéfinie (une seule classe dans un pli, par exemple) lève `ErreurMetrique`,
  jamais un résultat trompeur ;
- en classification binaire, la classe positive est celle de la spécification (`SpecProjet.classe_positive`),
  fixée en S3 avec la métrique ; ce module ne la déduit jamais en silence : sans elle, précision, rappel,
  F1, précision moyenne, ROC AUC et coût lèvent `ErreurMetrique`. `proposer_classe_positive` ne fait
  qu'une proposition (0 et 1, puis un lexique de mots d'anomalie, puis la rareté), que l'écran montre ;
- au delà de deux classes, précision, rappel, F1, ROC AUC et précision moyenne sont des moyennes macro
  (une classe, une voix) ;
- précision, rappel et F1 sont indéfinis (NaN et note, comme le ROC AUC) dans un pli sans aucune ligne de
  la classe positive : un zéro trompeur n'est jamais rendu ;
- la précision, le rappel et le F1 se lisent sur la classe d'intérêt : l'exactitude seule trompe quand
  une classe domine.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mecaml.core import renvois
from mecaml.core.types import Probleme

__all__ = [
    "BAS",
    "HAUT",
    "METRIQUES",
    "NEUTRE",
    "Entrees",
    "ErreurMetrique",
    "Metrique",
    "analyser",
    "calculer",
    "calculer_standard",
    "cout_au_seuil",
    "cout_binaire",
    "evaluer_modele",
    "indice_stabilite_population",
    "matrice_confusion",
    "metrique",
    "proposer_classe_positive",
    "scorer",
    "scores_du_modele",
    "valider",
]

HAUT, BAS, NEUTRE = "haut", "bas", "neutre"
SANS_CIBLE = Probleme.SANS_CIBLE
REG, CLS = Probleme.REGRESSION, Probleme.CLASSIFICATION
ECHANTILLON_SILHOUETTE = 5000


class ErreurMetrique(ValueError):
    """Métrique inconnue, mal paramétrée, ou indéfinie sur ces données."""


@dataclass(frozen=True, eq=False)
class Entrees:
    """Données d'un calcul de métrique ; chaque métrique prend celles dont elle a besoin.

    `scores` : en binaire, un vecteur de scores de la classe positive ou une matrice à deux colonnes ;
    en multiclasse, une matrice dont les colonnes suivent `classes`. `modele` : dernier modèle ajusté.
    """

    y_vrai: Any = None
    y_pred: Any = None
    scores: Any = None
    classes: Sequence[Any] | None = None
    classe_positive: Any = None
    seuil: float | None = None
    x: Any = None
    etiquettes: Any = None
    etiquettes_ref: Any = None
    modele: Any = None
    graine: int = 0


@dataclass(frozen=True, eq=False)
class Metrique:
    """Métrique du registre : libellé, périmètre, sens, calcul, renvoi au livre."""

    id: str
    libelle: str
    problemes: frozenset[Probleme]
    meilleur: str
    fonction: Callable[..., float] | None
    ref_livre: str
    texte_id: str
    parametres: tuple[str, ...] = ()

    @property
    def scalaire(self) -> bool:
        """Faux pour la matrice de confusion, qui n'est pas un nombre."""
        return self.fonction is not None


# ============================================================
# OUTILS
# ============================================================
def _exiger(e: Entrees, *noms: str) -> None:
    manquants = [nom for nom in noms if getattr(e, nom) is None]
    if manquants:
        raise ErreurMetrique(f"Donnée absente pour cette métrique : {', '.join(manquants)}.")


def _fini(valeur: Any) -> float:
    resultat = float(valeur)
    if not math.isfinite(resultat):
        raise ErreurMetrique("Métrique indéfinie sur ces données.")
    return resultat


# Lexique de la classe d'intérêt : on cherche le défaut, la panne, le rebut, non la pièce saine.
_MOTS_PROBLEME = frozenset(
    [
        "defaut",
        "defauts",
        "defectueux",
        "defectueuse",
        "rebut",
        "rebuts",
        "panne",
        "pannes",
        "critique",
        "critiques",
        "anomalie",
        "anomalies",
        "anormal",
        "anormale",
        "fissure",
        "fissures",
        "alarme",
        "echec",
        "mauvais",
        "mauvaise",
        "defaillant",
        "defaillance",
        "present",
        "presente",
        "positif",
        "positive",
        "oui",
        "vrai",
        "true",
        "yes",
    ]
)
_MOTS_SAINS = frozenset(
    [
        "conforme",
        "conformes",
        "stable",
        "stables",
        "sain",
        "saine",
        "ok",
        "bon",
        "bonne",
        "normal",
        "normale",
        "ras",
        "valide",
        "negatif",
        "negative",
        "faux",
        "false",
        "no",
    ]
)
_NEGATEURS = frozenset(["non", "pas", "sans", "aucun", "aucune", "absent", "absente", "absence"])


def _jetons(classe: Any) -> set[str]:
    """Mots d'une étiquette, sans accent ni majuscule (« Pas_de_défaut » donne pas, de, defaut)."""
    texte = unicodedata.normalize("NFKD", str(classe)).encode("ascii", "ignore").decode().lower()
    return {mot for mot in re.split(r"[^a-z0-9]+", texte) if mot}


def _signe_lexical(classe: Any) -> int:
    """+1 si l'étiquette désigne le problème, -1 si elle désigne l'état sain, 0 si rien ne tranche."""
    jetons = _jetons(classe)
    nie = bool(jetons & _NEGATEURS)
    probleme, sain = bool(jetons & _MOTS_PROBLEME), bool(jetons & _MOTS_SAINS)
    if probleme == sain:
        return -1 if nie and not probleme else 0
    return (1 if probleme else -1) * (-1 if nie else 1)


def proposer_classe_positive(y: Any) -> tuple[Any, str]:
    """Classe positive proposée pour une cible binaire, et d'où vient la proposition.

    Origine : « zero_un » (étiquettes 0 et 1, ou faux et vrai : la classe 1), « lexique » (un mot d'anomalie
    comme défaut, rebut, panne, critique, « non conforme », présent), sinon « rarete » (la classe la plus
    rare, la dernière dans l'ordre alphabétique à égalité). Ce n'est qu'une proposition : la classe
    positive du projet est celle que l'utilisateur fixe en S3 dans la spécification.
    """
    valeurs, effectifs = np.unique(np.asarray(y), return_counts=True)
    if len(valeurs) != 2:
        raise ErreurMetrique(f"Classe positive définie pour deux classes, {len(valeurs)} présentes.")
    natives = [v.item() if isinstance(v, np.generic) else v for v in valeurs]
    if valeurs.tolist() in ([0, 1], [False, True]):
        return natives[1], "zero_un"
    signes = [_signe_lexical(v) for v in natives]
    if signes[0] != signes[1]:
        return natives[int(np.argmax(signes))], "lexique"
    rare = int(np.argmin(effectifs[::-1]))
    return natives[::-1][rare], "rarete"


def _positive(e: Entrees) -> Any:
    if e.classe_positive is None:
        raise ErreurMetrique(
            "Classe positive non fixée : la choisir en S3 avec la métrique "
            "(`SpecProjet.classe_positive`), elle n'est jamais déduite en silence."
        )
    return e.classe_positive


def _classes(e: Entrees) -> list[Any]:
    if e.classes is not None:
        return list(e.classes)
    return np.unique(np.asarray(e.y_vrai)).tolist()


def _est_binaire(e: Entrees) -> bool:
    return len(_classes(e)) == 2


def _scores_positifs(e: Entrees) -> np.ndarray:
    """Scores de la classe positive (binaire) : vecteur ou colonne de la matrice."""
    scores = np.asarray(e.scores, dtype=float)
    if scores.ndim == 1:
        return scores
    classes = _classes(e)
    if scores.shape[1] != len(classes):
        raise ErreurMetrique("Les colonnes de scores ne suivent pas les classes.")
    return scores[:, classes.index(_positive(e))]


def _binariser(e: Entrees) -> tuple[np.ndarray, np.ndarray]:
    """Indicatrices (n, k) des classes présentes dans le pli et scores (n, k) correspondants."""
    classes = _classes(e)
    scores = np.asarray(e.scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ErreurMetrique("Une matrice de scores, une colonne par classe, est nécessaire.")
    y = np.asarray(e.y_vrai)
    indicatrices = np.column_stack([(y == c).astype(int) for c in classes])
    utiles = [k for k in range(len(classes)) if 0 < indicatrices[:, k].sum() < len(y)]
    if not utiles:
        raise ErreurMetrique("Une seule classe dans ce pli : métrique indéfinie.")
    return indicatrices[:, utiles], scores[:, utiles]


# ============================================================
# RÉGRESSION
# ============================================================
def _r2(e: Entrees) -> float:
    from sklearn.metrics import r2_score

    _exiger(e, "y_vrai", "y_pred")
    if len(e.y_vrai) < 2:
        raise ErreurMetrique("R² indéfini avec moins de deux lignes.")
    return _fini(r2_score(e.y_vrai, e.y_pred))


def _mae(e: Entrees) -> float:
    from sklearn.metrics import mean_absolute_error

    _exiger(e, "y_vrai", "y_pred")
    return _fini(mean_absolute_error(e.y_vrai, e.y_pred))


def _rmse(e: Entrees) -> float:
    from sklearn.metrics import root_mean_squared_error

    _exiger(e, "y_vrai", "y_pred")
    return _fini(root_mean_squared_error(e.y_vrai, e.y_pred))


def _mape(e: Entrees) -> float:
    """Erreur absolue moyenne en proportion de la valeur vraie ; indéfinie si une valeur vraie est nulle."""
    _exiger(e, "y_vrai", "y_pred")
    vrai = np.asarray(e.y_vrai, dtype=float)
    if np.any(vrai == 0):
        raise ErreurMetrique("MAPE indéfinie : au moins une valeur vraie est nulle (division par zéro).")
    return _fini(np.mean(np.abs((vrai - np.asarray(e.y_pred, dtype=float)) / vrai)))


# ============================================================
# CLASSIFICATION
# ============================================================
def _exactitude(e: Entrees) -> float:
    from sklearn.metrics import accuracy_score

    _exiger(e, "y_vrai", "y_pred")
    return _fini(accuracy_score(e.y_vrai, e.y_pred))


def _exactitude_equilibree(e: Entrees) -> float:
    """Moyenne des rappels par classe réellement présente (calcul direct : aucune alerte de scikit-learn)."""
    _exiger(e, "y_vrai", "y_pred")
    vrai, predit = np.asarray(e.y_vrai), np.asarray(e.y_pred)
    return _fini(np.mean([np.mean(predit[vrai == classe] == classe) for classe in np.unique(vrai)]))


def _moyenne_prf(e: Entrees, fonction: Callable[..., float]) -> float:
    """Précision, rappel ou F1 : sur la classe positive en binaire, macro au delà.

    Binaire : indéfini (et non zéro) si le pli ne contient aucune ligne de la classe positive.
    """
    _exiger(e, "y_vrai", "y_pred")
    classes = _classes(e)
    if len(classes) == 2:
        positive = _positive(e)
        if not np.any(np.asarray(e.y_vrai) == positive):
            raise ErreurMetrique(
                f"Aucune ligne de la classe positive {positive!r} dans ce pli : "
                "précision, rappel et F1 sont indéfinis."
            )
        return _fini(fonction(e.y_vrai, e.y_pred, pos_label=positive, average="binary", zero_division=np.nan))
    return _fini(fonction(e.y_vrai, e.y_pred, average="macro", zero_division=np.nan))


def _precision(e: Entrees) -> float:
    from sklearn.metrics import precision_score

    return _moyenne_prf(e, precision_score)


def _rappel(e: Entrees) -> float:
    from sklearn.metrics import recall_score

    return _moyenne_prf(e, recall_score)


def _f1(e: Entrees) -> float:
    from sklearn.metrics import f1_score

    return _moyenne_prf(e, f1_score)


def _roc_auc(e: Entrees) -> float:
    from sklearn.metrics import roc_auc_score

    _exiger(e, "y_vrai", "scores")
    if _est_binaire(e):
        y = (np.asarray(e.y_vrai) == _positive(e)).astype(int)
        if y.min() == y.max():
            raise ErreurMetrique("Une seule classe dans ce pli : ROC AUC indéfini.")
        return _fini(roc_auc_score(y, _scores_positifs(e)))
    indicatrices, scores = _binariser(e)
    return _fini(np.mean([roc_auc_score(indicatrices[:, k], scores[:, k]) for k in range(scores.shape[1])]))


def _precision_moyenne(e: Entrees) -> float:
    from sklearn.metrics import average_precision_score

    _exiger(e, "y_vrai", "scores")
    if _est_binaire(e):
        y = (np.asarray(e.y_vrai) == _positive(e)).astype(int)
        if y.sum() == 0:
            raise ErreurMetrique("Aucune ligne positive dans ce pli : précision moyenne indéfinie.")
        return _fini(average_precision_score(y, _scores_positifs(e)))
    indicatrices, scores = _binariser(e)
    moyenne = np.mean(
        [average_precision_score(indicatrices[:, k], scores[:, k]) for k in range(scores.shape[1])]
    )
    return _fini(moyenne)


def _cout_des_booleens(reel: np.ndarray, predit: np.ndarray, cout_fp: float, cout_fn: float) -> float:
    if cout_fp < 0 or cout_fn < 0 or not (math.isfinite(cout_fp) and math.isfinite(cout_fn)):
        raise ErreurMetrique("Les coûts doivent être des nombres positifs ou nuls.")
    if len(reel) == 0:
        raise ErreurMetrique("Coût indéfini sans aucune ligne.")
    faux_positifs = int(np.sum(~reel & predit))
    faux_negatifs = int(np.sum(reel & ~predit))
    return (cout_fp * faux_positifs + cout_fn * faux_negatifs) / len(reel)


def cout_binaire(
    y_vrai: Any, y_pred: Any, classe_positive: Any, cout_fp: float = 1.0, cout_fn: float = 1.0
) -> float:
    """Coût moyen par ligne : faux positifs et faux négatifs pondérés par leur coût."""
    reel = np.asarray(y_vrai) == classe_positive
    return _cout_des_booleens(reel, np.asarray(y_pred) == classe_positive, cout_fp, cout_fn)


def cout_au_seuil(
    y_vrai: Any,
    scores_positifs: Any,
    classe_positive: Any,
    seuil: float,
    cout_fp: float = 1.0,
    cout_fn: float = 1.0,
) -> float:
    """Coût moyen quand on déclare positif tout score au moins égal au seuil."""
    reel = np.asarray(y_vrai) == classe_positive
    return _cout_des_booleens(reel, np.asarray(scores_positifs, dtype=float) >= seuil, cout_fp, cout_fn)


def _cout(e: Entrees, fp: float = 1.0, fn: float = 1.0) -> float:
    _exiger(e, "y_vrai")
    if not _est_binaire(e):
        raise ErreurMetrique("Le coût de décision se définit pour deux classes ; regrouper les classes.")
    positive = _positive(e)
    if e.seuil is not None and e.scores is not None:
        return _fini(cout_au_seuil(e.y_vrai, _scores_positifs(e), positive, e.seuil, fp, fn))
    _exiger(e, "y_pred")
    return _fini(cout_binaire(e.y_vrai, e.y_pred, positive, fp, fn))


def matrice_confusion(y_vrai: Any, y_pred: Any, classes: Iterable[Any] | None = None) -> np.ndarray:
    """Matrice de confusion : lignes = classes réelles, colonnes = classes prédites (ordre de `classes`)."""
    from sklearn.metrics import confusion_matrix

    etiquettes = list(classes) if classes is not None else np.unique(np.asarray(y_vrai)).tolist()
    return confusion_matrix(y_vrai, y_pred, labels=etiquettes)


# ============================================================
# SANS CIBLE
# ============================================================
def _silhouette(e: Entrees) -> float:
    from sklearn.metrics import silhouette_score

    _exiger(e, "x", "etiquettes")
    etiquettes = np.asarray(e.etiquettes)
    garde = etiquettes != -1  # le bruit de DBSCAN (étiquette -1) ne forme pas un groupe
    n_groupes = len(np.unique(etiquettes[garde]))
    if n_groupes < 2 or garde.sum() <= n_groupes:
        raise ErreurMetrique("Silhouette indéfinie : au moins deux groupes et plus de points que de groupes.")
    x = np.asarray(e.x)[garde]
    taille = ECHANTILLON_SILHOUETTE if len(x) > ECHANTILLON_SILHOUETTE else None
    return _fini(silhouette_score(x, etiquettes[garde], sample_size=taille, random_state=e.graine))


def _rand_ajuste(e: Entrees) -> float:
    from sklearn.metrics import adjusted_rand_score

    _exiger(e, "etiquettes", "etiquettes_ref")
    return _fini(adjusted_rand_score(e.etiquettes_ref, e.etiquettes))


def _variance_expliquee(e: Entrees) -> float:
    _exiger(e, "modele")
    rapports = getattr(e.modele, "explained_variance_ratio_", None)
    if rapports is None:
        raise ErreurMetrique("La variance expliquée se lit sur une analyse en composantes principales.")
    return _fini(np.sum(rapports))


def _taux_anomalies(e: Entrees) -> float:
    _exiger(e, "etiquettes")
    return _fini(np.mean(np.asarray(e.etiquettes) == -1))


# ============================================================
# REGISTRE
# ============================================================
def _m(
    id: str,
    libelle: str,
    problemes: Iterable[Probleme],
    meilleur: str,
    fonction: Any,
    parametres: tuple[str, ...] = (),
) -> Metrique:
    """Métrique du registre ; son renvoi au livre est lu dans la table unique (`renvois.METRIQUES`)."""
    return Metrique(
        id,
        libelle,
        frozenset(problemes),
        meilleur,
        fonction,
        renvois.metrique(id),
        f"metrique.{id}",
        parametres,
    )


METRIQUES: tuple[Metrique, ...] = (
    _m("r2", "R² (coefficient de détermination)", [REG], HAUT, _r2),
    _m("mae", "Erreur absolue moyenne (MAE)", [REG], BAS, _mae),
    _m("rmse", "Racine de l'erreur quadratique moyenne (RMSE)", [REG], BAS, _rmse),
    _m("mape", "Erreur absolue moyenne en proportion de la valeur (MAPE)", [REG], BAS, _mape),
    _m("exactitude", "Exactitude", [CLS], HAUT, _exactitude),
    _m("exactitude_equilibree", "Exactitude équilibrée", [CLS], HAUT, _exactitude_equilibree),
    _m("precision", "Précision", [CLS], HAUT, _precision),
    _m("rappel", "Rappel", [CLS], HAUT, _rappel),
    _m("f1", "Score F1", [CLS], HAUT, _f1),
    _m("roc_auc", "Aire sous la courbe ROC", [CLS], HAUT, _roc_auc),
    _m(
        "precision_moyenne",
        "Précision moyenne (aire sous la courbe précision rappel)",
        [CLS],
        HAUT,
        _precision_moyenne,
    ),
    _m("matrice_confusion", "Matrice de confusion", [CLS], NEUTRE, None),
    _m(
        "cout",
        "Coût de décision (faux positifs et faux négatifs pondérés)",
        [CLS],
        BAS,
        _cout,
        ("fp", "fn"),
    ),
    _m("silhouette", "Coefficient de silhouette", [SANS_CIBLE], HAUT, _silhouette),
    _m("rand_ajuste", "Indice de Rand ajusté", [SANS_CIBLE], HAUT, _rand_ajuste),
    _m("variance_expliquee", "Variance expliquée cumulée", [SANS_CIBLE], HAUT, _variance_expliquee),
    _m("taux_anomalies", "Part des points déclarés anormaux", [SANS_CIBLE], NEUTRE, _taux_anomalies),
)
_PAR_ID = {m.id: m for m in METRIQUES}

# Métriques calculées d'office à chaque évaluation (sans paramètre), par problème.
STANDARD: dict[Probleme, tuple[str, ...]] = {
    REG: ("r2", "mae", "rmse", "mape"),
    CLS: ("exactitude", "exactitude_equilibree", "precision", "rappel", "f1", "roc_auc", "precision_moyenne"),
}
_RE_PARAMETRE = re.compile(r"^(?P<nom>[a-z]+)=(?P<valeur>[0-9.eE+]+)$")


def metrique(id_metrique: str) -> Metrique:
    """Métrique d'identifiant donné (sans paramètre) ; `ErreurMetrique` sinon."""
    try:
        return _PAR_ID[id_metrique]
    except KeyError:
        raise ErreurMetrique(
            f"Métrique inconnue : {id_metrique!r}. Connues : {', '.join(_PAR_ID)}."
        ) from None


def analyser(texte: str) -> tuple[Metrique, dict[str, float]]:
    """Décompose « cout:fp=1,fn=10 » en métrique et paramètres ; refuse un paramètre inconnu."""
    if not isinstance(texte, str) or not texte.strip():
        raise ErreurMetrique("La métrique est vide : en choisir une en S3.")
    id_metrique, _, reste = texte.strip().partition(":")
    m = metrique(id_metrique)
    parametres: dict[str, float] = {}
    for morceau in filter(None, (p.strip() for p in reste.split(","))):
        trouve = _RE_PARAMETRE.match(morceau)
        if trouve is None or trouve["nom"] not in m.parametres:
            raise ErreurMetrique(
                f"Paramètre {morceau!r} invalide pour {m.id} (permis : {', '.join(m.parametres) or 'aucun'})."
            )
        parametres[trouve["nom"]] = float(trouve["valeur"])
    return m, parametres


def valider(texte: str, probleme: Probleme) -> Metrique:
    """Vérifie que la métrique est un nombre applicable au problème ; rend la métrique."""
    m, _ = analyser(texte)
    if probleme not in m.problemes:
        valides = ", ".join(x.id for x in METRIQUES if probleme in x.problemes and x.scalaire)
        raise ErreurMetrique(f"La métrique {m.id} ne s'applique pas à un problème de {probleme} ({valides}).")
    if not m.scalaire:
        raise ErreurMetrique(
            f"{m.libelle} se lit sur un tableau, pas un nombre : choisir une autre métrique."
        )
    return m


def calculer(texte: str, **donnees: Any) -> float:
    """Valeur d'une métrique (nom éventuellement paramétré) sur les données fournies par mots-clés."""
    m, parametres = analyser(texte)
    if m.fonction is None:
        raise ErreurMetrique(f"{m.libelle} se calcule par `matrice_confusion`, pas comme un nombre.")
    return m.fonction(Entrees(**donnees), **parametres)


def calculer_standard(probleme: Probleme, notes: list[str] | None = None, **donnees: Any) -> dict[str, float]:
    """Métriques sans paramètre du problème ; une métrique indéfinie vaut NaN et laisse une note."""
    resultats: dict[str, float] = {}
    for id_metrique in STANDARD.get(probleme, ()):
        try:
            resultats[id_metrique] = calculer(id_metrique, **donnees)
        except ErreurMetrique as erreur:
            resultats[id_metrique] = math.nan
            if notes is not None:
                notes.append(f"{id_metrique} : {erreur}")
    return resultats


# ============================================================
# MODÈLE AJUSTÉ
# ============================================================
def scores_du_modele(modele: Any, x: Any, classes: Sequence[Any] | None = None, positive: Any = None) -> Any:
    """Scores continus d'un modèle ajusté : probabilités, sinon fonction de décision, sinon None.

    Binaire : vecteur de la classe positive. Multiclasse : matrice dont les colonnes suivent `classes`.
    """
    if hasattr(modele, "predict_proba"):
        scores = np.asarray(modele.predict_proba(x), dtype=float)
    elif hasattr(modele, "decision_function"):
        scores = np.asarray(modele.decision_function(x), dtype=float)
    else:
        return None
    ordre = list(getattr(modele, "classes_", classes if classes is not None else []))
    if scores.ndim == 1:  # fonction de décision binaire : positive pour la seconde classe du modèle
        return scores if not ordre or positive is None or ordre[-1] == positive else -scores
    if classes is not None and ordre and list(classes) != ordre:
        scores = _etaler_sur_les_classes(scores, ordre, list(classes))
        ordre = list(classes)
    if len(ordre) == 2 and positive is not None:
        return scores[:, ordre.index(positive)]
    return scores


def _etaler_sur_les_classes(scores: np.ndarray, ordre: list[Any], classes: list[Any]) -> np.ndarray:
    """Scores étalés sur toutes les classes quand le modèle n'en a vu qu'une partie (pli sans une classe)."""
    complet = np.full((scores.shape[0], len(classes)), float(np.min(scores)) - 1.0)
    for colonne, classe in enumerate(ordre):
        if classe in classes:
            complet[:, classes.index(classe)] = scores[:, colonne]
    return complet


def evaluer_modele(
    modele: Any,
    x: Any,
    y: Any,
    texte: str,
    *,
    classes: Sequence[Any] | None = None,
    classe_positive: Any = None,
) -> float:
    """Valeur d'une métrique de régression ou de classification pour un modèle ajusté et des lignes."""
    m, _ = analyser(texte)
    donnees: dict[str, Any] = {"y_vrai": np.asarray(y), "y_pred": modele.predict(x)}
    if CLS in m.problemes:
        donnees.update(classes=classes, classe_positive=classe_positive)
        donnees["scores"] = scores_du_modele(modele, x, classes, classe_positive)
    return calculer(texte, **donnees)


def scorer(
    texte: str, *, classes: Sequence[Any] | None = None, classe_positive: Any = None
) -> Callable[..., float]:
    """Évaluateur au format scikit-learn `(modele, x, y)`, plus grand est meilleur (signe changé si besoin).

    S'utilise comme `scoring=` d'une courbe d'apprentissage ou d'une importance par permutation.
    """
    m, _ = analyser(texte)
    signe = -1.0 if m.meilleur == BAS else 1.0

    def evaluateur(modele: Any, x: Any, y: Any) -> float:
        try:
            return signe * evaluer_modele(
                modele, x, y, texte, classes=classes, classe_positive=classe_positive
            )
        except ErreurMetrique:
            return math.nan

    return evaluateur


# ============================================================
# STABILITÉ D'UNE DISTRIBUTION (surveillance de dérive, rapport)
# ============================================================
def indice_stabilite_population(reference: Any, courant: Any, n_classes: int = 10) -> float:
    """Indice de stabilité de population (PSI) entre un échantillon de référence et un échantillon courant.

    Les classes sont les déciles (par défaut) de la référence ; sous 0,1 la distribution est stable, de 0,1 à
    0,25 elle bouge, au delà elle a changé. Outil du rapport : l'application ne surveille pas un modèle en
    production (voir la couverture du livre).
    """
    vieux = np.asarray(reference, dtype=float)
    neuf = np.asarray(courant, dtype=float)
    vieux, neuf = vieux[~np.isnan(vieux)], neuf[~np.isnan(neuf)]
    if len(vieux) < 2 or len(neuf) < 1:
        raise ErreurMetrique("PSI indéfini : échantillons trop courts.")
    bornes = np.unique(np.quantile(vieux, np.linspace(0, 1, n_classes + 1)[1:-1]))
    if len(np.unique(vieux)) < 2 or len(bornes) == 0:
        raise ErreurMetrique("PSI indéfini : la référence est constante.")

    def parts(x: np.ndarray) -> np.ndarray:
        effectifs = np.bincount(np.searchsorted(bornes, x), minlength=len(bornes) + 1)
        return np.clip(effectifs / len(x), 1e-4, None)

    p, q = parts(vieux), parts(neuf)
    return _fini(np.sum((q - p) * np.log(q / p)))
