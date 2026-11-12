"""
MécaML : lecture d'un modèle sur des lignes qu'il n'a pas vues, hors pli en S7 et sur le test en S8.

Deux sources de lignes jamais utilisées pour ajuster le modèle :

- hors pli : les prédictions de chaque ligne d'apprentissage faites par le modèle ajusté sans elle
  (`Resultat.modeles[i].predictions`), qui donnent résidus, matrice de confusion, courbes ROC et précision
  rappel, et la courbe des scores selon le seuil ;
- sur le test : le modèle choisi est ajusté sur tout l'apprentissage avant l'ouverture du coffre, puis
  `preparer_test` rend la fonction que `Tester` appelle avec une copie des lignes de test (I1). Rien n'est
  ajusté sur ces lignes, rien n'en est gardé que les prédictions et les scores.

Les figures sont construites en `matplotlib.figure.Figure`, sans moteur graphique ; l'interface et le rapport
les affichent ou les écrivent. Le seuil d'une classification binaire est un choix de décision (`decision`) :
il se lit ici sur les prédictions hors pli.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from mecaml.core import libelles, metrics, split, train
from mecaml.core.catalog import EntreeCatalogue, entree
from mecaml.core.preprocess import colonnes_d_entree
from mecaml.core.tasks import collecter_avertissements
from mecaml.core.types import ChoixTechnique, CoffreTest, Decision, Probleme, SpecProjet, Table

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "Apprentissage",
    "CourbeApprentissage",
    "Diagnostic",
    "ErreurEvaluation",
    "ResultatTest",
    "apprentissage_du_projet",
    "courbe_d_apprentissage",
    "diagnostiquer_lignes",
    "diagnostiquer_validation",
    "empreinte_predictions",
    "figure_confusion",
    "figure_courbe_apprentissage",
    "figure_regression",
    "figure_roc_pr",
    "figure_seuil",
    "predire_au_seuil",
    "preparer_test",
]

POINTS_MAX = 1500  # points tracés au plus dans un nuage (tirage fixe, pour des figures reproductibles)
COULEUR = "#1f4e79"
COULEUR_2 = "#c55a11"
GRIS = "#7f7f7f"


class ErreurEvaluation(ValueError):
    """Évaluation impossible ou incohérente ; le message dit quoi corriger."""


# ============================================================
# FONCTIONS PURES, COPIÉES DANS LE SCRIPT EXPORTÉ
# ============================================================
def predire_au_seuil(scores: Any, classes: Sequence[Any], classe_positive: Any, seuil: float) -> np.ndarray:
    """Étiquettes prédites quand on déclare positif tout score au moins égal au seuil (deux classes)."""
    if len(classes) != 2:
        raise ValueError("Un seuil se règle pour deux classes ; regrouper les classes.")
    autre = next(c for c in classes if c != classe_positive)
    return np.where(np.asarray(scores, dtype=float) >= seuil, classe_positive, autre)


def empreinte_predictions(predictions: Any) -> str:
    """SHA-256 des prédictions (nombres en flottants, étiquettes en texte), pour les comparer à l'octet."""
    valeurs = np.asarray(predictions)
    h = hashlib.sha256()
    if valeurs.dtype.kind in "fiub":
        h.update(np.ascontiguousarray(valeurs, dtype="<f8").tobytes())
    else:
        h.update("\n".join(map(str, valeurs.tolist())).encode("utf-8"))
    return h.hexdigest()


# ============================================================
# LIGNES D'APPRENTISSAGE DU PROJET
# ============================================================
class Apprentissage(NamedTuple):
    """Lignes d'apprentissage qui ont une cible, et ce qui sert à les lire comme `train` les lit."""

    donnees: pd.DataFrame  # lignes d'apprentissage avec cible, toutes colonnes
    X: pd.DataFrame  # colonnes d'entrée du pipeline
    y: np.ndarray
    contexte: pd.DataFrame | None  # lignes d'apprentissage sans cible (séries), colonnes d'entrée
    positions: np.ndarray  # position de chaque ligne de `donnees` dans la table
    n_sans_cible: int


def apprentissage_du_projet(spec: SpecProjet, table: Table, coffre: CoffreTest | None) -> Apprentissage:
    """Lignes d'apprentissage du coffre, séparées en lignes avec cible et lignes sans cible (contexte)."""
    if not spec.probleme.avec_cible:
        raise ErreurEvaluation("Un projet sans cible n'a pas de lignes d'apprentissage à part du test.")
    if coffre is None:
        raise ErreurEvaluation("Aucun jeu de test mis sous scellé : découper les données (S4) d'abord.")
    if not coffre.repose_sur(table) or coffre.decoupage != spec.decoupage():
        raise ErreurEvaluation("Le coffre n'a pas été construit pour cette table et ce découpage.")
    brut = coffre.apprentissage()
    presentes = brut[spec.cible].notna().to_numpy()
    avec, sans = split.separer_sans_cible(brut, spec.cible)
    colonnes = list(colonnes_d_entree(spec))
    brute = avec[spec.cible]
    y = brute.to_numpy(dtype=float) if spec.probleme is Probleme.REGRESSION else brute.to_numpy()
    contexte = sans[colonnes] if len(sans) else None
    return Apprentissage(
        avec, avec[colonnes], y, contexte, coffre.positions_apprentissage()[presentes], len(sans)
    )


# ============================================================
# DIAGNOSTIC SUR DES LIGNES JAMAIS VUES PAR L'AJUSTEMENT
# ============================================================
@dataclass(frozen=True, eq=False)
class Diagnostic:
    """Prédictions et vérités de lignes que le modèle n'a pas vues, avec les lectures usuelles.

    `origine` : « validation » (prédictions hors pli de l'apprentissage) ou « test ». Seules les lignes qui
    ont une cible et une prédiction y figurent. `scores` : score de la classe positive (deux classes) ou
    matrice (une colonne par classe) ; None si le modèle n'en donne pas. `plis` : numéro du pli de
    validation de chaque ligne (None sur le test).
    """

    probleme: Probleme
    modele_id: str
    origine: str
    metrique: str
    y_vrai: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray | None
    classes: tuple[Any, ...] | None
    classe_positive: Any
    seuil: float | None
    positions: np.ndarray
    plis: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Nombre de lignes lues."""
        return len(self.y_vrai)

    @property
    def binaire(self) -> bool:
        """Vrai pour une classification à deux classes munie de scores."""
        return self.classes is not None and len(self.classes) == 2 and self.scores is not None

    @property
    def residus(self) -> np.ndarray:
        """Écart vrai moins prédit (régression)."""
        if self.probleme is not Probleme.REGRESSION:
            raise ErreurEvaluation("Les résidus se lisent en régression.")
        return np.asarray(self.y_vrai, dtype=float) - np.asarray(self.y_pred, dtype=float)

    def resume(self) -> dict[str, float]:
        """Chiffres de lecture : biais, dispersion et écart maximal des résidus, ou taux d'erreur."""
        if self.probleme is Probleme.REGRESSION:
            r = self.residus
            return {
                "biais": float(np.mean(r)),
                "ecart_type_des_residus": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                "mae": float(np.mean(np.abs(r))),
                "rmse": float(np.sqrt(np.mean(r**2))),
                "ecart_maximal": float(np.max(np.abs(r))),
            }
        erreurs = np.asarray(self.y_vrai) != np.asarray(self.y_pred)
        return {"taux_d_erreur": float(np.mean(erreurs)), "erreurs": float(np.sum(erreurs))}

    def matrice_confusion(self) -> np.ndarray:
        """Matrice de confusion : lignes = classes réelles, colonnes = classes prédites (ordre `classes`)."""
        if self.classes is None:
            raise ErreurEvaluation("La matrice de confusion se lit en classification.")
        return metrics.matrice_confusion(self.y_vrai, self.y_pred, self.classes)

    def _positifs(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.binaire:
            raise ErreurEvaluation(
                "Les courbes ROC et précision rappel se lisent pour deux classes avec scores."
            )
        return np.asarray(self.y_vrai) == self.classe_positive, np.asarray(self.scores, dtype=float)

    def courbe_roc(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(taux de faux positifs, taux de vrais positifs, aire sous la courbe) ; deux classes."""
        from sklearn.metrics import roc_auc_score, roc_curve

        reel, scores = self._positifs()
        if reel.all() or not reel.any():
            raise ErreurEvaluation("Une seule classe parmi les lignes lues : la courbe ROC est indéfinie.")
        fpr, tpr, _ = roc_curve(reel, scores)
        return fpr, tpr, float(roc_auc_score(reel, scores))

    def courbe_pr(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(précision, rappel, précision moyenne) ; deux classes."""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        reel, scores = self._positifs()
        if not reel.any():
            raise ErreurEvaluation("Aucune ligne positive parmi les lignes lues : la courbe est indéfinie.")
        precision, rappel, _ = precision_recall_curve(reel, scores)
        return precision, rappel, float(average_precision_score(reel, scores))

    def courbe_seuil(self, cout_fp: float = 1.0, cout_fn: float = 1.0, points: int = 101) -> pd.DataFrame:
        """Précision, rappel, F1, exactitude et coût moyen selon le seuil (deux classes avec scores).

        Un point par seuil d'une grille de quantiles des scores : tout score au moins égal au seuil est
        déclaré positif. Le coût moyen est celui de `metrics.cout_au_seuil`.
        """
        reel, scores = self._positifs()
        if not 2 <= points <= 1001:
            raise ErreurEvaluation("La courbe selon le seuil compte de 2 à 1001 points.")
        seuils = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, points)))
        lignes = []
        for s in seuils:
            predit = scores >= s
            tp = int(np.sum(reel & predit))
            fp = int(np.sum(~reel & predit))
            fn = int(np.sum(reel & ~predit))
            tn = len(reel) - tp - fp - fn
            lignes.append(
                {
                    "seuil": float(s),
                    "precision": tp / (tp + fp) if tp + fp else math.nan,
                    "rappel": tp / (tp + fn) if tp + fn else math.nan,
                    "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan,
                    "exactitude": (tp + tn) / len(reel),
                    "cout": metrics.cout_au_seuil(reel, scores, True, float(s), cout_fp, cout_fn),
                    "positifs_predits": tp + fp,
                }
            )
        return pd.DataFrame(lignes)


def _numero_de_pli(resultat: train.Resultat) -> np.ndarray:
    numero = np.full(len(resultat.positions), -1, dtype=int)
    for k, (_, val) in enumerate(resultat.plis):
        numero[val] = k
    return numero


def diagnostiquer_validation(
    resultat: train.Resultat, id_modele: str, *, seuil: float | None = None
) -> Diagnostic:
    """Diagnostic hors pli d'un modèle comparé en S6 ; `seuil` règle une classification binaire.

    Les lignes jamais validées (la fenêtre croissante d'une série ne valide pas les premières) n'y figurent
    pas.
    """
    modele = resultat.modele(id_modele)
    if not modele.ok or modele.predictions is None:
        raise ErreurEvaluation(
            f"Le modèle {id_modele} n'a pas de prédictions hors pli : {modele.erreur or ''}."
        )
    if resultat.y_vrai is None:
        raise ErreurEvaluation("Un projet sans cible n'a pas de prédictions hors pli.")
    garde = resultat.lignes_validees()
    y_vrai = np.asarray(resultat.y_vrai)[garde]
    predictions = np.asarray(modele.predictions)[garde]
    scores = None if modele.scores_continus is None else np.asarray(modele.scores_continus)[garde]
    probleme = resultat.spec.probleme
    if seuil is not None:
        predictions = _appliquer_le_seuil(
            predictions, scores, resultat.classes, resultat.classe_positive, seuil
        )
    return Diagnostic(
        probleme=probleme,
        modele_id=id_modele,
        origine="validation",
        metrique=resultat.metrique,
        y_vrai=y_vrai,
        y_pred=predictions,
        scores=scores,
        classes=resultat.classes,
        classe_positive=resultat.classe_positive,
        seuil=seuil,
        positions=resultat.positions[garde],
        plis=_numero_de_pli(resultat)[garde],
    )


def _appliquer_le_seuil(
    predictions: np.ndarray, scores: np.ndarray | None, classes: Any, positive: Any, seuil: float
) -> np.ndarray:
    if classes is None or len(classes) != 2 or scores is None or positive is None:
        raise ErreurEvaluation("Un seuil se règle pour deux classes dont le modèle donne les scores.")
    if np.asarray(scores).ndim != 1:
        raise ErreurEvaluation("Un seuil se règle sur un score par ligne ; ce modèle n'en donne pas.")
    return predire_au_seuil(scores, classes, positive, seuil)


# ============================================================
# TEST : MODÈLE AJUSTÉ AVANT L'OUVERTURE, LIGNES LUES UNE FOIS
# ============================================================
@dataclass(frozen=True, eq=False)
class ResultatTest:
    """Ce que rend l'évaluation sur le test (S8) : scores, diagnostic, importances.

    Aucune ligne de test n'y est gardée : seuls en sortent les cibles, les prédictions et les scores des
    lignes qui ont une cible. `n_sans_cible` compte les lignes de test retirées faute de cible.
    """

    modele_id: str
    metrique: str
    scores: dict[str, float]
    n_lignes: int
    n_sans_cible: int
    seuil: float | None
    diagnostic: Diagnostic | None
    importances: pd.DataFrame | None
    empreinte_predictions: str
    avertissements: tuple[str, ...] = ()
    classe_positive: Any = None
    modele: train.ModeleAjuste | None = field(
        default=None, repr=False
    )  # le modèle évalué, ajusté avant l'ouverture

    def valeur(self) -> float:
        """Score de test de la métrique du projet."""
        return self.scores.get(self.metrique, math.nan)

    def tableau(self) -> pd.DataFrame:
        """Scores de test, une ligne par métrique."""
        return pd.DataFrame({"metrique": list(self.scores), "valeur": list(self.scores.values())})


def _scores_des_predictions(
    spec: SpecProjet, y: np.ndarray, predictions: np.ndarray, scores: Any, classes: Any, positive: Any
) -> dict[str, float]:
    """Métriques du problème et métrique du projet ; une métrique indéfinie vaut nan."""
    donnees = {
        "y_vrai": y,
        "y_pred": predictions,
        "scores": scores,
        "classes": classes,
        "classe_positive": positive,
    }
    valeurs = metrics.calculer_standard(spec.probleme, None, **donnees)
    if spec.metrique and spec.metrique not in valeurs:
        try:
            valeurs[spec.metrique] = metrics.calculer(spec.metrique, **donnees)
        except metrics.ErreurMetrique:
            valeurs[spec.metrique] = math.nan
    return valeurs


def diagnostiquer_lignes(
    modele: train.ModeleAjuste, spec: SpecProjet, lignes: pd.DataFrame, seuil: float | None = None
) -> tuple[dict[str, float], Diagnostic, int]:
    """(scores, diagnostic, lignes sans cible) du modèle ajusté sur des lignes qui ne l'ont pas ajusté.

    Toutes les lignes sont prédites (les variables temporelles ont besoin de celles sans cible), seules
    celles qui ont une cible sont notées. Sans seuil, les scores sont ceux de `train.evaluer_lignes`.
    """
    if not spec.probleme.avec_cible:
        raise ErreurEvaluation("L'évaluation sur des lignes de test suppose une cible.")
    presentes = lignes[spec.cible].notna().to_numpy()
    variables = lignes[list(colonnes_d_entree(spec))]
    predictions = np.asarray(modele.pipeline.predict(variables))
    continus = None
    if modele.classes is not None:
        brut = metrics.scores_du_modele(modele.pipeline, variables, modele.classes, modele.classe_positive)
        continus = None if brut is None else np.asarray(brut)
    if seuil is not None:
        predictions = _appliquer_le_seuil(
            predictions, continus, modele.classes, modele.classe_positive, seuil
        )
    y = lignes.loc[presentes, spec.cible].to_numpy(
        dtype=float if spec.probleme is Probleme.REGRESSION else None
    )
    retenues = predictions[presentes]
    scores_retenus = None if continus is None else continus[presentes]
    valeurs = _scores_des_predictions(
        spec, y, retenues, scores_retenus, modele.classes, modele.classe_positive
    )
    diagnostic = Diagnostic(
        probleme=spec.probleme,
        modele_id=modele.choix.id,
        origine="test",
        metrique=spec.metrique or "",
        y_vrai=y,
        y_pred=retenues,
        scores=scores_retenus,
        classes=modele.classes,
        classe_positive=modele.classe_positive,
        seuil=seuil,
        positions=np.flatnonzero(presentes),
    )
    return valeurs, diagnostic, int((~presentes).sum())


def _choix_de(spec: SpecProjet, id_modele: str) -> ChoixTechnique:
    for choix in spec.modeles:
        if choix.id == id_modele:
            return choix
    raise ErreurEvaluation(
        f"Le modèle {id_modele} ne figure pas parmi les modèles comparés : {[c.id for c in spec.modeles]}."
    )


def preparer_test(
    spec: SpecProjet,
    table: Table,
    coffre: CoffreTest,
    decision: Decision,
    *,
    importances: bool = True,
    repetitions: int = 5,
    registre: Iterable[EntreeCatalogue] | None = None,
) -> Callable[[pd.DataFrame], ResultatTest]:
    """Ajuste le modèle de la décision sur tout l'apprentissage, puis rend la fonction que `Tester` appelle.

    L'ajustement se fait ici, avant l'ouverture du coffre : s'il échoue, le test reste scellé. La fonction
    rendue reçoit la copie des lignes de test, ne fait que prédire et noter, et rend un `ResultatTest`. Le
    seuil de la décision règle une classification binaire.
    """
    if not spec.probleme.avec_cible:
        raise ErreurEvaluation("Un projet sans cible n'a pas de jeu de test.")
    choix = _choix_de(spec, decision.modele_id)
    with collecter_avertissements() as captees:
        modele = train.ajuster_final(spec, choix, table, coffre, registre=registre)
    avertissements = tuple(sorted({*modele.avertissements, *captees}))

    def evaluer(lignes: pd.DataFrame) -> ResultatTest:
        valeurs, diagnostic, sans_cible = diagnostiquer_lignes(modele, spec, lignes, decision.seuil)
        lues = None
        if importances:
            from mecaml.core import explain

            lues = explain.importances_sur_lignes(modele, spec, lignes, repetitions=repetitions)
        return ResultatTest(
            modele_id=decision.modele_id,
            metrique=spec.metrique or "",
            scores=valeurs,
            n_lignes=diagnostic.n,
            n_sans_cible=sans_cible,
            seuil=decision.seuil,
            diagnostic=diagnostic,
            importances=lues,
            empreinte_predictions=empreinte_predictions(diagnostic.y_pred),
            avertissements=avertissements,
            classe_positive=modele.classe_positive,
            modele=modele,
        )

    return evaluer


# ============================================================
# COURBE D'APPRENTISSAGE (S7)
# ============================================================
@dataclass(frozen=True, eq=False)
class CourbeApprentissage:
    """Score d'apprentissage et de validation selon la taille de l'échantillon d'apprentissage.

    Les scores sont ceux de la métrique du projet, dans son sens naturel (une erreur reste une erreur).
    """

    modele_id: str
    metrique: str
    sens: str
    tailles: np.ndarray
    apprentissage: np.ndarray  # (tailles, plis)
    validation: np.ndarray

    def moyennes(self) -> pd.DataFrame:
        """Moyenne et écart type par taille, d'apprentissage et de validation."""
        return pd.DataFrame(
            {
                "taille": self.tailles,
                "apprentissage": np.nanmean(self.apprentissage, axis=1),
                "apprentissage_ecart_type": np.nanstd(self.apprentissage, axis=1),
                "validation": np.nanmean(self.validation, axis=1),
                "validation_ecart_type": np.nanstd(self.validation, axis=1),
            }
        )

    def ecart_final(self) -> float:
        """Écart entre validation et apprentissage à la taille maximale (grand : apprentissage par cœur)."""
        m = self.moyennes().iloc[-1]
        return float(abs(m["validation"] - m["apprentissage"]))


def courbe_d_apprentissage(
    resultat: train.Resultat,
    id_modele: str,
    table: Table,
    coffre: CoffreTest,
    *,
    points: int = 8,
    registre: Iterable[EntreeCatalogue] | None = None,
) -> CourbeApprentissage:
    """Courbe d'apprentissage du modèle sur les plis de la comparaison (lignes d'apprentissage seules)."""
    spec = resultat.spec
    donnees = apprentissage_du_projet(spec, table, coffre)
    choix = _choix_de(spec, id_modele)
    pipeline = train.construire_pipeline(spec, choix, registre, donnees.contexte)
    scoring = metrics.scorer(
        resultat.metrique, classes=resultat.classes, classe_positive=resultat.classe_positive
    )
    calculer = entree("validation.courbe_apprentissage", registre).construire(spec.graine, n_points=points)
    with collecter_avertissements():
        courbe = calculer(pipeline, donnees.X, donnees.y, cv=list(resultat.plis), scoring=scoring)
    signe = -1.0 if resultat.meilleur == metrics.BAS else 1.0
    return CourbeApprentissage(
        id_modele,
        resultat.metrique,
        resultat.meilleur,
        np.asarray(courbe["tailles"]),
        signe * np.asarray(courbe["app"]),
        signe * np.asarray(courbe["val"]),
    )


# ============================================================
# FIGURES
# ============================================================
def _nouvelle_figure(largeur: float, hauteur: float) -> Figure:
    from matplotlib.figure import Figure

    return Figure(figsize=(largeur, hauteur), layout="constrained", dpi=100)


def _style(axe: Any) -> None:
    axe.spines[["top", "right"]].set_visible(False)
    axe.grid(True, color="#e5e5e5", linewidth=0.6)
    axe.set_axisbelow(True)


def _echantillon(n: int) -> np.ndarray:
    """Indices d'au plus POINTS_MAX points, tirés avec une graine fixe."""
    if n <= POINTS_MAX:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, POINTS_MAX, replace=False))


def _titre(diag: Diagnostic) -> str:
    origine = "hors pli" if diag.origine == "validation" else "test"
    return f"{libelles.technique(diag.modele_id)} ({origine}, {diag.n} lignes)"


def figure_regression(diag: Diagnostic) -> Figure:
    """Valeurs vraies et prédites, résidus selon la prédiction, histogramme des résidus."""
    r = diag.residus  # lève ErreurEvaluation hors régression, avant toute conversion
    vrai, predit = np.asarray(diag.y_vrai, float), np.asarray(diag.y_pred, float)
    k = _echantillon(len(vrai))
    fig = _nouvelle_figure(10.5, 3.4)
    a1, a2, a3 = fig.subplots(1, 3)
    a1.scatter(predit[k], vrai[k], s=8, alpha=0.5, color=COULEUR, edgecolors="none")
    bornes = [min(vrai.min(), predit.min()), max(vrai.max(), predit.max())]
    a1.plot(bornes, bornes, color=COULEUR_2, linewidth=1)
    a1.set(xlabel="valeur prédite", ylabel="valeur vraie", title="Vrai et prédit")
    a2.scatter(predit[k], r[k], s=8, alpha=0.5, color=COULEUR, edgecolors="none")
    a2.axhline(0.0, color=COULEUR_2, linewidth=1)
    a2.set(xlabel="valeur prédite", ylabel="résidu (vrai moins prédit)", title="Résidus")
    a3.hist(r, bins=30, color=COULEUR, edgecolor="white")
    a3.axvline(0.0, color=COULEUR_2, linewidth=1)
    a3.set(xlabel="résidu", ylabel="lignes", title="Répartition des résidus")
    for axe in (a1, a2, a3):
        _style(axe)
    fig.suptitle(_titre(diag), fontsize=10)
    return fig


def figure_confusion(diag: Diagnostic) -> Figure:
    """Matrice de confusion : effectifs et part de chaque classe réelle."""
    matrice = diag.matrice_confusion()
    totaux = matrice.sum(axis=1, keepdims=True)
    parts = np.divide(matrice, totaux, out=np.zeros(matrice.shape, dtype=float), where=totaux > 0)
    k = len(diag.classes or ())
    fig = _nouvelle_figure(max(4.2, 1.1 * k + 2.6), max(3.6, 0.9 * k + 2.2))
    axe = fig.subplots()
    axe.imshow(parts, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(k):
        for j in range(k):
            clair = parts[i, j] > 0.55
            axe.text(
                j,
                i,
                f"{matrice[i, j]}\n{100 * parts[i, j]:.0f} %".replace(".", ","),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if clair else "black",
            )
    etiquettes = [str(c) for c in diag.classes or ()]
    axe.set_xticks(range(k), etiquettes, rotation=30, ha="right")
    axe.set_yticks(range(k), etiquettes)
    axe.set(xlabel="classe prédite", ylabel="classe réelle", title=_titre(diag))
    return fig


def figure_roc_pr(diag: Diagnostic) -> Figure:
    """Courbe ROC et courbe précision rappel (deux classes)."""
    fpr, tpr, aire = diag.courbe_roc()
    precision, rappel, moyenne = diag.courbe_pr()
    fig = _nouvelle_figure(8.0, 3.6)
    a1, a2 = fig.subplots(1, 2)
    a1.plot(fpr, tpr, color=COULEUR, label=f"aire = {aire:.3f}".replace(".", ","))
    a1.plot([0, 1], [0, 1], color=GRIS, linestyle=":", linewidth=1)
    a1.set(xlabel="taux de faux positifs", ylabel="taux de vrais positifs", title="Courbe ROC")
    a2.plot(rappel, precision, color=COULEUR, label=f"précision moyenne = {moyenne:.3f}".replace(".", ","))
    part = float(np.mean(np.asarray(diag.y_vrai) == diag.classe_positive))
    a2.axhline(part, color=GRIS, linestyle=":", linewidth=1)
    a2.set(xlabel="rappel", ylabel="précision", title="Précision et rappel")
    for axe in (a1, a2):
        _style(axe)
        axe.legend(loc="lower right" if axe is a1 else "upper right", frameon=False, fontsize=8)
    fig.suptitle(_titre(diag), fontsize=10)
    return fig


def figure_seuil(courbe: pd.DataFrame, seuil: float | None = None, seuil_cout: float | None = None) -> Figure:
    """Précision, rappel et F1 selon le seuil, et coût moyen ; le seuil choisi est tracé."""
    fig = _nouvelle_figure(9.0, 3.4)
    a1, a2 = fig.subplots(1, 2)
    for colonne, couleur in (("precision", COULEUR), ("rappel", COULEUR_2), ("f1", GRIS)):
        a1.plot(courbe["seuil"], courbe[colonne], label=colonne.replace("f1", "F1"), color=couleur)
    a1.set(xlabel="seuil sur le score", ylabel="valeur", title="Précision, rappel, F1")
    a2.plot(courbe["seuil"], courbe["cout"], color=COULEUR)
    a2.set(xlabel="seuil sur le score", ylabel="coût moyen par ligne", title="Coût de décision")
    for axe in (a1, a2):
        _style(axe)
        if seuil is not None:
            axe.axvline(seuil, color="black", linewidth=1, linestyle="dashed")
    if seuil_cout is not None:
        a2.axvline(seuil_cout, color=COULEUR_2, linewidth=1, linestyle=":")
    a1.legend(frameon=False, fontsize=8)
    return fig


def figure_courbe_apprentissage(courbe: CourbeApprentissage) -> Figure:
    """Score d'apprentissage et de validation selon le nombre de lignes d'apprentissage."""
    m = courbe.moyennes()
    fig = _nouvelle_figure(5.6, 3.6)
    axe = fig.subplots()
    for colonne, nom, couleur in (
        ("apprentissage", "apprentissage", COULEUR_2),
        ("validation", "validation", COULEUR),
    ):
        axe.plot(m["taille"], m[colonne], marker="o", markersize=3, color=couleur, label=nom)
        axe.fill_between(
            m["taille"],
            m[colonne] - m[f"{colonne}_ecart_type"],
            m[colonne] + m[f"{colonne}_ecart_type"],
            color=couleur,
            alpha=0.15,
            linewidth=0,
        )
    axe.set(
        xlabel="lignes d'apprentissage",
        ylabel=libelles.metrique_courte(courbe.metrique),
        title=f"Courbe d'apprentissage : {libelles.technique(courbe.modele_id)}",
    )
    _style(axe)
    axe.legend(frameon=False, fontsize=8)
    return fig
