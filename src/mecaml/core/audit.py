"""
MécaML : profil des colonnes et audit des données.

L'audit est unique : il se déroule en S2 (sans objectif), puis la session le rejoue en
S3 (avec l'objectif) et en S4 (avec le coffre de test). `auditer` est l'`Auditeur` à trois
arguments (table, spécification ou None, coffre ou None) et rend des `Constat`.

Règles de sécurité de l'audit :

- sans spécification, rien ne dépend d'une cible ;
- avec spécification, seules les colonnes d'entrée (rôles numérique et catégoriel) sont
  jugées, plus la cible ; une colonne ignorée ou identifiant ne lève aucun constat ;
- avec le coffre, tout calcul qui touche la cible ou les variables se restreint aux
  lignes d'apprentissage ; seules les colonnes de groupe et de date des lignes de test
  sont lues, pour vérifier que le découpage tient sa structure (I4).

Les codes de constat sont uniques et stables (« I3.fuite:taux_rebuts_pct ») : ce sont les
clés de l'acquittement. Les renvois `ref_livre` sont des identifiants de technique du livre, lus dans
la table unique `mecaml.core.renvois`.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from mecaml.core import cible_passee
from mecaml.core.catalog.ecretage_tukey import valeurs_aberrantes
from mecaml.core.guards import controle_cible, decimale, detecter_probleme, nombre_lisible, renvoi
from mecaml.core.io import ATTR_FORMULES, ATTR_LIGNES_VIDES, ATTR_NOTES
from mecaml.core.split import controler_decoupage, controler_part_test
from mecaml.core.types import (
    GRAINE_CONTRAT,
    CoffreTest,
    Constat,
    Niveau,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
)

__all__ = [
    "ProfilColonne",
    "auditer",
    "probleme_propose",
    "profiler",
    "proposer_cible",
    "proposer_roles",
    "proposer_structure",
    "scores_fuite",
]

# ============================================================
# SEUILS ET LEXIQUES
# ============================================================
SEUIL_FUITE = 0.99  # association avec la cible : au delà, la variable contient la réponse
NATURE_RANG = "corrélation de rang"  # relation monotone (Spearman) sans être linéaire (Pearson)
ECHANTILLON_MAX = 20_000  # lignes tirées (graine de la spécification) pour les indicateurs de fuite
MODALITES_MAX = 50
N_MIN_STAT = 30
PART_MANQUANTS_ALERTE = 0.05
PART_MANQUANTS_MASSIFS = 0.60
PART_DOUBLONS_ALERTE = 0.01
# Valeurs hors des bornes de Tukey qui méritent un constat d'information : une part et un nombre minimaux
# (une colonne gaussienne n'atteint presque jamais les deux).
PART_ABERRANTES_INFO = 0.02
N_ABERRANTES_MIN = 5
N_LIGNES_PETIT = 100
N_MIN_UNICITE = 20  # en dessous, « toutes distinctes » ne prouve rien
CROISEMENT_MAX = 2_000_000  # modalités de la variable fois modalités de la cible, pour le V de Cramér

# Mots « forts » : avec des valeurs distinctes, ils désignent un identifiant. Mots « faibles » : ils exigent
# en plus des entiers presque consécutifs (ou du texte), car `num_tours` ou `ref` peuvent être des mesures.
MOTS_IDENTIFIANT_FORTS = frozenset({"id", "ids", "identifiant", "numero", "uuid", "guid"})
MOTS_IDENTIFIANT_FAIBLES = frozenset({"num", "index", "ref"})
MOTS_IDENTIFIANT = MOTS_IDENTIFIANT_FORTS | MOTS_IDENTIFIANT_FAIBLES
# Unités suivies dans le temps : confirmées par un repère de temps qui repart dans chaque unité.
MOTS_UNITE = frozenset(
    {
        "machine",
        "machines",
        "outil",
        "outils",
        "lot",
        "lots",
        "eprouvette",
        "eprouvettes",
        "moteur",
        "moteurs",
        "pompe",
        "pompes",
        "equipement",
        "unite",
        "sequence",
        "banc",
        "poste",
        "piece",
        "essai",
    }
)
PART_DISTINCTS_IDENTIFIANT = (
    0.99  # part de valeurs distinctes qui confirme un identifiant désigné par son nom
)
PREUVE_SOEUR_LIGNES_MIN = 5  # lignes d'une modalité qui impliquent la cible, au moins
PREUVE_SOEUR_HASARD = 1e-6  # probabilité, au hasard, qu'une telle modalité implique la cible
MOTS_DATE = frozenset(
    {
        "date",
        "time",
        "datetime",
        "timestamp",
        "horodatage",
        "heure",
        "heures",
        "instant",
        "cycle",
        "temps",
        "jour",
    }
)
MOTS_ETIQUETTE_MODELE = frozenset(
    {
        "cluster",
        "clusters",
        "gmm",
        "kmeans",
        "dbscan",
        "pred",
        "predit",
        "predite",
        "prediction",
        "proba",
        "probabilite",
    }
)
MOTS_PERSONNEL = frozenset(
    {
        "genre",
        "sexe",
        "gender",
        "sex",
        "ethnie",
        "ethnicite",
        "religion",
        "nationalite",
        "handicap",
        "naissance",
        "prenom",
    }
)
MOTS_PERSONNE = frozenset({"operateur", "soudeur", "employe", "salarie", "personne", "ouvrier", "technicien"})
MOTS_CIBLE_FORTS = frozenset({"cible", "target", "label", "etiquette"})
MOTS_CIBLE = frozenset(
    {
        "classe",
        "class",
        "etat",
        "defaut",
        "panne",
        "rul",
        "resistance",
        "qualite",
        "presence",
        "alarme",
        "stress",
    }
)

_RE_MOTS = re.compile(r"[^0-9A-Za-z]+|(?<=[a-z])(?=[A-Z])")
_RE_SANS_TITRE = re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE)
_RE_CODE_ILLICITE = re.compile(r"[^A-Za-z0-9_.\-]+")


def q(nom: str) -> str:
    """Nom de colonne entre guillemets français."""
    return f"« {nom} »"


def mots(nom: str) -> tuple[str, ...]:
    """Mots d'un nom de colonne, sans accent, en minuscules (« Piece_ID » donne « piece », « id »)."""
    sans = "".join(c for c in unicodedata.normalize("NFKD", nom) if not unicodedata.combining(c))
    return tuple(m.lower() for m in _RE_MOTS.split(sans) if m)


def _normalise(nom: str) -> str:
    return "_".join(mots(nom))


# ============================================================
# PROFIL DES COLONNES
# ============================================================
@dataclass(frozen=True)
class ProfilColonne:
    """Ce que l'on voit d'une colonne, et le rôle que l'on propose (l'utilisateur confirme)."""

    nom: str
    famille: str  # « nombre », « texte », « date », « booleen », « vide »
    role_propose: Role
    raison: str
    preuve_identifiant: str | None  # « nom » ou « unicite » si la colonne ressemble à un identifiant
    n_lignes: int
    n_manquants: int
    n_distincts: int
    minimum: float | None = None
    maximum: float | None = None
    moyenne: float | None = None
    modalites: tuple[str, ...] = ()  # modalités les plus fréquentes (colonnes à peu de modalités)
    nombres_en_texte: bool = False
    flottante: bool = False  # nombres à partie décimale (un flottant tout entier suit les règles des entiers)
    n_hors_tukey: int = 0  # valeurs hors des bornes de Tukey (grandeurs continues, 30 valeurs au moins)

    @property
    def part_manquants(self) -> float:
        """Part des lignes sans valeur."""
        return self.n_manquants / self.n_lignes if self.n_lignes else 0.0

    @property
    def constante(self) -> bool:
        """Vrai si la colonne n'a qu'une valeur (ou aucune)."""
        return self.n_distincts <= 1

    @property
    def reelle(self) -> bool:
        """Grandeur continue : nombres à partie décimale, ou entiers à plus de 50 valeurs."""
        return self.famille == "nombre" and (self.flottante or self.n_distincts > MODALITES_MAX)


def _famille(s: pd.Series, n_valides: int) -> str:
    if n_valides == 0:
        return "vide"
    if pd.api.types.is_bool_dtype(s):
        return "booleen"
    if pd.api.types.is_numeric_dtype(s):
        return "nombre"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "texte"


def _croissante(s: pd.Series) -> bool:
    """Vrai si au moins 90 % des pas successifs sont strictement croissants (indice d'ordre)."""
    v = s.dropna().to_numpy(dtype=float)
    return len(v) >= 3 and float(np.mean(np.diff(v) > 0)) >= 0.9


def _en_nombres(s: pd.Series) -> bool:
    """Vrai si une colonne de texte contient surtout des nombres écrits en lettres et chiffres (« 12,5 »)."""
    echantillon = s.dropna().head(500).astype(str).str.strip().str.replace(",", ".", regex=False)
    if echantillon.empty:
        return False
    return float(pd.to_numeric(echantillon, errors="coerce").notna().mean()) >= 0.9


def _decimale(s: pd.Series) -> bool:
    """Vrai si la colonne est flottante et qu'une de ses valeurs finies au moins a une partie décimale.

    Un flottant dont toutes les valeurs finies sont entières est un entier lu en flottant (une case vide
    suffit) : il suit les règles des entiers.
    """
    if not pd.api.types.is_float_dtype(s):
        return False
    valeurs = s.dropna().to_numpy(dtype=float)
    finies = valeurs[np.isfinite(valeurs)]
    return bool(np.any(finies != np.round(finies)))


def _discret(s: pd.Series, famille: str) -> bool:
    """Texte ou entiers, flottants tout entiers compris : seule une grandeur à partie décimale n'est jamais
    un identifiant."""
    return famille == "texte" or (famille == "nombre" and not _decimale(s))


def _hors_tukey(s: pd.Series) -> int:
    """Nombre de valeurs hors des bornes de Tukey (règle du catalogue : 1,5 écart interquartile)."""
    return int(valeurs_aberrantes(s.to_numpy(dtype=float, na_value=np.nan)).sum())


def _consecutif(s: pd.Series, n_val: int) -> bool:
    """Entiers presque consécutifs : l'étendue des valeurs ne dépasse pas deux fois leur nombre."""
    return float(s.max() - s.min() + 1) <= 2 * n_val


def _role_identifiant_par_le_nom(
    nom: str, s: pd.Series, famille: str, n_val: int, nun: int
) -> tuple[Role, str, str] | tuple[Role, str, None] | None:
    """Rôle que le nom évoque (identifiant ou unité répétée), confirmé par le type et la cardinalité."""
    m = set(mots(nom))
    fort = bool(m & MOTS_IDENTIFIANT_FORTS) or bool(_RE_SANS_TITRE.match(nom))
    if not (fort or m & MOTS_IDENTIFIANT_FAIBLES) or not _discret(s, famille):
        return None
    part = nun / n_val
    if part >= PART_DISTINCTS_IDENTIFIANT and (fort or famille == "texte" or _consecutif(s, n_val)):
        return (
            Role.IDENTIFIANT,
            "le nom évoque un identifiant et les valeurs sont toutes (ou presque) distinctes",
            "nom",
        )
    if not fort:
        return None
    if part >= 0.5:
        role = Role.NUMERIQUE if famille == "nombre" else Role.CATEGORIELLE
        return role, "le nom évoque un identifiant mais des valeurs se répètent", "nom_partiel"
    return Role.GROUPE, "identifiant répété : unité suivie (machine, lot)", None


def _role(nom: str, s: pd.Series, famille: str, n: int, n_val: int, nun: int) -> tuple[Role, str, str | None]:
    """(rôle proposé, raison, preuve d'identifiant)."""
    if famille == "vide":
        return Role.IGNOREE, "colonne vide", None
    if nun <= 1:
        return Role.IGNOREE, "colonne constante", None
    par_le_nom = _role_identifiant_par_le_nom(nom, s, famille, n_val, nun)
    if par_le_nom is not None:
        return par_le_nom
    m = set(mots(nom))
    if famille == "date":
        return Role.DATE, "colonne de dates", None
    if famille == "nombre" and m & MOTS_DATE and _croissante(s):
        return Role.DATE, "indice d'ordre croissant (temps, cycle)", None
    unique = n_val == n and nun == n and n >= N_MIN_UNICITE
    if unique and _discret(s, famille):
        return Role.IDENTIFIANT, "toutes les valeurs sont distinctes", "unicite"
    if famille == "nombre":
        return Role.NUMERIQUE, "nombre", None
    return Role.CATEGORIELLE, "catégories", None


def _profil_colonne(nom: str, s: pd.Series) -> ProfilColonne:
    n = len(s)
    n_val = int(s.notna().sum())
    nun = int(s.nunique(dropna=True))
    famille = _famille(s, n_val)
    role, raison, preuve = _role(nom, s, famille, n, n_val, nun)
    mini = maxi = moy = None
    if famille == "nombre":
        mini, maxi, moy = float(s.min()), float(s.max()), float(s.mean())
    modalites: tuple[str, ...] = ()
    if famille in ("texte", "booleen") and 1 <= nun <= MODALITES_MAX:
        modalites = tuple(str(k) for k in s.value_counts().head(5).index)
    profil = ProfilColonne(
        nom=nom,
        famille=famille,
        role_propose=role,
        raison=raison,
        preuve_identifiant=preuve,
        n_lignes=n,
        n_manquants=n - n_val,
        n_distincts=nun,
        minimum=mini,
        maximum=maxi,
        moyenne=moy,
        modalites=modalites,
        nombres_en_texte=famille == "texte" and nun > 1 and _en_nombres(s),
        flottante=_decimale(s),
    )
    if profil.reelle and not profil.constante and n_val >= N_MIN_STAT:
        profil = replace(profil, n_hors_tukey=_hors_tukey(s))
    return profil


def _part_croissante(pas: pd.Series) -> float:
    """Part des pas strictement positifs (nombres ou durées)."""
    zero = pd.Timedelta(0) if pd.api.types.is_timedelta64_dtype(pas) else 0
    return float((pas > zero).mean()) if len(pas) else 0.0


def _suivi_par_unite(unite: pd.Series, repere: pd.Series) -> bool:
    """Le repère croît dans chaque unité (90 % des pas) et chaque unité forme une série à part.

    Une série à part : le repère repart d'une unité à l'autre (il ne croît pas sur tout le fichier), ou les
    lignes d'une unité se suivent en bloc. Un journal où les unités alternent ligne à ligne n'en est pas une.
    """
    ok = (unite.notna() & repere.notna()).to_numpy()
    if ok.sum() < N_MIN_STAT:
        return False
    u, d = unite[ok], repere[ok]
    dans_l_unite = _part_croissante(d.groupby(u, sort=False).diff().dropna())
    repart = _part_croissante(d.diff().dropna()) < 0.999
    en_blocs = int((u != u.shift()).sum()) <= 1.1 * u.nunique()
    return dans_l_unite >= 0.9 and (repart or en_blocs)


def _confirmer_unites(profils: list[ProfilColonne], series: dict[str, pd.Series]) -> list[ProfilColonne]:
    """Colonne au nom d'unité (machine, outil...) dont le repère de temps repart par modalité : groupe."""
    if any(p.role_propose is Role.GROUPE for p in profils):
        return profils
    reperes = [
        i
        for i, p in enumerate(profils)
        if p.famille in ("nombre", "date") and (p.role_propose is Role.DATE or set(mots(p.nom)) & MOTS_DATE)
    ]
    for i, p in enumerate(profils):
        n_val = p.n_lignes - p.n_manquants
        candidate = (
            set(mots(p.nom)) & MOTS_UNITE
            and p.role_propose in (Role.NUMERIQUE, Role.CATEGORIELLE)
            and not p.flottante
            and 2 <= p.n_distincts <= n_val / 2
        )
        if not candidate:
            continue
        for j in reperes:
            if j != i and _suivi_par_unite(series[p.nom], series[profils[j].nom]):
                repere = profils[j]
                profils[i] = replace(
                    p,
                    role_propose=Role.GROUPE,
                    raison=f"unité suivie : « {repere.nom} » est une série par modalité",
                )
                if repere.role_propose is not Role.DATE:
                    profils[j] = replace(
                        repere, role_propose=Role.DATE, raison="repère de temps de chaque unité"
                    )
                return profils
    return profils


def profiler(table: Table, positions: np.ndarray | None = None) -> tuple[ProfilColonne, ...]:
    """Profil de chaque colonne (sur les lignes `positions` si elles sont données)."""
    df = table.donnees
    series = {c: (df[c] if positions is None else df[c].iloc[positions]) for c in df.columns}
    profils = [_profil_colonne(c, s) for c, s in series.items()]
    return tuple(_confirmer_unites(profils, series))


def proposer_cible(profils: Sequence[ProfilColonne]) -> str | None:
    """Colonne proposée comme cible d'après son nom et sa place ; None si rien ne s'impose.

    Une étiquette issue d'un modèle (`cluster_gmm`) n'est jamais proposée : c'est un résultat, non une mesure.
    """
    meilleure, cle_max = None, (0, -99, False)
    dernier = len(profils) - 1
    for i, p in enumerate(profils):
        if p.role_propose in (Role.IDENTIFIANT, Role.DATE, Role.GROUPE, Role.IGNOREE):
            continue
        if _nom_etiquette_modele(p.nom):
            continue
        m = mots(p.nom)
        nom_evocateur = (3 if set(m) & MOTS_CIBLE_FORTS else 0) + (2 if set(m) & MOTS_CIBLE else 0)
        # À évocation égale : le nom le plus court (« Panne » avant « Panne_outil »), puis le dernier.
        cle = (nom_evocateur, -len(m), i == dernier)
        if (nom_evocateur or i == dernier) and cle >= cle_max:
            meilleure, cle_max = p.nom, cle
    return meilleure


def proposer_structure(roles: Sequence[tuple[str, Role]]) -> Structure:
    """Structure déduite des rôles : groupe seul, date seule, les deux, ou lignes indépendantes."""
    groupe = any(r is Role.GROUPE for _, r in roles)
    date = any(r is Role.DATE for _, r in roles)
    if groupe and date:
        return Structure.GROUPEE_CHRONOLOGIQUE
    if groupe:
        return Structure.GROUPEE
    return Structure.CHRONOLOGIQUE if date else Structure.INDEPENDANTE


def _nom_etiquette_modele(nom: str) -> bool:
    return bool(set(mots(nom)) & MOTS_ETIQUETTE_MODELE)


def _nom_personnel(nom: str) -> bool:
    m = set(mots(nom))
    return bool(m & MOTS_PERSONNEL) or ("age" in m and bool(m & MOTS_PERSONNE))


def _nom_soeur(nom: str, cible: str) -> bool:
    """Nom dérivé de la cible (« Panne_outil » pour « Panne ») ou dont la cible est dérivée."""
    a, b = _normalise(nom), _normalise(cible)
    return bool(a) and bool(b) and a != b and (a.startswith(b + "_") or b.startswith(a + "_"))


def preuve_soeur(colonne: pd.Series, cible: pd.Series) -> tuple[object, object, int] | None:
    """(valeur, valeur de la cible, lignes) si une modalité de la colonne implique la cible sur ses lignes.

    La modalité compte au moins `PREUVE_SOEUR_LIGNES_MIN` lignes et l'implication ne doit pas être due au
    hasard (probabilité inférieure à `PREUVE_SOEUR_HASARD`, d'après la fréquence de la valeur de la cible) :
    une colonne `Panne_outil` dont la valeur 1 donne toujours `Panne` = 1 est une sœur ; une température
    ambiante, non.
    """
    ok = (colonne.notna() & cible.notna()).to_numpy()
    x, y = colonne[ok], cible[ok]
    if len(x) < N_MIN_STAT or x.nunique() > MODALITES_MAX:
        return None
    frequences = y.value_counts(normalize=True)
    meilleure = None
    for valeur, groupe in y.groupby(x, observed=True):
        n = len(groupe)
        if n < PREUVE_SOEUR_LIGNES_MIN or groupe.nunique() != 1:
            continue
        t = groupe.iloc[0]
        base = float(frequences.get(t, 0.0))
        if 0.0 < base < 1.0 and n * math.log10(base) < math.log10(PREUVE_SOEUR_HASARD):
            meilleure = max(meilleure or (valeur, t, n), (valeur, t, n), key=lambda e: e[2])
    return meilleure


# ============================================================
# INDICATEURS DE FUITE PAR LA CIBLE
# ============================================================
def _pearson_colonnes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Pearson| de chaque colonne de `x` (n, p) avec `y` (n,), sans valeur manquante."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    normes = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(xc.T @ yc) / normes
    return np.nan_to_num(r, nan=0.0)


def _correlations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(|Pearson|, |Spearman|) de chaque colonne de `x` avec `y`, sur les paires complètes."""
    from scipy.stats import rankdata

    pearson, spearman = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    completes = np.isfinite(x).all(axis=0)
    if completes.any():
        xa = x[:, completes]
        pearson[completes] = _pearson_colonnes(xa, y)
        spearman[completes] = _pearson_colonnes(rankdata(xa, axis=0), rankdata(y))
    for j in np.flatnonzero(~completes):
        ok = np.isfinite(x[:, j])
        if ok.sum() >= N_MIN_STAT:
            xj, yj = x[ok, j][:, None], y[ok]
            pearson[j] = _pearson_colonnes(xj, yj)[0]
            spearman[j] = _pearson_colonnes(rankdata(xj, axis=0), rankdata(yj))[0]
    return pearson, spearman


def _nature_correlation(pearson: float, spearman: float) -> tuple[str, float]:
    """Linéaire presque parfaite : fuite ; monotone seulement : peut être la loi étudiée."""
    if pearson >= SEUIL_FUITE or pearson >= spearman:
        return "corrélation", pearson
    return NATURE_RANG, spearman


def _separation(valeurs: np.ndarray, modalites: pd.Series) -> float:
    """Part des lignes dont la valeur n'est couverte que par l'intervalle de sa propre modalité."""
    ok = np.isfinite(valeurs) & modalites.notna().to_numpy()
    if ok.sum() < N_MIN_STAT:
        return 0.0
    v = valeurs[ok]
    bornes = (
        pd.DataFrame({"v": v, "g": modalites[ok].to_numpy()})
        .groupby("g", observed=True)["v"]
        .agg(["min", "max"])
    )
    if not 2 <= len(bornes) <= MODALITES_MAX:
        return 0.0
    couvertures = np.zeros(v.size, dtype=np.int32)
    for bas, haut in bornes.itertuples(index=False):
        couvertures += (v >= bas) & (v <= haut)
    return float(np.mean(couvertures == 1))


def _cramer(a: pd.Series, b: pd.Series) -> float:
    """V de Cramér corrigé du biais (Bergsma) : pas de fausse alerte sur les modalités nombreuses."""
    if a.nunique() * b.nunique() > CROISEMENT_MAX:
        return 0.0
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    r, k = table.shape
    if min(r, k) < 2 or n < N_MIN_STAT:
        return 0.0
    attendu = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    phi2 = float(((table - attendu) ** 2 / attendu).sum() / n)
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denominateur = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denominateur)) if denominateur > 0 else 0.0


def _lignes_echantillon(n_table: int, positions: np.ndarray | None, graine: int) -> np.ndarray:
    """Positions absolues des lignes échantillonnées (au plus `ECHANTILLON_MAX`, tirage à graine fixe)."""
    base = np.arange(n_table) if positions is None else np.asarray(positions)
    if base.size > ECHANTILLON_MAX:
        base = base[np.sort(np.random.default_rng(graine).choice(base.size, ECHANTILLON_MAX, replace=False))]
    return base


def scores_fuite(
    table: Table,
    cible: str,
    colonnes: Sequence[str],
    *,
    positions: np.ndarray | None = None,
    graine: int = GRAINE_CONTRAT,
) -> dict[str, tuple[str, float]]:
    """Association de chaque colonne avec la cible : (nature de l'indicateur, valeur entre 0 et 1).

    Corrélation pour deux grandeurs continues, séparation pour une grandeur et une
    catégorie, V de Cramér pour deux catégories. Les calculs portent sur au plus
    `ECHANTILLON_MAX` lignes tirées parmi `positions` (toutes si None), les lignes sans
    cible étant écartées.
    """
    df = table.donnees
    base = _lignes_echantillon(len(df), positions, graine)
    base = base[df[cible].iloc[base].notna().to_numpy()]
    if base.size < N_MIN_STAT:
        return {}
    y = df[cible].iloc[base].reset_index(drop=True)
    y_reelle = _profil_colonne(cible, y).reelle
    y_num = y.to_numpy(dtype=float, na_value=np.nan) if y_reelle else None
    series = {c: df[c].iloc[base].reset_index(drop=True) for c in colonnes if c != cible}
    profils = {c: _profil_colonne(c, s) for c, s in series.items()}
    resultat: dict[str, tuple[str, float]] = {}
    reelles = [c for c, p in profils.items() if p.reelle and not p.constante]
    if y_reelle and reelles:
        matrice = np.column_stack([series[c].to_numpy(dtype=float, na_value=np.nan) for c in reelles])
        pearson, spearman = _correlations(matrice, y_num)
        for c, p, s in zip(reelles, pearson, spearman, strict=True):
            resultat[c] = _nature_correlation(float(p), float(s))
    for c, p in profils.items():
        if c in resultat or p.constante:
            continue
        s = series[c]
        if p.reelle:
            resultat[c] = ("séparation", _separation(s.to_numpy(dtype=float, na_value=np.nan), y))
        elif y_reelle:
            resultat[c] = ("séparation", _separation(y_num, s))
        else:
            resultat[c] = ("V de Cramér", _cramer(s, y))
    return resultat


# ============================================================
# CONTEXTE D'AUDIT
# ============================================================
@dataclass
class _Contexte:
    """Ce que voit l'audit : la table, la spécification, les lignes autorisées, le profil."""

    table: Table
    spec: SpecProjet | None
    coffre: CoffreTest | None
    lignes: np.ndarray | None  # positions des lignes d'apprentissage (None : toutes)
    profils: dict[str, ProfilColonne]
    n: int
    codes: set[str] = field(default_factory=set)

    @property
    def cible(self) -> str | None:
        return self.spec.cible if self.spec is not None else None

    @property
    def eligibles(self) -> tuple[str, ...]:
        """Colonnes qui pourraient entrer dans le modèle."""
        if self.spec is None:
            return tuple(self.profils)
        return tuple(c for c in self.spec.variables() if c in self.profils)

    def code(self, prefixe: str, colonne: str | None = None) -> str:
        """Code unique : « prefixe » ou « prefixe:colonne » (caractères non ASCII remplacés)."""
        brut = prefixe if colonne is None else f"{prefixe}:{_RE_CODE_ILLICITE.sub('_', _ascii(colonne))}"
        code, k = brut, 1
        while code in self.codes:
            k += 1
            code = f"{brut}-{k}"
        self.codes.add(code)
        return code

    def serie(self, nom: str) -> pd.Series:
        s = self.table.donnees[nom]
        return s if self.lignes is None else s.iloc[self.lignes]


def _ascii(texte: str) -> str:
    return unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode("ascii") or "col"


def _contexte(table: Table, spec: SpecProjet | None, coffre: CoffreTest | None) -> _Contexte:
    lignes = coffre.positions_apprentissage() if coffre is not None else None
    profils = {p.nom: p for p in profiler(table, lignes)}
    n = table.n_lignes if lignes is None else int(lignes.size)
    return _Contexte(table, spec, coffre, lignes, profils, n)


def _lister(noms: Sequence[str], maxi: int = 5) -> str:
    vus = ", ".join(q(c) for c in noms[:maxi])
    return vus if len(noms) <= maxi else f"{vus} et {len(noms) - maxi} autre(s)"


def _c(niveau: Niveau, code: str, message: str, action: str = "", ref: str | None = None) -> Constat:
    return Constat(niveau=niveau, code=code, message=message, action=action, ref_livre=ref)


# ============================================================
# CONSTATS
# ============================================================
def _constats_formules_et_lignes_vides(ctx: _Contexte) -> list[Constat]:
    """Formules du classeur (lues en valeurs) et lignes sans aucune valeur, d'après la lecture du fichier."""
    attrs = ctx.table.donnees.attrs
    constats: list[Constat] = []
    formules, sans_valeur = attrs.get(ATTR_FORMULES, (0, 0))
    if sans_valeur:
        constats.append(
            _c(
                Niveau.ALERTE,
                ctx.code("IO.formules_sans_valeur"),
                f"{nombre_lisible(sans_valeur)} cellule(s) du classeur contiennent une formule sans valeur "
                "calculée : elles sont lues comme manquantes (une formule n'est jamais évaluée).",
                "Ouvrir le classeur dans le tableur, l'enregistrer pour fixer les valeurs, puis le "
                "recharger ; ou acquitter si ces manquants sont acceptables.",
                renvoi("manquants"),
            )
        )
    elif formules:
        constats.append(
            _c(
                Niveau.INFO,
                ctx.code("IO.formules_lues_en_valeurs"),
                f"{nombre_lisible(formules)} cellule(s) du classeur contiennent une formule : leur valeur "
                "calculée par le tableur est lue, la formule n'est pas évaluée.",
            )
        )
    vides = attrs.get(ATTR_LIGNES_VIDES, 0)
    if vides:
        constats.append(
            _c(
                Niveau.ALERTE,
                ctx.code("DONNEES.lignes_vides"),
                f"{nombre_lisible(vides)} ligne(s) du fichier n'ont aucune valeur : elles gardent leur place "
                "(les positions du fichier sont conservées) mais ne portent aucune information.",
                "Les supprimer dans le tableur avant de charger le fichier, ou acquitter : en apprentissage "
                "supervisé, elles partent avec les lignes sans cible.",
                renvoi("profil"),
            )
        )
    return constats


def _constats_chargement(ctx: _Contexte) -> list[Constat]:
    constats = _constats_formules_et_lignes_vides(ctx)
    notes = ctx.table.donnees.attrs.get(ATTR_NOTES, ())
    if notes:
        constats.append(
            _c(
                Niveau.INFO,
                ctx.code("IO.remarques_lecture"),
                "À la lecture du fichier : " + " ; ".join(notes) + ".",
            )
        )
    if ctx.n < N_LIGNES_PETIT:
        constats.append(
            _c(
                Niveau.INFO,
                ctx.code("DONNEES.echantillon_petit"),
                f"Seulement {ctx.n} ligne(s) visibles pour l'apprentissage : scores très variables.",
                "",
                renvoi("echantillon"),
            )
        )
    return constats


def _constat_identifiant(ctx: _Contexte, p: ProfilColonne) -> list[Constat]:
    """Colonne qui ressemble à un identifiant : information, puis blocage ou alerte comme variable."""
    c = p.nom
    if ctx.spec is None:
        return [
            _c(
                Niveau.INFO,
                ctx.code("I3.identifiant", c),
                f"La colonne {q(c)} semble être un identifiant ({p.raison}) : elle sera proposée avec le "
                "rôle "
                "« identifiant » et exclue des variables.",
                "",
                renvoi("fuite"),
            )
        ]
    niveau = Niveau.BLOQUANT if p.preuve_identifiant == "nom" else Niveau.ALERTE
    return [
        _c(
            niveau,
            ctx.code("I3.identifiant", c),
            f"La colonne {q(c)} ressemble à un identifiant ({p.raison}) : comme variable, le modèle "
            "retiendrait les lignes par cœur au lieu d'apprendre.",
            "Passer son rôle à « identifiant » (ou « ignorée ») à l'étape Objectif."
            + ("" if niveau is Niveau.BLOQUANT else " Acquitter si c'est bien une mesure."),
            renvoi("fuite"),
        )
    ]


def _constats_identifiants(ctx: _Contexte) -> list[Constat]:
    constats: list[Constat] = []
    for c in ctx.eligibles:
        p = ctx.profils[c]
        if p.role_propose is Role.IDENTIFIANT or (
            p.preuve_identifiant == "nom_partiel" and ctx.spec is not None
        ):
            constats += _constat_identifiant(ctx, p)
        elif p.role_propose is Role.GROUPE and ctx.spec is not None:
            constats.append(
                _c(
                    Niveau.ALERTE,
                    ctx.code("I3.identifiant_repete", c),
                    f"La colonne {q(c)} identifie une unité qui revient sur plusieurs lignes : comme "
                    "variable, "
                    "le modèle apprendrait les unités et non le phénomène, et le test serait trop facile.",
                    "Lui donner le rôle « groupe » (découpage par unités), ou « ignorée ».",
                    renvoi("fuite"),
                )
            )
    return constats


def _constats_nom_de_colonne(ctx: _Contexte, c: str) -> list[Constat]:
    """Étiquette issue d'un modèle, attribut personnel : ce que le nom de la colonne laisse craindre."""
    constats: list[Constat] = []
    if _nom_etiquette_modele(c):
        constats.append(
            _c(
                Niveau.ALERTE,
                ctx.code("I3.etiquette_modele", c),
                f"La colonne {q(c)} semble produite par un modèle (regroupement, prédiction) : l'utiliser "
                "comme variable enchaîne deux modèles et peut propager une fuite.",
                "La retirer des variables, ou acquitter si elle est connue et fiable à la prévision.",
                renvoi("fuite"),
            )
        )
    if _nom_personnel(c):
        constats.append(
            _c(
                Niveau.ALERTE,
                ctx.code("I3.personnel", c),
                f"La colonne {q(c)} décrit une personne : l'utiliser peut introduire une discrimination sans "
                "lien avec la grandeur mesurée.",
                "La retirer des variables, ou acquitter en connaissance de cause.",
                renvoi("personnel"),
            )
        )
    return constats


def _constat_contenu_de_colonne(ctx: _Contexte, p: ProfilColonne) -> list[Constat]:
    """Colonne constante ou vide, nombres écrits en texte, catégories trop nombreuses (au plus un constat)."""
    c = p.nom
    if p.constante:
        vide = "vide" if p.famille == "vide" else "constante (une seule valeur)"
        return [
            _c(
                Niveau.ALERTE,
                ctx.code("DONNEES.constante", c),
                f"La colonne {q(c)} est {vide} : elle n'apporte aucune information.",
                "Passer son rôle à « ignorée ».",
                renvoi("profil"),
            )
        ]
    if p.nombres_en_texte:
        return [
            _c(
                Niveau.ALERTE,
                ctx.code("DONNEES.nombres_en_texte", c),
                f"La colonne {q(c)} contient des nombres écrits comme du texte : elle serait traitée comme "
                "une catégorie.",
                "Corriger le format dans le tableur (virgule décimale, unités collées aux nombres), puis "
                "recharger.",
                renvoi("profil"),
            )
        ]
    if p.famille == "texte" and p.n_distincts > MODALITES_MAX and p.preuve_identifiant is None:
        return [
            _c(
                Niveau.ALERTE,
                ctx.code("DONNEES.cardinalite", c),
                f"La colonne {q(c)} compte {nombre_lisible(p.n_distincts)} catégories : l'encodage créerait "
                "autant de colonnes.",
                "Regrouper les catégories rares dans le tableur, ou passer son rôle à « ignorée ».",
                renvoi("cardinalite"),
            )
        ]
    return []


def _constats_nature_des_colonnes(ctx: _Contexte) -> list[Constat]:
    """Pour chaque colonne d'entrée : ce que son nom puis son contenu appellent comme alerte."""
    constats: list[Constat] = []
    for c in ctx.eligibles:
        constats += _constats_nom_de_colonne(ctx, c)
        constats += _constat_contenu_de_colonne(ctx, ctx.profils[c])
    return constats


def _constats_manquants(ctx: _Contexte) -> list[Constat]:
    concernees = [
        ctx.profils[c] for c in ctx.eligibles if ctx.profils[c].n_manquants and not ctx.profils[c].constante
    ]
    if not concernees:
        return []
    constats: list[Constat] = []
    for p in concernees:
        if p.part_manquants >= PART_MANQUANTS_MASSIFS:
            constats.append(
                _c(
                    Niveau.ALERTE,
                    ctx.code("DONNEES.manquants_massifs", p.nom),
                    f"La colonne {q(p.nom)} est vide à {decimale(100 * p.part_manquants, 0)} % : une "
                    "imputation "
                    "inventerait l'essentiel de ses valeurs.",
                    "La passer à « ignorée », ou la garder en connaissance de cause (acquitter).",
                    renvoi("manquants"),
                )
            )
    triees = sorted(concernees, key=lambda p: -p.part_manquants)
    pire = triees[0].part_manquants
    detail = ", ".join(f"{q(p.nom)} ({decimale(100 * p.part_manquants)} %)" for p in triees[:5])
    niveau = Niveau.ALERTE if pire >= PART_MANQUANTS_ALERTE else Niveau.INFO
    constats.append(
        _c(
            niveau,
            ctx.code("DONNEES.manquants"),
            f"{len(concernees)} colonne(s) ont des valeurs manquantes : {detail}"
            + (" et d'autres." if len(triees) > 5 else "."),
            "Les compléter se fera dans le prétraitement, à l'intérieur de la validation croisée ; "
            "acquitter pour confirmer que ces manquants sont connus.",
            renvoi("manquants"),
        )
    )
    return constats


def _hachage_lignes(ctx: _Contexte, colonnes: Sequence[str]) -> np.ndarray:
    """Empreinte de chaque ligne sur les colonnes données, calculée colonne par colonne."""
    acc = np.zeros(ctx.n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for c in colonnes:
            h = pd.util.hash_pandas_object(ctx.serie(c), index=False).to_numpy(dtype=np.uint64)
            acc = acc * np.uint64(1_000_003) ^ h
    return acc


def _constats_doublons(ctx: _Contexte) -> list[Constat]:
    colonnes = [c for c in ctx.eligibles if ctx.profils[c].role_propose is not Role.IDENTIFIANT]
    if ctx.cible is not None and ctx.cible in ctx.profils:
        colonnes.append(ctx.cible)
    if not colonnes or ctx.n < 2:
        return []
    n_doubles = int(pd.Series(_hachage_lignes(ctx, colonnes)).duplicated().sum())
    if not n_doubles:
        return []
    part = n_doubles / ctx.n
    niveau = Niveau.ALERTE if part >= PART_DOUBLONS_ALERTE else Niveau.INFO
    return [
        _c(
            niveau,
            ctx.code("DONNEES.doublons"),
            f"{nombre_lisible(n_doubles)} ligne(s) répètent exactement une autre ligne "
            f"({decimale(100 * part)} %) : "
            "une ligne copiée peut se trouver des deux côtés du découpage et gonfler le score.",
            "Supprimer les doublons dans le tableur avant de charger le fichier, ou acquitter si les "
            "répétitions "
            "sont des mesures répétées voulues.",
            renvoi("doublons"),
        )
    ]


def _constats_aberrantes(ctx: _Contexte) -> list[Constat]:
    """Valeurs hors des bornes de Tukey : une information du profil, qui ne change rien aux données.

    Sont signalées les grandeurs continues proposées comme numériques dont au moins `PART_ABERRANTES_INFO`
    des valeurs (et `N_ABERRANTES_MIN` au moins) sortent des bornes ; sur les lignes visibles de l'audit.
    """
    trouvees = [
        p
        for p in (ctx.profils[c] for c in ctx.eligibles)
        if p.role_propose is Role.NUMERIQUE
        and p.n_hors_tukey >= N_ABERRANTES_MIN
        and p.n_hors_tukey >= PART_ABERRANTES_INFO * (p.n_lignes - p.n_manquants)
    ]
    if not trouvees:
        return []
    triees = sorted(trouvees, key=lambda p: -p.n_hors_tukey / (p.n_lignes - p.n_manquants))
    detail = ", ".join(
        f"{q(p.nom)} ({nombre_lisible(p.n_hors_tukey)} valeurs, "
        f"{decimale(100 * p.n_hors_tukey / (p.n_lignes - p.n_manquants))} %)"
        for p in triees[:5]
    )
    return [
        _c(
            Niveau.INFO,
            ctx.code("DONNEES.valeurs_aberrantes"),
            f"{len(trouvees)} colonne(s) ont des valeurs hors des bornes de Tukey, à plus de 1,5 fois "
            f"l'écart interquartile sous le premier quartile ou au dessus du troisième : {detail}"
            + (" et d'autres." if len(triees) > 5 else "."),
            "Les examiner dans le jeu : erreur de saisie à corriger, ou cas réels à garder ; l'écrêtage de "
            "Tukey du prétraitement les ramène aux bornes apprises sur l'apprentissage.",
            renvoi("aberrantes"),
        )
    ]


def _constats_cible_passee(ctx: _Contexte) -> list[Constat]:
    """Cible décalée ou persistance choisies : BLOQUANT sans déclaration, ou sur une cible qui décompte.

    La spécification d'un projet qui n'a pas encore choisi de modèle ne porte aucun de ces choix : rien n'est
    mesuré. Seules les lignes d'apprentissage se lisent quand le coffre existe.
    """
    spec = ctx.spec
    if spec is None or not spec.probleme.avec_cible or not cible_passee.lit_la_cible_passee(spec):
        return []
    donnees = ctx.table.donnees if ctx.lignes is None else ctx.table.donnees.iloc[ctx.lignes]
    constats = list(cible_passee.constats(spec, donnees))
    ctx.codes.update(c.code for c in constats)
    return constats


def _constat_non_citees(ctx: _Contexte) -> list[Constat]:
    if ctx.spec is None:
        return []
    ecartees = [c for c in ctx.table.colonnes if c not in ctx.spec.colonnes_citees()]
    if not ecartees:
        return []
    return [
        _c(
            Niveau.INFO,
            ctx.code("DONNEES.colonnes_non_citees"),
            f"{len(ecartees)} colonne(s) ne sont citées par aucun rôle et seront ignorées : "
            f"{_lister(ecartees)}.",
            "",
            renvoi("profil"),
        )
    ]


def _constat_soeur(ctx: _Contexte, c: str, cible: str, preuve: tuple[object, object, int] | None) -> Constat:
    """Cible sœur reconnue au nom : bloquante si les valeurs le confirment, sinon alerte à acquitter."""
    if preuve is not None:
        valeur, t, n = preuve
        return _c(
            Niveau.BLOQUANT,
            ctx.code("I3.soeur", c),
            f"La colonne {q(c)} porte un nom dérivé de la cible {q(cible)} et sa valeur {valeur} implique "
            f"{q(cible)} = {t} sur ses {nombre_lisible(n)} lignes : c'est une cible sœur (le même phénomène "
            "mesuré autrement), donc la réponse.",
            "La passer à « ignorée » : elle ne sera pas connue au moment de la prévision.",
            renvoi("fuite"),
        )
    return _c(
        Niveau.ALERTE,
        ctx.code("I3.soeur", c),
        f"La colonne {q(c)} porte un nom dérivé de la cible {q(cible)}, mais aucune de ses valeurs "
        "n'implique la cible : ce peut être une grandeur indépendante (une température ambiante pour une "
        "température mesurée).",
        "Acquitter si c'est un autre phénomène, connu avant la mesure de la cible ; sinon la passer à « "
        "ignorée ».",
        renvoi("fuite"),
    )


def _constat_association(ctx: _Contexte, c: str, cible: str, nature: str, valeur: float) -> Constat:
    """Association presque parfaite avec la cible : fuite si linéaire, alerte si seulement monotone."""
    if nature == NATURE_RANG:
        return _c(
            Niveau.ALERTE,
            ctx.code("I3.relation_monotone", c),
            f"La colonne {q(c)} suit la cible {q(cible)} de façon monotone presque parfaite "
            f"({nature} {decimale(valeur, 3)}) : c'est soit la loi étudiée, soit une autre sortie du même "
            "calcul.",
            "Acquitter si c'est une entrée du phénomène ; la passer à « ignorée » si elle est mesurée en "
            "même "
            "temps que la cible. Si la corrélation linéaire sur l'apprentissage dépasse 0,99, le constat "
            "devient bloquant.",
            renvoi("relation_monotone"),
        )
    return _c(
        Niveau.BLOQUANT,
        ctx.code("I3.fuite", c),
        f"La colonne {q(c)} contient déjà la réponse ({nature} {decimale(valeur, 3)} avec {q(cible)}).",
        "La passer à « ignorée » : elle ne sera pas connue au moment de la prévision.",
        renvoi("fuite"),
    )


def _constats_fuite(ctx: _Contexte) -> list[Constat]:
    spec = ctx.spec
    if spec is None or not spec.probleme.avec_cible or spec.cible not in ctx.profils:
        return []
    cible = spec.cible
    # Un identifiant a déjà son constat (I3.identifiant) : inutile de le mesurer une seconde fois.
    candidates = [
        c for c in ctx.eligibles if c != cible and ctx.profils[c].role_propose is not Role.IDENTIFIANT
    ]
    scores = scores_fuite(ctx.table, cible, candidates, positions=ctx.lignes, graine=spec.graine)
    constats: list[Constat] = []
    for c in candidates:
        nature, valeur = scores.get(c, ("", 0.0))
        soeur = _nom_soeur(c, cible)
        preuve = preuve_soeur(ctx.serie(c), ctx.serie(cible)) if soeur else None
        if preuve is not None:
            constats.append(_constat_soeur(ctx, c, cible, preuve))
        elif valeur >= SEUIL_FUITE:
            constats.append(_constat_association(ctx, c, cible, nature, valeur))
        elif soeur:
            constats.append(_constat_soeur(ctx, c, cible, None))
    return constats


def _constats_cible(ctx: _Contexte) -> list[Constat]:
    spec = ctx.spec
    if spec is None or not spec.probleme.avec_cible or spec.cible not in ctx.profils:
        return []
    return list(
        controle_cible(
            ctx.serie(spec.cible),
            spec.cible,
            spec.probleme,
            n_plis=spec.n_plis,
            part_test=spec.part_test,
            apres_decoupage=ctx.coffre is not None,
        )
    )


def _constats_temps(ctx: _Contexte) -> list[Constat]:
    spec = ctx.spec
    if spec is None:
        return [
            _c(
                Niveau.INFO,
                ctx.code("TEMPS.colonne_date", p.nom),
                f"La colonne {q(p.nom)} est un repère de temps ({p.raison}) : si les lignes se suivent "
                "dans le temps, "
                "le découpage doit être chronologique.",
                "",
                renvoi("temps_repere"),
            )
            for p in ctx.profils.values()
            if p.role_propose is Role.DATE
        ]
    constats: list[Constat] = []
    if not spec.structure.chronologique and spec.colonnes_de(Role.DATE):
        dates = ", ".join(q(c) for c in spec.colonnes_de(Role.DATE))
        constats.append(
            _c(
                Niveau.ALERTE,
                ctx.code("TEMPS.fuite_chronologique"),
                f"Une colonne de temps est déclarée ({dates}) mais le découpage ignore l'ordre : un "
                "découpage aléatoire "
                "mélange passé et futur, et le test devient plus facile que la réalité.",
                "Choisir une structure chronologique à l'étape Objectif, ou acquitter si l'ordre n'a pas "
                "d'importance.",
                renvoi("temps_validation"),
            )
        )
    for c in ctx.eligibles:
        if ctx.profils[c].role_propose is Role.DATE:
            constats.append(
                _c(
                    Niveau.ALERTE,
                    ctx.code("TEMPS.date_variable", c),
                    f"La colonne {q(c)} est un repère de temps utilisé comme variable : le modèle "
                    "reconnaîtrait la "
                    "période au lieu du phénomène.",
                    "Lui donner le rôle « date » (et un découpage chronologique), ou « ignorée ».",
                    renvoi("temps_validation"),
                )
            )
    return constats


def _constats_decoupage(ctx: _Contexte) -> list[Constat]:
    """Au S4, le découpage tient sa structure (I4) et aucune classe ne manque au test ni à un pli (I7).

    Le contrôle est celui du découpage (`split.controler_decoupage`), source unique, rejoué ici. Il ne lit que
    les colonnes de groupe et de date et les classes de la cible, jamais une autre valeur du test.
    """
    spec, coffre = ctx.spec, ctx.coffre
    if spec is None or coffre is None:
        return []
    if coffre.mode_challenge:
        if spec.structure is Structure.INDEPENDANTE:
            return []
        return [
            _c(
                Niveau.INFO,
                ctx.code("I4.non_verifiable"),
                "Le test est fourni à part : la structure du découpage (groupes, dates) est contrôlée "
                "à la construction du coffre, non rejouée ici.",
            )
        ]
    constats = [*controler_decoupage(ctx.table, spec, coffre), *controler_part_test(spec, coffre)]
    ctx.codes.update(c.code for c in constats)
    return constats


# ============================================================
# POINT D'ENTRÉE
# ============================================================
def auditer(table: Table, spec: SpecProjet | None, coffre: CoffreTest | None) -> tuple[Constat, ...]:
    """Audit des données : constats de niveau bloquant, alerte ou information, à codes uniques.

    C'est l'`Auditeur` de la session (voir `mecaml.core.session`). Sans spécification (S2),
    le profil seul est jugé ; avec elle (S3), les colonnes d'entrée et la cible ; avec le
    coffre (S4), uniquement les lignes d'apprentissage (sauf groupes et dates, pour I4).
    """
    ctx = _contexte(table, spec, coffre)
    constats = [
        *_constats_chargement(ctx),
        *_constats_cible(ctx),
        *_constats_identifiants(ctx),
        *_constats_fuite(ctx),
        *_constats_nature_des_colonnes(ctx),
        *_constats_manquants(ctx),
        *_constats_aberrantes(ctx),
        *_constats_doublons(ctx),
        *_constats_temps(ctx),
        *_constats_cible_passee(ctx),
        *_constats_decoupage(ctx),
        *_constat_non_citees(ctx),
    ]
    return tuple(constats)


# ============================================================
# PROPOSITION DE RÔLES
# ============================================================
@dataclass(frozen=True, eq=False)
class PropositionRoles:
    """Rôles proposés et raison de chaque choix (pourquoi une colonne est écartée, laquelle est la date)."""

    roles: tuple[tuple[str, Role], ...]
    raisons: dict[str, str]

    def raison_de(self, colonne: str) -> str:
        """Raison du rôle proposé pour une colonne."""
        return self.raisons[colonne]


def _ecarter_par_le_nom_ou_le_contenu(
    profils: Sequence[ProfilColonne], roles: dict[str, Role], raisons: dict[str, str]
) -> None:
    for p in profils:
        if p.role_propose not in (Role.NUMERIQUE, Role.CATEGORIELLE):
            continue
        if _nom_etiquette_modele(p.nom):
            motif = "écartée : étiquette issue d'un modèle (regroupement, prédiction)"
        elif _nom_personnel(p.nom):
            motif = "écartée : attribut personnel"
        elif p.nombres_en_texte:
            motif = "écartée : nombres écrits comme du texte, à corriger dans le fichier"
        else:
            continue
        roles[p.nom], raisons[p.nom] = Role.IGNOREE, motif


def _ecarter_les_fuites(
    table: Table, cible: str, roles: dict[str, Role], raisons: dict[str, str], graine: int
) -> None:
    entrees = [c for c, r in roles.items() if r in (Role.NUMERIQUE, Role.CATEGORIELLE)]
    df = table.donnees
    for c in entrees:
        if _nom_soeur(c, cible):
            preuve = preuve_soeur(df[c], df[cible])
            if preuve is not None:
                roles[c] = Role.IGNOREE
                raisons[c] = f"écartée : cible sœur (la valeur {preuve[0]} implique {cible} = {preuve[1]})"
    restantes = [c for c in entrees if roles[c] is not Role.IGNOREE]
    for c, (nature, valeur) in scores_fuite(table, cible, restantes, graine=graine).items():
        if valeur >= SEUIL_FUITE and nature != NATURE_RANG:
            roles[c], raisons[c] = (
                Role.IGNOREE,
                f"écartée : fuite par la cible ({nature} {decimale(valeur, 3)})",
            )


def proposition_roles(
    table: Table, cible: str | None = None, *, graine: int = GRAINE_CONTRAT
) -> PropositionRoles:
    """Rôles proposés pour chaque colonne, avec la raison ; l'utilisateur les confirme ou les change.

    Identifiant, date et groupe d'après le profil ; colonne vide ou constante, étiquette issue d'un modèle,
    attribut personnel, cible sœur confirmée par les valeurs et fuite par la cible : « ignorée ». Une
    spécification bâtie sur cette proposition passe donc l'audit sans constat bloquant. Une colonne au nom de
    cible sœur sans preuve dans les valeurs reste une variable, que l'audit signale par une alerte.
    """
    profils = profiler(table)
    roles: dict[str, Role] = {p.nom: p.role_propose for p in profils}
    raisons: dict[str, str] = {p.nom: p.raison for p in profils}
    _ecarter_par_le_nom_ou_le_contenu(profils, roles, raisons)
    _un_seul(roles, Role.DATE, raisons)
    _un_seul(roles, Role.GROUPE, raisons)
    if cible is not None:
        if cible not in roles:
            raise KeyError(f"Cible absente de la table : {cible!r}.")
        roles[cible], raisons[cible] = Role.CIBLE, "cible choisie"
        _ecarter_les_fuites(table, cible, roles, raisons, graine)
    return PropositionRoles(tuple(sorted(roles.items())), raisons)


def proposer_roles(
    table: Table, cible: str | None = None, *, graine: int = GRAINE_CONTRAT
) -> tuple[tuple[str, Role], ...]:
    """Rôles proposés (voir `proposition_roles` pour les raisons)."""
    return proposition_roles(table, cible, graine=graine).roles


def _un_seul(roles: dict[str, Role], role: Role, raisons: dict[str, str]) -> None:
    """Une structure n'admet qu'une date et qu'un groupe : les suivants sont ignorés."""
    for c in [c for c, r in roles.items() if r is role][1:]:
        roles[c], raisons[c] = (
            Role.IGNOREE,
            f"écartée : une seule colonne de rôle « {role.value} » est permise",
        )


def probleme_propose(table: Table, cible: str) -> Probleme:
    """Problème proposé pour la cible (classification si peu de modalités, régression sinon)."""
    return detecter_probleme(table.donnees[cible])[0]
