"""
MécaML : garde-fous du cœur.

Bornes de taille (I9), fichiers acceptés, techniques coupées au delà d'une taille,
cible exploitable (I7) et confinement des écritures (I10). Ce module ne dépend que
des types du cœur et de la table des renvois : il ne lit aucun fichier et n'écrit nulle part.

Chaque refus est une `ErreurDonnees` dont le message dit quoi faire.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mecaml.core import renvois
from mecaml.core.tasks import ErreurAnnulation
from mecaml.core.types import LIGNES_MAX_LENTES, Constat, Niveau, Probleme

if TYPE_CHECKING:
    from mecaml.core.catalog import EntreeCatalogue

__all__ = [
    "COLONNES_MAX",
    "EXTENSIONS_ACCEPTEES",
    "LIGNES_LECTURE_EN_FIL",
    "LIGNES_MAX",
    "LIGNES_MAX_LENTES",
    "TAILLE_FICHIER_MAX_OCTETS",
    "TAILLE_XLSX_DECOMPRESSE_MAX_OCTETS",
    "ErreurAnnulee",
    "ErreurDonnees",
    "ResumeCible",
    "chemin_dans_dossier",
    "controle_cible",
    "decimale",
    "detecter_probleme",
    "extension_acceptee",
    "lecture_en_fil_requise",
    "lignes_sans_cible",
    "nombre_lisible",
    "renvoi",
    "resumer_cible",
    "technique_coupee",
    "verifier_bornes",
    "verifier_taille_fichier",
]

# ============================================================
# BORNES (I9)
# ============================================================
LIGNES_MAX = 200_000
COLONNES_MAX = 500
LIGNES_LECTURE_EN_FIL = 50_000  # au delà : lecture xlsx dans un fil, avec progression
TAILLE_FICHIER_MAX_OCTETS = 500 * 2**20
TAILLE_XLSX_DECOMPRESSE_MAX_OCTETS = 2 * 2**30
EXTENSIONS_ACCEPTEES = (".xlsx", ".csv")


def renvoi(theme: str) -> str | None:
    """Identifiant `T<cc>-<s>-<k>` du livre pour un thème de constat (None si le livre n'en parle pas).

    Lit la table unique des renvois (`mecaml.core.renvois`).
    """
    return renvois.theme(theme)


class ErreurDonnees(ValueError):
    """Données ou fichier refusés ; le message dit quoi faire."""


# Une seule exception d'annulation pour la lecture et les calculs : celle des tâches.
ErreurAnnulee = ErreurAnnulation


def nombre_lisible(n: int) -> str:
    """Entier avec espaces fines insécables : 200 000."""
    return f"{n:,}".replace(",", " ")


def decimale(x: float, chiffres: int = 1) -> str:
    """Nombre avec virgule décimale : 3,39."""
    return f"{x:.{chiffres}f}".replace(".", ",")


def lecture_en_fil_requise(n_lignes: int) -> bool:
    """Vrai si la lecture d'un classeur de cette taille doit se faire dans un fil, avec progression."""
    return n_lignes > LIGNES_LECTURE_EN_FIL


def extension_acceptee(chemin: str | Path) -> str:
    """Extension en minuscules (« .xlsx » ou « .csv »), sinon refus explicite."""
    chemin = Path(chemin)
    ext = chemin.suffix.lower()
    if ext in EXTENSIONS_ACCEPTEES:
        return ext
    attendu = " ou ".join(EXTENSIONS_ACCEPTEES)
    trouve = f"« {ext} »" if ext else "sans extension"
    raise ErreurDonnees(
        f"Format non pris en charge ({trouve}) : seuls les fichiers {attendu} sont acceptés. "
        "Enregistrer le jeu de données en .xlsx ou en .csv depuis le tableur, puis le recharger."
    )


def verifier_taille_fichier(taille_octets: int, nom: str = "Le fichier") -> None:
    """Refuse un fichier trop volumineux ou vide."""
    if taille_octets <= 0:
        raise ErreurDonnees(f"{nom} est vide : choisir un fichier qui contient des données.")
    if taille_octets > TAILLE_FICHIER_MAX_OCTETS:
        raise ErreurDonnees(
            f"{nom} pèse {taille_octets / 2**20:.0f} Mio, au delà de la limite de "
            f"{TAILLE_FICHIER_MAX_OCTETS / 2**20:.0f} Mio. Retirer les colonnes inutiles ou "
            "échantillonner les lignes avant de le charger."
        )


def verifier_bornes(n_lignes: int, n_colonnes: int) -> None:
    """Refuse un jeu hors des bornes (200 000 lignes, 500 colonnes) ou inexploitable."""
    if n_colonnes > COLONNES_MAX:
        raise ErreurDonnees(
            f"Le jeu compte {nombre_lisible(n_colonnes)} colonnes, au delà de la limite de "
            f"{nombre_lisible(COLONNES_MAX)}. Retirer les colonnes inutiles dans le tableur, "
            "puis recharger le fichier."
        )
    if n_lignes > LIGNES_MAX:
        raise ErreurDonnees(
            f"Le jeu compte {nombre_lisible(n_lignes)} lignes, au delà de la limite de "
            f"{nombre_lisible(LIGNES_MAX)}. Échantillonner ou filtrer les lignes dans le tableur, "
            "puis recharger le fichier."
        )
    if n_lignes < 1:
        raise ErreurDonnees("Le jeu ne contient aucune ligne de données sous la ligne d'en-tête.")
    if n_colonnes < 2:
        raise ErreurDonnees(
            "Le jeu ne contient qu'une colonne. Vérifier le séparateur du fichier .csv "
            "(virgule, point-virgule, tabulation) ou la feuille choisie du classeur."
        )


def technique_coupee(
    id_technique: str, n_lignes: int, registre: Iterable[EntreeCatalogue] | None = None
) -> str | None:
    """Message de refus si la technique est trop lente pour `n_lignes`, sinon None.

    Le critère est celui du registre (`EntreeCatalogue.coupee` : coût LOURD au delà de `LIGNES_MAX_LENTES`) ;
    une technique inconnue du registre n'est jamais coupée ici. C'est le seul message de refus d'une
    technique trop lente : l'entraînement (`train.verifier_bornes`) et la validation par exclusion d'une
    ligne (`split.plis_validation`) le rendent tel quel.
    """
    from mecaml.core.catalog import entree

    try:
        coupee = entree(id_technique, registre).coupee(n_lignes)
    except KeyError:
        return None
    if not coupee:
        return None
    return (
        f"La technique {id_technique} est coupée au delà de {nombre_lisible(LIGNES_MAX_LENTES)} lignes "
        f"(le jeu d'apprentissage en compte {nombre_lisible(n_lignes)}) : son temps de calcul croît plus "
        "vite que le nombre de lignes. Choisir une technique plus légère (arbre, forêt, gradient boosting "
        "pour un modèle, plis aléatoires pour une validation) ou travailler sur un échantillon d'au plus "
        f"{nombre_lisible(LIGNES_MAX_LENTES)} lignes."
    )


# ============================================================
# CONFINEMENT DES ÉCRITURES (I10)
# ============================================================
_RE_CARACTERES_INTERDITS = re.compile(r'[<>:"|?*\x00-\x1f]')
_NOMS_RESERVES = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}
)


def chemin_dans_dossier(dossier: str | Path, nom: str) -> Path:
    """Chemin d'un fichier à écrire, garanti sous `dossier` (le dossier choisi par l'utilisateur)."""
    if not isinstance(nom, str) or not nom.strip():
        raise ErreurDonnees("Nom de fichier à écrire vide.")
    win, posix = PureWindowsPath(nom), PurePosixPath(nom)
    if win.anchor or posix.is_absolute() or ".." in win.parts or ".." in posix.parts:
        raise ErreurDonnees(
            f"Le nom « {nom} » sort du dossier choisi : donner un simple nom de fichier, "
            "sans chemin absolu ni « .. »."
        )
    if _RE_CARACTERES_INTERDITS.search(nom):
        raise ErreurDonnees(
            f'Le nom « {nom} » contient un caractère interdit (: < > " | ? * ou de contrôle).'
        )
    for partie in win.parts:
        if partie.rstrip(". ") != partie or partie.split(".")[0].lower() in _NOMS_RESERVES:
            raise ErreurDonnees(f"Le nom « {nom} » n'est pas un nom de fichier valide sous Windows.")
    base = Path(dossier)
    if not base.is_dir():
        raise ErreurDonnees(f"Le dossier de sortie « {base} » n'existe pas : le choisir ou le créer d'abord.")
    base = base.resolve()
    cible = (base / nom).resolve()
    if cible != base and base not in cible.parents:
        raise ErreurDonnees(f"Le nom « {nom} » sort du dossier choisi (lien symbolique ou raccourci).")
    return cible


# ============================================================
# CIBLE (I7)
# ============================================================
PART_CLASSE_RARE = 0.05
_MODALITES_CLASSIFICATION = 10
_DISTINCTES_CONTINUE = 20
_DISTINCTES_ENTIER_CONTINU = 50


@dataclass(frozen=True)
class ResumeCible:
    """Ce que l'on voit de la cible : valeurs présentes, manquantes, classes."""

    n_total: int
    n_valides: int
    n_classes: int
    numerique: bool
    entiere: bool  # nombres tous entiers
    plus_petite_nom: str  # classe la plus rare (vide si aucune valeur)
    plus_petite_effectif: int

    @property
    def n_manquantes(self) -> int:
        """Lignes sans cible."""
        return self.n_total - self.n_valides

    @property
    def n_distinctes(self) -> int:
        """Nombre de valeurs distinctes présentes (égal au nombre de classes)."""
        return self.n_classes


def resumer_cible(serie: pd.Series) -> ResumeCible:
    """Résumé de la cible, calculé sans modifier la série."""
    valides = serie.dropna()
    numerique = bool(pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie))
    entiere = bool(
        numerique and len(valides) > 0 and bool(np.all(np.mod(valides.to_numpy(dtype=float), 1.0) == 0))
    )
    comptes = valides.value_counts()
    rare_nom, rare_n = (str(comptes.index[-1]), int(comptes.iloc[-1])) if len(comptes) else ("", 0)
    return ResumeCible(
        n_total=len(serie),
        n_valides=len(valides),
        n_classes=len(comptes),
        numerique=numerique,
        entiere=entiere,
        plus_petite_nom=rare_nom,
        plus_petite_effectif=rare_n,
    )


def lignes_sans_cible(donnees: pd.DataFrame, cible: str) -> int:
    """Nombre de lignes dont la cible est manquante (elles sont retirées avant l'apprentissage)."""
    return int(donnees[cible].isna().sum())


def detecter_probleme(serie: pd.Series) -> tuple[Probleme, str]:
    """Problème proposé pour cette cible, avec la raison ; l'utilisateur confirme."""
    r = resumer_cible(serie)
    if r.n_valides == 0:
        return Probleme.CLASSIFICATION, "aucune valeur présente : la cible est à corriger"
    if not r.numerique:
        return Probleme.CLASSIFICATION, f"cible catégorielle ({r.n_classes} modalités)"
    if r.n_classes <= 2:
        return Probleme.CLASSIFICATION, f"cible à {r.n_classes} valeur(s) distincte(s)"
    if r.entiere and r.n_classes <= _MODALITES_CLASSIFICATION:
        return Probleme.CLASSIFICATION, f"cible entière à {r.n_classes} valeurs : des classes codées"
    return Probleme.REGRESSION, f"cible numérique ({r.n_classes} valeurs distinctes)"


def _constat(niveau: Niveau, code: str, message: str, action: str, ref: str | None) -> Constat:
    return Constat(niveau=niveau, code=code, message=message, action=action, ref_livre=ref)


def _incoherence(r: ResumeCible, probleme: Probleme, cible: str) -> list[Constat]:
    """Cible incohérente avec le problème choisi."""
    q = f"« {cible} »"
    if probleme is Probleme.CLASSIFICATION:
        continue_ = r.numerique and (
            (not r.entiere and r.n_distinctes > _DISTINCTES_CONTINUE)
            or (r.entiere and r.n_distinctes > _DISTINCTES_ENTIER_CONTINU)
        )
        if continue_:
            return [
                _constat(
                    Niveau.BLOQUANT,
                    "I7.cible_continue",
                    f"La cible {q} prend {nombre_lisible(r.n_distinctes)} valeurs numériques distinctes : "
                    "c'est une grandeur continue, pas des classes.",
                    "Choisir le problème « régression », ou découper la cible en classes avant de la "
                    "charger.",
                    renvoi("cible_classification"),
                )
            ]
        if r.numerique and r.entiere and r.n_distinctes > _DISTINCTES_CONTINUE:
            return [
                _constat(
                    Niveau.ALERTE,
                    "I7.cible_entiere_nombreuse",
                    f"La cible {q} compte {r.n_distinctes} valeurs entières distinctes : "
                    "autant de classes est inhabituel.",
                    "Vérifier qu'il s'agit bien de classes ; sinon choisir « régression ».",
                    renvoi("cible_classification"),
                )
            ]
        return []
    if not r.numerique:
        return [
            _constat(
                Niveau.BLOQUANT,
                "I7.cible_textuelle",
                f"La cible {q} contient du texte : une régression prédit un nombre.",
                "Choisir le problème « classification », ou corriger la colonne si elle devait être "
                "numérique.",
                renvoi("cible_regression"),
            )
        ]
    if r.n_distinctes <= 2:
        return [
            _constat(
                Niveau.ALERTE,
                "I7.cible_binaire_en_regression",
                f"La cible {q} ne prend que {r.n_distinctes} valeur(s) : c'est plutôt une classification.",
                "Passer en « classification », ou acquitter si une probabilité continue est voulue.",
                renvoi("cible_regression"),
            )
        ]
    return []


def _classes(
    r: ResumeCible, cible: str, n_plis: int, part_test: float, apres_decoupage: bool
) -> list[Constat]:
    """Contrôles de la classification : nombre de classes, classe rare, classe plus petite que les plis."""
    q = f"« {cible} »"
    if r.n_classes < 2:
        return [
            _constat(
                Niveau.BLOQUANT,
                "I7.moins_de_deux_classes",
                f"La cible {q} n'a que {r.n_classes} classe(s) : rien à distinguer.",
                "Choisir une autre cible, ou charger un jeu qui contient au moins deux classes.",
                renvoi("cible_classification"),
            )
        ]
    constats: list[Constat] = []
    petite = r.plus_petite_effectif
    nom_petite = r.plus_petite_nom
    reste = petite if apres_decoupage else int(petite * (1.0 - part_test))
    if petite < n_plis or (apres_decoupage and reste < n_plis):
        constats.append(
            _constat(
                Niveau.BLOQUANT,
                "I7.classe_plus_petite_que_les_plis",
                f"La classe « {nom_petite} » de la cible {q} compte {petite} ligne(s) : "
                f"moins que les {n_plis} plis de la validation croisée.",
                f"Réduire le nombre de plis à {max(2, petite)} au plus, regrouper les classes rares, "
                "ou rassembler davantage de lignes.",
                renvoi("plis"),
            )
        )
    elif reste < n_plis:
        constats.append(
            _constat(
                Niveau.ALERTE,
                "I7.classe_trop_petite_apres_decoupage",
                f"La classe « {nom_petite} » de la cible {q} compte {petite} ligne(s) : après mise à part "
                f"du test, il en restera environ {reste}, moins que les {n_plis} plis.",
                "Réduire la part de test ou le nombre de plis, ou regrouper les classes rares.",
                renvoi("plis"),
            )
        )
    # Rare : moins de 5 % des lignes et moins de la moitié d'une part égale (30 classes à 3 % ne le sont pas).
    if r.n_valides and petite / r.n_valides < min(PART_CLASSE_RARE, 0.5 / r.n_classes):
        constats.append(
            _constat(
                Niveau.ALERTE,
                "I7.classe_rare",
                f"La classe « {nom_petite} » ne représente que {decimale(100 * petite / r.n_valides, 2)} % "
                f"des lignes de la cible {q} : l'exactitude sera trompeuse.",
                "Choisir une métrique adaptée (rappel, précision moyenne, F1) et garder la stratification.",
                renvoi("classe_rare"),
            )
        )
    return constats


def controle_cible(
    serie: pd.Series,
    cible: str,
    probleme: Probleme,
    *,
    n_plis: int = 5,
    part_test: float = 0.2,
    apres_decoupage: bool = False,
) -> tuple[Constat, ...]:
    """Constats sur la cible : manquantes, incohérence avec le problème, classes (I7).

    `apres_decoupage` vrai : `serie` ne contient que les lignes d'apprentissage.
    """
    r = resumer_cible(serie)
    q = f"« {cible} »"
    if r.n_valides == 0:
        return (
            _constat(
                Niveau.BLOQUANT,
                "I7.cible_vide",
                f"La cible {q} n'a aucune valeur renseignée.",
                "Choisir une autre colonne comme cible.",
                renvoi("cible"),
            ),
        )
    constats: list[Constat] = []
    if r.n_manquantes:
        constats.append(
            _constat(
                Niveau.ALERTE,
                "I7.cible_manquante",
                f"{nombre_lisible(r.n_manquantes)} ligne(s) n'ont pas de valeur pour la cible {q} "
                f"({decimale(100 * r.n_manquantes / r.n_total)} %) : elles seront retirées de "
                "l'apprentissage "
                "et comptées dans le rapport.",
                "Acquitter si ce retrait convient, sinon compléter la cible dans le tableur.",
                renvoi("cible"),
            )
        )
    incoherences = _incoherence(r, probleme, cible)
    constats.extend(incoherences)
    if probleme is Probleme.CLASSIFICATION and not any(c.bloquant for c in incoherences):
        constats.extend(_classes(r, cible, n_plis, part_test, apres_decoupage))
    return tuple(constats)
