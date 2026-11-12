"""
MécaML : types partagés du cœur.

Énumérations et structures de données communes à tous les modules du cœur.
Règle d'or : les spécifications sont gelées et normalisées (rôles triés par
colonne, paramètres triés par nom, listes changées en tuples), donc comparables
et hachables ; les données (`Table`) et les objets ajustés sont partagés par
référence, jamais copiés d'un état du projet à l'autre. Une empreinte permet de
vérifier à tout moment que la table n'a pas été modifiée.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "GRAINE_CONTRAT",
    "LIGNES_MAX_LENTES",
    "METRIQUES_CLASSE_POSITIVE",
    "REFERENCES_NAIVES",
    "VALIDATIONS",
    "VALIDATIONS_PAR_STRUCTURE",
    "VALIDATIONS_STRATIFIEES",
    "ChoixTechnique",
    "CoffreTest",
    "Constat",
    "Decision",
    "ErreurScelle",
    "Niveau",
    "Probleme",
    "Role",
    "SpecProjet",
    "Structure",
    "Table",
    "empreinte_dataframe",
    "identifiant_metrique",
    "validations_permises",
]


# ============================================================
# ÉNUMÉRATIONS
# ============================================================
class Role(StrEnum):
    """Rôle d'une colonne dans le projet."""

    CIBLE = "cible"
    NUMERIQUE = "numerique"
    CATEGORIELLE = "categorielle"
    DATE = "date"  # date ou indice d'ordre temporel (temps, cycle)
    GROUPE = "groupe"  # unité suivie dans le temps ou regroupée (machine, lot)
    IDENTIFIANT = "identifiant"  # jamais utilisé comme variable
    IGNOREE = "ignoree"


class Structure(StrEnum):
    """Structure des lignes, dont se déduit le découpage apprentissage et test."""

    INDEPENDANTE = "independante"
    GROUPEE = "groupee"
    CHRONOLOGIQUE = "chronologique"
    GROUPEE_CHRONOLOGIQUE = "groupee_chronologique"

    @property
    def groupee(self) -> bool:
        """Vrai si les lignes sont regroupées par une colonne de rôle GROUPE."""
        return self in (Structure.GROUPEE, Structure.GROUPEE_CHRONOLOGIQUE)

    @property
    def chronologique(self) -> bool:
        """Vrai si les lignes sont ordonnées par une colonne de rôle DATE."""
        return self in (Structure.CHRONOLOGIQUE, Structure.GROUPEE_CHRONOLOGIQUE)


class Probleme(StrEnum):
    """Nature du problème posé au jeu de données."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SANS_CIBLE = "sans_cible"  # réduction de dimension, regroupement, anomalies

    @property
    def avec_cible(self) -> bool:
        """Vrai pour la régression et la classification."""
        return self is not Probleme.SANS_CIBLE


class Niveau(StrEnum):
    """Gravité d'un constat d'audit."""

    BLOQUANT = "bloquant"  # interdit d'avancer tant qu'il subsiste
    ALERTE = "alerte"  # doit être acquittée avant d'avancer
    INFO = "info"


# ============================================================
# CONSTAT D'AUDIT
# ============================================================
_RE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$")


@dataclass(frozen=True)
class Constat:
    """Constat d'audit : ce qui est vu, ce qu'il faut faire, où le livre l'explique.

    `code` identifie le constat de façon unique dans un audit (par exemple
    « I3.identifiant:id_piece ») : c'est la clé de l'acquittement.
    """

    niveau: Niveau
    code: str
    message: str
    action: str = ""
    ref_livre: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.niveau, Niveau):
            raise TypeError("Constat : `niveau` doit être un `Niveau`.")
        if not isinstance(self.code, str) or not _RE_CODE.match(self.code):
            raise ValueError(f"Constat : code invalide {self.code!r} (lettres, chiffres, « _ . : - »).")
        if not self.message.strip():
            raise ValueError(f"Constat {self.code} : message vide.")
        if self.niveau is not Niveau.INFO and not self.action.strip():
            raise ValueError(f"Constat {self.code} : un constat {self.niveau} doit proposer une action.")

    @property
    def bloquant(self) -> bool:
        """Vrai si le constat interdit d'avancer."""
        return self.niveau is Niveau.BLOQUANT


# ============================================================
# TABLE DE DONNÉES
# ============================================================
def empreinte_dataframe(df: pd.DataFrame) -> str:
    """Empreinte SHA-256 du contenu, des colonnes, des types et de l'index."""
    h = hashlib.sha256()
    h.update(repr(df.shape).encode("utf-8"))
    h.update(repr([(str(c), str(t)) for c, t in df.dtypes.items()]).encode("utf-8"))
    if len(df.columns) and len(df):
        lignes = pd.util.hash_pandas_object(df, index=True)
        h.update(np.ascontiguousarray(lignes.to_numpy()).tobytes())
    else:
        h.update(repr(list(df.index)).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class Table:
    """Jeu de données chargé, partagé par référence entre les états du projet.

    La dataclass est gelée mais le DataFrame reste un objet pandas modifiable :
    aucun module du cœur ne doit le modifier, ce que `est_intacte` vérifie.
    L'égalité est l'identité : deux tables sont égales si c'est le même objet.
    """

    donnees: pd.DataFrame = field(repr=False)
    nom: str
    feuille: str | None = None
    empreinte: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.donnees, pd.DataFrame):
            raise TypeError("Table : `donnees` doit être un DataFrame pandas.")
        colonnes = list(self.donnees.columns)
        non_textes = [c for c in colonnes if not isinstance(c, str)]
        if non_textes:
            raise ValueError(
                f"Table : noms de colonnes non textuels {non_textes[:5]!r} ; "
                "convertir les noms en texte au chargement."
            )
        doublons = sorted({c for c in colonnes if colonnes.count(c) > 1})
        if doublons:
            raise ValueError(f"Table : colonnes en double {doublons[:5]!r} ; renommer ces colonnes.")
        object.__setattr__(self, "empreinte", empreinte_dataframe(self.donnees))

    def __repr__(self) -> str:
        return f"Table(nom={self.nom!r}, feuille={self.feuille!r}, forme={self.forme})"

    @property
    def n_lignes(self) -> int:
        """Nombre de lignes."""
        return int(self.donnees.shape[0])

    @property
    def n_colonnes(self) -> int:
        """Nombre de colonnes."""
        return int(self.donnees.shape[1])

    @property
    def forme(self) -> tuple[int, int]:
        """(lignes, colonnes)."""
        return self.n_lignes, self.n_colonnes

    @property
    def colonnes(self) -> tuple[str, ...]:
        """Noms des colonnes, dans l'ordre du fichier."""
        return tuple(self.donnees.columns)

    def taille_octets(self) -> int:
        """Mémoire occupée par les données, chaînes comprises."""
        return int(self.donnees.memory_usage(index=True, deep=True).sum())

    def est_intacte(self) -> bool:
        """Vrai si le contenu n'a pas changé depuis le chargement."""
        return empreinte_dataframe(self.donnees) == self.empreinte


# ============================================================
# CHOIX D'UNE TECHNIQUE
# ============================================================
_RE_ID_TECHNIQUE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
_SCALAIRES = (str, bool, int, float, type(None))


def _normaliser_valeur(valeur: Any, technique: str, nom: str) -> Any:
    """Valeur de paramètre normalisée : scalaire simple, ou tuple de valeurs normalisées."""
    if isinstance(valeur, np.generic):
        valeur = valeur.item()
    if isinstance(valeur, float) and not math.isfinite(valeur):
        raise ValueError(
            f"ChoixTechnique {technique} : paramètre {nom!r} non fini ; laisser la valeur par défaut "
            "de la technique plutôt que de passer une valeur manquante."
        )
    if isinstance(valeur, _SCALAIRES):
        return valeur
    if isinstance(valeur, (list, tuple)):
        return tuple(_normaliser_valeur(v, technique, nom) for v in valeur)
    raise TypeError(
        f"ChoixTechnique {technique} : paramètre {nom!r} de type {type(valeur).__name__} non sérialisable "
        "(permis : texte, nombre, booléen, None, liste ou tuple de ces valeurs)."
    )


@dataclass(frozen=True)
class ChoixTechnique:
    """Technique retenue (identifiant du registre), ses paramètres et ses colonnes.

    Les paramètres sont triés par nom et leurs listes deviennent des tuples : deux
    choix équivalents sont égaux, hachables, et survivent à un aller-retour JSON.
    `colonnes` vide signifie : toutes les colonnes concernées par la technique.
    """

    id: str
    parametres: tuple[tuple[str, Any], ...] = ()
    colonnes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _RE_ID_TECHNIQUE.match(self.id):
            raise ValueError(f"ChoixTechnique : identifiant invalide {self.id!r} (forme « famille.nom »).")
        paires = (
            list(self.parametres.items()) if isinstance(self.parametres, Mapping) else list(self.parametres)
        )
        if any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in paires):
            raise TypeError(f"ChoixTechnique {self.id} : `parametres` attend des paires (nom, valeur).")
        noms = [nom for nom, _ in paires]
        if any(not isinstance(nom, str) or not nom for nom in noms) or len(set(noms)) != len(noms):
            raise ValueError(f"ChoixTechnique {self.id} : noms de paramètres invalides ou en double.")
        normalises = tuple(sorted((nom, _normaliser_valeur(v, self.id, nom)) for nom, v in paires))
        if isinstance(self.colonnes, str) or any(not isinstance(c, str) for c in self.colonnes):
            raise ValueError(f"ChoixTechnique {self.id} : `colonnes` attend des noms de colonnes.")
        object.__setattr__(self, "parametres", normalises)
        object.__setattr__(self, "colonnes", tuple(self.colonnes))

    @property
    def famille(self) -> str:
        """Premier segment de l'identifiant (« reference », « regression »...)."""
        return self.id.split(".", 1)[0]

    @classmethod
    def creer(cls, id: str, colonnes: Sequence[str] = (), **parametres: Any) -> ChoixTechnique:
        """Construit un choix à partir de paramètres nommés."""
        return cls(id=id, parametres=tuple(parametres.items()), colonnes=tuple(colonnes))

    def parametres_dict(self) -> dict[str, Any]:
        """Paramètres sous forme de dictionnaire (copie)."""
        return dict(self.parametres)


# ============================================================
# SPÉCIFICATION DU PROJET
# ============================================================
# Sources uniques du socle.
# Graine par défaut : celle des bases de test du dépôt d'exercices, pour que le
# découpage proposé coïncide avec celui des challenges.
GRAINE_CONTRAT = 42
# Au delà de ce nombre de lignes, les techniques de coût LOURD (SVC, SVR, OneClassSVM,
# exclusion d'une ligne) sont coupées : leur temps de calcul croît comme le carré des lignes.
LIGNES_MAX_LENTES = 20_000

# Références naïves : la moyenne (régression) et la classe majoritaire (classification). Elles seules
# servent de base de lecture des scores (I6) ; une persistance, même de la famille « reference », ne les
# remplace jamais, car elle lit la cible passée et suppose une déclaration du projet.
REFERENCES_NAIVES: Mapping[Probleme, str] = MappingProxyType(
    {
        Probleme.REGRESSION: "reference.dummy_regression",
        Probleme.CLASSIFICATION: "reference.dummy_classification",
    }
)

# Méthodes de validation croisée (identifiants du registre) permises par structure : une
# validation qui place une même unité des deux côtés d'un pli, ou le futur dans
# l'apprentissage, rend le score optimiste.
VALIDATIONS_PAR_STRUCTURE: Mapping[Structure, frozenset[str]] = MappingProxyType(
    {
        Structure.INDEPENDANTE: frozenset(
            {"validation.kfold", "validation.kfold_stratifie", "validation.exclusion_unique"}
        ),
        Structure.GROUPEE: frozenset(
            {"validation.plis_groupes", "validation.plis_groupes_stratifies", "validation.groupes_aleatoires"}
        ),
        Structure.CHRONOLOGIQUE: frozenset({"validation.serie_temporelle"}),
        Structure.GROUPEE_CHRONOLOGIQUE: frozenset(
            {"validation.plis_groupes", "validation.plis_groupes_stratifies", "validation.groupes_aleatoires"}
        ),
    }
)
VALIDATIONS: frozenset[str] = frozenset().union(*VALIDATIONS_PAR_STRUCTURE.values())
# Validations stratifiées : la cible doit être une classe.
VALIDATIONS_STRATIFIEES: frozenset[str] = frozenset(
    {"validation.kfold_stratifie", "validation.plis_groupes_stratifies"}
)
# Métriques de classification qui dépendent de la classe dite positive (identifiant avant « : »).
METRIQUES_CLASSE_POSITIVE: frozenset[str] = frozenset(
    {"precision", "rappel", "f1", "precision_moyenne", "cout"}
)

_MOTIFS_VALIDATION = {
    Structure.INDEPENDANTE: "suppose des unités suivies ou un ordre temporel que la structure indépendante "
    "ne déclare pas ; déclarer la structure en S3",
    Structure.GROUPEE: "place des lignes d'une même unité des deux côtés d'un pli, et le score devient "
    "optimiste",
    Structure.CHRONOLOGIQUE: "met des lignes postérieures dans l'apprentissage de chaque pli, et le score "
    "devient optimiste",
    Structure.GROUPEE_CHRONOLOGIQUE: "place des lignes d'une même unité des deux côtés d'un pli, et le "
    "score devient optimiste",
}


def identifiant_metrique(texte: str) -> str:
    """Identifiant d'une métrique écrite avec ses paramètres (« cout:fp=1,fn=10 » donne « cout »)."""
    return texte.split(":", 1)[0].strip()


def validations_permises(structure: Structure, probleme: Probleme) -> frozenset[str]:
    """Méthodes de validation compatibles avec la structure (stratifiées : classification seule)."""
    permises = VALIDATIONS_PAR_STRUCTURE[structure]
    if probleme is not Probleme.CLASSIFICATION:
        permises = permises - VALIDATIONS_STRATIFIEES
    return permises if probleme.avec_cible else frozenset()


def _normaliser_classe(valeur: Any) -> Any:
    """Classe positive : texte, entier, booléen ou flottant fini ; scalaires numpy convertis."""
    if isinstance(valeur, np.generic):
        valeur = valeur.item()
    if isinstance(valeur, str) and valeur.strip():
        return valeur
    if isinstance(valeur, (bool, int)) or (isinstance(valeur, float) and math.isfinite(valeur)):
        return valeur
    raise TypeError(
        f"SpecProjet : classe positive {valeur!r} invalide ; donner une valeur de la cible "
        "(texte, entier, booléen ou nombre fini)."
    )


def _est_entier(valeur: object) -> bool:
    return isinstance(valeur, int) and not isinstance(valeur, bool)


@dataclass(frozen=True)
class SpecProjet:
    """Spécification gelée du projet : objectif, découpage, prétraitement, modèles.

    Un seul `SpecProjet` construit le pipeline et le script exporté. Les rôles sont
    triés par nom de colonne : l'ordre de saisie ne change ni l'égalité ni les clés.
    Une colonne absente de `roles` est ignorée. La métrique est fixée avec
    l'objectif, avant tout calcul, pour qu'aucun choix ne se fasse au vu des scores.
    La graine vaut par défaut `GRAINE_CONTRAT` (42).

    `validation` : méthode de validation croisée du registre (famille `validation`), ou
    None pour une méthode déduite de la structure ; une méthode incompatible avec la
    structure est refusée. `classe_positive` : classe dont on mesure la précision, le
    rappel, le F1 ou le coût ; fixée avec la métrique en S3, jamais déduite en silence
    (la session l'exige pour une classification binaire évaluée par une de ces métriques).
    """

    probleme: Probleme
    cible: str | None = None
    roles: tuple[tuple[str, Role], ...] = ()
    metrique: str | None = None
    structure: Structure = Structure.INDEPENDANTE
    part_test: float = 0.2
    n_plis: int = 5
    graine: int = GRAINE_CONTRAT
    pretraitement: tuple[ChoixTechnique, ...] = ()
    modeles: tuple[ChoixTechnique, ...] = ()
    validation: str | None = None
    classe_positive: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        self._verifier_roles()
        object.__setattr__(self, "roles", tuple(sorted(self.roles, key=lambda paire: paire[0])))
        self._verifier_objectif()
        self._verifier_decoupage()
        self._verifier_validation()
        self._verifier_classe_positive()
        for nom in ("pretraitement", "modeles"):
            valeurs = getattr(self, nom)
            if not isinstance(valeurs, tuple) or any(not isinstance(c, ChoixTechnique) for c in valeurs):
                raise TypeError(f"SpecProjet : `{nom}` doit être un tuple de `ChoixTechnique`.")

    def _verifier_roles(self) -> None:
        if not isinstance(self.roles, tuple):
            raise TypeError("SpecProjet : `roles` doit être un tuple de paires (colonne, Role).")
        for paire in self.roles:
            if (
                not isinstance(paire, tuple)
                or len(paire) != 2
                or not isinstance(paire[0], str)
                or not isinstance(paire[1], Role)
            ):
                raise TypeError(f"SpecProjet : rôle invalide {paire!r}, attendu (colonne, Role).")
        colonnes = [c for c, _ in self.roles]
        if len(set(colonnes)) != len(colonnes):
            raise ValueError("SpecProjet : une colonne reçoit plusieurs rôles.")

    def _verifier_objectif(self) -> None:
        if not isinstance(self.probleme, Probleme):
            raise TypeError("SpecProjet : `probleme` doit être un `Probleme`.")
        cibles_roles = self.colonnes_de(Role.CIBLE)
        if self.probleme.avec_cible:
            if not isinstance(self.cible, str) or not self.cible:
                raise ValueError(f"SpecProjet : un problème de {self.probleme} exige une cible.")
            if any(c != self.cible for c in cibles_roles):
                raise ValueError("SpecProjet : le rôle CIBLE est réservé à la colonne cible.")
            if self.role_de(self.cible) not in (None, Role.CIBLE):
                raise ValueError(f"SpecProjet : la cible {self.cible!r} ne peut pas avoir un autre rôle.")
            if not isinstance(self.metrique, str) or not self.metrique:
                raise ValueError("SpecProjet : choisir la métrique avec l'objectif, avant tout calcul.")
        elif self.cible is not None or cibles_roles:
            raise ValueError("SpecProjet : un problème sans cible ne désigne aucune cible.")

    def _verifier_decoupage(self) -> None:
        if not isinstance(self.structure, Structure):
            raise TypeError("SpecProjet : `structure` doit être une `Structure`.")
        if self.structure.groupee and len(self.colonnes_de(Role.GROUPE)) != 1:
            raise ValueError(
                "SpecProjet : une structure groupée exige exactement une colonne de rôle GROUPE."
            )
        if self.structure.chronologique and len(self.colonnes_de(Role.DATE)) != 1:
            raise ValueError(
                "SpecProjet : une structure chronologique exige exactement une colonne de rôle DATE."
            )
        part = self.part_test
        if isinstance(part, bool) or not isinstance(part, (int, float)) or not math.isfinite(part):
            raise ValueError("SpecProjet : `part_test` doit être un nombre.")
        if not 0.0 < float(part) < 1.0:
            raise ValueError(f"SpecProjet : `part_test` = {part} hors de l'intervalle ouvert (0, 1).")
        if not _est_entier(self.n_plis) or self.n_plis < 2:
            raise ValueError("SpecProjet : `n_plis` doit être un entier au moins égal à 2.")
        if not _est_entier(self.graine) or self.graine < 0:
            raise ValueError("SpecProjet : `graine` doit être un entier positif ou nul.")

    def _verifier_validation(self) -> None:
        v = self.validation
        if v is None:
            return
        if not isinstance(v, str) or v not in VALIDATIONS:
            raise ValueError(
                f"SpecProjet : validation {v!r} inconnue ; choisir parmi {sorted(VALIDATIONS)}, "
                "ou None pour une validation déduite de la structure."
            )
        if not self.probleme.avec_cible:
            raise ValueError(
                "SpecProjet : un problème sans cible n'a pas de validation croisée ; laisser None."
            )
        permises = validations_permises(self.structure, self.probleme)
        if v in VALIDATIONS_STRATIFIEES and self.probleme is not Probleme.CLASSIFICATION:
            raise ValueError(
                f"SpecProjet : la validation {v} stratifie des classes, la cible d'une {self.probleme} n'en "
                f"a pas ; choisir parmi {sorted(permises)}, ou None."
            )
        if v not in permises:
            raise ValueError(
                f"SpecProjet : la validation {v} {_MOTIFS_VALIDATION[self.structure]} ; avec la structure "
                f"{self.structure}, choisir parmi {sorted(permises)}, ou None pour la validation déduite "
                "de la structure."
            )

    def _verifier_classe_positive(self) -> None:
        if self.classe_positive is None:
            return
        if self.probleme is not Probleme.CLASSIFICATION:
            raise ValueError(
                "SpecProjet : la classe positive n'a de sens qu'en classification ; laisser None."
            )
        object.__setattr__(self, "classe_positive", _normaliser_classe(self.classe_positive))

    def exige_classe_positive(self) -> bool:
        """Vrai si la métrique dépend de la classe positive (exigée dès que la cible a deux classes)."""
        return (
            self.probleme is Probleme.CLASSIFICATION
            and isinstance(self.metrique, str)
            and identifiant_metrique(self.metrique) in METRIQUES_CLASSE_POSITIVE
        )

    # Lecture
    def role_de(self, colonne: str) -> Role | None:
        """Rôle explicite d'une colonne, ou None si elle n'est pas citée."""
        for nom, role in self.roles:
            if nom == colonne:
                return role
        return None

    def colonnes_de(self, role: Role) -> tuple[str, ...]:
        """Colonnes qui portent un rôle donné, par ordre alphabétique."""
        return tuple(nom for nom, r in self.roles if r is role)

    @property
    def colonne_date(self) -> str | None:
        """Colonne d'ordre temporel, si la structure est chronologique."""
        return self.colonnes_de(Role.DATE)[0] if self.structure.chronologique else None

    @property
    def colonne_groupe(self) -> str | None:
        """Colonne de groupe, si la structure est groupée."""
        return self.colonnes_de(Role.GROUPE)[0] if self.structure.groupee else None

    def variables(self) -> tuple[str, ...]:
        """Colonnes d'entrée du modèle (rôles NUMERIQUE et CATEGORIELLE), par ordre alphabétique."""
        return tuple(nom for nom, r in self.roles if r in (Role.NUMERIQUE, Role.CATEGORIELLE))

    def colonnes_citees(self) -> frozenset[str]:
        """Toutes les colonnes que la spécification nomme."""
        noms = {nom for nom, _ in self.roles}
        if self.cible is not None:
            noms.add(self.cible)
        for choix in self.pretraitement + self.modeles:
            noms.update(choix.colonnes)
        return frozenset(noms)

    # Clés de comparaison, indépendantes de l'ordre de saisie des rôles
    def objectif(self) -> tuple[Any, ...]:
        """Ce que fixe l'état OBJECTIF : problème, cible, rôles, métrique, classe positive."""
        return (self.probleme, self.cible, self.roles, self.metrique, self.classe_positive)

    def decoupage(self) -> tuple[Any, ...]:
        """Ce que fixe l'état DÉCOUPÉ : structure, part de test, plis, graine, validation."""
        return (self.structure, float(self.part_test), self.n_plis, self.graine, self.validation)

    def cle_audit(self) -> tuple[Any, ...]:
        """Ce dont dépend l'audit : l'audit est rejoué quand cette clé change."""
        return self.objectif() + self.decoupage()

    def avec(self, **changements: Any) -> SpecProjet:
        """Copie modifiée (la spécification elle-même est gelée)."""
        return replace(self, **changements)

    # Sérialisation
    def vers_dict(self) -> dict[str, Any]:
        """Représentation simple, sérialisable en JSON (les tuples y deviennent des listes)."""

        def choix(c: ChoixTechnique) -> dict[str, Any]:
            return {"id": c.id, "parametres": dict(c.parametres), "colonnes": list(c.colonnes)}

        return {
            "probleme": self.probleme.value,
            "cible": self.cible,
            "roles": [[nom, role.value] for nom, role in self.roles],
            "metrique": self.metrique,
            "structure": self.structure.value,
            "part_test": float(self.part_test),
            "n_plis": self.n_plis,
            "graine": self.graine,
            "pretraitement": [choix(c) for c in self.pretraitement],
            "modeles": [choix(c) for c in self.modeles],
            "validation": self.validation,
            "classe_positive": self.classe_positive,
        }

    @classmethod
    def depuis_dict(cls, d: Mapping[str, Any]) -> SpecProjet:
        """Reconstruit une spécification écrite par `vers_dict`."""

        def choix(e: Mapping[str, Any]) -> ChoixTechnique:
            return ChoixTechnique(
                id=e["id"],
                parametres=tuple(dict(e.get("parametres", {})).items()),
                colonnes=tuple(e.get("colonnes", ())),
            )

        return cls(
            probleme=Probleme(d["probleme"]),
            cible=d.get("cible"),
            roles=tuple((str(nom), Role(role)) for nom, role in d.get("roles", ())),
            metrique=d.get("metrique"),
            structure=Structure(d.get("structure", Structure.INDEPENDANTE.value)),
            part_test=d.get("part_test", 0.2),
            n_plis=d.get("n_plis", 5),
            graine=d.get("graine", GRAINE_CONTRAT),
            pretraitement=tuple(choix(e) for e in d.get("pretraitement", ())),
            modeles=tuple(choix(e) for e in d.get("modeles", ())),
            validation=d.get("validation"),
            classe_positive=d.get("classe_positive"),
        )


# ============================================================
# DÉCISION
# ============================================================
@dataclass(frozen=True)
class Decision:
    """Choix motivé du modèle, pris sur la validation croisée, avant tout test.

    C'est le jeton exigé pour ouvrir le coffre de test.
    """

    modele_id: str
    metrique: str
    score_validation: float
    seuil: float | None = None
    justification: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.modele_id, str) or not self.modele_id.strip():
            raise ValueError("Decision : `modele_id` est vide.")
        if not isinstance(self.metrique, str) or not self.metrique.strip():
            raise ValueError("Decision : `metrique` est vide.")
        score = self.score_validation
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Decision : `score_validation` doit être un nombre fini.")
        if self.seuil is not None:
            s = self.seuil
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0.0 <= float(s) <= 1.0:
                raise ValueError("Decision : `seuil` doit être compris entre 0 et 1.")


# ============================================================
# COFFRE DE TEST
# ============================================================
class ErreurScelle(PermissionError):
    """Tentative d'accès au jeu de test sans décision."""


class CoffreTest:
    """Jeu de test scellé au découpage, ouvert une fois après la décision.

    Aucune méthode ne rend les lignes de test avant `ouvrir(decision)`. Chaque
    ouverture est comptée ; au delà de la première, le score de test est dit
    optimiste, car le test a déjà influencé les choix. Le coffre garde le découpage
    (`SpecProjet.decoupage()`) pour lequel il a été construit.
    """

    __slots__ = ("_decoupage", "_ouvertures", "_positions_app", "_positions_test", "_table", "_table_test")

    def __init__(self) -> None:
        raise TypeError("CoffreTest : utiliser CoffreTest.par_positions ou CoffreTest.par_fichier.")

    @classmethod
    def _nouveau(cls, table: Table, decoupage: tuple[Any, ...]) -> CoffreTest:
        if not isinstance(decoupage, tuple) or not decoupage:
            raise TypeError("CoffreTest : `decoupage` attend le tuple `SpecProjet.decoupage()`.")
        coffre = object.__new__(cls)
        coffre._table = table
        coffre._decoupage = decoupage
        coffre._positions_test = None
        coffre._positions_app = None
        coffre._table_test = None
        coffre._ouvertures = 0
        return coffre

    @classmethod
    def par_positions(
        cls, table: Table, positions_test: Sequence[int] | np.ndarray, *, decoupage: tuple[Any, ...]
    ) -> CoffreTest:
        """Coffre dont le test est un sous-ensemble de lignes de `table` (positions 0..n-1)."""
        if not isinstance(table, Table):
            raise TypeError("CoffreTest : `table` doit être une `Table`.")
        pos = np.asarray(positions_test)
        if pos.ndim != 1 or pos.size == 0 or not np.issubdtype(pos.dtype, np.integer):
            raise ValueError("CoffreTest : positions de test vides ou non entières.")
        if pos.min() < 0 or pos.max() >= table.n_lignes:
            raise ValueError("CoffreTest : position de test hors de la table.")
        pos = np.sort(pos.astype(np.int64))
        if np.any(np.diff(pos) == 0):
            raise ValueError("CoffreTest : position de test en double.")
        if pos.size >= table.n_lignes:
            raise ValueError("CoffreTest : le test ne peut pas contenir toutes les lignes.")
        masque = np.ones(table.n_lignes, dtype=bool)
        masque[pos] = False
        coffre = cls._nouveau(table, decoupage)
        coffre._positions_test = pos
        coffre._positions_app = np.flatnonzero(masque)
        pos.setflags(write=False)
        coffre._positions_app.setflags(write=False)
        return coffre

    @classmethod
    def par_fichier(
        cls, table_apprentissage: Table, table_test: Table, *, decoupage: tuple[Any, ...]
    ) -> CoffreTest:
        """Coffre dont le test est fourni à part (mode challenge)."""
        if not isinstance(table_apprentissage, Table) or not isinstance(table_test, Table):
            raise TypeError("CoffreTest : les deux jeux doivent être des `Table`.")
        if table_test.n_lignes == 0:
            raise ValueError("CoffreTest : le jeu de test fourni est vide.")
        manquantes = sorted(set(table_apprentissage.colonnes) - set(table_test.colonnes))
        en_trop = sorted(set(table_test.colonnes) - set(table_apprentissage.colonnes))
        if manquantes or en_trop:
            raise ValueError(
                "CoffreTest : colonnes différentes entre apprentissage et test "
                f"(absentes du test : {manquantes[:5]}, absentes de l'apprentissage : {en_trop[:5]})."
            )
        coffre = cls._nouveau(table_apprentissage, decoupage)
        positions = np.arange(table_apprentissage.n_lignes, dtype=np.int64)
        positions.setflags(write=False)
        coffre._positions_app = positions
        coffre._table_test = table_test
        return coffre

    def __repr__(self) -> str:
        return (
            f"CoffreTest(apprentissage={self.n_apprentissage}, test={self.n_test}, "
            f"ouvertures={self._ouvertures})"
        )

    def __reduce__(self) -> Any:
        raise TypeError("CoffreTest : un coffre scellé ne se sérialise pas.")

    # Lecture autorisée avant l'ouverture
    @property
    def n_test(self) -> int:
        """Nombre de lignes de test (le compte n'en révèle aucune)."""
        if self._table_test is not None:
            return self._table_test.n_lignes
        return int(self._positions_test.size)

    @property
    def n_apprentissage(self) -> int:
        """Nombre de lignes d'apprentissage."""
        return int(self._positions_app.size)

    @property
    def mode_challenge(self) -> bool:
        """Vrai si le test vient d'un fichier ou d'une feuille à part."""
        return self._table_test is not None

    @property
    def decoupage(self) -> tuple[Any, ...]:
        """Découpage (`SpecProjet.decoupage()`) pour lequel le coffre a été construit."""
        return self._decoupage

    @property
    def empreinte_test(self) -> str:
        """Empreinte des données d'où vient le test : la table, ou le fichier de test à part."""
        source = self._table if self._table_test is None else self._table_test
        return source.empreinte

    @property
    def ouvertures(self) -> int:
        """Nombre d'ouvertures du coffre."""
        return self._ouvertures

    @property
    def score_optimiste(self) -> bool:
        """Vrai dès la seconde ouverture : le test a déjà guidé des choix."""
        return self._ouvertures > 1

    def repose_sur(self, table: Table) -> bool:
        """Vrai si le coffre a été construit sur cette table (identité)."""
        return self._table is table

    def positions_apprentissage(self) -> np.ndarray:
        """Positions des lignes d'apprentissage dans la table (lecture seule)."""
        return self._positions_app

    def apprentissage(self) -> pd.DataFrame:
        """Lignes d'apprentissage : seules lignes visibles avant la décision (copie)."""
        return self._table.donnees.iloc[self._positions_app]

    # Ouverture
    def ouvrir(self, decision: Decision) -> pd.DataFrame:
        """Rend une copie des lignes de test ; exige une `Decision` ; compte l'ouverture."""
        if not isinstance(decision, Decision):
            raise ErreurScelle(
                "Le jeu de test reste scellé : choisir d'abord le modèle sur la validation "
                "croisée, puis ouvrir le test une seule fois."
            )
        self._ouvertures += 1
        if self._table_test is not None:
            return self._table_test.donnees.copy()
        return self._table.donnees.iloc[self._positions_test]
