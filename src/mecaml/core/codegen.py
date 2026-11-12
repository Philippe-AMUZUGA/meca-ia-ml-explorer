"""
MécaML : script scikit-learn équivalent au projet, engendré depuis la spécification qui construit le pipeline.

Source unique (décision 11) : le script ne réécrit aucune étape à la main. Les pipelines sont ceux de
`train.construire_pipeline`, lus et écrits en code Python par un sérialiseur d'estimateurs ; les fonctions du
paquet dont le résultat doit être identique (empreintes, variables temporelles, métriques) sont reprises
telles quelles dans l'annexe du script, extraites du code source par analyse syntaxique. Le script est
autonome : il n'importe pas `mecaml`, seulement numpy, pandas et scikit-learn.

Il vérifie, à chaque lancement, trois empreintes (données lues, positions de test, plis de validation) :
si une bibliothèque ou un fichier a changé, il s'arrête en le disant plutôt que de rendre des scores
différents en silence. Il refait le découpage, valide chaque modèle sur les mêmes plis, puis évalue le
modèle retenu sur le test.

Ceinture à l'export : un projet dont un choix lit la cible passée sans le déclarer, ou sur une cible qui
décompte, n'est pas exporté (`cible_passee.constats`). Quand le projet déclare la cible du pas précédent
connue (`cible_connue=True`), l'en-tête du script l'écrit comme une hypothèse du projet.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import math
import platform
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mecaml import __version__
from mecaml.core import cible_passee, evaluate, features_ts, libelles, metrics, preprocess, split, train
from mecaml.core import types as types_du_coeur
from mecaml.core.catalog import OPTIONNELLES, ecretage_tukey, persistance
from mecaml.core.guards import chemin_dans_dossier, decimale
from mecaml.core.types import CoffreTest, Decision, Probleme, SpecProjet, Structure, Table

if TYPE_CHECKING:
    from mecaml.core.io import Chargement

__all__ = [
    "NOM_DU_SCRIPT",
    "ErreurExport",
    "Extrait",
    "Lecture",
    "ecrire_script",
    "empreinte_plis",
    "environnement",
    "extraire_du_paquet",
    "generer_script",
    "lecture_du_fichier",
    "pipelines_du_projet",
    "rendre",
    "retrouver_decoupage",
    "verifier_ceinture",
]

NOM_DU_SCRIPT = "script_projet.py"
LARGEUR = 100  # largeur visée du code engendré
RETRAIT = 4
OPT = (
    "-" * 2
)  # préfixe des options du script, écrit ainsi pour que ce fichier reste sans double trait d'union


class ErreurExport(ValueError):
    """Export impossible ou refusé ; le message dit quoi corriger."""


# ============================================================
# ENVIRONNEMENT ET EMPREINTES
# ============================================================
def environnement() -> dict[str, str]:
    """Versions qui comptent pour reproduire les scores : Python, numpy, pandas, scipy, scikit-learn."""
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def empreinte_plis(plis: Sequence[tuple[Any, Any]]) -> str:
    """SHA-256 des plis de validation : positions d'apprentissage puis de validation, pli après pli."""
    h = hashlib.sha256()
    for app, val in plis:
        for positions in (app, val):
            h.update(np.asarray(positions, dtype="<i8").tobytes())
            h.update(b"|")
        h.update(b"#")
    return h.hexdigest()


# ============================================================
# CODE DU PAQUET REPRIS DANS LE SCRIPT (EXTRACTION PAR ANALYSE SYNTAXIQUE)
# ============================================================
@dataclass(frozen=True)
class _SourceDuModule:
    """Ce que l'on sait du code source d'un module : définitions de premier niveau et imports."""

    nom: str
    lignes: tuple[str, ...]
    definitions: Mapping[str, ast.stmt]  # nom défini au premier niveau vers son noeud
    imports: Mapping[str, tuple[str, bool]]  # nom lié vers (instruction d'import, vient de mecaml)


def _instruction_d_import(noeud: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    suite = f"{alias.name} as {alias.asname}" if alias.asname else alias.name
    if isinstance(noeud, ast.Import):
        return f"import {suite}"
    return f"from {'.' * noeud.level}{noeud.module or ''} import {suite}"


def _noms_definis(noeud: ast.stmt) -> list[str]:
    if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [noeud.name]
    if isinstance(noeud, ast.Assign):
        cibles = noeud.targets
    elif isinstance(noeud, ast.AnnAssign):
        cibles = [noeud.target]
    else:
        cibles = []
    return [c.id for c in cibles if isinstance(c, ast.Name)]


def _vient_du_paquet(noeud: ast.Import | ast.ImportFrom) -> bool:
    if isinstance(noeud, ast.ImportFrom):
        return noeud.level > 0 or (noeud.module or "").split(".")[0] == "mecaml"
    return any(a.name.split(".")[0] == "mecaml" for a in noeud.names)


@functools.cache
def _analyser_module(module: ModuleType) -> _SourceDuModule:
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as erreur:
        raise ErreurExport(
            f"Le code source du module {module.__name__} est introuvable : le script exporté en reprend des "
            "fonctions. Installer l'application avec ses fichiers .py (dans un exécutable, collecter aussi "
            "les sources du paquet mecaml)."
        ) from erreur
    definitions: dict[str, ast.stmt] = {}
    imports: dict[str, tuple[str, bool]] = {}
    for noeud in ast.parse(source).body:
        if isinstance(noeud, (ast.Import, ast.ImportFrom)):
            if isinstance(noeud, ast.ImportFrom) and noeud.module == "__future__":
                continue
            for alias in noeud.names:
                lie = alias.asname or alias.name.split(".")[0]
                imports[lie] = (_instruction_d_import(noeud, alias), _vient_du_paquet(noeud))
        else:
            for nom in _noms_definis(noeud):
                definitions[nom] = noeud
    return _SourceDuModule(module.__name__, tuple(source.splitlines()), definitions, imports)


class _ChercheurDeNoms(ast.NodeVisitor):
    """Noms lus dans un noeud, hors annotations (jamais évaluées : `from __future__ import annotations`)."""

    def __init__(self) -> None:
        self.noms: set[str] = set()

    def visit_Name(self, noeud: ast.Name) -> None:
        if isinstance(noeud.ctx, ast.Load):
            self.noms.add(noeud.id)

    def visit_arg(self, noeud: ast.arg) -> None:
        return None  # annotation d'un paramètre

    def visit_AnnAssign(self, noeud: ast.AnnAssign) -> None:
        if noeud.value is not None:
            self.visit(noeud.value)

    def _fonction(self, noeud: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for d in noeud.decorator_list:
            self.visit(d)
        for defaut in (*noeud.args.defaults, *(d for d in noeud.args.kw_defaults if d is not None)):
            self.visit(defaut)
        for instruction in noeud.body:
            self.visit(instruction)

    def visit_FunctionDef(self, noeud: ast.FunctionDef) -> None:
        self._fonction(noeud)

    def visit_AsyncFunctionDef(self, noeud: ast.AsyncFunctionDef) -> None:
        self._fonction(noeud)


def _noms_lus(noeud: ast.AST) -> set[str]:
    chercheur = _ChercheurDeNoms()
    chercheur.visit(noeud)
    return chercheur.noms


@dataclass(frozen=True)
class Extrait:
    """Code repris d'un module du paquet : instructions d'import et définitions, dans l'ordre du module."""

    imports: tuple[str, ...]
    code: str


def _debut(noeud: ast.stmt) -> int:
    decorateurs = getattr(noeud, "decorator_list", [])
    return min([noeud.lineno, *(d.lineno for d in decorateurs)])


def _fermeture(source: _SourceDuModule, racines: Sequence[str]) -> list[str]:
    """Noms de premier niveau à reprendre : les racines et tout ce qu'elles lisent, dans l'ordre du module."""
    absentes = [r for r in racines if r not in source.definitions]
    if absentes:
        raise ErreurExport(f"Module {source.nom} : définition(s) à reprendre introuvable(s) : {absentes}.")
    retenus: set[str] = set()
    a_voir = list(racines)
    while a_voir:
        nom = a_voir.pop()
        if nom in retenus:
            continue
        retenus.add(nom)
        lus = _noms_lus(source.definitions[nom])
        a_voir.extend(n for n in lus if n in source.definitions and n not in retenus)
    ordre = {nom: k for k, nom in enumerate(source.definitions)}
    return sorted(retenus, key=ordre.__getitem__)


def extraire_du_paquet(
    module: ModuleType, racines: Sequence[str], renommer: Mapping[str, str] | None = None
) -> Extrait:
    """Définitions `racines` d'un module et tout ce dont elles dépendent, prêtes à coller dans un script.

    Refus (`ErreurExport`) si l'une dépend d'un nom importé de `mecaml` : elle ne serait pas autonome.
    `renommer` change des noms de premier niveau (deux modules peuvent définir le même nom privé).
    """
    source = _analyser_module(module)
    imports: set[str] = set()
    blocs: list[str] = []
    for nom in _fermeture(source, racines):
        noeud = source.definitions[nom]
        for lu in _noms_lus(noeud):
            if lu in source.imports:
                instruction, interne = source.imports[lu]
                if interne:
                    raise ErreurExport(
                        f"{source.nom}.{nom} dépend de {lu}, importé du paquet : non autonome pour le script."
                    )
                imports.add(instruction)
        blocs.append("\n".join(source.lignes[_debut(noeud) - 1 : noeud.end_lineno]))
    code = "\n\n\n".join(blocs)
    for ancien, nouveau in (renommer or {}).items():
        code = re.sub(rf"\b{re.escape(ancien)}\b", nouveau, code)
    return Extrait(tuple(sorted(imports)), code)


@dataclass(frozen=True)
class _Reprise:
    """Une définition du paquet que le script reprend : où elle vit, ce qu'elle emporte, sous quel nom."""

    module: ModuleType
    racines: tuple[str, ...]
    renommer: Mapping[str, str] = field(default_factory=dict)
    affectation: str = ""  # instruction qui lie le nom public d'une classe créée au premier usage


REPRISES: Mapping[str, _Reprise] = {
    "empreinte_dataframe": _Reprise(types_du_coeur, ("empreinte_dataframe",)),
    "empreinte_positions": _Reprise(split, ("empreinte_positions",)),
    "empreinte_plis": _Reprise(sys.modules[__name__], ("empreinte_plis",)),
    "empreinte_predictions": _Reprise(evaluate, ("empreinte_predictions",)),
    "predire_au_seuil": _Reprise(evaluate, ("predire_au_seuil",)),
    "normaliser_manquants": _Reprise(preprocess, ("normaliser_manquants",)),
    "contexte_de": _Reprise(
        preprocess, ("_contexte_de_la_branche",), {"_contexte_de_la_branche": "contexte_de"}
    ),
    "VariablesTemporelles": _Reprise(
        features_ts, ("_classe_transformeur",), affectation="VariablesTemporelles = _classe_transformeur()"
    ),
    "EcretageTukey": _Reprise(
        ecretage_tukey,
        ("_classe_transformeur",),
        {"_classe_transformeur": "_classe_ecretage_tukey"},
        "EcretageTukey = _classe_ecretage_tukey()",
    ),
    "Persistance": _Reprise(persistance, ("_classe_modele",), affectation="Persistance = _classe_modele()"),
}


def annexe_des(noms: Iterable[str]) -> Extrait:
    """Code du paquet à coller dans l'annexe d'un script pour les noms demandés (ceux de `REPRISES`)."""
    imports: set[str] = set()
    blocs: list[str] = []
    for nom in REPRISES:  # l'ordre du tableau est celui de l'annexe
        if nom not in set(noms):
            continue
        reprise = REPRISES[nom]
        extrait = extraire_du_paquet(reprise.module, reprise.racines, reprise.renommer)
        imports.update(extrait.imports)
        blocs.append(extrait.code + (f"\n\n\n{reprise.affectation}" if reprise.affectation else ""))
    return Extrait(tuple(sorted(imports)), "\n\n\n".join(blocs))


# ============================================================
# ÉCRITURE DE CODE PYTHON : IMPORTS, VALEURS, ESTIMATEURS
# ============================================================
class _Imports:
    """Imports d'un script : fusionnés, triés (bibliothèque standard, puis le reste)."""

    def __init__(self) -> None:
        self.modules: set[str] = set()  # « import x » ou « import x as y »
        self.noms: dict[str, set[str]] = {}  # « from m import a, b »

    def instruction(self, texte: str) -> None:
        """Ajoute une instruction d'import complète."""
        arbre = ast.parse(texte).body[0]
        if isinstance(arbre, ast.Import):
            self.modules.add(texte)
        elif isinstance(arbre, ast.ImportFrom):
            self.noms.setdefault(arbre.module or "", set()).update(
                a.name + (f" as {a.asname}" if a.asname else "") for a in arbre.names
            )

    def depuis(self, module: str, nom: str) -> None:
        """Ajoute `from module import nom`."""
        self.noms.setdefault(module, set()).add(nom)

    def lignes(self) -> list[str]:
        """Instructions triées, un groupe pour la bibliothèque standard, un pour le reste."""
        groupes: list[list[str]] = [[], []]
        for texte in sorted(self.modules, key=str.lower):
            racine = texte.split()[1].split(".")[0]
            groupes[0 if racine in sys.stdlib_module_names else 1].append(texte)
        depuis: list[list[str]] = [[], []]
        for module in sorted(self.noms, key=str.lower):
            texte = f"from {module} import {', '.join(sorted(self.noms[module], key=str.lower))}"
            depuis[0 if module.split(".")[0] in sys.stdlib_module_names else 1].append(texte)
        sortie: list[str] = []
        for g, d in zip(groupes, depuis, strict=True):
            if g or d:
                sortie += [*g, *d, ""]
        return sortie[:-1]


@dataclass
class _Appel:
    nom: str
    args: list[tuple[str | None, Any]] = field(default_factory=list)


@dataclass
class _Suite:
    ouvrant: str
    fermant: str
    elements: list[Any] = field(default_factory=list)


@dataclass
class _Paire:
    cle: str
    valeur: Any


def _a_plat(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, _Paire):
        return f"{doc.cle}: {_a_plat(doc.valeur)}"
    if isinstance(doc, _Appel):
        dedans = ", ".join((f"{cle}=" if cle else "") + _a_plat(v) for cle, v in doc.args)
        return f"{doc.nom}({dedans})"
    dedans = ", ".join(_a_plat(e) for e in doc.elements)
    if doc.ouvrant == "(" and len(doc.elements) == 1:
        dedans += ","
    return f"{doc.ouvrant}{dedans}{doc.fermant}"


def rendre(doc: Any, indent: int = 0, prefixe: int = 0, largeur: int = LARGEUR) -> str:
    """Texte du document : sur une ligne s'il tient dans la largeur, sinon un élément par ligne."""
    plat = _a_plat(doc)
    if isinstance(doc, str) or indent + prefixe + len(plat) + 1 <= largeur:
        return plat
    marge, pas = " " * indent, " " * (indent + RETRAIT)
    if isinstance(doc, _Paire):
        return f"{doc.cle}: " + rendre(doc.valeur, indent, prefixe + len(doc.cle) + 2, largeur)
    if isinstance(doc, _Appel):
        lignes = [f"{doc.nom}("]
        for cle, valeur in doc.args:
            etiquette = f"{cle}=" if cle else ""
            lignes.append(pas + etiquette + rendre(valeur, indent + RETRAIT, len(etiquette), largeur) + ",")
        return "\n".join([*lignes, f"{marge})"])
    if not doc.elements:
        return doc.ouvrant + doc.fermant
    lignes = [doc.ouvrant]
    lignes += [pas + rendre(e, indent + RETRAIT, 0, largeur) + "," for e in doc.elements]
    return "\n".join([*lignes, f"{marge}{doc.fermant}"])


def _chaine(texte: str) -> str:
    return json.dumps(texte, ensure_ascii=False)


def _nombre(valeur: float) -> str:
    if math.isnan(valeur):
        return "np.nan"
    if math.isinf(valeur):
        return "np.inf" if valeur > 0 else "-np.inf"
    return repr(valeur)


# Classes dont le premier paramètre s'écrit sans son nom.
_PREMIER_ARGUMENT = {
    "Pipeline": "steps",
    "ColumnTransformer": "transformers",
    "FeatureUnion": "transformer_list",
}


def _chemin_public(objet: Any) -> str | None:
    """Module public (sans segment privé) qui expose la classe ou la fonction, s'il existe."""
    module = getattr(objet, "__module__", "") or ""
    nom = getattr(objet, "__name__", "")
    segments = module.split(".")
    for k in range(len(segments), 0, -1):
        if any(s.startswith("_") for s in segments[:k]):
            continue
        candidat = ".".join(segments[:k])
        if getattr(sys.modules.get(candidat), nom, None) is objet:
            return candidat
    return None


def _egaux(a: Any, b: Any) -> bool:
    """Deux valeurs de paramètre identiques (même type et même écriture, nan égal à nan)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    try:
        return repr(a) == repr(b)
    except Exception:  # un objet dont la représentation échoue est dit différent
        return False


def _parametres_modifies(est: Any) -> dict[str, Any]:
    """Paramètres de construction qui diffèrent des défauts de la classe (comme `repr` de scikit-learn)."""
    parametres = est.get_params(deep=False)
    init = getattr(est.__init__, "deprecated_original", est.__init__)
    signature = inspect.signature(init).parameters
    modifies = {}
    for nom, v in parametres.items():
        defaut = signature[nom].default if nom in signature else inspect.Parameter.empty
        if defaut is inspect.Parameter.empty or not _egaux(v, defaut):
            modifies[nom] = v
    return modifies


@dataclass
class _Ecrivain:
    """Écrit en Python les valeurs de paramètres et les estimateurs d'un pipeline.

    `graine` : la valeur de `random_state` égale à la graine du projet s'écrit `GRAINE`.
    `contexte` : nom de la variable qui porte les lignes d'apprentissage sans cible ; `contexte_reel` ces
    lignes, pour vérifier que chaque table de contexte d'un estimateur se reconstruit à l'identique.
    """

    graine: int
    imports: _Imports
    contexte_reel: pd.DataFrame | None = None
    contexte: str = "contexte"
    embarques: set[str] = field(default_factory=set)

    def valeur(self, v: Any, nom: str = "") -> Any:
        """Document d'une valeur de paramètre (`nom` : celui du paramètre, pour la graine)."""
        if isinstance(v, np.generic):
            v = v.item()
        if v is None or isinstance(v, (bool, str, int)):
            if nom == "random_state" and isinstance(v, int) and not isinstance(v, bool) and v == self.graine:
                return "GRAINE"
            return _chaine(v) if isinstance(v, str) else repr(v)
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                self.imports.instruction("import numpy as np")
            return _nombre(v)
        if isinstance(v, (list, tuple)):
            ouvrant, fermant = ("[", "]") if isinstance(v, list) else ("(", ")")
            return _Suite(ouvrant, fermant, [self.valeur(e) for e in v])
        if isinstance(v, dict):
            return _Suite("{", "}", [_Paire(_chaine(str(k)), self.valeur(e)) for k, e in v.items()])
        if isinstance(v, type) and issubclass(v, np.generic):
            self.imports.instruction("import numpy as np")
            return f"np.{v.__name__}"
        if isinstance(v, pd.DataFrame):
            return self._contexte(v)
        if isinstance(v, functools.partial):
            args = [(None, self.valeur(v.func))] + [(None, self.valeur(a)) for a in v.args]
            args += [(cle, self.valeur(x, cle)) for cle, x in (v.keywords or {}).items()]
            self.imports.depuis("functools", "partial")
            return _Appel("partial", args)
        if hasattr(v, "get_params"):
            return self.estimateur(v)
        if callable(v):
            return self._fonction(v)
        raise ErreurExport(
            f"Paramètre de type {type(v).__name__} non reproductible dans le script ({v!r:.60}) : "
            "choisir une technique à paramètres simples (nombres, textes, listes)."
        )

    def _fonction(self, f: Callable[..., Any]) -> str:
        module = getattr(f, "__module__", "") or ""
        nom = getattr(f, "__name__", "")
        if module.startswith("mecaml"):
            if nom != "normaliser_manquants":
                raise ErreurExport(f"Fonction {module}.{nom} non reproductible dans le script.")
            self.embarques.add(nom)
            return nom
        public = _chemin_public(f)
        if public is None:
            raise ErreurExport(f"Fonction {module}.{nom} non importable par un chemin public.")
        self.imports.depuis(public, nom)
        return nom

    def _contexte(self, df: pd.DataFrame) -> str:
        """Lignes de contexte d'une branche chronologique : colonnes de la variable, index remis à zéro."""
        colonnes = [str(c) for c in df.columns]
        reel = self.contexte_reel
        attendu = None if reel is None or len(reel) == 0 else reel[colonnes].reset_index(drop=True)
        if attendu is None or not df.equals(attendu):
            raise ErreurExport(
                "Le contexte des séries (lignes d'apprentissage sans cible) ne se reconstruit pas à "
                "l'identique dans le script : l'export est refusé."
            )
        self.embarques.add("contexte_de")
        return f"contexte_de({self.contexte}, {_a_plat(self.valeur(colonnes))})"

    def estimateur(self, est: Any) -> Any:
        """Document `Classe(paramètres modifiés)` d'un estimateur scikit-learn ou du paquet."""
        classe = type(est)
        nom = classe.__name__
        if classe.__module__.startswith("mecaml"):
            self.embarques.add(nom)
        else:
            public = _chemin_public(classe)
            if public is None:
                raise ErreurExport(f"Classe {classe.__module__}.{nom} non importable par un chemin public.")
            self.imports.depuis(public, nom)
        premier = _PREMIER_ARGUMENT.get(nom)
        args = [
            (None if cle == premier else cle, self.valeur(v, cle))
            for cle, v in _parametres_modifies(est).items()
        ]
        args.sort(key=lambda a: a[0] is not None)
        return _Appel(nom, args)


# ============================================================
# CE QU'IL FAUT SAVOIR DU PROJET POUR LE REFAIRE
# ============================================================
@dataclass(frozen=True)
class Lecture:
    """Comment relire le fichier de données pour retrouver la table du projet."""

    fichier: str
    feuille: str | None
    csv: bool
    encodage: str | None = None
    separateur: str | None = None
    decimale: str | None = None
    colonnes_dates: tuple[str, ...] = ()


def _options_csv(lecture: Lecture) -> list[tuple[str, str, str | None]]:
    """(option de `read_csv`, constante du script, valeur) ; une valeur inconnue n'est pas écrite."""
    return [
        ("sep", "SEPARATEUR", lecture.separateur),
        ("decimal", "DECIMALE", lecture.decimale),
        ("encoding", "ENCODAGE", lecture.encodage),
    ]


def lecture_du_fichier(table: Table, chargement: Chargement | None = None) -> Lecture:
    """Lecture du fichier d'origine de la table ; un `.csv` demande le `Chargement` pour son séparateur."""
    csv = table.nom.lower().endswith(".csv")
    dates = tuple(c for c in table.donnees.columns if pd.api.types.is_datetime64_any_dtype(table.donnees[c]))
    if csv and chargement is not None:
        return Lecture(
            table.nom, None, True, chargement.encodage, chargement.separateur, chargement.decimal, dates
        )
    return Lecture(table.nom, table.feuille, csv, None, None, None, dates)


def verifier_ceinture(spec: SpecProjet, table: Table, coffre: CoffreTest | None) -> None:
    """Refuse l'export d'un projet dont la cible passée n'est pas déclarée, ou dont la cible décompte.

    Seules les lignes d'apprentissage se lisent. `Exporter` n'examine pas son contenu : c'est ici que se tient
    la garde, dans `generer_script` comme dans le rapport. Un réseau Keras n'a pas de script équivalent (le
    script ne dépend que de scikit-learn) : l'export le refuse.
    """
    sans_script = {e.id for e in OPTIONNELLES}
    reseaux = [libelles.technique(c.id) for c in spec.modeles if c.id in sans_script]
    if reseaux:
        raise ErreurExport(
            f"Export impossible : {', '.join(reseaux)} n'a pas de script Python équivalent. "
            "Retirer ce modèle de la comparaison (étape 6) pour exporter."
        )
    if not spec.probleme.avec_cible:
        return
    if coffre is None or not coffre.repose_sur(table):
        raise ErreurExport("Export impossible sans le jeu de test mis sous scellé de cette table (S4).")
    avec, _ = split.separer_sans_cible(coffre.apprentissage(), spec.cible)
    refus = cible_passee.constats(spec, avec)
    if refus:
        raise ErreurExport("Export refusé : " + " ".join(f"{c.message} {c.action}" for c in refus))


def retrouver_decoupage(table: Table, spec: SpecProjet, coffre: CoffreTest) -> split.Decoupage:
    """Découpage déclaré du coffre : la méthode de `split` dont les positions de test sont les siennes."""
    total = coffre.n_test + coffre.n_apprentissage
    if coffre.mode_challenge:
        colonne = spec.colonne_groupe or spec.colonne_date
        return split.Decoupage(
            split.Methode.FICHIER,
            coffre.n_test / total,
            spec.graine,
            colonne,
            coffre.n_test,
            coffre.empreinte_test,
        )
    test = np.setdiff1d(np.arange(table.n_lignes), coffre.positions_apprentissage())
    empreinte = split.empreinte_positions(test)
    deduite = split.methode_deduite(spec)
    candidats = [deduite, *sorted(split.methodes_permises(spec) - {deduite}, key=str)]
    for methode in candidats:
        colonne = _colonne_du_decoupage(spec, methode)
        try:
            positions = split.positions_test(
                table.donnees, methode, spec.part_test, spec.graine, colonne or spec.cible, strate=spec.cible
            )
        except split.ErreurDecoupage:
            continue
        if positions.size == test.size and split.empreinte_positions(positions) == empreinte:
            return split.Decoupage(methode, spec.part_test, spec.graine, colonne, positions.size, empreinte)
    raise ErreurExport(
        "Le jeu de test du coffre ne correspond à aucune méthode de découpage de l'application : le refaire "
        "avec `split.sceller` (S4), ou transmettre le découpage déclaré à l'export."
    )


def _colonne_du_decoupage(spec: SpecProjet, methode: split.Methode) -> str | None:
    if methode in (split.Methode.UNITES, split.Methode.UNITES_STRATIFIEES):
        return spec.colonne_groupe
    if methode is split.Methode.CHRONOLOGIQUE:
        return spec.colonne_date
    return None


def contexte_d_apprentissage(
    spec: SpecProjet, table: Table, coffre: CoffreTest | None
) -> pd.DataFrame | None:
    """Lignes d'apprentissage sans cible (colonnes d'entrée) qui complètent les séries ; None si aucune."""
    if not spec.probleme.avec_cible or coffre is None:
        return None
    return evaluate.apprentissage_du_projet(spec, table, coffre).contexte


def pipelines_du_projet(spec: SpecProjet, table: Table, coffre: CoffreTest | None) -> dict[str, Any]:
    """Pipeline non ajusté de chaque modèle comparé, construit comme le construit `train.entrainer`."""
    contexte = contexte_d_apprentissage(spec, table, coffre)
    identifiants = [c.id for c in spec.modeles]
    doubles = sorted({i for i in identifiants if identifiants.count(i) > 1})
    if doubles:
        raise ErreurExport(
            f"Deux choix de la même technique ({doubles}) : le script les distingue par leur identifiant."
        )
    return {c.id: train.construire_pipeline(spec, c, None, contexte) for c in spec.modeles}


# ============================================================
# GABARITS DU SCRIPT
# ============================================================
ENTETE = Template('''\
"""Script équivalent du projet « $titre », engendré par MécaML Explorer $version.

Il refait, avec pandas et scikit-learn seuls (l'application n'est pas nécessaire) : la lecture du fichier de
données, le découpage apprentissage et test, le prétraitement et les modèles tels que l'application les a
construits, la validation croisée sur l'apprentissage, puis l'évaluation du modèle retenu sur le test.

$resume

Reproductibilité : la graine est unique (GRAINE). Le script vérifie trois empreintes (données lues,
positions de test, plis de validation) et s'arrête si l'une diffère ; l'option ${o}ignorer-empreintes le
laisse continuer.
Environnement de l'export : $environnement.
$hypothese
Le jeu de test a servi une fois à juger le modèle retenu : relancer ce script ne le rend pas neuf, et toute
décision prise en le regardant rend le score de test optimiste.

Utilisation : python $nom_script [${o}donnees FICHIER] [${o}test FICHIER] [${o}sortie DOSSIER]
[${o}ignorer-empreintes]
Dépendances : numpy, pandas, scikit-learn, et openpyxl pour lire un classeur .xlsx.
"""
''')

LECTURE = Template('''\
def lire_donnees(chemin, feuille=FEUILLE):
    """Lit le fichier de données comme l'application : $explication."""
    chemin = Path(chemin)
    if not chemin.is_file():
        raise SystemExit(
            f"Fichier de données introuvable : {chemin}. Le placer à côté du script, "
            "ou donner son chemin avec ${o}donnees."
        )
$lecture
    for colonne in COLONNES_DATES:  # texte ISO lu comme des dates, comme dans l'application
        if not pd.api.types.is_datetime64_any_dtype(donnees[colonne]):
            donnees[colonne] = pd.to_datetime(donnees[colonne], format="ISO8601", errors="coerce")
    return donnees


def verifier_donnees(donnees, ignorer):
    """Arrête le script si le fichier lu n'est pas celui du projet (empreinte différente)."""
    if not ignorer and empreinte_dataframe(donnees) != EMPREINTE_DONNEES:
        raise SystemExit(
            "Les données lues ne sont pas celles du projet (empreinte différente) : fichier modifié, "
            "autre feuille ou autre version de pandas. L'option ${o}ignorer-empreintes continue malgré "
            "tout, mais les scores peuvent changer."
        )
''')

SEPARER = Template('''\
def separer(donnees, options):
    """Lignes d'apprentissage et lignes de test (le découpage est celui de l'application)."""
    test = positions_de_test(donnees)
    if not options.ignorer_empreintes and empreinte_positions(test) != EMPREINTE_TEST:
        raise SystemExit(
            "Le découpage refait ne donne pas le jeu de test du projet (empreinte différente) : "
            "version de scikit-learn différente ou données modifiées."
        )
    apprentissage = np.setdiff1d(np.arange(len(donnees)), test)
    return donnees.iloc[apprentissage], donnees.iloc[test]
''')

SEPARER_FICHIER = Template('''\
def separer(donnees, options):
    """Lignes d'apprentissage (tout le fichier) et lignes de test (fournies à part, comme dans le projet)."""
    if not options.test:
        raise SystemExit("Ce projet a un jeu de test fourni à part : donner son chemin avec ${o}test.")
    lignes_test = lire_donnees(options.test, FEUILLE_TEST)
    if not options.ignorer_empreintes and empreinte_dataframe(lignes_test) != EMPREINTE_TEST:
        raise SystemExit("Le jeu de test lu n'est pas celui du projet (empreinte différente).")
    return donnees, lignes_test
''')

PREPARATION = Template('''\
def preparer_apprentissage(apprentissage):
    """Lignes avec cible (variables et cible) ; les lignes sans cible ne servent qu'au contexte des séries."""
    presentes = apprentissage[CIBLE].notna().to_numpy()
    avec_cible, sans_cible = apprentissage[presentes], apprentissage[~presentes]
    y = avec_cible[CIBLE].to_numpy($dtype)
    contexte = sans_cible[ENTREES] if len(sans_cible) else None
    return avec_cible, avec_cible[ENTREES], y, contexte


def verifier_plis(plis, ignorer):
    """Arrête le script si les plis refaits ne sont pas ceux du projet."""
    if not ignorer and empreinte_plis(plis) != EMPREINTE_PLIS:
        raise SystemExit(
            "Les plis de validation refaits ne sont pas ceux du projet (empreinte différente) : "
            "version de scikit-learn différente ou données modifiées."
        )
''')

METRIQUES_BASE = Template('''\
def fonctions_metriques():
    """Fonctions de métrique du paquet, reprises dans l'annexe."""
    return {$fonctions}


def calculer_metrique(nom, parametres, donnees):
    """Une métrique ; indéfinie (une seule classe dans un pli, par exemple), elle vaut nan."""
    try:
        return fonctions_metriques()[nom](Entrees(**donnees), **parametres)
    except ErreurMetrique:
        return float("nan")
''')

METRIQUES_DU_PROJET = Template('''\
def calculer_metriques(y_vrai, y_pred, scores=None):
    """Métriques du problème et métrique du projet, comme l'application les calcule."""
    donnees = dict(
        y_vrai=y_vrai, y_pred=y_pred, scores=scores, classes=CLASSES, classe_positive=CLASSE_POSITIVE
    )
    valeurs = {nom: calculer_metrique(nom, {}, donnees) for nom in METRIQUES_STANDARD}
    if METRIQUE not in valeurs:
        valeurs[METRIQUE] = calculer_metrique(METRIQUE_ID, PARAMETRES_METRIQUE, donnees)
    return valeurs
''')

VALIDATION = Template('''\
def valider_modele(nom, X, y, plis, contexte):
    """Un ajustement par pli sur l'apprentissage du pli, noté sur le pli de validation."""
    pipeline = construire_pipeline(nom, contexte)
    resultats = {}
    for app, val in plis:
        modele = clone(pipeline)
        modele.fit(X.iloc[app], y[app])
        x_val = X.iloc[val]
        predictions = modele.predict(x_val)
        scores = scores_du_modele(modele, x_val, CLASSES, CLASSE_POSITIVE) if CLASSES else None
        for cle, valeur in calculer_metriques(y[val], predictions, scores).items():
            resultats.setdefault(cle, []).append(float(valeur))
    return resultats


def evaluer_sur_le_test(nom, X, y, contexte, lignes_test, seuil):
    """Ajuste le modèle retenu sur l'apprentissage entier et le note sur le test (sans-cible comptées)."""
    modele = construire_pipeline(nom, contexte)
    modele.fit(X, y)
    presentes = lignes_test[CIBLE].notna().to_numpy()
    variables = lignes_test[ENTREES]
    predictions = np.asarray(modele.predict(variables))
    scores = None
    if CLASSES:
        scores = scores_du_modele(modele, variables, CLASSES, CLASSE_POSITIVE)
        scores = None if scores is None else np.asarray(scores)
    if seuil is not None:
        predictions = predire_au_seuil(scores, CLASSES, CLASSE_POSITIVE, seuil)
    y_test = lignes_test.loc[presentes, CIBLE].to_numpy($dtype)
    retenues = predictions[presentes]
    valeurs = calculer_metriques(y_test, retenues, None if scores is None else scores[presentes])
    return {
        "scores": {cle: float(v) for cle, v in valeurs.items()},
        "n_lignes": int(presentes.sum()),
        "n_sans_cible": int((~presentes).sum()),
        "empreinte_predictions": empreinte_predictions(retenues),
    }
''')

SANS_CIBLE = Template('''\
def ajuster_sans_cible(nom, X):
    """Ajuste un modèle sans cible sur toutes les lignes ; une valeur par score, étiquettes comptées."""
    pipeline = construire_pipeline(nom)
    etiquettes = np.asarray(pipeline.fit_predict(X)) if hasattr(pipeline, "fit_predict") else None
    if etiquettes is None:
        pipeline.fit(X)
    final = pipeline[-1]
    anomalie = nom in ANOMALIES
    transformees = pipeline[:-1].transform(X) if etiquettes is not None and not anomalie else None
    noms = []
    if etiquettes is not None:
        noms.append("taux_anomalies" if anomalie else "silhouette")
    if hasattr(final, "explained_variance_ratio_"):
        noms.append("variance_expliquee")
    valeurs = {}
    for cle in noms:
        donnees = dict(
            modele=final, etiquettes=etiquettes, x=transformees, etiquettes_ref=None, graine=GRAINE
        )
        valeurs[cle] = calculer_metrique(cle, {}, donnees)
    return {cle: [float(v)] for cle, v in valeurs.items()}
''')

SORTIE = Template('''\
def lire_options(argv=None):
    """Options de la ligne de commande."""
    analyseur = argparse.ArgumentParser(description="Refait le projet avec scikit-learn.")
    analyseur.add_argument("${o}donnees", default=DONNEES, help="fichier de données (par défaut : DONNEES)")
$option_test    analyseur.add_argument("${o}sortie", default=None, help="dossier de resultats_script.json")
    analyseur.add_argument(
        "${o}ignorer-empreintes",
        action="store_true",
        help="continue même si les empreintes (données, test, plis) diffèrent",
    )
    return analyseur.parse_args(argv)


def en_json(valeur):
    """Valeur écrite en JSON : nombres de numpy en nombres Python, nan en null."""
    if isinstance(valeur, dict):
        return {str(cle): en_json(v) for cle, v in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return [en_json(v) for v in valeur]
    if isinstance(valeur, (np.floating, float)):
        return None if np.isnan(valeur) else float(valeur)
    if isinstance(valeur, np.integer):
        return int(valeur)
    return valeur


def ecrire_resultats(dossier, resultats):
    """Écrit resultats_script.json dans le dossier donné (créé au besoin)."""
    if dossier is None:
        return
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    texte = json.dumps(en_json(resultats), indent=2, ensure_ascii=False, sort_keys=True)
    (dossier / "resultats_script.json").write_text(texte + "\\n", encoding="utf-8")
''')

MAIN_AVEC_CIBLE = Template('''\
def afficher_validation(resultats):
    """Tableau de la validation croisée : moyenne et écart type des plis de la métrique du projet."""
    print()
    print(f"Validation croisée ({N_PLIS} plis, {VALIDATION}), métrique {METRIQUE} :")
    for nom, scores in resultats.items():
        valeurs = [v for v in scores[METRIQUE] if not np.isnan(v)]
        moyenne = float(np.mean(valeurs)) if valeurs else float("nan")
        ecart = float(np.std(valeurs, ddof=1)) if len(valeurs) > 1 else 0.0
        total = len(scores[METRIQUE])
        print(f"  {nom:<46} {moyenne:>12.4f} +/- {ecart:.4f}  ({len(valeurs)} plis sur {total})")


def main(argv=None):
    """Refait le projet : lecture, découpage, validation croisée, puis test du modèle retenu."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    options = lire_options(argv)
    donnees = lire_donnees(options.donnees)
    verifier_donnees(donnees, options.ignorer_empreintes)
    apprentissage, lignes_test = separer(donnees, options)
    avec_cible, X, y, contexte = preparer_apprentissage(apprentissage)
    plis = plis_de_validation(avec_cible)
    verifier_plis(plis, options.ignorer_empreintes)
    print(
        f"{len(donnees)} lignes lues ; {len(avec_cible)} d'apprentissage avec cible, "
        f"{len(lignes_test)} de test."
    )
    resultats = {
        "graine": GRAINE,
        "environnement": ENVIRONNEMENT,
        "empreintes": {"donnees": EMPREINTE_DONNEES, "test": EMPREINTE_TEST, "plis": EMPREINTE_PLIS},
        "validation": {nom: valider_modele(nom, X, y, plis, contexte) for nom in MODELES},
    }
    afficher_validation(resultats["validation"])
    if MODELE_RETENU is not None:
        resultats["decision"] = {"modele": MODELE_RETENU, "seuil": SEUIL}
        resultats["test"] = evaluer_sur_le_test(MODELE_RETENU, X, y, contexte, lignes_test, SEUIL)
        test = resultats["test"]
        print()
        print(f"Test du modèle retenu ({MODELE_RETENU}), {test['n_lignes']} lignes :")
        for cle, valeur in test["scores"].items():
            print(f"  {cle:<28} {valeur:>12.4f}")
    ecrire_resultats(options.sortie, resultats)

''')

MAIN_SANS_CIBLE = Template('''\
def main(argv=None):
    """Refait le projet sans cible : lecture, puis un ajustement par modèle sur toutes les lignes."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    options = lire_options(argv)
    donnees = lire_donnees(options.donnees)
    verifier_donnees(donnees, options.ignorer_empreintes)
    X = donnees[ENTREES]
    print(f"{len(donnees)} lignes lues, {len(ENTREES)} variables.")
    resultats = {
        "graine": GRAINE,
        "environnement": ENVIRONNEMENT,
        "empreintes": {"donnees": EMPREINTE_DONNEES},
        "validation": {nom: ajuster_sans_cible(nom, X) for nom in MODELES},
    }
    print()
    for nom, scores in resultats["validation"].items():
        lecture = ", ".join(f"{cle} = {valeur[0]:.4f}" for cle, valeur in scores.items())
        print(f"  {nom:<46} {lecture}")
    if MODELE_RETENU is not None:
        resultats["decision"] = {"modele": MODELE_RETENU, "seuil": None}
        print(f"\\nModèle retenu : {MODELE_RETENU}")
    ecrire_resultats(options.sortie, resultats)

''')


# ============================================================
# CONSTRUCTION DU SCRIPT
# ============================================================
PROBLEMES_FR = {
    Probleme.REGRESSION: "régression",
    Probleme.CLASSIFICATION: "classification",
    Probleme.SANS_CIBLE: "sans cible (regroupement, réduction de dimension, anomalies)",
}
STRUCTURES_FR = {
    Structure.INDEPENDANTE: "lignes indépendantes",
    Structure.GROUPEE: "lignes groupées par unité",
    Structure.CHRONOLOGIQUE: "lignes ordonnées dans le temps",
    Structure.GROUPEE_CHRONOLOGIQUE: "unités suivies dans le temps",
}
LIBELLES_VALIDATION = {
    "validation.kfold": "plis aléatoires",
    "validation.kfold_stratifie": "plis aléatoires gardant les proportions des classes",
    "validation.exclusion_unique": "exclusion d'une ligne à la fois",
    "validation.plis_groupes": "plis d'unités entières",
    "validation.plis_groupes_stratifies": "plis d'unités entières gardant les proportions des classes",
    "validation.groupes_aleatoires": "tirages d'unités entières",
    "validation.serie_temporelle": "fenêtre croissante sur les dates",
}
CODE_DES_PLIS = {
    "validation.kfold": (
        "KFold",
        "KFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(np.zeros(n))",
    ),
    "validation.kfold_stratifie": (
        "StratifiedKFold",
        "StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(np.zeros(n), classes)",
    ),
    "validation.exclusion_unique": ("LeaveOneOut", "LeaveOneOut().split(np.zeros(n))"),
    "validation.plis_groupes": (
        "GroupKFold",
        "GroupKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(np.zeros(n), groups=unites)",
    ),
    "validation.plis_groupes_stratifies": (
        "StratifiedGroupKFold",
        "StratifiedGroupKFold(n_splits=N_PLIS, shuffle=True, random_state=GRAINE).split(\n"
        "        np.zeros(n), classes, groups=unites\n    )",
    ),
    "validation.groupes_aleatoires": (
        "GroupShuffleSplit",
        "GroupShuffleSplit(n_splits=N_PLIS, test_size=0.2, random_state=GRAINE).split(\n"
        "        np.zeros(n), groups=unites\n    )",
    ),
}
LIBELLES_DECOUPAGE = {
    split.Methode.ALEATOIRE: "lignes tirées au hasard",
    split.Methode.STRATIFIE: "lignes tirées au hasard en gardant les proportions des classes",
    split.Methode.UNITES: "unités entières (aucune unité des deux côtés)",
    split.Methode.UNITES_STRATIFIEES: "unités entières, classes réparties entre apprentissage et test",
    split.Methode.CHRONOLOGIQUE: "dernières dates, coupure entre deux dates distinctes",
    split.Methode.FICHIER: "jeu de test fourni à part",
}


def _echapper_docstring(texte: str) -> str:
    return texte.replace("\\", "/").replace('"""', "'''")


def _liste(noms: Iterable[Any]) -> _Suite:
    return _Suite("[", "]", [_chaine(n) if isinstance(n, str) else repr(n) for n in noms])


@dataclass
class _Constructeur:
    """Assemble le script d'un projet : constantes, fonctions du projet, annexe du paquet."""

    spec: SpecProjet
    table: Table
    coffre: CoffreTest | None
    decoupage: split.Decoupage | None
    decision: Decision | None
    lecture: Lecture
    env: Mapping[str, str]
    fichier_test: Lecture | None = None
    imports: _Imports = field(default_factory=_Imports)
    reprises: list[Extrait] = field(default_factory=list)
    embarques: set[str] = field(default_factory=set)

    # Constantes
    def _affecter(self, nom: str, valeur: Any, ecrivain: _Ecrivain) -> str:
        doc = ecrivain.valeur(valeur)
        return f"{nom} = {rendre(doc, 0, len(nom) + 3)}"

    def _constantes(self, ecrivain: _Ecrivain) -> str:
        s, d = self.spec, self.decoupage
        lecture = self.lecture
        lignes = [
            self._affecter("GRAINE", s.graine, ecrivain),
            self._affecter("ENVIRONNEMENT", dict(self.env), ecrivain),
            "",
            "# Projet tel que l'application l'a enregistré (voir SpecProjet.vers_dict).",
            self._affecter("PROJET", s.vers_dict(), ecrivain),
            "",
            f"DONNEES = {_chaine(lecture.fichier)}",
            f"FEUILLE = {_chaine(lecture.feuille) if lecture.feuille else 'None'}",
            self._affecter("COLONNES_DATES", list(lecture.colonnes_dates), ecrivain),
            f"EMPREINTE_DONNEES = {_chaine(self.table.empreinte)}",
        ]
        if lecture.csv:
            lignes += [
                f"{constante} = {_chaine(valeur)}" for _, constante, valeur in _options_csv(lecture) if valeur
            ]
        lignes += ["", self._affecter("ENTREES", list(preprocess.colonnes_d_entree(s)), ecrivain)]
        if s.probleme.avec_cible:
            lignes += self._constantes_avec_cible(ecrivain, d)
        lignes += [
            "",
            self._affecter("MODELES", [c.id for c in s.modeles], ecrivain),
            f"MODELE_RETENU = {_chaine(self.decision.modele_id) if self.decision else 'None'}",
        ]
        if s.probleme.avec_cible:
            seuil = self.decision.seuil if self.decision and self.decision.seuil is not None else None
            lignes.append(f"SEUIL = {seuil!r}")
        return "\n".join(lignes)

    def _constantes_avec_cible(self, ecrivain: _Ecrivain, d: split.Decoupage | None) -> list[str]:
        s = self.spec
        texte = s.metrique or ""
        metrique, parametres = metrics.analyser(texte)
        classes, positive = self._classes()
        lignes = [
            f"CIBLE = {_chaine(s.cible or '')}",
            f"PROBLEME = {_chaine(s.probleme.value)}",
            f"METRIQUE = {_chaine(texte)}",
            f"METRIQUE_ID = {_chaine(metrique.id)}",
            self._affecter("PARAMETRES_METRIQUE", parametres, ecrivain),
            self._affecter("METRIQUES_STANDARD", list(metrics.STANDARD.get(s.probleme, ())), ecrivain),
            self._affecter("CLASSES", None if classes is None else tuple(classes), ecrivain),
            self._affecter("CLASSE_POSITIVE", positive, ecrivain),
            "",
            f"PART_TEST = {float(s.part_test)!r}",
            f"N_PLIS = {s.n_plis}",
            f"VALIDATION = {_chaine(split.validation_appliquee(s))}",
        ]
        if s.colonne_groupe:
            lignes.append(f"UNITES = {_chaine(s.colonne_groupe)}")
        if s.colonne_date:
            lignes.append(f"DATE = {_chaine(s.colonne_date)}")
        if d is not None:
            lignes += self._constantes_decoupage(ecrivain, d)
        return lignes

    def _constantes_decoupage(self, ecrivain: _Ecrivain, d: split.Decoupage) -> list[str]:
        lignes = [f"EMPREINTE_TEST = {_chaine(d.empreinte_sha256)}"]
        if d.methode is split.Methode.STRATIFIE:
            lignes.append(f"STRATE = {_chaine(d.colonne_groupe or self.spec.cible or '')}")
        if d.methode is split.Methode.FICHIER:
            feuille = self.fichier_test.feuille if self.fichier_test else None
            lignes.append(f"FEUILLE_TEST = {_chaine(feuille) if feuille else 'None'}")
        if d.methode is split.Methode.UNITES_STRATIFIEES and self.coffre is not None:
            unites = self._unites_de_test()
            lignes.append(self._affecter("UNITES_TEST", unites, ecrivain))
        lignes.append(f"EMPREINTE_PLIS = {_chaine(self._empreinte_des_plis())}")
        return lignes

    def _classes(self) -> tuple[list[Any] | None, Any]:
        """Classes de la cible dans l'apprentissage et classe positive, comme `train` les fixe."""
        s = self.spec
        if s.probleme is not Probleme.CLASSIFICATION:
            return None, None
        y = evaluate.apprentissage_du_projet(s, self.table, self.coffre).y
        classes = np.unique(y).tolist()
        if len(classes) != 2:
            return classes, None
        if s.classe_positive is not None:
            return classes, s.classe_positive
        return classes, metrics.proposer_classe_positive(y)[0]

    def _unites_de_test(self) -> list[Any]:
        test = np.setdiff1d(np.arange(self.table.n_lignes), self.coffre.positions_apprentissage())
        return self.table.donnees[self.spec.colonne_groupe].iloc[test].drop_duplicates().tolist()

    def _empreinte_des_plis(self) -> str:
        donnees = evaluate.apprentissage_du_projet(self.spec, self.table, self.coffre)
        try:
            return empreinte_plis(split.plis_validation(donnees.donnees, self.spec))
        except split.ErreurDecoupage as erreur:
            raise ErreurExport(f"Les plis de validation du projet ne se refont pas : {erreur}") from erreur

    # Lecture et découpage
    def _code_lecture(self) -> str:
        lec = self.lecture
        self.imports.instruction("from pathlib import Path")
        self.imports.instruction("import pandas as pd")
        if lec.csv:
            options = [f"{cle}={constante}" for cle, constante, valeur in _options_csv(lec) if valeur]
            arguments = ", ".join(["chemin", *options, "low_memory=False"])
            code = f"    donnees = pd.read_csv(\n        {arguments}\n    )"
            explication = (
                "fichier .csv avec le séparateur, la décimale et l'encodage du projet"
                if options
                else "fichier .csv aux options de pandas (virgule, point décimal, UTF-8)"
            )
        else:
            code = "    donnees = pd.read_excel(chemin, sheet_name=feuille if feuille is not None else 0)"
            explication = "feuille du classeur .xlsx choisie dans le projet"
        self.embarques.add("empreinte_dataframe")
        return LECTURE.substitute(lecture=code, explication=explication, o=OPT)

    def _code_decoupage(self) -> str:
        d = self.decoupage
        if d is None:
            raise ErreurExport("Découpage du projet inconnu : le script ne peut pas le refaire.")
        self.embarques.add("empreinte_positions")
        if d.methode is split.Methode.FICHIER:
            self.embarques.add("empreinte_dataframe")
            return SEPARER_FICHIER.substitute(o=OPT)
        return self._fonction_positions(d) + "\n\n\n" + SEPARER.substitute()

    def _fonction_positions(self, d: split.Decoupage) -> str:
        imports = self.imports
        libelle = LIBELLES_DECOUPAGE[d.methode]
        entete = (
            f'def positions_de_test(donnees):\n    """Positions de test (à partir de 0) : {libelle}."""\n'
        )
        if d.methode is split.Methode.ALEATOIRE:
            imports.depuis("sklearn.model_selection", "train_test_split")
            return entete + (
                "    positions = np.arange(len(donnees))\n"
                "    _, test = train_test_split(positions, test_size=PART_TEST, random_state=GRAINE)\n"
                "    return np.sort(test)"
            )
        if d.methode is split.Methode.STRATIFIE:
            imports.depuis("sklearn.model_selection", "train_test_split")
            return entete + (
                "    positions = np.arange(len(donnees))\n"
                "    strates = donnees[STRATE]\n"
                "    if strates.isna().any():  # un manque forme sa strate, ou rejoint la plus fréquente\n"
                "        codes, _ = pd.factorize(strates, sort=True)\n"
                "        manque = codes < 0\n"
                "        if manque.sum() >= 2:\n"
                "            codes[manque] = codes.max() + 1\n"
                "        else:\n"
                "            codes[manque] = np.bincount(codes[~manque]).argmax()\n"
                "        strates = codes\n"
                "    _, test = train_test_split(\n"
                "        positions, test_size=PART_TEST, random_state=GRAINE, stratify=strates\n"
                "    )\n"
                "    return np.sort(test)"
            )
        if d.methode is split.Methode.UNITES:
            imports.depuis("sklearn.model_selection", "GroupShuffleSplit")
            return entete + (
                "    positions = np.arange(len(donnees))\n"
                "    tirage = GroupShuffleSplit(n_splits=1, test_size=PART_TEST, random_state=GRAINE)\n"
                "    _, test = next(tirage.split(positions, groups=donnees[UNITES]))\n"
                "    return np.sort(test)"
            )
        if d.methode is split.Methode.CHRONOLOGIQUE:
            imports.instruction("import math")
            return entete + (
                "    valeurs = donnees[DATE].to_numpy()\n"
                "    n = len(valeurs)\n"
                '    coupure = np.sort(valeurs, kind="stable")[n - math.ceil(PART_TEST * n)]\n'
                "    return np.flatnonzero(valeurs >= coupure)"
            )
        return entete + "    return np.flatnonzero(donnees[UNITES].isin(UNITES_TEST).to_numpy())"

    def _code_plis(self) -> str:
        validation = split.validation_appliquee(self.spec)
        libelle = LIBELLES_VALIDATION[validation]
        entete = (
            f'def plis_de_validation(avec_cible):\n    """Plis de validation croisée : {libelle}."""\n'
            "    n = len(avec_cible)\n"
        )
        if self.spec.colonne_groupe and validation in {
            "validation.plis_groupes",
            "validation.plis_groupes_stratifies",
            "validation.groupes_aleatoires",
        }:
            entete += "    unites = avec_cible[UNITES]\n"
        if validation in {"validation.kfold_stratifie", "validation.plis_groupes_stratifies"}:
            entete += "    classes, _ = pd.factorize(avec_cible[CIBLE], sort=True)\n"
        self.embarques.add("empreinte_plis")
        if validation == "validation.serie_temporelle":
            self.imports.depuis("sklearn.model_selection", "TimeSeriesSplit")
            return entete + (
                "    valeurs = avec_cible[DATE].to_numpy()\n"
                "    distinctes = np.unique(valeurs)\n"
                "    rang = np.searchsorted(distinctes, valeurs)\n"
                "    plis = []\n"
                "    for app, val in TimeSeriesSplit(n_splits=N_PLIS).split(distinctes):\n"
                "        apprentissage = np.flatnonzero(rang <= app.max())\n"
                "        validation = np.flatnonzero((rang >= val.min()) & (rang <= val.max()))\n"
                "        plis.append((apprentissage, validation))\n"
                "    return [(np.asarray(a, dtype=np.int64), np.asarray(v, dtype=np.int64)) for a, v in plis]"
            )
        classe, appel = CODE_DES_PLIS[validation]
        self.imports.depuis("sklearn.model_selection", classe)
        return entete + (
            f"    decoupes = {appel}\n"
            "    return [(np.asarray(a, dtype=np.int64), np.asarray(v, dtype=np.int64)) for a, v in decoupes]"
        )

    # Pipelines
    def _code_pipelines(self, ecrivain: _Ecrivain) -> str:
        pipelines = pipelines_du_projet(self.spec, self.table, self.coffre)
        self.imports.depuis("sklearn.pipeline", "Pipeline")
        fonctions: dict[str, tuple[str, str]] = {}  # code des étapes vers (nom de fonction, texte)
        branches: list[str] = []
        for choix in self.spec.modeles:
            *etapes, (_, modele) = pipelines[choix.id].steps
            code_etapes = rendre(ecrivain.valeur([tuple(e) for e in etapes]), 4, len("return "))
            if code_etapes not in fonctions:
                persistance_ = choix.id == cible_passee.PERSISTANCE
                nom = "etapes_de_la_persistance" if persistance_ else "etapes_du_projet"
                if any(n == nom for n, _ in fonctions.values()):
                    nom = f"etapes_{len(fonctions) + 1}"
                fonctions[code_etapes] = (
                    nom,
                    "Prétraitement de la persistance : la cible décalée d'un pas."
                    if persistance_
                    else "Prétraitement du projet, par rôle de colonne, puis sélection éventuelle.",
                )
            nom_fonction = fonctions[code_etapes][0]
            pipeline = _Appel(
                "Pipeline",
                [
                    (
                        None,
                        _Suite(
                            "[",
                            "]",
                            [
                                f"*{nom_fonction}(contexte)",
                                _Suite("(", ")", ['"modele"', ecrivain.valeur(modele)]),
                            ],
                        ),
                    )
                ],
            )
            branches.append(
                f"    if nom == {_chaine(choix.id)}:\n        return {rendre(pipeline, 8, len('return '))}"
            )
        blocs = []
        for code, (nom, texte) in fonctions.items():
            blocs.append(f'def {nom}(contexte):\n    """{texte}"""\n    return {code}')
        construire = (
            "def construire_pipeline(nom, contexte=None):\n"
            '    """Pipeline non ajusté du modèle `nom` : le prétraitement, puis le modèle."""\n'
            + "\n".join(branches)
            + '\n    raise KeyError(f"Modèle inconnu : {nom}")'
        )
        return "\n\n\n".join([*blocs, construire])

    # Métriques
    def _code_metriques(self) -> str:
        s = self.spec
        if s.probleme.avec_cible:
            ids = list(metrics.STANDARD.get(s.probleme, ()))
            metrique, _ = metrics.analyser(s.metrique or "")
            if metrique.id not in ids:
                ids.append(metrique.id)
        else:
            ids = ["silhouette", "variance_expliquee", "taux_anomalies"]
        noms = {i: metrics.metrique(i).fonction.__name__ for i in ids}
        racines = ["Entrees", "ErreurMetrique", *noms.values()]
        if s.probleme is Probleme.CLASSIFICATION:
            racines.append("scores_du_modele")
        fonctions = ", ".join(f"{_chaine(i)}: {n}" for i, n in noms.items())
        self.reprises.append(extraire_du_paquet(metrics, racines))
        base = METRIQUES_BASE.substitute(fonctions=fonctions)
        return base + "\n\n" + METRIQUES_DU_PROJET.substitute() if s.probleme.avec_cible else base

    # Annexe
    def _annexe(self, ecrivain: _Ecrivain) -> str:
        noms = set(ecrivain.embarques) | self.embarques
        if self.spec.probleme.avec_cible:
            noms.add("empreinte_predictions")
            if self.decision is not None and self.decision.seuil is not None:
                noms.add("predire_au_seuil")
        reprise = annexe_des(noms)
        blocs = [reprise, *self.reprises]
        for e in blocs:
            for instruction in e.imports:
                self.imports.instruction(instruction)
        return "\n\n\n".join(e.code for e in blocs if e.code)

    # Assemblage
    def construire(self) -> str:
        s = self.spec
        ecrivain = _Ecrivain(s.graine, self.imports, contexte_d_apprentissage(s, self.table, self.coffre))
        for instruction in ("import argparse", "import json", "import sys", "import numpy as np"):
            self.imports.instruction(instruction)
        constantes = self._constantes(ecrivain)
        corps = [self._code_lecture()]
        if s.probleme.avec_cible:
            corps += self._corps_avec_cible(ecrivain)
        else:
            corps += self._corps_sans_cible(ecrivain)
        annexe = self._annexe(ecrivain)
        bandeau = "# " + "=" * 76
        parties = [
            self._entete(),
            "from __future__ import annotations",
            "\n".join(self.imports.lignes()),
            f"{bandeau}\n# Réglages du projet\n{bandeau}\n{constantes}",
            f"{bandeau}\n# Étapes du projet\n{bandeau}\n\n" + "\n\n\n".join(c.strip("\n") for c in corps),
            f"{bandeau}\n# Annexe : code du paquet MécaML repris tel quel\n{bandeau}\n\n{annexe}",
            # Le lancement suit l'annexe : `main` appelle des fonctions qui y sont définies.
            'if __name__ == "__main__":\n    main()',
        ]
        separateurs = ["\n\n", "\n\n", "\n\n\n", "\n\n\n", "\n\n\n", "\n\n\n"]
        texte = parties[0].rstrip("\n")
        for sep, partie in zip(separateurs, parties[1:], strict=True):
            texte += sep + partie.strip("\n")
        return texte + "\n"

    def _corps_avec_cible(self, ecrivain: _Ecrivain) -> list[str]:
        dtype = "dtype=float" if self.spec.probleme is Probleme.REGRESSION else ""
        self.imports.depuis("sklearn.base", "clone")
        self.embarques.add("empreinte_plis")
        corps = [
            self._code_decoupage(),
            PREPARATION.substitute(dtype=dtype),
            self._code_plis(),
            self._code_pipelines(ecrivain),
            self._code_metriques(),
            VALIDATION.substitute(dtype=dtype),
            SORTIE.substitute(o=OPT, option_test=self._option_test()),
            MAIN_AVEC_CIBLE.substitute(),
        ]
        return corps

    def _corps_sans_cible(self, ecrivain: _Ecrivain) -> list[str]:
        self.imports.depuis("sklearn.base", "clone")
        return [
            self._code_pipelines(ecrivain),
            self._code_metriques(),
            "ANOMALIES = " + rendre(_liste(train.ANOMALIES), 0, len("ANOMALIES = ")),
            SANS_CIBLE.substitute(),
            SORTIE.substitute(o=OPT, option_test=""),
            MAIN_SANS_CIBLE.substitute(),
        ]

    def _option_test(self) -> str:
        if self.decoupage is not None and self.decoupage.methode is split.Methode.FICHIER:
            return (
                f'    analyseur.add_argument("{OPT}test", default=None, '
                'help="jeu de test fourni à part (celui du projet)")\n'
            )
        return ""

    def _entete(self) -> str:
        s = self.spec
        lecture = self.lecture
        if s.probleme.avec_cible:
            resume = (
                f"Problème : {PROBLEMES_FR[s.probleme]} de « {_echapper_docstring(s.cible or '')} », "
                f"métrique {s.metrique}, {STRUCTURES_FR[s.structure]}, part de test "
                f"{decimale(s.part_test, 2)}, {s.n_plis} plis de validation."
            )
        else:
            resume = (
                f"Problème : {PROBLEMES_FR[s.probleme]}, {len(s.variables())} variables, "
                f"métrique {s.metrique or 'silhouette'}."
            )
        declarees = sorted({c.id for c in cible_passee.choix_concernes(s)})
        hypothese = ""
        if declarees:
            hypothese = (
                "Hypothèse déclarée par le projet : la cible du pas précédent est connue au moment de la "
                f"prévision (cible_connue=True pour {', '.join(declarees)}).\n"
            )
        env = ", ".join(f"{k} {v}" for k, v in self.env.items())
        return ENTETE.substitute(
            titre=_echapper_docstring(lecture.fichier),
            version=__version__,
            resume=resume,
            hypothese=hypothese,
            environnement=env,
            nom_script=NOM_DU_SCRIPT,
            o=OPT,
        )


def generer_script(
    spec: SpecProjet,
    table: Table,
    coffre: CoffreTest | None = None,
    *,
    decoupage: split.Decoupage | None = None,
    decision: Decision | None = None,
    chargement: Chargement | None = None,
    fichier_test: Lecture | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Script Python autonome qui refait le projet : mêmes pipelines, mêmes plis, mêmes scores.

    `decoupage` : celui que `split.sceller` a rendu (retrouvé du coffre s'il manque). `decision` : le
    modèle retenu et son seuil (sans elle, le script valide les modèles sans évaluer le test). `chargement`
    donne le séparateur et l'encodage d'un `.csv`. Refus (`ErreurExport`) si la cible passée n'est pas
    déclarée ou si elle décompte.
    """
    verifier_ceinture(spec, table, coffre)
    if not spec.modeles:
        raise ErreurExport("Aucun modèle comparé : choisir des modèles en S6 avant d'exporter.")
    if spec.probleme.avec_cible:
        if coffre is None:
            raise ErreurExport("Export impossible sans le jeu de test mis sous scellé (S4).")
        decoupage = decoupage or retrouver_decoupage(table, spec, coffre)
    constructeur = _Constructeur(
        spec,
        table,
        coffre,
        decoupage,
        decision,
        lecture_du_fichier(table, chargement),
        env or environnement(),
        fichier_test,
    )
    texte = constructeur.construire()
    try:
        ast.parse(texte)
    except SyntaxError as erreur:
        raise ErreurExport(
            f"Le script engendré n'est pas du Python valide (ligne {erreur.lineno}) : {erreur.msg}."
        ) from erreur
    return texte


def ecrire_script(texte: str, dossier: str | Path, nom: str = NOM_DU_SCRIPT) -> Path:
    """Écrit le script dans le dossier choisi (jamais ailleurs) et rend son chemin."""
    chemin = chemin_dans_dossier(dossier, nom)
    chemin.write_text(texte, encoding="utf-8", newline="\n")
    return chemin
