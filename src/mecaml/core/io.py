"""
MécaML : chargement des jeux de données (.xlsx et .csv).

Règles : seuls `.xlsx` et `.csv` sont lus ; l'encodage et le séparateur d'un `.csv`
sont détectés ; un classeur est lu en valeurs (une formule est lue par sa valeur
mise en cache, jamais évaluée ni exécutée) ; les bornes (200 000 lignes, 500
colonnes) sont contrôlées avant de lire tout le fichier ; la lecture d'un classeur
signale sa progression et peut être annulée, pour tourner dans un fil sans bloquer
l'interface. Rien n'est écrit, aucun accès réseau, aucun objet sérialisé chargé.

Les remarques de lecture (formules sans valeur, colonnes renommées, lignes vides comptées)
sont rendues avec la table et rangées dans `donnees.attrs`, où l'audit les retrouve. Les lignes
gardent leur position, comme à la lecture par `read_excel` ou `read_csv`.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import re
import time
import unicodedata
import warnings
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

import pandas as pd

from mecaml.core.guards import (
    COLONNES_MAX,
    LIGNES_MAX,
    TAILLE_XLSX_DECOMPRESSE_MAX_OCTETS,
    ErreurAnnulee,
    ErreurDonnees,
    extension_acceptee,
    nombre_lisible,
    verifier_bornes,
    verifier_taille_fichier,
)
from mecaml.core.tasks import collecter_avertissements
from mecaml.core.types import Table

__all__ = [
    "ATTR_FORMULES",
    "ATTR_LIGNES_VIDES",
    "ATTR_NOTES",
    "Annulation",
    "Chargement",
    "Progression",
    "Sonde",
    "charger",
    "empreinte_fichier",
    "feuilles_challenge",
    "lister_feuilles",
    "sonder",
]

Progression = Callable[[float, str], None]


class JetonAnnulation(Protocol):
    """Jeton des tâches (`tasks.Annulation`) : `verifier()` lève l'exception d'annulation commune."""

    def verifier(self) -> None: ...


Annulation = Callable[[], bool] | JetonAnnulation

ATTR_NOTES = "mecaml_notes"  # tuple de remarques de lecture, dans `DataFrame.attrs`
ATTR_FORMULES = "mecaml_formules"  # (formules, formules sans valeur en cache)
ATTR_LIGNES_VIDES = "mecaml_lignes_vides"  # nombre de lignes sans aucune valeur

_PAS_PROGRESSION = 5_000
_BLOC_OCTETS = 4 * 2**20
_ECHANTILLON_OCTETS = 64 * 2**10
_ERREURS_TABLEUR = frozenset(
    {"#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!", "#VALUE!", "#GETTING_DATA"}
)
_RE_DATE_ISO = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?")
_RE_DECIMAL_VIRGULE = re.compile(r"^-?\d+,\d+$")
_RE_DECIMAL_POINT = re.compile(r"^-?\d+\.\d+$")
_RE_DIMENSION = re.compile(rb'<dimension ref="[A-Z]+\d+:[A-Z]+(\d+)"')
_RE_FORMULE = re.compile(rb"<f[\s>/]")
_RE_FORMULE_SANS_VALEUR = re.compile(rb"(?:</f>|<f[^>]*/>)\s*(?:<v\s*/>|<v></v>)?\s*</c>")
_CHEVAUCHEMENT = 256
_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass(frozen=True)
class Sonde:
    """Ce que l'on sait d'un fichier avant de le lire en entier."""

    extension: str
    taille_octets: int
    feuilles: tuple[str, ...]  # vide pour un .csv
    lignes_estimees: int | None  # d'après l'en-tête du classeur, sinon None


@dataclass(frozen=True)
class Chargement:
    """Résultat de la lecture : la table et ce qu'il faut en dire à l'utilisateur."""

    table: Table
    notes: tuple[str, ...]
    feuilles: tuple[str, ...]
    encodage: str | None  # .csv seulement
    separateur: str | None  # .csv seulement
    decimal: str | None  # .csv seulement
    formules: int  # cellules à formule (lues par leur valeur en cache)
    formules_sans_valeur: int  # formules dont aucune valeur n'est en cache : lues comme manquantes
    duree_s: float
    sha256: str  # empreinte du fichier : à comparer à la liste blanche du dépôt d'exercices (mode challenge)


# ============================================================
# OUTILS COMMUNS
# ============================================================
def _vers_chemin(chemin: str | Path) -> Path:
    chemin = Path(chemin)
    try:
        if not chemin.exists():
            raise ErreurDonnees(
                f"Fichier introuvable : {chemin}. Vérifier le chemin ou le choisir à nouveau."
            )
        if not chemin.is_file():
            raise ErreurDonnees(f"« {chemin.name} » n'est pas un fichier : choisir un fichier .xlsx ou .csv.")
        taille = chemin.stat().st_size
    except OSError as erreur:
        raise _erreur_systeme(erreur, chemin) from erreur
    extension_acceptee(chemin)
    verifier_taille_fichier(taille, f"« {chemin.name} »")
    return chemin


def _erreur_systeme(erreur: OSError, chemin: Path) -> ErreurDonnees:
    if isinstance(erreur, PermissionError):
        return ErreurDonnees(
            f"« {chemin.name} » est verrouillé ou protégé (souvent ouvert dans Excel) : "
            "le fermer, ou en faire une copie, puis recharger."
        )
    return ErreurDonnees(f"« {chemin.name} » n'a pas pu être lu ({erreur.strerror or erreur}).")


def _signaler(progression: Progression | None, fraction: float, message: str) -> None:
    if progression is not None:
        progression(max(0.0, min(1.0, fraction)), message)


def _verifier_annulation(annuler: Annulation | None) -> None:
    """Interrompt si l'appelable rend vrai, ou si le jeton des tâches le demande (même exception)."""
    if annuler is None:
        return
    verifier = getattr(annuler, "verifier", None)
    if callable(verifier):
        verifier()
    elif annuler():
        raise ErreurAnnulee("Chargement annulé.")


_MODULES_DE_LECTURE = r"(openpyxl|pandas)(\.|$)|mecaml\.core\.io$"


@contextmanager
def _avertissements_de_lecture() -> Iterator[list[str]]:
    """Avertissements des bibliothèques de lecture, rassemblés pour le fil courant.

    Chaque avertissement s'écrit « Catégorie : message ». Pas de `warnings.catch_warnings`, qui n'est pas
    sûr entre fils : le collecteur de `tasks` est local au fil, et le filtre ajouté (idempotent) rend ces
    avertissements systématiques.
    """
    warnings.filterwarnings("always", module=_MODULES_DE_LECTURE)
    with collecter_avertissements() as rassembles:
        yield rassembles


def _sans_accent(texte: str) -> str:
    return (
        "".join(c for c in unicodedata.normalize("NFKD", texte) if not unicodedata.combining(c))
        .lower()
        .strip()
    )


def _entetes(valeurs: list[object]) -> tuple[list[str], list[str]]:
    """Noms de colonnes textuels et uniques ; remarques sur les renommages."""
    noms: list[str] = []
    notes: list[str] = []
    vus: dict[str, int] = {}
    for i, v in enumerate(valeurs):
        brut = "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
        nom = brut or f"Unnamed: {i}"
        if not brut:
            notes.append(f"colonne {i + 1} sans titre, nommée « {nom} »")
        if nom in vus:
            vus[nom] += 1
            nouveau = f"{nom}.{vus[nom]}"
            notes.append(f"titre « {nom} » en double, la répétition devient « {nouveau} »")
            nom = nouveau
        vus.setdefault(nom, 0)
        noms.append(nom)
    return noms, notes


def _dates_iso(df: pd.DataFrame, notes: list[str]) -> None:
    """Colonnes de texte dont toutes les valeurs sont des dates ISO : converties en dates."""
    for nom in df.columns:
        s = df[nom]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        valides = s.dropna()
        if valides.empty or not all(isinstance(v, str) for v in valides.iloc[:50]):
            continue
        if not valides.astype(str).str.fullmatch(_RE_DATE_ISO).all():
            continue
        with _avertissements_de_lecture():
            converti = pd.to_datetime(s, format="ISO8601", errors="coerce")
        if converti.notna().sum() == s.notna().sum():
            df[nom] = converti
            notes.append(f"colonne « {nom} » convertie en dates (format ISO)")


def _erreurs_de_tableur(df: pd.DataFrame, notes: list[str]) -> None:
    """Cellules en erreur (#DIV/0!, #N/A...) lues comme manquantes."""
    total = 0
    for nom in df.columns:
        s = df[nom]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        masque = s.isin(_ERREURS_TABLEUR)
        n = int(masque.sum())
        if n:
            total += n
            df[nom] = s.mask(masque).infer_objects()
    if total:
        notes.append(
            f"{nombre_lisible(total)} cellule(s) en erreur de tableur (#DIV/0!, #N/A...) lues comme "
            "manquantes"
        )


def _finaliser(df: pd.DataFrame, notes: list[str], formules: tuple[int, int]) -> pd.DataFrame:
    """Colonnes sans titre ni valeur retirées, lignes vides comptées (jamais retirées), remarques rangées.

    Les lignes gardent leur position : le mode challenge désigne le test par des positions de lignes
    comptées comme le fait `read_excel` ou `read_csv`.
    """
    _erreurs_de_tableur(df, notes)
    sans_titre = [c for c in df.columns if str(c).startswith("Unnamed: ") and df[c].isna().all()]
    if sans_titre:
        df = df.drop(columns=sans_titre)
        notes.append(f"{len(sans_titre)} colonne(s) sans titre et sans valeur retirée(s)")
    n_vides = int(df.isna().all(axis=1).sum())
    if n_vides:
        notes.append(f"{nombre_lisible(n_vides)} ligne(s) sans aucune valeur conservée(s) à leur place")
    _dates_iso(df, notes)
    df.attrs[ATTR_NOTES] = tuple(notes)
    df.attrs[ATTR_FORMULES] = formules
    df.attrs[ATTR_LIGNES_VIDES] = n_vides
    return df


# ============================================================
# CLASSEUR .xlsx
# ============================================================
def _verifier_classeur(zf: zipfile.ZipFile, chemin: Path) -> None:
    """Refus d'un classeur à macros, à entités XML, ou démesuré une fois décompressé."""
    noms = zf.namelist()
    if "xl/workbook.xml" not in noms:
        raise ErreurDonnees(
            f"« {chemin.name} » n'est pas un classeur .xlsx valide : l'ouvrir dans le tableur et "
            "l'enregistrer à nouveau."
        )
    if any(n.lower().endswith("vbaproject.bin") for n in noms):
        raise ErreurDonnees(
            f"« {chemin.name} » contient des macros, qui ne sont jamais exécutées. "
            "L'enregistrer en .xlsx sans macros (ou en .csv), puis le recharger."
        )
    if sum(i.file_size for i in zf.infolist()) > TAILLE_XLSX_DECOMPRESSE_MAX_OCTETS:
        raise ErreurDonnees(
            f"« {chemin.name} » est démesuré une fois décompressé (plus de "
            f"{TAILLE_XLSX_DECOMPRESSE_MAX_OCTETS / 2**30:.0f} Gio) : retirer des lignes ou des colonnes."
        )
    for nom in noms:
        if nom.endswith((".xml", ".rels")):
            with zf.open(nom) as flux:
                debut = flux.read(4096)
            if b"<!DOCTYPE" in debut or b"<!ENTITY" in debut:
                raise ErreurDonnees(
                    f"« {chemin.name} » contient des déclarations XML (entités) qui ne sont jamais "
                    "interprétées : "
                    "le ré-enregistrer depuis le tableur."
                )


def _ouvrir_zip(chemin: Path) -> zipfile.ZipFile:
    try:
        with chemin.open("rb") as f:
            signature = f.read(8)
        if signature.startswith(b"\xd0\xcf\x11\xe0"):
            raise ErreurDonnees(
                f"« {chemin.name} » est un ancien format (.xls) ou un classeur protégé par mot de passe : "
                "l'enregistrer en .xlsx sans mot de passe, ou en .csv."
            )
        zf = zipfile.ZipFile(chemin)
    except zipfile.BadZipFile as erreur:
        raise ErreurDonnees(
            f"« {chemin.name} » n'est pas un classeur .xlsx valide (fichier abîmé ou d'un autre format) : "
            "l'ouvrir dans le tableur et l'enregistrer à nouveau."
        ) from erreur
    except OSError as erreur:
        raise _erreur_systeme(erreur, chemin) from erreur
    try:
        _verifier_classeur(zf, chemin)
    except BaseException:
        zf.close()
        raise
    return zf


def _parties_feuilles(zf: zipfile.ZipFile) -> dict[str, str]:
    """Nom de chaque feuille visible vers le chemin de sa partie XML dans le classeur."""
    try:
        classeur = ElementTree.fromstring(zf.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    except (KeyError, ElementTree.ParseError) as erreur:
        raise ErreurDonnees(
            "Le classeur est abîmé (liste des feuilles illisible) : le ré-enregistrer."
        ) from erreur
    cibles = {r.get("Id"): r.get("Target", "") for r in rels.findall("p:Relationship", _NS)}
    parties: dict[str, str] = {}
    for feuille in classeur.findall("m:sheets/m:sheet", _NS):
        if feuille.get("state") in ("hidden", "veryHidden"):
            continue
        cible = cibles.get(feuille.get(f"{{{_NS['r']}}}id"), "")
        chemin = cible.lstrip("/") if cible.startswith("/") else f"xl/{cible}"
        parties[feuille.get("name", "")] = chemin
    return parties


def _lignes_declarees(zf: zipfile.ZipFile, partie: str) -> int | None:
    with zf.open(partie) as flux:
        trouve = _RE_DIMENSION.search(flux.read(8192))
    return int(trouve.group(1)) if trouve else None


def _compter_formules(zf: zipfile.ZipFile, partie: str, annuler: Annulation | None) -> tuple[int, int]:
    """(formules, formules sans valeur en cache) de la feuille, par balayage du XML sans l'interpréter."""
    total = sans_valeur = 0
    reste = b""
    with zf.open(partie) as flux:
        while True:
            _verifier_annulation(annuler)
            bloc = flux.read(_BLOC_OCTETS)
            tampon = reste + bloc
            limite = len(tampon) if not bloc else max(0, len(tampon) - _CHEVAUCHEMENT)
            total += sum(1 for m in _RE_FORMULE.finditer(tampon) if m.start() < limite)
            sans_valeur += sum(1 for m in _RE_FORMULE_SANS_VALEUR.finditer(tampon) if m.start() < limite)
            if not bloc:
                return total, sans_valeur
            reste = tampon[limite:]


def empreinte_fichier(chemin: str | Path) -> str:
    """Empreinte SHA-256 du fichier, lue par blocs (le contenu certifié, non les positions de lignes)."""
    h = hashlib.sha256()
    try:
        with Path(chemin).open("rb") as f:
            while bloc := f.read(_BLOC_OCTETS):
                h.update(bloc)
    except OSError as erreur:
        raise _erreur_systeme(erreur, Path(chemin)) from erreur
    return h.hexdigest()


def lister_feuilles(chemin: str | Path) -> tuple[str, ...]:
    """Feuilles visibles d'un classeur (vide pour un .csv)."""
    return sonder(chemin).feuilles


def sonder(chemin: str | Path) -> Sonde:
    """Contrôle l'extension, la taille et la structure du fichier sans lire les données."""
    chemin = _vers_chemin(chemin)
    taille = chemin.stat().st_size
    if chemin.suffix.lower() == ".csv":
        return Sonde(".csv", taille, (), None)
    zf = _ouvrir_zip(chemin)
    try:
        parties = _parties_feuilles(zf)
        premiere = next(iter(parties.values()), None)
        lignes = _lignes_declarees(zf, premiere) if premiere and premiere in zf.namelist() else None
    finally:
        zf.close()
    return Sonde(".xlsx", taille, tuple(parties), lignes)


def feuilles_challenge(feuilles: tuple[str, ...]) -> tuple[str, str] | None:
    """(feuille d'apprentissage, feuille de test) si le classeur suit la convention du mode challenge."""
    normalisees = {_sans_accent(f): f for f in feuilles}
    for app, test in (("apprentissage", "test"), ("train", "test")):
        if app in normalisees and test in normalisees:
            return normalisees[app], normalisees[test]
    return None


def _colonnes_et_lignes(
    ws, progression: Progression | None, annuler: Annulation | None, estimees: int | None
):
    """Lit l'en-tête et les lignes d'une feuille en lecture seule ; borne, signale, peut s'interrompre."""
    entete: list[object] | None = None
    largeur = 0
    lignes: list[tuple] = []
    en_attente = (
        0  # lignes vides rencontrées : conservées si une ligne de données suit, ignorées en fin de feuille
    )
    for k, ligne in enumerate(ws.iter_rows(values_only=True), 1):
        if entete is None:
            if ligne.count(None) == len(ligne):
                continue
            largeur = max(i for i, v in enumerate(ligne) if v is not None) + 1
            if largeur > COLONNES_MAX:
                verifier_bornes(1, largeur)
            entete = list(ligne[:largeur])
            continue
        if k % _PAS_PROGRESSION == 0:
            _verifier_annulation(annuler)
            total = max(estimees or 0, k)
            _signaler(progression, 0.97 * k / total, f"Lecture : {nombre_lisible(len(lignes))} lignes")
        if ligne.count(None) == len(ligne):
            en_attente += 1
            continue
        if en_attente:  # même lecture que `read_excel` : une ligne vide intérieure garde sa position
            lignes.extend([(None,) * largeur] * en_attente)
            en_attente = 0
        lignes.append(ligne[:largeur])
        if len(lignes) > LIGNES_MAX:
            verifier_bornes(len(lignes), largeur)
    if entete is None:
        raise ErreurDonnees("La feuille choisie est vide : choisir une autre feuille du classeur.")
    return entete, lignes, en_attente


def _serie(valeurs: tuple) -> pd.Series:
    """Colonne typée par pandas ; sans aucune valeur, elle est flottante (comme à la lecture d'un .csv)."""
    s = pd.Series(list(valeurs))
    return s.astype("float64") if len(s) and s.isna().all() else s


def _charger_xlsx(
    chemin: Path, feuille: str | None, progression: Progression | None, annuler: Annulation | None
):
    from openpyxl import load_workbook  # import à l'usage : démarrage rapide du cœur

    zf = _ouvrir_zip(chemin)
    try:
        parties = _parties_feuilles(zf)
        if not parties:
            raise ErreurDonnees("Le classeur ne contient aucune feuille visible.")
        if feuille is None:
            feuille = next(iter(parties))
        elif feuille not in parties:
            raise ErreurDonnees(
                f"Feuille « {feuille} » absente du classeur. Feuilles disponibles : {', '.join(parties)}."
            )
        partie = parties[feuille]
        estimees = _lignes_declarees(zf, partie) if partie in zf.namelist() else None
        if estimees is not None and estimees > LIGNES_MAX + 1:
            verifier_bornes(estimees - 1, 2)
        _signaler(progression, 0.0, "Analyse du classeur")
        formules = _compter_formules(zf, partie, annuler) if partie in zf.namelist() else (0, 0)
    finally:
        zf.close()
    try:
        with _avertissements_de_lecture() as avertissements:
            classeur = load_workbook(chemin, read_only=True, data_only=True, keep_links=False)
            try:
                entete, lignes, finales = _colonnes_et_lignes(
                    classeur[feuille], progression, annuler, estimees
                )
            finally:
                classeur.close()
    except OSError as erreur:
        raise _erreur_systeme(erreur, chemin) from erreur
    except (KeyError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as erreur:
        if isinstance(erreur, ErreurDonnees):
            raise
        raise ErreurDonnees(
            f"« {chemin.name} » est illisible ({erreur}) : le ré-enregistrer depuis le tableur."
        ) from erreur
    noms, notes = _entetes(entete)
    verifier_bornes(len(lignes), len(noms))
    if finales:
        notes.append(f"{nombre_lisible(finales)} ligne(s) vide(s) en fin de feuille ignorée(s)")
    notes.extend(f"avertissement du lecteur de classeur : {a}" for a in avertissements)
    colonnes = list(zip(*lignes, strict=True)) if lignes else [() for _ in noms]
    df = pd.DataFrame({nom: _serie(col) for nom, col in zip(noms, colonnes, strict=True)})
    if formules[1]:
        notes.append(
            f"{nombre_lisible(formules[1])} formule(s) sans valeur calculée dans le fichier : lues comme "
            "manquantes"
        )
    return _finaliser(df, notes, formules), feuille, tuple(parties)


# ============================================================
# FICHIER .csv
# ============================================================
def _balayer(chemin: Path, encodage: str, annuler: Annulation | None) -> tuple[bytes, int, bool]:
    """Lit le fichier par blocs : décodage strict, échantillon, nombre de lignes, présence d'octets nuls."""
    decodeur = codecs.getincrementaldecoder(encodage)(errors="strict")
    echantillon = b""
    n_nl = n_cr = 0
    nul = False
    with chemin.open("rb") as f:
        while bloc := f.read(_BLOC_OCTETS):
            _verifier_annulation(annuler)
            if not echantillon:
                echantillon = bloc[:_ECHANTILLON_OCTETS]
            decodeur.decode(bloc)
            n_nl += bloc.count(b"\n")
            n_cr += bloc.count(b"\r")
            nul = nul or b"\x00" in bloc
        decodeur.decode(b"", final=True)
    return echantillon, (n_nl if n_nl else n_cr), nul


def _detecter_encodage(chemin: Path, annuler: Annulation | None) -> tuple[str, bytes, int]:
    """(encodage, échantillon, nombre de lignes) : UTF-8, UTF-16 (avec BOM), puis cp1252, puis latin-1."""
    with chemin.open("rb") as f:
        debut = f.read(4)
    if debut.startswith(codecs.BOM_UTF8):
        candidats = ["utf-8-sig"]
    elif debut.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidats = ["utf-16"]
    else:
        candidats = ["utf-8", "cp1252"]
    for encodage in candidats:
        try:
            echantillon, lignes, nul = _balayer(chemin, encodage, annuler)
        except UnicodeDecodeError:
            continue
        if nul and encodage != "utf-16":
            raise ErreurDonnees(
                f"« {chemin.name} » contient des octets nuls : ce n'est pas un fichier texte. "
                "Vérifier que c'est bien un export .csv."
            )
        return encodage, echantillon, lignes
    echantillon, lignes, _ = _balayer(chemin, "latin-1", annuler)
    return "latin-1", echantillon, lignes


def _texte_echantillon(echantillon: bytes, encodage: str) -> str:
    texte = echantillon.decode(encodage, errors="ignore")
    if len(echantillon) >= _ECHANTILLON_OCTETS and "\n" in texte:
        texte = texte[: texte.rfind("\n")]  # la dernière ligne est coupée
    return texte.lstrip("﻿")


def _detecter_separateur(texte: str) -> str:
    """Séparateur dont le nombre de champs est le plus stable sur les premières lignes."""
    lignes = texte.splitlines()[:50]
    meilleur, score_max = ",", (-1.0, -1)
    for sep in (",", ";", "\t", "|"):
        tailles = [len(r) for r in csv.reader(io.StringIO("\n".join(lignes)), delimiter=sep) if r]
        if not tailles:
            continue
        mode = max(set(tailles), key=tailles.count)
        score = (tailles.count(mode) / len(tailles) if mode > 1 else 0.0, mode)
        if score > score_max:
            meilleur, score_max = sep, score
    return meilleur


def _detecter_decimal(texte: str, sep: str) -> str:
    """« , » si les nombres de l'échantillon s'écrivent à la française, sinon « . »."""
    if sep == ",":
        return "."
    virgule = point = 0
    for ligne in texte.splitlines()[1:200]:
        for champ in ligne.split(sep):
            champ = champ.strip().strip('"')
            virgule += bool(_RE_DECIMAL_VIRGULE.match(champ))
            point += bool(_RE_DECIMAL_POINT.match(champ))
    return "," if virgule >= 3 and virgule > point else "."


def _charger_csv(chemin: Path, progression: Progression | None, annuler: Annulation | None):
    _signaler(progression, 0.0, "Analyse du fichier")
    encodage, echantillon, lignes = _detecter_encodage(chemin, annuler)
    texte = _texte_echantillon(echantillon, encodage)
    sep = _detecter_separateur(texte)
    decimal = _detecter_decimal(texte, sep)
    entete = next(csv.reader(io.StringIO(texte), delimiter=sep), [])
    if len(entete) > COLONNES_MAX:
        verifier_bornes(1, len(entete))
    if (
        lignes > 2 * LIGNES_MAX + 1
    ):  # marge pour les textes sur plusieurs lignes ; contrôle exact après lecture
        verifier_bornes(lignes - 1, len(entete))
    _signaler(progression, 0.2, "Lecture du fichier .csv")
    try:
        with _avertissements_de_lecture() as avertissements:
            df = pd.read_csv(chemin, sep=sep, decimal=decimal, encoding=encodage, low_memory=False)
    except pd.errors.EmptyDataError as erreur:
        raise ErreurDonnees(
            f"« {chemin.name} » est vide : choisir un fichier qui contient des données."
        ) from erreur
    except pd.errors.ParserError as erreur:
        ligne = re.search(r"line (\d+)", str(erreur))
        ou = f" (vers la ligne {ligne.group(1)})" if ligne else ""
        raise ErreurDonnees(
            f"« {chemin.name} » est mal formé{ou}. Vérifier que toutes les lignes ont le même nombre de "
            "champs et que les textes contenant le séparateur sont entre guillemets."
        ) from erreur
    except (UnicodeDecodeError, OSError) as erreur:
        raise ErreurDonnees(f"« {chemin.name} » n'a pas pu être lu ({erreur}).") from erreur
    # Titres lus tels quels dans le fichier : pandas a déjà nommé les vides et les doublons, on le signale.
    brut = entete if len(entete) == len(df.columns) else list(df.columns)
    noms, notes = _entetes(brut)
    df.columns = noms
    verifier_bornes(len(df), len(df.columns))
    notes.extend(f"avertissement du lecteur : {a}" for a in avertissements)
    return _finaliser(df, notes, (0, 0)), encodage, sep, decimal


# ============================================================
# POINT D'ENTRÉE
# ============================================================
def charger(
    chemin: str | Path,
    *,
    feuille: str | None = None,
    progression: Progression | None = None,
    annuler: Annulation | None = None,
) -> Chargement:
    """Lit un jeu de données .xlsx ou .csv et rend la table avec les remarques de lecture.

    `progression(fraction, message)` est appelée pendant la lecture. `annuler` est soit un appelable
    interrogé régulièrement, qui interrompt la lecture (`ErreurAnnulee`) quand il rend vrai, soit le jeton
    d'une `Tache` (`tasks.Annulation`), dont `verifier()` lève l'exception commune `ErreurAnnulation`.
    Lève `ErreurDonnees`, au message actionnable, pour tout refus. Aucun état n'est gardé d'un appel à
    l'autre : la fonction peut tourner dans un fil, deux lectures simultanées ne se gênent pas, et les
    avertissements des bibliothèques sont rassemblés fil par fil (jamais `warnings.catch_warnings`).
    """
    debut = time.perf_counter()
    chemin = _vers_chemin(chemin)
    if chemin.suffix.lower() == ".csv":
        df, encodage, sep, decimal = _charger_csv(chemin, progression, annuler)
        nom_feuille, feuilles = None, ()
    else:
        df, nom_feuille, feuilles = _charger_xlsx(chemin, feuille, progression, annuler)
        encodage = sep = decimal = None
    table = Table(df, nom=chemin.name, feuille=nom_feuille)
    _signaler(progression, 1.0, f"{nombre_lisible(table.n_lignes)} lignes, {table.n_colonnes} colonnes")
    formules = df.attrs.get(ATTR_FORMULES, (0, 0))
    return Chargement(
        table=table,
        notes=tuple(df.attrs.get(ATTR_NOTES, ())),
        feuilles=tuple(feuilles),
        encodage=encodage,
        separateur=sep,
        decimal=decimal,
        formules=formules[0],
        formules_sans_valeur=formules[1],
        duree_s=time.perf_counter() - debut,
        sha256=empreinte_fichier(chemin),
    )
