"""Prépare le dossier `examples/` : les jeux sont lus dans le dépôt du livre, empreinte vérifiée.

Le dépôt du livre (`Meca-IA_Ed2026`) publie un `manifest.json` qui donne l'empreinte SHA-256 de chaque jeu.
Ce script refuse tout fichier dont l'empreinte diffère, copie trois jeux tels quels, prend les vingt premières
machines du jeu de séries chronologiques, écrit `examples/empreintes.json`, puis exporte le projet d'exemple
(`examples/projet_exemple/`) avec le code de l'application.

Usage : python build/preparer_exemples.py <dossier du dépôt du livre> [-p pour refaire le projet d'exemple]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
EXEMPLES = RACINE / "examples"
LIMITE_OCTETS = 1_000_000  # les quatre jeux ensemble restent sous 1 Mo
MACHINES_RETENUES = 20
# (identifiant dans le manifeste, nom du fichier d'exemple, extrait de séries ou non)
JEUX = (
    ("Traitement_Acier", "MecaIA-Traitement_Acier.xlsx", False),
    ("Controle_Conformite", "MecaIA-Controle_Conformite.xlsx", False),
    ("Anomalies_Pompe", "MecaIA-Anomalies_Pompe.xlsx", False),
    ("TimeSeries_RUL", "MecaIA-TimeSeries_RUL_extrait.xlsx", True),
)
PROJET_JEU = "MecaIA-Traitement_Acier.xlsx"
# Le projet rejouable et le classeur des chiffres : le rapport, le script et le modèle se refont par le rejeu.
PROJET_FICHIERS = ("projet.json", "rapport.xlsx")


def sha256(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def lire_manifeste(depot: Path) -> dict[str, dict]:
    """Jeux du manifeste du dépôt du livre, par identifiant."""
    manifeste = json.loads((depot / "manifest.json").read_text(encoding="utf-8"))
    return {jeu["id"]: jeu for jeu in manifeste["jeux"]}


def source_verifiee(depot: Path, jeu: dict) -> Path:
    """Chemin du jeu dans le dépôt du livre, après contrôle de l'empreinte du manifeste."""
    chemin = depot / jeu["fichier"]
    if not chemin.is_file():
        raise SystemExit(f"Jeu introuvable : {chemin}")
    if sha256(chemin) != jeu["sha256"]:
        raise SystemExit(f"Empreinte différente de celle du manifeste : {chemin.name}")
    return chemin


def extrait_de_series(source: Path, destination: Path) -> None:
    """Écrit les `MACHINES_RETENUES` premières machines ; un fichier déjà identique en contenu est gardé."""
    table = pd.read_excel(source)
    machines = table["ID_Machine"].drop_duplicates().tolist()[:MACHINES_RETENUES]
    extrait = table[table["ID_Machine"].isin(machines)].reset_index(drop=True)
    if destination.is_file() and pd.read_excel(destination).equals(extrait):
        return
    extrait.to_excel(destination, index=False)


def preparer_jeux(depot: Path) -> list[dict]:
    """Copie ou extrait les quatre jeux ; rend leurs empreintes pour `empreintes.json`."""
    manifeste = lire_manifeste(depot)
    EXEMPLES.mkdir(exist_ok=True)
    lignes = []
    for identifiant, nom, extrait in JEUX:
        jeu = manifeste[identifiant]
        source = source_verifiee(depot, jeu)
        destination = EXEMPLES / nom
        if extrait:
            extrait_de_series(source, destination)
        else:
            shutil.copyfile(source, destination)
        lignes.append(
            {
                "fichier": nom,
                "sha256": sha256(destination),
                "octets": destination.stat().st_size,
                "source": jeu["fichier"],
                "sha256_source": jeu["sha256"],
                "extrait": f"machines 1 à {MACHINES_RETENUES}" if extrait else None,
            }
        )
    total = sum(ligne["octets"] for ligne in lignes)
    if total >= LIMITE_OCTETS:
        raise SystemExit(f"Les jeux d'exemple pèsent {total} octets : la limite est {LIMITE_OCTETS}.")
    return lignes


def refaire_le_projet() -> list[str]:
    """Rejoue le jeu d'acier avec la ligne de commande de l'application et garde les fichiers retenus."""
    sys.path.insert(0, str(RACINE / "src"))
    from mecaml import cli

    cible = EXEMPLES / "projet_exemple"
    with tempfile.TemporaryDirectory(prefix="mecaml_exemple_") as temporaire:
        travail = Path(temporaire)
        proposition = travail / "projet_propose.json"
        jeu = EXEMPLES / PROJET_JEU
        code = cli.principal(["proposer", str(jeu), "-s", str(proposition)])
        code = code or cli.principal(
            ["rejouer", str(proposition), "-d", str(jeu), "-a", "-o", str(travail / "sortie")]
        )
        if code:
            raise SystemExit("Le projet d'exemple n'a pas pu être exporté.")
        shutil.rmtree(cible, ignore_errors=True)
        cible.mkdir()
        for nom in PROJET_FICHIERS:
            shutil.copyfile(travail / "sortie" / nom, cible / nom)
    return list(PROJET_FICHIERS)


def principal(argv: list[str] | None = None) -> int:
    analyseur = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    analyseur.add_argument("depot", type=Path, help="dossier du dépôt du livre (manifest.json et donnees/)")
    analyseur.add_argument("-p", dest="projet", action="store_true", help="refaire aussi le projet d'exemple")
    options = analyseur.parse_args(argv)
    lignes = preparer_jeux(options.depot)
    projet = refaire_le_projet() if options.projet else None
    fiche = {"jeux": lignes, "projet_exemple": {"jeu": PROJET_JEU, "fichiers": projet}}
    ancienne = EXEMPLES / "empreintes.json"
    if projet is None and ancienne.is_file():
        fiche["projet_exemple"] = json.loads(ancienne.read_text(encoding="utf-8"))["projet_exemple"]
    ancienne.write_text(
        json.dumps(fiche, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    for ligne in lignes:
        print(f"{ligne['fichier']} : {ligne['octets']} octets, empreinte de la source vérifiée")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
