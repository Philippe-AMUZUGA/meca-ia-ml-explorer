"""Essai d'installation : environnement virtuel neuf, `pip install .` sans réseau, essai automatique.

Le dépôt est copié dans un dossier temporaire (l'installation ne laisse donc aucun dossier de construction
dans le dépôt). L'environnement voit les bibliothèques déjà installées sur le poste (numpy, pandas,
scikit-learn, scipy, matplotlib, openpyxl, joblib) : pip n'a rien à télécharger, et on le lui interdit
(option no-index de pip). Ce qui est vérifié : le paquet se construit et s'installe, ses textes sont
dans la roue, la commande `mecaml` existe, et `mecaml -t` mène un jeu d'exemple jusqu'à l'export.

Limite : les bibliothèques ne sont pas réinstallées depuis zéro. Pour un essai complet, avec téléchargement,
créer l'environnement sans `system_site_packages` et installer `requirements.txt` avant `pip install .`.

Usage : python build/essai_environnement_propre.py
Code de retour : 0 si tout réussit, 1 sinon.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
OPTION = "-" * 2  # préfixe des options longues de pip, écrit sans double trait d'union littéral
IGNORES = shutil.ignore_patterns(
    "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info", "dist", "travail"
)


def executer(commande: list[str], dossier: Path, **environnement: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONUTF8": "1", **environnement}
    return subprocess.run(commande, cwd=dossier, env=env, capture_output=True, text=True, encoding="utf-8")


def etape(nom: str, resultat: subprocess.CompletedProcess[str], attendu: str | None = None) -> bool:
    sortie = (resultat.stdout + resultat.stderr).strip()
    reussi = resultat.returncode == 0 and (attendu is None or attendu in sortie)
    print(f"{'réussi' if reussi else 'ÉCHEC'} : {nom}")
    if not reussi:
        print(sortie[-1500:])
    return reussi


def principal() -> int:
    with tempfile.TemporaryDirectory(prefix="mecaml_propre_") as temporaire:
        base = Path(temporaire)
        copie = base / "depot"
        shutil.copytree(RACINE, copie, ignore=IGNORES)
        shutil.rmtree(copie / "build" / "travail", ignore_errors=True)
        venv.create(base / "env", system_site_packages=True, with_pip=True)
        python = str(base / "env" / "Scripts" / "python.exe")
        neutre = base  # ni le dépôt ni ses sources : le paquet doit venir de l'environnement
        installation = executer(
            [python, "-m", "pip", "install", "-q", OPTION + "no-index", OPTION + "no-build-isolation", "."],
            copie,
        )
        reussites = [etape("pip install . sans réseau", installation)]
        if reussites[0]:
            lieu = executer([python, "-c", "import mecaml; print(mecaml.__file__)"], neutre)
            reussites.append(etape("le paquet vient de l'environnement neuf", lieu, "env"))
            textes = executer([python, "-c", "from mecaml.ui.textes import t; print(t('app.titre'))"], neutre)
            reussites.append(etape("textes TOML présents dans la roue", textes, "Explorer"))
            icone = executer(
                [
                    python,
                    "-c",
                    "from mecaml.ui.ressources import chemin_asset as c; print(c('mecaml_icone.ico'))",
                ],
                neutre,
            )
            reussites.append(etape("icônes présentes dans la roue", icone, "site-packages"))
            version = executer([python, "-m", "mecaml", "-V"], neutre)
            reussites.append(etape("python -m mecaml -V", version, "MécaML Explorer"))
            commande = executer([str(base / "env" / "Scripts" / "mecaml.exe"), "-V"], neutre)
            reussites.append(etape("commande mecaml", commande, "MécaML Explorer"))
            essai = executer([python, "-m", "mecaml", "-t"], copie)
            reussites.append(etape("mecaml -t (interface, parcours complet)", essai, "Essai réussi"))
    return 0 if all(reussites) else 1


if __name__ == "__main__":
    sys.exit(principal())
