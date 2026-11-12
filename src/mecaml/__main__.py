"""Lance l'application : `python -m mecaml` ; l'option `-t` rejoue un parcours complet, sans geste."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import multiprocessing
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from mecaml import __version__

JEU_DE_L_ESSAI = "MecaIA-Traitement_Acier.xlsx"
DELAI_DE_L_ESSAI_S = 300.0
MODULES_REQUIS = ("tkinter", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "openpyxl", "joblib")
VERSION_PYTHON = (3, 12)


def _analyseur() -> argparse.ArgumentParser:
    analyseur = argparse.ArgumentParser(
        prog="mecaml",
        description="MécaML Explorer : application guidée d'apprentissage automatique, données tabulaires.",
    )
    analyseur.add_argument("-V", action="version", version=f"MécaML Explorer {__version__}")
    analyseur.add_argument(
        "-t",
        dest="essai",
        action="store_true",
        help="essai automatique : ouvre la fenêtre, mène un jeu d'exemple jusqu'à l'export, puis la ferme",
    )
    analyseur.add_argument(
        "-s", dest="sortie", help="avec -t : dossier où garder l'export (sinon un dossier temporaire, effacé)"
    )
    return analyseur


def _dire(texte: str) -> None:
    """Écrit sur la sortie standard si elle existe (un exécutable sans console n'en a pas)."""
    with contextlib.suppress(AttributeError, OSError, ValueError):
        print(texte)


def modules_manquants() -> list[str]:
    """Bibliothèques nécessaires à l'application que ce Python ne trouve pas."""
    return [nom for nom in MODULES_REQUIS if importlib.util.find_spec(nom) is None]


def verifier_poste() -> str | None:
    """Le motif, en clair, pour lequel l'application ne peut pas démarrer ; None si tout est prêt."""
    version = ".".join(str(n) for n in VERSION_PYTHON)
    if sys.version_info[:2] < VERSION_PYTHON:
        trouve = f"{sys.version_info[0]}.{sys.version_info[1]}"
        return f"MécaML Explorer demande Python {version} ou plus récent ; ce poste fournit Python {trouve}."
    manquants = modules_manquants()
    if "tkinter" in manquants:
        return (
            "Ce Python n'a pas Tk (tkinter). Réinstallez Python depuis python.org "
            "en laissant cochée l'option « tcl/tk and IDLE »."
        )
    if manquants:
        return (
            f"Il manque des bibliothèques : {', '.join(manquants)}. "
            "Dans le dossier de l'application, lancer : python -m pip install -r requirements.txt"
        )
    return None


def _attendre(app, delai: float) -> None:
    """Fait tourner la boucle Tk jusqu'à la fin du calcul en cours, sans dépasser le délai."""
    fin = time.monotonic() + delai

    def verifier() -> None:
        if not app.occupe or time.monotonic() > fin:
            app.quit()
        else:
            app.after(15, verifier)

    app.after(1, verifier)
    app.mainloop()
    app.update()
    if app.occupe:
        raise TimeoutError(f"calcul encore en cours après {delai:.0f} s")


def _parcours(app, chemin: Path, sortie: Path) -> list[str]:
    """Les gestes d'un utilisateur, de l'accueil à l'export ; rend les fichiers écrits."""
    from mecaml.core.session import Etat

    ecrans = app.ecrans

    def etape(attendu: Etat, nom: str) -> None:
        _attendre(app, DELAI_DE_L_ESSAI_S)
        if app.ctrl.etat < attendu:
            raise RuntimeError(
                f"{nom} : état {app.ctrl.etat.name} au lieu de {attendu.name} ({app.banniere.texte})"
            )
        _dire(f"  {nom} : réussi")

    ecrans[0].ouvrir(chemin)
    etape(Etat.DONNEES, "données")
    ecrans[1].auditer()
    etape(Etat.AUDIT, "audit")
    ecrans[2].continuer()
    _attendre(app, DELAI_DE_L_ESSAI_S)
    ecrans[3].fixer()
    etape(Etat.OBJECTIF, "objectif")
    app.afficher(4)
    ecrans[4].sceller()
    etape(Etat.DECOUPE, "découpage sous scellé")
    app.acquitter(frozenset(c.code for c in app.ctrl.session.alertes_en_attente()))
    app.afficher(5)
    _attendre(app, DELAI_DE_L_ESSAI_S)
    ecrans[5].appliquer()
    etape(Etat.PRETRAITEMENT, "prétraitement")
    app.afficher(6)
    _attendre(app, DELAI_DE_L_ESSAI_S)
    ecrans[6].entrainer()
    etape(Etat.MODELES, "modèles")
    app.afficher(7)
    _attendre(app, DELAI_DE_L_ESSAI_S)
    ecrans[7].retenir()
    etape(Etat.DIAGNOSTIC, "diagnostic")
    app.afficher(8)
    ecrans[8].ouvrir()
    etape(Etat.TESTE, "test")
    app.afficher(9)
    ecrans[9].exporter(sortie)
    etape(Etat.EXPORTE, "export")
    return sorted(p.name for p in sortie.iterdir())


def essai_automatique(sortie: str | None = None) -> int:
    """Ouvre la vraie fenêtre, mène le jeu d'exemple jusqu'à l'export puis ferme ; 0 si tout a réussi."""
    from mecaml.ui.app import Application
    from mecaml.ui.ressources import exemples

    jeu = next((p for p in exemples() if p.name == JEU_DE_L_ESSAI), None)
    if jeu is None:
        _dire(f"Essai impossible : {JEU_DE_L_ESSAI} est introuvable dans le dossier des exemples.")
        return 1
    dossier = Path(sortie) if sortie else Path(tempfile.mkdtemp(prefix="mecaml_essai_"))
    dossier.mkdir(parents=True, exist_ok=True)
    app = Application(confirmer=lambda _texte: True, precharger=False)
    erreurs: list[BaseException] = []
    app.report_callback_exception = lambda _type, valeur, _trace: erreurs.append(valeur)
    try:
        _dire(f"MécaML Explorer {__version__} : essai automatique sur {jeu.name}")
        fichiers = _parcours(app, jeu, dossier)
        if erreurs:
            raise RuntimeError(f"exception dans un rappel de l'interface : {erreurs[0]}")
        if not {"rapport.html", "projet.json"} <= set(fichiers):
            raise RuntimeError(f"export incomplet : {fichiers}")
    except (RuntimeError, TimeoutError) as erreur:
        _dire(f"Essai en échec : {erreur}")
        return 1
    finally:
        app.destroy()
        if not sortie:
            shutil.rmtree(dossier, ignore_errors=True)
    _dire(f"Essai réussi : {len(fichiers)} fichiers exportés ({', '.join(fichiers)}).")
    return 0


def principal(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée : sans option, ouvre l'application ; avec -t, l'essai automatique."""
    for flux in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            flux.reconfigure(encoding="utf-8", errors="replace")
    options = _analyseur().parse_args(argv)
    motif = verifier_poste()
    if motif is not None:
        _dire(motif)
        return 1
    if options.essai:
        return essai_automatique(options.sortie)
    from mecaml.ui.app import lancer

    lancer()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()  # exécutable autonome : un processus fils ne rouvre pas la fenêtre
    sys.exit(principal())
