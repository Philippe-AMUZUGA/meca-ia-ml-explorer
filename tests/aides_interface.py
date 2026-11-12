"""Aides des tests de l'interface : fenêtre, attente bornée du calcul, parcours scripté, captures."""

from __future__ import annotations

import gc
import os
import time
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mecaml.core.session import Etat
from mecaml.core.types import ChoixTechnique
from mecaml.ui.app import Application

DELAI_S = 300.0
DOSSIER_CAPTURES = os.environ.get("MECAML_CAPTURES")


def ouvrir_application(**options: Any) -> Application:
    """Fenêtre de test (aucune boîte de dialogue : la confirmation répond oui) ; ignore le test sans écran."""
    options.setdefault("precharger", False)
    options.setdefault("confirmer", lambda _texte: True)
    try:
        app = Application(**options)
    except tk.TclError as erreur:
        pytest.skip(f"Tk indisponible : {erreur}")
    erreurs: list[BaseException] = []
    app.report_callback_exception = lambda exc, val, tb: erreurs.append(val)
    app.erreurs_de_rappel = erreurs  # type: ignore[attr-defined]
    return app


def attendre(app: Application, delai: float = DELAI_S) -> None:
    """Fait tourner la boucle Tk jusqu'à la fin du calcul en cours, sans jamais dépasser le délai.

    La boucle est la vraie (`mainloop`) : un objet Tk libéré dans le fil de calcul est rendu au fil de Tk.
    """
    fin = time.monotonic() + delai

    def verifier() -> None:
        if not app.occupe or time.monotonic() > fin:
            app.quit()
        else:
            app.after(15, verifier)

    app.after(1, verifier)
    app.mainloop()
    app.update()
    assert not app.occupe, f"calcul encore en cours après {delai} s"
    erreurs = getattr(app, "erreurs_de_rappel", [])
    assert not erreurs, f"exception dans un rappel de l'interface : {erreurs[:1]}"


def chauffer(app: Application, delai: float = DELAI_S) -> None:
    """Lance le préchargement de l'accueil (scikit-learn, graphiques) et l'attend, boucle Tk vivante."""
    from mecaml.core.tasks import Tache, precharger_scikit_learn
    from mecaml.ui.app import _chauffer_les_graphiques

    taches = [precharger_scikit_learn(), Tache(_chauffer_les_graphiques, nom="graphiques").demarrer()]
    fin = time.monotonic() + delai

    def verifier() -> None:
        if all(t.terminee for t in taches) or time.monotonic() > fin:
            app.quit()
        else:
            app.after(20, verifier)

    app.after(1, verifier)
    app.mainloop()
    assert all(t.terminee for t in taches), "préchargement trop long"


def fermer(app: Application) -> None:
    """Ferme la fenêtre puis libère ses objets Tk dans le fil principal (jamais dans un fil de calcul)."""
    app.destroy()
    gc.collect()


def capturer(app: Application, nom: str) -> None:
    """Capture de la fenêtre dans le dossier désigné par MECAML_CAPTURES (sinon, ne fait rien)."""
    if not DOSSIER_CAPTURES:
        return
    import ctypes

    from PIL import ImageGrab

    app.update()
    time.sleep(0.3)
    app.update()
    # Contenu de la fenêtre elle-même (PrintWindow), jamais celui d'une autre fenêtre qui la couvrirait.
    identifiant = app.winfo_id()
    fenetre = ctypes.windll.user32.GetParent(identifiant) or identifiant
    image = ImageGrab.grab(window=fenetre)
    dossier = Path(DOSSIER_CAPTURES)
    dossier.mkdir(parents=True, exist_ok=True)
    image.save(dossier / f"{nom}.png")


def mener_jusqu_a(
    app: Application,
    chemin: Path,
    etat: Etat,
    *,
    cible: str | None = None,
    sans_cible: bool = False,
    modeles: Sequence[ChoixTechnique] | None = None,
    pretraitement: Sequence[ChoixTechnique] | None = None,
    acquitter: bool = True,
    prefixe: str = "",
    avant_objectif: Callable[[Application], None] | None = None,
    apres_etape: Callable[[Etat], None] | None = None,
) -> Etat:
    """Parcours d'un utilisateur par les gestes de l'interface, jusqu'à l'état demandé ; rend l'état atteint.

    Les écrans reçoivent les mêmes appels que leurs boutons ; `modeles` et `pretraitement` remplacent la
    proposition par défaut comme le ferait un réglage dans la liste.
    """
    ecrans = app.ecrans

    def atteint(minimum: Etat) -> None:
        """L'étape précédente a bien eu lieu ; sinon le motif affiché par l'interface est celui du refus."""
        assert app.ctrl.etat >= minimum, (
            f"état {app.ctrl.etat.name} au lieu de {minimum.name} : {app.banniere.texte}"
        )
        if apres_etape is not None:
            apres_etape(app.ctrl.etat)

    app.afficher(0)
    capturer(app, f"{prefixe}s0_accueil")
    ecrans[0].ouvrir(chemin)
    attendre(app)
    atteint(Etat.DONNEES)
    capturer(app, f"{prefixe}s1_donnees")
    if etat <= Etat.DONNEES:
        return app.ctrl.etat
    ecrans[1].auditer()
    attendre(app)
    atteint(Etat.AUDIT)
    capturer(app, f"{prefixe}s2_audit")
    if etat <= Etat.AUDIT:
        return app.ctrl.etat
    ecrans[2].continuer()
    attendre(app)
    if sans_cible:
        ecrans[3].var_probleme.set("sans_cible")
        ecrans[3].changer_probleme()
        attendre(app)
    elif cible is not None:
        ecrans[3].var_cible.set(cible)
        ecrans[3].proposer(cible)
        attendre(app)
    if avant_objectif is not None:
        avant_objectif(app)
    ecrans[3].fixer()
    attendre(app)
    atteint(Etat.OBJECTIF)
    capturer(app, f"{prefixe}s3_objectif")
    if etat <= Etat.OBJECTIF:
        return app.ctrl.etat
    app.afficher(4)
    ecrans[4].sceller()
    attendre(app)
    atteint(Etat.DECOUPE)
    capturer(app, f"{prefixe}s4_decoupage")
    if acquitter:
        app.acquitter(frozenset(c.code for c in app.ctrl.session.alertes_en_attente()))
    if etat <= Etat.DECOUPE:
        return app.ctrl.etat
    app.afficher(5)
    attendre(app)
    if pretraitement is not None:
        ecrans[5].liste.definir(pretraitement)
    capturer(app, f"{prefixe}s5_pretraitement")
    ecrans[5].appliquer()
    attendre(app)
    atteint(Etat.PRETRAITEMENT)
    if etat <= Etat.PRETRAITEMENT:
        return app.ctrl.etat
    app.afficher(6)
    attendre(app)
    if modeles is not None:
        ecrans[6].liste.definir(modeles)
    ecrans[6].entrainer()
    attendre(app)
    atteint(Etat.MODELES)
    capturer(app, f"{prefixe}s6_modeles")
    if etat <= Etat.MODELES:
        return app.ctrl.etat
    app.afficher(7)
    attendre(app)
    capturer(app, f"{prefixe}s7_diagnostic")
    ecrans[7].retenir()
    attendre(app)
    atteint(Etat.DIAGNOSTIC)
    if etat <= Etat.DIAGNOSTIC or app.ctrl.session.sans_cible:
        return app.ctrl.etat
    app.afficher(8)
    ecrans[8].ouvrir()
    attendre(app)
    atteint(Etat.TESTE)
    capturer(app, f"{prefixe}s8_test")
    return app.ctrl.etat


def exporter(app: Application, dossier: Path, prefixe: str = "") -> list[str]:
    """Export par l'écran S9 ; rend les noms des fichiers écrits."""
    app.afficher(9)
    app.ecrans[9].exporter(dossier)
    attendre(app)
    capturer(app, f"{prefixe}s9_export")
    return sorted(p.name for p in Path(dossier).iterdir())


@dataclass
class Instantane:
    """État du contrôleur à un instant du parcours, pour rouvrir un écran dans l'état voulu."""

    session: Any
    chargement: Any
    chargement_test: Any
    profils: Any
    decoupage: Any
    vues: dict[str, Any]
    export: Any


def prendre(app: Application) -> Instantane:
    c = app.ctrl
    return Instantane(
        c.session, c.chargement, c.chargement_test, c.profils, c.decoupage, dict(c.vues), c.export
    )


def restaurer(app: Application, instantane: Instantane) -> None:
    c = app.ctrl
    c.session, c.chargement, c.chargement_test = (
        instantane.session,
        instantane.chargement,
        instantane.chargement_test,
    )
    c.recherches = []
    c.profils, c.decoupage, c.vues, c.export = (
        instantane.profils,
        instantane.decoupage,
        dict(instantane.vues),
        instantane.export,
    )
