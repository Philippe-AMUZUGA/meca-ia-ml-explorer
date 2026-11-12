"""
MécaML : fenêtre principale.

Une colonne d'étapes à gauche, l'écran de l'étape au centre, un pied de page (état, progression,
annulation) comme l'application de référence. L'interface envoie des actions à la session et affiche ce
qu'elle rend ; tout calcul passe par un fil (`worker.Pilote`) et la boucle Tk n'attend jamais.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any

from mecaml import __version__
from mecaml.core.session import Etat
from mecaml.core.tasks import Annulation, Fonction, Progression, Tache, precharger_scikit_learn
from mecaml.ui import theme
from mecaml.ui.controleur import Controleur
from mecaml.ui.ressources import chemin_asset
from mecaml.ui.screens.accueil import Accueil
from mecaml.ui.screens.base import Ecran
from mecaml.ui.screens.s1_donnees import EcranDonnees
from mecaml.ui.screens.s2_audit import EcranAudit
from mecaml.ui.screens.s3_objectif import EcranObjectif
from mecaml.ui.screens.s4_decoupage import EcranDecoupage
from mecaml.ui.screens.s5_pretraitement import EcranPretraitement
from mecaml.ui.screens.s6_modeles import EcranModeles
from mecaml.ui.screens.s7_diagnostic import EcranDiagnostic
from mecaml.ui.screens.s8_test import EcranTest
from mecaml.ui.screens.s9_export import EcranExport
from mecaml.ui.textes import t
from mecaml.ui.widgets import Banniere
from mecaml.ui.worker import Pilote, message_d_erreur

__all__ = ["Application", "lancer"]

ECRANS: tuple[type[Ecran], ...] = (
    Accueil,
    EcranDonnees,
    EcranAudit,
    EcranObjectif,
    EcranDecoupage,
    EcranPretraitement,
    EcranModeles,
    EcranDiagnostic,
    EcranTest,
    EcranExport,
)


class Application(tk.Tk):
    """Fenêtre de MécaML Explorer."""

    def __init__(
        self,
        controleur: Controleur | None = None,
        *,
        confirmer: Callable[[str], bool] | None = None,
        precharger: bool = True,
    ) -> None:
        super().__init__()
        self.title(t("app.titre"))
        self.configure(bg=theme.BG)
        self.resizable(True, True)
        largeur = min(1180, self.winfo_screenwidth() - 20)
        hauteur = min(740, self.winfo_screenheight() - 110)
        self.geometry(f"{largeur}x{hauteur}+0+0")
        self.minsize(min(960, largeur), min(560, hauteur))
        theme.appliquer(self)
        self.ctrl = controleur or Controleur()
        self._confirmer = confirmer or (lambda texte: messagebox.askokcancel(t("app.titre"), texte))
        self.statut = tk.StringVar(value=t("pied.pret"))
        self.progression = tk.DoubleVar(value=0.0)
        self.pilote = Pilote(self, self._montrer_progression, self._basculer)
        self.ecrans: dict[int, Ecran] = {}
        self._rail: dict[int, ttk.Button] = {}
        self._courant = 0
        self._icone()
        self._construire()
        self.afficher(0)
        self.prechargement = precharger_scikit_learn() if precharger else None
        self.prechauffage = (
            Tache(_chauffer_les_graphiques, nom="graphiques").demarrer() if precharger else None
        )

    # Construction
    def _icone(self) -> None:
        png = chemin_asset("mecaml_icone.png")
        try:
            if png is not None:
                self._image_icone = tk.PhotoImage(file=str(png))
                self.iconphoto(True, self._image_icone)
        except tk.TclError:
            pass

    def _construire(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        externe = ttk.Frame(self, padding=(16, 14, 16, 8))
        externe.grid(row=0, column=0, sticky="nsew")
        externe.grid_rowconfigure(1, weight=1)
        externe.grid_columnconfigure(1, weight=1)
        ttk.Label(externe, text=t("app.nom"), style="Titre.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self._rail_etapes(externe).grid(row=1, column=0, sticky="ns", padx=(0, 14), pady=(10, 0))
        droite = ttk.Frame(externe)
        droite.grid(row=1, column=1, sticky="nsew", pady=(10, 0))
        droite.grid_rowconfigure(1, weight=1)
        droite.grid_columnconfigure(0, weight=1)
        self.banniere = Banniere(droite)
        self.banniere.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.banniere.masquer()  # invisible tant qu'aucun message ne s'affiche
        zone = ttk.Frame(droite)
        zone.grid(row=1, column=0, sticky="nsew")
        zone.grid_rowconfigure(0, weight=1)
        zone.grid_columnconfigure(0, weight=1)
        for numero, classe in enumerate(ECRANS):
            ecran = classe(zone, self)
            ecran.grid(row=0, column=0, sticky="nsew")
            self.ecrans[numero] = ecran
        self._pied(externe).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _rail_etapes(self, parent: tk.Misc) -> ttk.Frame:
        rail = ttk.Frame(parent)
        for numero in range(len(ECRANS)):
            cle = "rail.accueil" if numero == 0 else f"rail.s{numero}"
            bouton = ttk.Button(
                rail, text=t(cle), style="Rail.TButton", width=22, command=lambda n=numero: self.afficher(n)
            )
            bouton.grid(row=numero, column=0, sticky="ew", pady=2)
            self._rail[numero] = bouton
        return rail

    def _pied(self, parent: tk.Misc) -> ttk.Frame:
        pied = ttk.Frame(parent)
        pied.grid_columnconfigure(0, weight=1)
        ttk.Separator(pied).grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Label(pied, textvariable=self.statut).grid(row=1, column=0, sticky="w")
        self.barre = ttk.Progressbar(pied, variable=self.progression, maximum=100, length=220)
        self.barre.grid(row=1, column=1, padx=10)
        self.bouton_annuler = ttk.Button(
            pied, text=t("pied.annuler"), command=self.pilote.annuler, state="disabled"
        )
        self.bouton_annuler.grid(row=1, column=2)
        ttk.Label(pied, text=t("pied.credit", version=__version__), style="Doux.TLabel").grid(
            row=1, column=3, padx=(14, 0)
        )
        return pied

    # Navigation
    def peut_afficher(self, numero: int) -> bool:
        """Vrai si l'étape est atteinte ou prochaine : la session en décide."""
        etat = self.ctrl.etat
        if numero == 0:
            return True
        if numero == int(Etat.EXPORTE):
            return etat >= Etat.TESTE or (self.ctrl.session.sans_cible and etat >= Etat.DIAGNOSTIC)
        return numero <= int(etat) + 1

    def afficher(self, numero: int) -> None:
        """Montre l'écran d'une étape atteinte ; la session est relue."""
        if not self.peut_afficher(numero):
            return
        self._courant = numero
        ecran = self.ecrans[numero]
        ecran.tkraise()
        ecran.rafraichir()
        self.mettre_a_jour_rail()
        ecran.mettre_a_jour_boutons()

    @property
    def courant(self) -> int:
        return self._courant

    def suivant(self) -> None:
        """Ouvre l'écran de l'étape qui suit l'état de la session."""
        numero = int(self.ctrl.etat) + 1
        if numero == int(Etat.TESTE) and self.ctrl.session.sans_cible:
            numero = int(Etat.EXPORTE)
        self.afficher(min(numero, len(ECRANS) - 1))

    def mettre_a_jour_rail(self) -> None:
        for numero, bouton in self._rail.items():
            actif = numero == self._courant
            bouton.configure(
                style="RailActif.TButton" if actif else "Rail.TButton",
                state="normal" if self.peut_afficher(numero) else "disabled",
            )

    # Calcul
    @property
    def occupe(self) -> bool:
        return self.pilote.occupe

    def _basculer(self, occupe: bool) -> None:
        self.configure(cursor="watch" if occupe else "")
        self.bouton_annuler.configure(state="normal" if occupe else "disabled")
        for ecran in self.ecrans.values():
            ecran.basculer(occupe)

    def _montrer_progression(self, fraction: float, message: str) -> None:
        self.progression.set(100.0 * fraction)
        if message:
            self.statut.set(message)
        elif not self.pilote.occupe:
            self.statut.set(t("pied.pret"))

    def executer(
        self,
        fonction: Fonction[Any],
        *,
        nom: str,
        au_succes: Callable[[Any], None],
        en_echec: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Lance un calcul dans un fil ; le résultat revient dans Tk, l'erreur s'affiche dans le bandeau."""
        self.banniere.masquer()

        def reussi(valeur: Any) -> None:
            au_succes(valeur)
            self.mettre_a_jour_rail()  # l'état a pu avancer : l'étape suivante se dégrise

        self.pilote.lancer(
            fonction,
            nom=nom,
            au_succes=reussi,
            en_echec=en_echec or (lambda erreur: self.erreur(erreur, etape=nom)),
            si_annulee=lambda: self.statut.set(t("pied.annule")),
        )

    def erreur(self, erreur: BaseException | str, *, etape: str | None = None) -> None:
        """Affiche le motif d'un refus du cœur tel quel ; une erreur de bibliothèque, en français."""
        texte = erreur if isinstance(erreur, str) else message_d_erreur(erreur, etape)
        self.banniere.afficher(texte, "bloquant")
        self.statut.set(texte[:140])

    def info(self, texte: str) -> None:
        self.banniere.afficher(texte, "bon")

    def confirmer(self, texte: str) -> bool:
        return bool(self._confirmer(texte))

    def acquitter(self, codes: frozenset[str]) -> None:
        """Acquitte des alertes (une action rapide de la session) et relit l'écran."""
        if not codes:
            return
        try:
            self.ctrl.acquitter(codes)
        except Exception as erreur:
            self.erreur(erreur)
            return
        self.afficher(self._courant)

    def fermer(self) -> None:
        self.pilote.annuler()
        self.destroy()


def _chauffer_les_graphiques(annulation: Annulation, avancer: Callable[[Progression], None]) -> None:
    """Importe matplotlib et dessine une figure d'essai à l'accueil : le premier tracé est rapide."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
    from matplotlib.figure import Figure

    annulation.verifier()
    figure = Figure(figsize=(4, 2), layout="constrained")
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1])
    axes.set_title("essai")
    FigureCanvasAgg(figure).draw()


def lancer() -> None:
    """Ouvre l'application (point d'entrée de `python -m mecaml.ui`)."""
    if sys.platform == "win32":
        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.fermer)
    app.mainloop()
