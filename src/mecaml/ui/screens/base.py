"""MécaML : socle commun des écrans (en-tête, aide de l'étape, corps défilant, boutons gérés)."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from mecaml.core.session import Session
from mecaml.ui.controleur import Controleur
from mecaml.ui.textes import t
from mecaml.ui.widgets import ScrollableFrame

if TYPE_CHECKING:
    from mecaml.ui.app import Application

__all__ = ["Ecran"]


class Ecran(ttk.Frame):
    """Un écran de l'application : relit la session à chaque affichage, n'applique aucune règle."""

    NUMERO = 0
    CLE = "accueil"

    def __init__(self, parent: tk.Misc, app: Application) -> None:
        super().__init__(parent)
        self.app = app
        self._boutons: list[tuple[ttk.Button, Callable[[], bool]]] = []
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        ttk.Label(self, text=t(f"rail.{self.CLE}"), style="Titre.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.defilement = ScrollableFrame(self)
        self.defilement.grid(row=1, column=0, sticky="nsew")
        self.corps = self.defilement.interieur
        self.corps.grid_columnconfigure(0, weight=1)
        self.construire(self.corps)

    # À écrire par chaque écran
    def construire(self, corps: ttk.Frame) -> None:
        """Pose les composants de l'écran."""

    def rafraichir(self) -> None:
        """Relit la session et remplit l'écran."""

    # Accès
    @property
    def ctrl(self) -> Controleur:
        return self.app.ctrl

    @property
    def session(self) -> Session:
        return self.app.ctrl.session

    # Boutons
    def bouton(
        self,
        parent: tk.Misc,
        cle: str,
        commande: Callable[[], Any],
        *,
        principal: bool = False,
        condition: Callable[[], bool] = lambda: True,
        **grille: Any,
    ) -> ttk.Button:
        """Bouton actif seulement si aucun calcul ne tourne et si la condition est vraie."""
        bouton = ttk.Button(
            parent, text=t(cle), command=commande, style="Accent.TButton" if principal else "TButton"
        )
        if grille:
            bouton.grid(**grille)
        self._boutons.append((bouton, condition))
        return bouton

    def basculer(self, occupe: bool) -> None:
        """Active ou grise les boutons pendant un calcul."""
        for bouton, condition in self._boutons:
            bouton.configure(state="disabled" if occupe or not condition() else "normal")

    def mettre_a_jour_boutons(self) -> None:
        self.basculer(self.app.occupe)
