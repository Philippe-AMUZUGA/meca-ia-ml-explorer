"""MécaML : écran d'accueil (état S0), principe de l'application et jeux d'exemple."""

from __future__ import annotations

from pathlib import Path
from tkinter import ttk

from mecaml import __version__
from mecaml.ui.ressources import exemples
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t
from mecaml.ui.widgets import carte

__all__ = ["Accueil"]


class Accueil(Ecran):
    """Premier écran : ce que fait l'application, ce qu'elle ne fait pas, et par où commencer."""

    NUMERO = 0
    CLE = "accueil"

    def construire(self, corps: ttk.Frame) -> None:
        for k, cle in enumerate(("ecran.accueil.principe", "ecran.accueil.confiance")):
            ttk.Label(corps, text=t(cle), wraplength=860, justify="left").grid(
                row=k, column=0, sticky="w", pady=(0, 6)
            )
        depart = carte(corps, t("ecran.accueil.depart.titre"))
        depart.grid(row=2, column=0, sticky="ew", pady=(4, 10))
        self.bouton(
            depart,
            "ecran.accueil.depart.choisir",
            lambda: self.app.afficher(1),
            principal=True,
            row=0,
            column=0,
            sticky="w",
            pady=(0, 8),
        )
        self.exemples = ttk.Frame(depart, style="Carte.TFrame")
        self.exemples.grid(row=1, column=0, sticky="ew")
        self._lister_exemples()
        ttk.Label(corps, text=t("ecran.accueil.version", version=__version__), style="Doux.TLabel").grid(
            row=3, column=0, sticky="w"
        )

    def _lister_exemples(self) -> None:
        classeurs = exemples()
        if not classeurs:
            ttk.Label(self.exemples, text=t("ecran.accueil.sans_exemple"), style="CarteDoux.TLabel").grid(
                row=0, column=0
            )
            return
        ttk.Label(self.exemples, text=t("ecran.accueil.exemples"), style="CarteGras.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        for k, chemin in enumerate(classeurs, start=1):
            bouton = ttk.Button(
                self.exemples,
                text=chemin.stem.removeprefix("MecaIA-"),
                command=lambda c=chemin: self.ouvrir(c),
            )
            bouton.grid(row=k, column=0, sticky="w", pady=2)
            self._boutons.append((bouton, lambda: True))

    def ouvrir(self, chemin: Path) -> None:
        """Ouvre un jeu d'exemple : l'écran des données le charge."""
        self.app.afficher(1)
        self.app.ecrans[1].charger(chemin)

    def rafraichir(self) -> None:
        self.mettre_a_jour_boutons()
