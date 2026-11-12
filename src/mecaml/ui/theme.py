"""
MécaML : thème de l'interface (palette, polices, styles ttk).

Le style est celui des applications de bureau de l'auteur : fond gris bleuté, cartes blanches à filet
fin, accent bleu acier, polices Segoe UI et Consolas, thème ttk « clam ».
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

__all__ = [
    "ACCENT",
    "ACCENT_FONCE",
    "BG",
    "BORDURE",
    "CARTE",
    "COULEURS_NIVEAU",
    "MONO",
    "POLICE",
    "TEXTE",
    "TEXTE_DOUX",
    "appliquer",
]

BG = "#EEF2F7"
CARTE = "#FFFFFF"
ACCENT = "#335C81"
ACCENT_FONCE = "#264A68"
TEXTE = "#18212F"
TEXTE_DOUX = "#5C6778"
BORDURE = "#D6DDE6"
ENTETE = "#E8EEF5"
SELECTION = "#CFE0F0"
ROUGE = "#B3261E"
ORANGE = "#9A5B00"
VERT = "#2E7D32"

POLICE = "Segoe UI"
POLICE_GRAS = "Segoe UI Semibold"
MONO = "Consolas"

# Couleur du texte d'un constat selon son niveau (valeurs de `Niveau`).
COULEURS_NIVEAU = {"bloquant": ROUGE, "alerte": ORANGE, "info": TEXTE_DOUX}


def appliquer(racine: tk.Misc) -> ttk.Style:
    """Installe le thème sur la fenêtre et rend le style configuré."""
    style = ttk.Style(racine)
    with contextlib.suppress(tk.TclError):
        style.theme_use("clam")
    _polices(racine)
    racine.option_add("*TCombobox*Listbox.font", f"{{{POLICE}}} 10")
    _cadres(style)
    _boutons(style)
    _tableaux(style)
    _champs(style)
    return style


def _polices(racine: tk.Misc) -> None:
    """Police des composants par les polices nommées de Tk, jamais par l'option `*Font`.

    L'option `*Font` fixe le `-font` de chaque composant ttk et l'emporte alors sur la police d'un style :
    les titres et les libellés en gras resteraient à la taille courante.
    """
    for nom in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkIconFont", "TkTooltipFont"):
        with contextlib.suppress(tk.TclError):
            tkfont.nametofont(nom, root=racine).configure(family=POLICE, size=10)


def _cadres(style: ttk.Style) -> None:
    style.configure("TFrame", background=BG)
    style.configure("Carte.TFrame", background=CARTE)
    style.configure("TLabel", background=BG, foreground=TEXTE)
    style.configure("Doux.TLabel", background=BG, foreground=TEXTE_DOUX)
    style.configure("Titre.TLabel", background=BG, foreground=TEXTE, font=(POLICE_GRAS, 20))
    style.configure("Sous.TLabel", background=BG, foreground=TEXTE_DOUX, font=(POLICE, 11))
    style.configure("Carte.TLabel", background=CARTE, foreground=TEXTE)
    style.configure("CarteDoux.TLabel", background=CARTE, foreground=TEXTE_DOUX)
    style.configure("CarteGras.TLabel", background=CARTE, foreground=TEXTE, font=(POLICE_GRAS, 10))
    style.configure("Bloquant.TLabel", background=BG, foreground=ROUGE, font=(POLICE_GRAS, 10))
    style.configure("Alerte.TLabel", background=BG, foreground=ORANGE, font=(POLICE_GRAS, 10))
    style.configure("CarteAlerte.TLabel", background=CARTE, foreground=ORANGE, font=(POLICE_GRAS, 10))
    style.configure("Bon.TLabel", background=BG, foreground=VERT, font=(POLICE_GRAS, 10))
    style.configure("Carte.TLabelframe", background=CARTE, bordercolor=BORDURE, relief="solid")
    style.configure("Carte.TLabelframe.Label", background=CARTE, foreground=TEXTE, font=(POLICE_GRAS, 11))
    style.configure("TSeparator", background=BORDURE)
    style.configure("TNotebook", background=BG, bordercolor=BORDURE)
    style.configure("TNotebook.Tab", background=ENTETE, foreground=TEXTE, padding=(12, 6))
    style.map("TNotebook.Tab", background=[("selected", CARTE)])
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=ENTETE, bordercolor=BORDURE)


def _boutons(style: ttk.Style) -> None:
    style.configure("TButton", font=(POLICE_GRAS, 10), padding=(10, 6))
    style.configure("Accent.TButton", background=ACCENT, foreground="#FFFFFF")
    style.map(
        "Accent.TButton",
        background=[("disabled", "#9DB2C5"), ("pressed", ACCENT_FONCE), ("active", ACCENT_FONCE)],
        foreground=[("disabled", "#F2F5F8")],
    )
    style.configure("Rail.TButton", font=(POLICE, 10), padding=(10, 8), anchor="w", background=ENTETE)
    style.map("Rail.TButton", background=[("active", SELECTION)], foreground=[("disabled", "#9AA5B4")])
    style.configure("RailActif.TButton", font=(POLICE_GRAS, 10), padding=(10, 8), anchor="w")
    style.configure("RailActif.TButton", background=ACCENT, foreground="#FFFFFF")
    style.map("RailActif.TButton", background=[("active", ACCENT_FONCE)])
    style.configure("TCheckbutton", background=CARTE, foreground=TEXTE)
    style.configure("TRadiobutton", background=CARTE, foreground=TEXTE)


def _tableaux(style: ttk.Style) -> None:
    style.configure("Treeview", background=CARTE, fieldbackground=CARTE, foreground=TEXTE, rowheight=26)
    style.map("Treeview", background=[("selected", SELECTION)], foreground=[("selected", TEXTE)])
    style.configure("Treeview.Heading", background=ENTETE, foreground=TEXTE, font=(POLICE_GRAS, 10))


def _champs(style: ttk.Style) -> None:
    style.configure("TEntry", fieldbackground=CARTE, bordercolor=BORDURE)
    style.configure("TCombobox", fieldbackground=CARTE, bordercolor=BORDURE)
    style.configure("TSpinbox", fieldbackground=CARTE, bordercolor=BORDURE)
