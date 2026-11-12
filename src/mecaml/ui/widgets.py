"""MécaML : composants graphiques partagés par les écrans (aucune règle de calcul ici)."""

from __future__ import annotations

import contextlib
import math
import tkinter as tk
from collections.abc import Callable, Iterable, Sequence
from tkinter import font as tkfont
from tkinter import ttk
from typing import Any

from mecaml.core.guards import decimale
from mecaml.core.livre import TECHNIQUES_LIVRE
from mecaml.core.types import Constat, Niveau
from mecaml.ui import theme
from mecaml.ui.textes import t

__all__ = [
    "PARTS_DE_TEST",
    "Banniere",
    "Colonne",
    "PanneauConstats",
    "Planche",
    "ScrollableFrame",
    "Tableau",
    "ZoneFigure",
    "ZoneTexte",
    "carte",
    "champ",
    "lire_decimal",
    "nombre",
    "renvoi_livre",
]

Colonne = tuple[str, str, int, str]  # (clé, titre, largeur en pixels, ancre)

PARTS_DE_TEST = tuple(decimale(centieme / 100, 2) for centieme in range(5, 55, 5))  # « 0,05 » à « 0,50 »


def lire_decimal(texte: str) -> float:
    """Nombre saisi avec une virgule ou un point décimal ; `ValueError` sinon."""
    return float(texte.strip().replace(",", "."))


def nombre(valeur: Any, chiffres: int = 3) -> str:
    """Nombre à la française (virgule), `n.d.` s'il est indéfini."""
    if valeur is None or (isinstance(valeur, float) and math.isnan(valeur)):
        return t("commun.indefini")
    if isinstance(valeur, bool):
        return t("commun.oui") if valeur else t("commun.non")
    if isinstance(valeur, int):
        return str(valeur)
    try:
        return decimale(float(valeur), chiffres)
    except (TypeError, ValueError):
        return str(valeur)


def renvoi_livre(identifiant: str | None) -> str:
    """Où le lecteur trouve la notion dans le livre : « chapitre 2, section 2.1 »."""
    technique = TECHNIQUES_LIVRE.get(identifiant) if identifiant else None
    if technique is None:
        return ""
    if "sans fiche" in technique.libelle:  # le livre n'imprime aucune carte pour cet identifiant
        return t("commun.renvoi_sans_fiche", chapitre=technique.chapitre, section=technique.section)
    return t("commun.renvoi", chapitre=technique.chapitre, section=technique.section, id=technique.id)


def carte(parent: tk.Misc, titre: str, **options: Any) -> ttk.LabelFrame:
    """Carte à filet fin, comme les cadres de l'application de référence."""
    return ttk.LabelFrame(
        parent, text=titre, style="Carte.TLabelframe", padding=options.pop("padding", 12), **options
    )


def champ(parent: tk.Misc, ligne: int, libelle: str, widget: tk.Widget, colonne: int = 0) -> None:
    """Pose un libellé et son champ sur une ligne de grille d'une carte."""
    ttk.Label(parent, text=libelle, style="CarteDoux.TLabel").grid(
        row=ligne, column=colonne, sticky="w", padx=(0, 10), pady=3
    )
    widget.grid(row=ligne, column=colonne + 1, sticky="ew", pady=3)


class ScrollableFrame(ttk.Frame):
    """Zone à défilement vertical : le contenu se place dans `interieur`."""

    def __init__(self, parent: tk.Misc, **options: Any) -> None:
        super().__init__(parent, **options)
        self.toile = tk.Canvas(self, bg=theme.BG, highlightthickness=0)
        self.barre = ttk.Scrollbar(self, orient="vertical", command=self.toile.yview)
        self.toile.configure(yscrollcommand=self.barre.set)
        self.interieur = ttk.Frame(self.toile)
        self._fenetre = self.toile.create_window((0, 0), window=self.interieur, anchor="nw")
        self.toile.grid(row=0, column=0, sticky="nsew")
        self.barre.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.interieur.bind(
            "<Configure>", lambda _e: self.toile.configure(scrollregion=self.toile.bbox("all"))
        )
        self.toile.bind("<Configure>", lambda e: self.toile.itemconfigure(self._fenetre, width=e.width))
        self.toile.bind("<Enter>", lambda _e: self.toile.bind_all("<MouseWheel>", self._molette))
        self.toile.bind("<Leave>", lambda _e: self.toile.unbind_all("<MouseWheel>"))

    def _molette(self, evenement: tk.Event) -> None:
        with contextlib.suppress(tk.TclError):
            self.toile.yview_scroll(int(-1 * (evenement.delta / 120)), "units")


def _police_des_entetes() -> tkfont.Font | None:
    """Police des en-têtes de tableau, pour mesurer les titres ; None si elle ne se lit pas."""
    try:
        nom = ttk.Style().lookup("Treeview.Heading", "font") or "TkHeadingFont"
        return tkfont.nametofont(nom) if nom in tkfont.names() else tkfont.Font(font=nom)
    except tk.TclError:
        return None


class Tableau(ttk.Frame):
    """Tableau (Treeview) à barres de défilement, rempli par des lignes de textes."""

    def __init__(
        self, parent: tk.Misc, colonnes: Sequence[Colonne], hauteur: int = 8, selection: str = "extended"
    ):
        super().__init__(parent)
        self.cles = [c[0] for c in colonnes]
        self.arbre = ttk.Treeview(
            self, columns=self.cles, show="headings", selectmode=selection, height=hauteur
        )
        police = _police_des_entetes()
        for rang, (cle, titre, largeur, ancre) in enumerate(colonnes):
            self.arbre.heading(cle, text=titre)
            if police is not None:
                largeur = max(largeur, police.measure(titre) + 32)  # un en-tête n'est jamais tronqué
            # seule la dernière colonne s'étire : les autres gardent leur largeur, le défilement fait le reste
            self.arbre.column(
                cle, width=largeur, anchor=ancre, stretch=rang == len(colonnes) - 1, minwidth=40
            )
        vertical = ttk.Scrollbar(self, orient="vertical", command=self.arbre.yview)
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.arbre.xview)
        self.arbre.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.arbre.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def vider(self) -> None:
        self.arbre.delete(*self.arbre.get_children())

    def ajouter(self, valeurs: Iterable[Any], iid: str | None = None, etiquettes: Sequence[str] = ()) -> str:
        """Ajoute une ligne ; rend son identifiant."""
        textes = tuple("" if v is None else str(v) for v in valeurs)
        return self.arbre.insert("", "end", iid=iid, values=textes, tags=tuple(etiquettes))

    def remplir(
        self, lignes: Iterable[Iterable[Any]], etiquettes: Callable[[int], Sequence[str]] | None = None
    ) -> None:
        self.vider()
        for k, valeurs in enumerate(lignes):
            self.ajouter(valeurs, iid=str(k), etiquettes=etiquettes(k) if etiquettes else ())

    def selection(self) -> tuple[str, ...]:
        return tuple(self.arbre.selection())

    def lignes(self) -> list[tuple[str, ...]]:
        """Contenu affiché, ligne par ligne."""
        return [tuple(self.arbre.item(i, "values")) for i in self.arbre.get_children()]

    def colorier(self, etiquette: str, couleur: str) -> None:
        self.arbre.tag_configure(etiquette, foreground=couleur)

    def lier_selection(self, rappel: Callable[[tuple[str, ...]], None]) -> None:
        self.arbre.bind("<<TreeviewSelect>>", lambda _e: rappel(self.selection()))


class PanneauConstats(ttk.Frame):
    """Liste des constats de l'audit par niveau, avec le détail du constat choisi."""

    def __init__(self, parent: tk.Misc, hauteur: int = 6) -> None:
        super().__init__(parent)
        colonnes: list[Colonne] = [
            ("niveau", t("constats.niveau"), 90, "w"),
            ("message", t("constats.message"), 440, "w"),
            ("statut", t("constats.statut"), 90, "w"),
            ("livre", t("constats.livre"), 250, "w"),
        ]
        self.tableau = Tableau(self, colonnes, hauteur=hauteur)
        self.tableau.grid(row=0, column=0, sticky="nsew")
        for niveau, couleur in theme.COULEURS_NIVEAU.items():
            self.tableau.colorier(niveau, couleur)
        self.detail = ttk.Label(self, text="", style="Carte.TLabel", wraplength=760, justify="left")
        self.detail.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.grid_columnconfigure(0, weight=1)
        self._constats: dict[str, Constat] = {}
        self._acquittes: frozenset[str] = frozenset()
        self.tableau.lier_selection(self._montrer_detail)

    def afficher(self, constats: Sequence[Constat], acquittes: frozenset[str] = frozenset()) -> None:
        self.tableau.vider()
        self._constats = {c.code: c for c in constats}
        self._acquittes = acquittes
        ordre = {Niveau.BLOQUANT: 0, Niveau.ALERTE: 1, Niveau.INFO: 2}
        for c in sorted(constats, key=lambda c: ordre[c.niveau]):
            statut = t("constats.acquittee") if c.code in acquittes else ""
            self.tableau.ajouter(
                (t(f"niveau.{c.niveau.value}"), c.message, statut, renvoi_livre(c.ref_livre)),
                iid=c.code,
                etiquettes=(c.niveau.value,),
            )
        self.detail.configure(text=t("constats.aucun") if not constats else t("constats.choisir"))

    def codes(self) -> tuple[str, ...]:
        """Codes des constats affichés, du plus grave au plus léger."""
        return tuple(self.tableau.arbre.get_children())

    def codes_alertes(self, tout: bool = False) -> frozenset[str]:
        """Codes des alertes choisies (ou de toutes les alertes en attente), jamais ceux des bloquants."""
        codes = list(self._constats) if tout else list(self.tableau.selection())
        return frozenset(
            c for c in codes if self._constats[c].niveau is Niveau.ALERTE and c not in self._acquittes
        )

    def _montrer_detail(self, choix: tuple[str, ...]) -> None:
        if not choix:
            return
        constat = self._constats[choix[0]]
        texte = constat.message + (f"\n{t('constats.action')} {constat.action}" if constat.action else "")
        livre = renvoi_livre(constat.ref_livre)
        if livre:
            texte += "\n" + t("commun.voir_livre", renvoi=livre) + "."
        self.detail.configure(text=texte)


class ZoneFigure(ttk.Frame):
    """Zone d'une figure matplotlib (construite par le cœur, sans moteur graphique)."""

    def __init__(self, parent: tk.Misc, largeur: int = 760, hauteur: int = 320) -> None:
        super().__init__(parent, style="Carte.TFrame")
        self._taille = (largeur, hauteur)
        self._toile: Any = None
        self.figure: Any = None
        self.grid_columnconfigure(0, weight=1)

    def afficher(self, figure: Any) -> None:
        """Remplace la figure affichée ; l'ancienne est libérée."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.effacer()
        self.figure = figure
        self._toile = FigureCanvasTkAgg(figure, master=self)
        graphique = self._toile.get_tk_widget()
        graphique.configure(width=self._taille[0], height=self._taille[1])
        graphique.grid(row=0, column=0, sticky="nsew")
        self._toile.draw()

    def effacer(self) -> None:
        if self._toile is not None:
            self._toile.get_tk_widget().destroy()
        self._toile = None
        self.figure = None


class ZoneTexte(ttk.Frame):
    """Texte en lecture seule, police à chasse fixe (arbre, script, mesures)."""

    def __init__(self, parent: tk.Misc, hauteur: int = 10) -> None:
        super().__init__(parent)
        self.texte = tk.Text(
            self,
            height=hauteur,
            wrap="word",
            font=(theme.MONO, 10),
            bg=theme.CARTE,
            relief="solid",
            borderwidth=1,
        )
        barre = ttk.Scrollbar(self, orient="vertical", command=self.texte.yview)
        self.texte.configure(yscrollcommand=barre.set, state="disabled")
        self.texte.grid(row=0, column=0, sticky="nsew")
        barre.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def definir(self, contenu: str) -> None:
        self.texte.configure(state="normal")
        self.texte.delete("1.0", "end")
        self.texte.insert("1.0", contenu)
        self.texte.configure(state="disabled")

    def lire(self) -> str:
        return self.texte.get("1.0", "end").strip()


class Banniere(ttk.Frame):
    """Bandeau de message en tête d'écran : erreur, avertissement ou information, avec fermeture."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.message = ttk.Label(self, text="", style="Bloquant.TLabel", wraplength=900, justify="left")
        self.message.grid(row=0, column=0, sticky="ew")
        ttk.Button(self, text=t("commun.fermer"), command=self.masquer).grid(row=0, column=1, padx=(10, 0))
        self.grid_columnconfigure(0, weight=1)
        self.grid_remove()

    def afficher(self, texte: str, niveau: str = "bloquant") -> None:
        style = {"bloquant": "Bloquant.TLabel", "alerte": "Alerte.TLabel", "bon": "Bon.TLabel"}.get(
            niveau, "TLabel"
        )
        self.message.configure(text=texte, style=style)
        self.grid()

    def masquer(self) -> None:
        self.message.configure(text="")
        self.grid_remove()

    @property
    def texte(self) -> str:
        return str(self.message.cget("text"))


class Planche(ttk.Frame):
    """Planche de lecture : figures, tableaux et textes d'une `Vue` calculée par le cœur, empilés."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style="Carte.TFrame")
        self.grid_columnconfigure(0, weight=1)
        self.figures: list[ZoneFigure] = []
        self.tableaux: list[Tableau] = []
        self.textes: list[ZoneTexte] = []

    def vider(self) -> None:
        for enfant in self.winfo_children():
            enfant.destroy()
        self.figures, self.tableaux, self.textes = [], [], []

    def afficher(self, vue: Any, titres: dict[str, str] | None = None) -> None:
        """Remplace le contenu par les figures, tableaux et textes de la vue."""
        self.vider()
        titres = titres or {}
        ligne = 0
        for nom, figure in vue.figures:
            ttk.Label(self, text=titres.get(nom, nom), style="CarteGras.TLabel").grid(
                row=ligne, column=0, sticky="w"
            )
            zone = ZoneFigure(self, hauteur=300)
            zone.grid(row=ligne + 1, column=0, sticky="ew", pady=(2, 10))
            zone.afficher(figure)
            self.figures.append(zone)
            ligne += 2
        for nom, donnees in vue.tableaux:
            ttk.Label(self, text=titres.get(nom, nom), style="CarteGras.TLabel").grid(
                row=ligne, column=0, sticky="w"
            )
            tableau = Tableau(
                self, [(str(c), str(c), 110, "w") for c in donnees.columns], hauteur=min(8, len(donnees) + 1)
            )
            tableau.remplir([_cellule(v) for v in ligne_] for ligne_ in donnees.itertuples(index=False))
            tableau.grid(row=ligne + 1, column=0, sticky="ew", pady=(2, 10))
            self.tableaux.append(tableau)
            ligne += 2
        for nom, contenu in vue.textes:
            ttk.Label(self, text=titres.get(nom, nom), style="CarteGras.TLabel").grid(
                row=ligne, column=0, sticky="w"
            )
            zone_texte = ZoneTexte(self, hauteur=10)
            zone_texte.definir(contenu)
            zone_texte.grid(row=ligne + 1, column=0, sticky="ew", pady=(2, 10))
            self.textes.append(zone_texte)
            ligne += 2


def _cellule(valeur: Any) -> str:
    if isinstance(valeur, float):
        return nombre(valeur, 4)
    return "" if valeur is None else str(valeur)
