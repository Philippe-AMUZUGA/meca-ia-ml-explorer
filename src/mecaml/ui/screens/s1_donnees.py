"""MécaML : écran S1, chargement du fichier, aperçu, bornes et mode challenge."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import pandas as pd

from mecaml.core import io as fichiers
from mecaml.core.guards import (
    COLONNES_MAX,
    EXTENSIONS_ACCEPTEES,
    LIGNES_MAX,
    ErreurDonnees,
    nombre_lisible,
)
from mecaml.core.session import Etat
from mecaml.ui.controleur import Chargee
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t
from mecaml.ui.widgets import Tableau, carte, champ, nombre

__all__ = ["EcranDonnees"]

COLONNES_APERCU = 24
LIGNES_APERCU = 100


class EcranDonnees(Ecran):
    """Choix du fichier, lecture dans un fil, aperçu de la table lue."""

    NUMERO = 1
    CLE = "s1"

    def construire(self, corps: ttk.Frame) -> None:
        self.chemin = tk.StringVar()
        self.feuille = tk.StringVar()
        self.chemin_test = tk.StringVar()
        self.challenge = tk.BooleanVar(value=True)
        self._feuilles_challenge: tuple[str, str] | None = None
        self._carte_fichier(corps).grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._carte_apercu(corps).grid(row=1, column=0, sticky="ew")

    def _carte_fichier(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s1.fichier.titre"))
        cadre.grid_columnconfigure(1, weight=1)
        ligne = ttk.Frame(cadre, style="Carte.TFrame")
        ligne.grid_columnconfigure(0, weight=1)
        ttk.Entry(ligne, textvariable=self.chemin).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.bouton(ligne, "ecran.s1.parcourir", self.parcourir, row=0, column=1)
        champ(cadre, 0, t("ecran.s1.fichier.chemin"), ligne)
        self.liste_feuilles = ttk.Combobox(cadre, textvariable=self.feuille, state="readonly", width=30)
        champ(cadre, 1, t("ecran.s1.fichier.feuille"), self.liste_feuilles)
        self.case_challenge = ttk.Checkbutton(cadre, text=t("ecran.s1.challenge"), variable=self.challenge)
        self.case_challenge.grid(row=2, column=1, sticky="w")
        self.case_challenge.grid_remove()
        test = ttk.Frame(cadre, style="Carte.TFrame")
        test.grid_columnconfigure(0, weight=1)
        ttk.Entry(test, textvariable=self.chemin_test).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.bouton(test, "ecran.s1.parcourir", self.parcourir_test, row=0, column=1)
        champ(cadre, 3, t("ecran.s1.fichier.test"), test)
        ttk.Label(
            cadre,
            text=t("ecran.s1.bornes", lignes=nombre_lisible(LIGNES_MAX), colonnes=COLONNES_MAX),
            style="CarteDoux.TLabel",
        ).grid(row=4, column=1, sticky="w", pady=(6, 0))
        self.bouton(
            cadre, "ecran.s1.charger", self.charger, principal=True, row=5, column=1, sticky="w", pady=(8, 0)
        )
        return cadre

    def _carte_apercu(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s1.apercu.titre"))
        cadre.grid_columnconfigure(0, weight=1)
        self.resume = ttk.Label(
            cadre, text=t("ecran.s1.vide"), style="Carte.TLabel", wraplength=820, justify="left"
        )
        self.resume.grid(row=0, column=0, sticky="w")
        self.apercu = Tableau(cadre, [("c", "", 80, "w")], hauteur=9)
        self.apercu.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.notes = ttk.Label(cadre, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.notes.grid(row=2, column=0, sticky="w")
        self.bouton(
            cadre,
            "ecran.s1.auditer",
            self.auditer,
            principal=True,
            condition=lambda: self.session.etat >= Etat.DONNEES,
            row=3,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        return cadre

    # Choix des fichiers
    def parcourir(self) -> None:
        motifs = " ".join(f"*{e}" for e in EXTENSIONS_ACCEPTEES)
        chemin = filedialog.askopenfilename(
            title=t("ecran.s1.parcourir_titre"), filetypes=[(t("ecran.s1.type_donnees"), motifs)]
        )
        if chemin:
            self.choisir(chemin)

    def parcourir_test(self) -> None:
        chemin = filedialog.askopenfilename(
            title=t("ecran.s1.parcourir_test_titre"),
            filetypes=[(t("ecran.s1.type_donnees"), " ".join(f"*{e}" for e in EXTENSIONS_ACCEPTEES))],
        )
        if chemin:
            self.chemin_test.set(chemin)

    def choisir(self, chemin: str | Path) -> None:
        """Retient un fichier : ses feuilles se listent, le mode challenge se reconnaît à ses feuilles."""
        self.chemin.set(str(chemin))
        try:
            feuilles = fichiers.lister_feuilles(chemin)
        except ErreurDonnees as erreur:
            self.app.erreur(erreur)
            return
        self.liste_feuilles.configure(values=feuilles)
        self.feuille.set(feuilles[0] if feuilles else "")
        self._feuilles_challenge = fichiers.feuilles_challenge(feuilles)
        if self._feuilles_challenge:
            self.case_challenge.grid()
        else:
            self.case_challenge.grid_remove()

    # Lecture
    def charger(self, chemin: str | Path | None = None) -> None:
        """Lit le fichier dans un fil, avec progression et annulation."""
        if chemin is not None:
            self.choisir(chemin)
        if not self.chemin.get():
            self.app.erreur(t("ecran.s1.aucun_fichier"))
            return
        feuille, feuille_test = self.feuille.get() or None, None
        chemin_test = self.chemin_test.get() or None
        if self._feuilles_challenge and self.challenge.get():
            feuille, feuille_test = self._feuilles_challenge
        travail = self.ctrl.travail_charger(self.chemin.get(), feuille, chemin_test, feuille_test)
        self.app.executer(travail, nom=t("ecran.s1.lecture"), au_succes=self._chargee)

    def _chargee(self, chargee: Chargee) -> None:
        self.ctrl.adopter_chargee(chargee)
        self.app.afficher(1)
        self.app.info(t("ecran.s1.lu", nom=chargee.chargement.table.nom))

    def auditer(self) -> None:
        self.app.executer(self.ctrl.travail_auditer(), nom=t("ecran.s2.audit"), au_succes=self._auditee)

    def _auditee(self, resultat) -> None:
        self.ctrl.adopter_audit(resultat)
        self.app.afficher(2)

    # Affichage
    def rafraichir(self) -> None:
        chargement = self.ctrl.chargement
        if chargement is None or self.session.table is None:
            self.apercu.vider()
            self.resume.configure(text=t("ecran.s1.vide"))
            self.notes.configure(text="")
        else:
            self._montrer(chargement)
        self.mettre_a_jour_boutons()

    def _montrer(self, chargement: fichiers.Chargement) -> None:
        table = chargement.table
        lignes = [
            t(
                "ecran.s1.forme",
                nom=table.nom,
                lignes=nombre_lisible(table.n_lignes),
                colonnes=table.n_colonnes,
            ),
            t(
                "ecran.s1.technique",
                octets=nombre_lisible(self.ctrl.octets),
                duree=nombre(chargement.duree_s, 2),
            ),
            t("ecran.s1.empreinte", sha=chargement.sha256[:16]),
        ]
        if self.ctrl.chargement_test is not None:
            test = self.ctrl.chargement_test.table
            lignes.append(t("ecran.s1.test_lu", lignes=nombre_lisible(test.n_lignes)))
        self.resume.configure(text="\n".join(lignes))
        self.notes.configure(text="\n".join(chargement.notes))
        self._remplir_apercu(table.donnees)

    def _remplir_apercu(self, donnees: pd.DataFrame) -> None:
        colonnes = list(donnees.columns[:COLONNES_APERCU])
        self.apercu.arbre.configure(columns=[str(c) for c in colonnes])
        for c in colonnes:
            self.apercu.arbre.heading(str(c), text=str(c))
            self.apercu.arbre.column(str(c), width=110, anchor="w", stretch=False)
        self.apercu.vider()
        for _, ligne in donnees[colonnes].head(LIGNES_APERCU).iterrows():
            self.apercu.ajouter([nombre(v, 4) if isinstance(v, float) else v for v in ligne])
