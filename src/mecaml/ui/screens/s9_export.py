"""MécaML : écran S9, export du rapport, du classeur, du modèle et du script équivalent."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from mecaml.core.report import FORMATS, NOMS, Export
from mecaml.core.session import Etat, Revenir, Session
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t
from mecaml.ui.widgets import Tableau, carte, champ

__all__ = ["EcranExport"]


class EcranExport(Ecran):
    """Tout ce qui sort de l'application s'écrit dans le dossier choisi, et nulle part ailleurs."""

    NUMERO = 9
    CLE = "s9"

    def construire(self, corps: ttk.Frame) -> None:
        self.dossier = tk.StringVar()
        self.formats = {f: tk.BooleanVar(value=True) for f in FORMATS}
        cadre = carte(corps, t("ecran.s9.dossier.titre"))
        cadre.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        cadre.grid_columnconfigure(1, weight=1)
        ligne = ttk.Frame(cadre, style="Carte.TFrame")
        ligne.grid_columnconfigure(0, weight=1)
        ttk.Entry(ligne, textvariable=self.dossier).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.bouton(ligne, "ecran.s1.parcourir", self.parcourir, row=0, column=1)
        champ(cadre, 0, t("ecran.s9.champ_dossier"), ligne)
        for k, format_ in enumerate(FORMATS, start=1):
            ttk.Checkbutton(
                cadre, text=t(f"ecran.s9.format.{format_}", nom=NOMS[format_]), variable=self.formats[format_]
            ).grid(row=k, column=1, sticky="w")
        ttk.Label(
            cadre, text=t("ecran.s9.note"), style="CarteDoux.TLabel", wraplength=820, justify="left"
        ).grid(row=len(FORMATS) + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.bouton(
            cadre,
            "ecran.s9.exporter",
            self.exporter,
            principal=True,
            condition=self._peut_exporter,
            row=len(FORMATS) + 2,
            column=1,
            sticky="w",
            pady=(10, 0),
        )
        self._carte_resultat(corps).grid(row=1, column=0, sticky="ew")

    def _carte_resultat(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s9.resultat.titre"))
        cadre.grid_columnconfigure(0, weight=1)
        self.resume = ttk.Label(
            cadre, text=t("ecran.s9.pas_encore"), style="Carte.TLabel", wraplength=820, justify="left"
        )
        self.resume.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        colonnes = [
            ("format", t("ecran.s9.col.format"), 120, "w"),
            ("fichier", t("ecran.s9.col.fichier"), 330, "w"),
            ("sha", t("ecran.s9.col.empreinte"), 280, "w"),
        ]
        self.tableau = Tableau(cadre, colonnes, hauteur=5)
        self.tableau.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.bouton(
            cadre,
            "ecran.s9.ouvrir_dossier",
            self.ouvrir_dossier,
            condition=lambda: self.ctrl.export is not None,
            row=2,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self.bouton(
            cadre,
            "ecran.s9.ouvrir_rapport",
            self.ouvrir_rapport,
            condition=self._a_un_rapport,
            row=2,
            column=1,
            padx=8,
            pady=(8, 0),
        )
        self.bouton(
            cadre,
            "ecran.s9.nouveau",
            self.nouveau,
            condition=lambda: self.session.etat >= Etat.EXPORTE,
            row=2,
            column=2,
            pady=(8, 0),
        )
        return cadre

    def _peut_exporter(self) -> bool:
        session = self.session
        return session.etat >= Etat.TESTE or (session.sans_cible and session.etat >= Etat.DIAGNOSTIC)

    def _a_un_rapport(self) -> bool:
        export = self.ctrl.export
        return export is not None and "html" in export.fichiers

    # Actions
    def parcourir(self) -> None:
        choisi = filedialog.askdirectory(title=t("ecran.s9.dossier_titre"))
        if choisi:
            self.dossier.set(choisi)

    def exporter(self, dossier: str | Path | None = None) -> None:
        """Écrit les fichiers dans le dossier choisi ; l'export est refusé s'il reste un constat à lever."""
        if dossier is not None:
            self.dossier.set(str(dossier))
        if not self.dossier.get().strip():
            self.app.erreur(t("ecran.s9.aucun_dossier"))
            return
        choisis = [f for f in FORMATS if self.formats[f].get()]
        travail = self.ctrl.travail_exporter(self.dossier.get(), choisis)
        self.app.executer(travail, nom=t("ecran.s9.export"), au_succes=self._exporte)

    def _exporte(self, resultat: tuple[Session, Export]) -> None:
        session, export = resultat
        self.ctrl.adopter(session)
        self.ctrl.export = export
        self.rafraichir()

    def ouvrir_dossier(self) -> None:
        if self.ctrl.export is not None and sys.platform == "win32":
            os.startfile(self.ctrl.export.dossier)

    def ouvrir_rapport(self) -> None:
        if self._a_un_rapport() and sys.platform == "win32":
            os.startfile(self.ctrl.export.fichiers["html"])

    def nouveau(self) -> None:
        self.app.executer(
            self.ctrl.travail_actions(Revenir(Etat.VIDE)),
            nom=t("ecran.s9.nouveau_projet"),
            au_succes=self._nouveau,
        )

    def _nouveau(self, session: Session) -> None:
        self.ctrl.adopter(session)
        self.app.afficher(0)

    # Affichage
    def rafraichir(self) -> None:
        export = self.ctrl.export
        self.tableau.vider()
        if export is None or self.session.etat < Etat.EXPORTE:
            self.resume.configure(text=t("ecran.s9.pas_encore"))
        else:
            self.resume.configure(
                text=t("ecran.s9.fait", dossier=str(export.dossier)) + self._avertissements(export)
            )
            for format_, chemin in export.fichiers.items():
                nom = Path(chemin).name
                self.tableau.ajouter((format_, nom, export.empreintes.get(nom, "")[:32]))
        self.mettre_a_jour_boutons()

    @staticmethod
    def _avertissements(export: Export) -> str:
        return "\n" + "\n".join(export.avertissements[:5]) if export.avertissements else ""
