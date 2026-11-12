"""MécaML : écran S4, découpage déduit de la structure et jeu de test mis sous scellé."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from mecaml.core import split
from mecaml.core.codegen import LIBELLES_DECOUPAGE, STRUCTURES_FR
from mecaml.core.guards import decimale, nombre_lisible
from mecaml.core.session import Etat
from mecaml.core.types import SpecProjet
from mecaml.ui.controleur import Scelle
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t
from mecaml.ui.widgets import PARTS_DE_TEST, PanneauConstats, carte, champ, lire_decimal, nombre

__all__ = ["EcranDecoupage"]


class EcranDecoupage(Ecran):
    """Jeu de test mis de côté avant toute lecture des variables, selon la structure des données."""

    NUMERO = 4
    CLE = "s4"

    def construire(self, corps: ttk.Frame) -> None:
        self.var_methode = tk.StringVar()
        self.var_part = tk.StringVar()
        self.var_plis = tk.StringVar()
        self.var_graine = tk.StringVar()
        self._methodes: dict[str, split.Methode] = {}
        self._carte_decoupage(corps).grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._carte_resultat(corps).grid(row=1, column=0, sticky="ew", pady=(0, 10))

    def _carte_decoupage(self, corps: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(corps, t("ecran.s4.decoupage.titre"))
        cadre.grid_columnconfigure(1, weight=1)
        self.structure = ttk.Label(cadre, text="", style="Carte.TLabel")
        champ(cadre, 0, t("ecran.s4.structure"), self.structure)
        self.liste_methode = ttk.Combobox(cadre, textvariable=self.var_methode, state="readonly")
        champ(cadre, 1, t("ecran.s4.methode"), self.liste_methode)
        champ(
            cadre,
            2,
            t("ecran.s3.part_test"),
            ttk.Spinbox(cadre, textvariable=self.var_part, values=PARTS_DE_TEST, width=8),
        )
        champ(
            cadre,
            3,
            t("ecran.s3.plis"),
            ttk.Spinbox(cadre, textvariable=self.var_plis, from_=2, to=10, width=8),
        )
        champ(cadre, 4, t("ecran.s3.graine"), ttk.Entry(cadre, textvariable=self.var_graine, width=10))
        self.note = ttk.Label(cadre, text="", style="CarteDoux.TLabel", wraplength=820, justify="left")
        self.note.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.bouton_sceller = self.bouton(
            cadre,
            "ecran.s4.sceller",
            self.sceller,
            principal=True,
            condition=lambda: self.session.etat >= Etat.OBJECTIF,
            row=6,
            column=1,
            sticky="w",
            pady=(8, 0),
        )
        return cadre

    def _carte_resultat(self, corps: ttk.Frame) -> ttk.LabelFrame:
        resultat = carte(corps, t("ecran.s4.resultat.titre"))
        resultat.grid_columnconfigure(0, weight=1)
        self.resume = ttk.Label(resultat, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.resume.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.constats = PanneauConstats(resultat)
        self.constats.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.bouton(
            resultat,
            "ecran.s2.acquitter_choix",
            lambda: self.app.acquitter(self.constats.codes_alertes()),
            condition=lambda: self.session.etat >= Etat.DECOUPE,
            row=2,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self.bouton(
            resultat,
            "ecran.s4.continuer",
            lambda: self.app.afficher(5),
            condition=lambda: self.session.etat >= Etat.DECOUPE,
            row=2,
            column=1,
            padx=8,
            pady=(8, 0),
        )
        return resultat

    def _spec(self) -> SpecProjet:
        return self.session.spec

    def sceller(self) -> None:
        """Met le test sous scellé : le cœur refuse un découpage qui mélange unités ou dates."""
        try:
            spec = self._spec().avec(
                part_test=lire_decimal(self.var_part.get()),
                n_plis=int(self.var_plis.get()),
                graine=int(self.var_graine.get()),
            )
        except (ValueError, TypeError) as erreur:
            self.app.erreur(erreur)
            return
        methode = self._methodes.get(self.var_methode.get())
        self.app.executer(
            self.ctrl.travail_sceller(spec, methode), nom=t("ecran.s4.scelle"), au_succes=self._scelle
        )

    def _scelle(self, resultat: Scelle) -> None:
        self.ctrl.adopter_scelle(resultat)
        self.rafraichir()

    # Affichage
    def rafraichir(self) -> None:
        spec = self.session.spec
        if spec is None:
            return
        self._champs(spec)
        self._resultat()
        self.mettre_a_jour_boutons()

    def _champs(self, spec: SpecProjet) -> None:
        self.structure.configure(text=STRUCTURES_FR[spec.structure])
        self.bouton_sceller.configure(
            text=t("ecran.s4.sceller" if spec.probleme.avec_cible else "ecran.s4.continuer_sans")
        )
        self.var_part.set(decimale(spec.part_test, 2))
        self.var_plis.set(str(spec.n_plis))
        self.var_graine.set(str(spec.graine))
        if not spec.probleme.avec_cible:
            self.liste_methode.configure(values=[], state="disabled")
            self.var_methode.set("")
            self.note.configure(text=t("ecran.s4.sans_cible"))
            return
        if self.ctrl.chargement_test is not None:
            self._methodes = {LIBELLES_DECOUPAGE[split.Methode.FICHIER]: split.Methode.FICHIER}
            self.liste_methode.configure(values=list(self._methodes), state="disabled")
            self.var_methode.set(LIBELLES_DECOUPAGE[split.Methode.FICHIER])
            self.note.configure(text=t("ecran.s4.challenge"))
            return
        permises = sorted(split.methodes_permises(spec), key=str)
        self._methodes = {LIBELLES_DECOUPAGE[m]: m for m in permises}
        self.liste_methode.configure(values=list(self._methodes), state="readonly")
        self.var_methode.set(LIBELLES_DECOUPAGE[split.methode_deduite(spec)])
        self.note.configure(text=t("ecran.s4.deduit"))

    def _resultat(self) -> None:
        session = self.session
        if session.etat < Etat.DECOUPE:
            self.resume.configure(text=t("ecran.s4.pas_encore"))
            self.constats.afficher((), frozenset())
            return
        coffre, decoupage = session.coffre, self.ctrl.decoupage
        if coffre is None:
            self.resume.configure(text=t("ecran.s4.sans_test"))
        else:
            lignes = [
                t(
                    "ecran.s4.effectifs",
                    app=nombre_lisible(coffre.n_apprentissage),
                    test=nombre_lisible(coffre.n_test),
                ),
                t("ecran.s4.part", part=nombre(100.0 * split.part_obtenue(coffre), 1)),
                t("ecran.s4.scellee"),
            ]
            if decoupage is not None:
                lignes.insert(1, t("ecran.s4.methode_prise", methode=LIBELLES_DECOUPAGE[decoupage.methode]))
                lignes.append(t("ecran.s4.empreinte", sha=decoupage.empreinte_sha256[:16]))
            self.resume.configure(text="\n".join(lignes))
        self.constats.afficher(session.constats, session.acquittes)
