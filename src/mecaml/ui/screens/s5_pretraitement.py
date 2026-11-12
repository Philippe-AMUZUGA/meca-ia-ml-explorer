"""MécaML : écran S5, prétraitement proposé par rôle et modifiable ; garde de la cible passée rejouée."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk
from typing import Any

from mecaml.core import catalog, cible_passee
from mecaml.core.catalog.entree import EntreeCatalogue
from mecaml.core.session import Etat, Pretraiter, Session
from mecaml.core.types import ChoixTechnique, SpecProjet
from mecaml.ui.choix import ListeChoix
from mecaml.ui.controleur import Verification
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t
from mecaml.ui.widgets import PanneauConstats, carte

__all__ = ["FAMILLES_PRETRAITEMENT", "EcranPretraitement"]

FAMILLES_PRETRAITEMENT = ("manquants", "encodage", "transformation", "selection", "chronologique")


class EcranPretraitement(Ecran):
    """Imputation, encodage, mise à l'échelle, sélection et variables temporelles, par rôle de colonne."""

    NUMERO = 5
    CLE = "s5"

    def construire(self, corps: ttk.Frame) -> None:
        self.var_cible_connue = tk.BooleanVar(value=False)
        self._cle: Any = None
        cadre = carte(corps, t("ecran.s5.chaine.titre"))
        cadre.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        cadre.grid_columnconfigure(0, weight=1)
        self.liste = ListeChoix(
            cadre,
            offertes=self._offertes,
            colonnes=self._colonnes,
            defauts=_defauts,
            au_changement=self.verifier,
        )
        self.liste.grid(row=0, column=0, sticky="ew")
        self.case = ttk.Checkbutton(
            cadre,
            text=t("ecran.s5.cible_connue"),
            variable=self.var_cible_connue,
            command=self._cible_connue_change,
        )
        self.case.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            cadre,
            text=t("ecran.s5.cible_connue_limite"),
            style="CarteDoux.TLabel",
            wraplength=820,
            justify="left",
        ).grid(row=2, column=0, sticky="w")
        self.bouton(cadre, "ecran.s5.proposer", self.proposer, row=3, column=0, sticky="w", pady=(8, 0))
        controles = carte(corps, t("ecran.s5.controles.titre"))
        controles.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        controles.grid_columnconfigure(0, weight=1)
        self.avis = ttk.Label(controles, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.avis.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.constats = PanneauConstats(controles, hauteur=4)
        self.constats.grid(row=1, column=0, sticky="ew")
        self.bouton(controles, "ecran.s5.verifier", self.verifier, row=2, column=0, sticky="w", pady=(8, 0))
        self.bouton(
            corps,
            "ecran.s5.appliquer",
            self.appliquer,
            principal=True,
            condition=lambda: self.session.etat >= Etat.DECOUPE,
            row=2,
            column=0,
            sticky="w",
        )

    # Catalogue offert
    def _offertes(self) -> Sequence[EntreeCatalogue]:
        spec = self.session.spec
        if spec is None:
            return ()
        entrees = [
            e for f in FAMILLES_PRETRAITEMENT for e in catalog.entrees(probleme=spec.probleme, famille=f)
        ]
        if not self.var_cible_connue.get():
            entrees = [e for e in entrees if e.id not in cible_passee.TECHNIQUES]
        return entrees

    def _colonnes(self) -> Sequence[str]:
        spec = self.session.spec
        return () if spec is None else spec.variables()

    def _cible_connue_change(self) -> None:
        self.liste.actualiser()

    # Proposition et contrôle
    def proposer(self) -> None:
        self.app.executer(
            self.ctrl.travail_proposer_pretraitement(),
            nom=t("ecran.s5.proposition"),
            au_succes=self._proposee,
        )

    def _proposee(self, choix: tuple[ChoixTechnique, ...]) -> None:
        self.liste.definir(choix)
        self.verifier()

    def _specification(self) -> SpecProjet:
        return self.session.spec.avec(pretraitement=self.liste.choix, modeles=())

    def verifier(self) -> None:
        """Rejoue l'audit sur la spécification complète : le refus de la cible passée se lit d.avance."""
        if self.session.spec is None or self.app.occupe:
            return
        try:
            spec = self._specification()
        except (ValueError, TypeError) as erreur:
            self.app.erreur(erreur)
            return
        self.app.executer(
            self.ctrl.travail_verifier(spec), nom=t("ecran.s5.controle"), au_succes=self._verifiee
        )

    def _verifiee(self, verification: Verification) -> None:
        self.avis.configure(
            text="\n".join(verification.avis) if verification.avis else t("ecran.s5.aucun_avis")
        )
        self.constats.afficher(verification.constats, self.session.acquittes)
        self.mettre_a_jour_boutons()

    def appliquer(self) -> None:
        """Retient le prétraitement : la session refuse tant qu'un constat bloquant ou une alerte reste."""
        try:
            spec = self._specification()
        except (ValueError, TypeError) as erreur:
            self.app.erreur(erreur)
            return
        self.app.executer(
            self.ctrl.travail_actions(Pretraiter(spec)),
            nom=t("ecran.s5.application"),
            au_succes=self._appliquee,
        )

    def _appliquee(self, session: Session) -> None:
        self.ctrl.adopter(session)
        self.app.suivant()

    # Affichage
    def rafraichir(self) -> None:
        spec = self.session.spec
        if spec is None or self.session.etat < Etat.DECOUPE:
            return
        cle = (self.session.table, spec.objectif(), spec.decoupage())
        if self.session.etat >= Etat.PRETRAITEMENT:
            self.var_cible_connue.set(
                any(cible_passee.declaree(c.parametres_dict()) for c in spec.pretraitement)
            )
            self.liste.definir(spec.pretraitement)
            self._cle = cle
        elif self._cle != cle:
            self._cle = cle
            self.liste.definir(())
            self.proposer()
        self.liste.actualiser()
        self.mettre_a_jour_boutons()

    def basculer(self, occupe: bool) -> None:
        super().basculer(occupe)
        self.liste.basculer(not occupe)


def _defauts(entree: EntreeCatalogue) -> dict[str, Any]:
    """Une technique qui lit la cible passée se pose avec sa déclaration : la case cochée l'a décidé."""
    return {cible_passee.PARAMETRE: True} if entree.id in cible_passee.TECHNIQUES else {}
