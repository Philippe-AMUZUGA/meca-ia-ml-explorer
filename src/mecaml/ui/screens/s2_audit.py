"""MécaML : écran S2, profil des colonnes et constats de l'audit, acquittement des alertes."""

from __future__ import annotations

from tkinter import ttk

from mecaml.core.audit import ProfilColonne
from mecaml.core.session import Etat
from mecaml.core.types import Constat
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t
from mecaml.ui.widgets import PanneauConstats, Tableau, carte, nombre

__all__ = ["EcranAudit"]

CODE_REPLIE = "DONNEES.valeurs_aberrantes"  # information repliée dans la colonne « hors bornes » du profil


class EcranAudit(Ecran):
    """Ce que l'audit voit dans le fichier, avant de choisir une cible."""

    NUMERO = 2
    CLE = "s2"

    def construire(self, corps: ttk.Frame) -> None:
        profil = carte(corps, t("ecran.s2.profil.titre"))
        profil.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        profil.grid_columnconfigure(0, weight=1)
        colonnes = [
            ("nom", t("ecran.s2.col.nom"), 170, "w"),
            ("famille", t("ecran.s2.col.famille"), 60, "w"),
            ("role", t("ecran.s2.col.role"), 130, "w"),
            ("manquants", t("ecran.s2.col.manquants"), 80, "e"),
            ("distincts", t("ecran.s2.col.distincts"), 70, "e"),
            ("minimum", t("ecran.s2.col.minimum"), 70, "e"),
            ("maximum", t("ecran.s2.col.maximum"), 70, "e"),
            ("hors", t("ecran.s2.col.hors"), 80, "e"),
            ("raison", t("ecran.s2.col.raison"), 200, "w"),
        ]
        self.profil = Tableau(profil, colonnes, hauteur=10)
        self.profil.grid(row=0, column=0, sticky="ew")
        constats = carte(corps, t("ecran.s2.constats.titre"))
        constats.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        constats.grid_columnconfigure(0, weight=1)
        self.constats = PanneauConstats(constats)
        self.constats.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.bouton(
            constats,
            "ecran.s2.acquitter_choix",
            lambda: self.app.acquitter(self.constats.codes_alertes()),
            row=1,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self.bouton(
            constats,
            "ecran.s2.acquitter_tout",
            lambda: self.app.acquitter(self.constats.codes_alertes(tout=True)),
            row=1,
            column=1,
            sticky="w",
            padx=8,
            pady=(8, 0),
        )
        self.replie = ttk.Label(constats, text="", style="CarteDoux.TLabel")
        self.replie.grid(row=1, column=2, sticky="e", pady=(8, 0))
        self.bouton(
            corps,
            "ecran.s2.continuer",
            self.continuer,
            principal=True,
            condition=lambda: self.session.etat >= Etat.AUDIT,
            row=2,
            column=0,
            sticky="w",
        )

    def continuer(self) -> None:
        """Ouvre l'écran de l'objectif : l'audit rejoué y tiendra compte de la cible choisie."""
        self.app.afficher(3)

    def rafraichir(self) -> None:
        profils = self.ctrl.profils
        self.profil.remplir(_ligne_de_profil(p) for p in profils)
        constats = self.session.constats
        visibles = [c for c in constats if not c.code.startswith(CODE_REPLIE)]
        self.constats.afficher(visibles, self.session.acquittes)
        self._repli(constats, visibles)
        self.mettre_a_jour_boutons()

    def _repli(self, constats: tuple[Constat, ...], visibles: list[Constat]) -> None:
        n = len(constats) - len(visibles)
        self.replie.configure(text=t("ecran.s2.replie", n=n) if n else "")


def _ligne_de_profil(p: ProfilColonne) -> tuple[str, ...]:
    return (
        p.nom,
        t(f"famille.{p.famille}"),
        t(f"role.{p.role_propose.value}"),
        nombre(100.0 * p.part_manquants, 1) + " %",
        str(p.n_distincts),
        nombre(p.minimum, 4) if p.minimum is not None else "",
        nombre(p.maximum, 4) if p.maximum is not None else "",
        str(p.n_hors_tukey) if p.n_hors_tukey else "",
        p.raison,
    )
