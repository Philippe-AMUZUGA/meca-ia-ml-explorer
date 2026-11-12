"""
MécaML : recherche d'hyperparamètres d'un modèle comparé (carte facultative de l'écran S6).

La recherche est celle du cœur (`train.rechercher`) : grille bornée de la technique, mêmes plis que la
comparaison, test toujours scellé. L'interface choisit le modèle et le mode, lance le calcul dans un fil et
montre le classement des essais ; le meilleur réglage se reporte dans la liste des modèles, à ré-entraîner.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from mecaml.core import catalog, libelles, train
from mecaml.core.types import ChoixTechnique
from mecaml.ui.choix import decrire_parametres
from mecaml.ui.textes import t, titre_technique
from mecaml.ui.widgets import Tableau, champ, nombre

if TYPE_CHECKING:
    from mecaml.ui.screens.s6_modeles import EcranModeles

__all__ = ["PanneauRecherche"]

MODES = ("grille", "aleatoire")


class PanneauRecherche(ttk.LabelFrame):
    """Choix du modèle et du mode, classement des essais, report du meilleur réglage."""

    def __init__(self, parent: tk.Misc, ecran: EcranModeles) -> None:
        super().__init__(parent, text=t("ecran.s6.recherche.titre"), style="Carte.TLabelframe", padding=12)
        self.ecran = ecran
        self.var_modele = tk.StringVar()
        self.var_mode = tk.StringVar(value=MODES[0])
        self.var_essais = tk.StringVar(value="20")
        self._ids: dict[str, str] = {}
        self._derniere: train.ResultatRecherche | None = None
        self.grid_columnconfigure(1, weight=1)
        self.liste_modele = ttk.Combobox(self, textvariable=self.var_modele, state="readonly")
        champ(self, 0, t("ecran.s6.recherche.modele"), self.liste_modele)
        modes = ttk.Frame(self, style="Carte.TFrame")
        for k, mode in enumerate(MODES):
            ttk.Radiobutton(
                modes, text=t(f"ecran.s6.recherche.mode.{mode}"), value=mode, variable=self.var_mode
            ).grid(row=0, column=k, padx=(0, 16))
        champ(self, 1, t("ecran.s6.recherche.champ_mode"), modes)
        champ(
            self,
            2,
            t("ecran.s6.recherche.essais"),
            ttk.Spinbox(self, textvariable=self.var_essais, from_=2, to=50, width=6),
        )
        self.bouton_chercher = ttk.Button(self, text=t("ecran.s6.recherche.chercher"), command=self.chercher)
        self.bouton_chercher.grid(row=3, column=1, sticky="w", pady=(6, 0))
        self.avis = ttk.Label(
            self, text=t("ecran.s6.recherche.pas_encore"), style="CarteDoux.TLabel", wraplength=800
        )
        self.avis.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        colonnes = [
            ("parametres", t("ecran.s6.recherche.col.parametres"), 420, "w"),
            ("moyenne", t("ecran.s6.recherche.col.moyenne"), 100, "e"),
            ("ecart", t("ecran.s6.recherche.col.ecart"), 100, "e"),
        ]
        self.tableau = Tableau(self, colonnes, hauteur=5, selection="browse")
        self.tableau.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.bouton_utiliser = ttk.Button(self, text=t("ecran.s6.recherche.utiliser"), command=self.utiliser)
        self.bouton_utiliser.grid(row=6, column=1, sticky="w", pady=(6, 0))

    # Modèles qui ont une grille
    def actualiser(self) -> None:
        """Modèles de la comparaison qui ont une grille bornée, hors référence naïve."""
        session = self.ecran.session
        resultat = session.resultats if session.etat >= self.ecran.NUMERO else None
        offerts = []
        if resultat is not None:
            entrees = [(c, catalog.entree(c.id)) for c in resultat.spec.modeles]
            offerts = [c for c, e in entrees if e.grille_bornee and e.famille != "reference"]
        self._ids = {titre_technique(c.id): c.id for c in offerts}
        self.liste_modele.configure(values=list(self._ids))
        if self.var_modele.get() not in self._ids:
            self.var_modele.set(next(iter(self._ids), ""))
        if self.ecran.ctrl.recherches:
            self._montrer(self.ecran.ctrl.recherches[-1])

    def basculer(self, actif: bool) -> None:
        etat = "normal" if actif and self._ids else "disabled"
        self.bouton_chercher.configure(state=etat)
        self.bouton_utiliser.configure(state="normal" if actif and self._derniere is not None else "disabled")

    # Calcul
    def chercher(self) -> None:
        """Recherche dans un fil annulable ; le classement s'affiche à la fin."""
        identifiant = self._ids.get(self.var_modele.get())
        if identifiant is None:
            return
        try:
            essais = int(self.var_essais.get())
        except ValueError:
            self.ecran.app.erreur(t("ecran.s6.recherche.nombre_invalide"))
            return
        travail = self.ecran.ctrl.travail_recherche(identifiant, self.var_mode.get(), essais)
        self.ecran.app.executer(travail, nom=t("ecran.s6.recherche.travail"), au_succes=self._trouvee)

    def _trouvee(self, resultat: train.ResultatRecherche) -> None:
        self.ecran.ctrl.recherches.append(resultat)
        self._montrer(resultat)
        self.ecran.mettre_a_jour_boutons()

    def _montrer(self, resultat: train.ResultatRecherche) -> None:
        self._derniere = resultat
        self.avis.configure(
            text=t("ecran.s6.recherche.optimiste", metrique=libelles.metrique_courte(resultat.metrique))
        )
        self.tableau.vider()
        for k, essai in enumerate(resultat.classement()):
            self.tableau.ajouter(
                (
                    decrire_parametres(essai.choix) or t("ecran.s6.recherche.defauts"),
                    nombre(essai.moyenne(resultat.metrique), 4),
                    nombre(essai.ecart_type(resultat.metrique), 4),
                ),
                iid=str(k),
            )

    # Report du meilleur réglage
    def utiliser(self) -> None:
        """Remplace le choix du modèle par le meilleur essai : à ré-entraîner pour le comparer aux autres."""
        resultat = self._derniere
        meilleur = None if resultat is None else resultat.meilleur
        if resultat is None or meilleur is None:
            return
        liste = self.ecran.liste
        remplace: list[ChoixTechnique] = [meilleur if c.id == meilleur.id else c for c in liste.choix]
        liste.definir(remplace)
        self.ecran.app.info(t("ecran.s6.recherche.reporte"))
