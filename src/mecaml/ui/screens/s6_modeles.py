"""MécaML : écran S6, catalogue filtré, référence naïve, validation croisée et tableau comparatif."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk
from typing import Any

from mecaml.core import cible_passee, decision, libelles, train
from mecaml.core.catalog import OPTIONNELLES
from mecaml.core.catalog.entree import EntreeCatalogue
from mecaml.core.codegen import LIBELLES_VALIDATION
from mecaml.core.guards import technique_coupee
from mecaml.core.session import Etat, Session
from mecaml.core.types import ChoixTechnique
from mecaml.ui.choix import ListeChoix
from mecaml.ui.controleur import Verification
from mecaml.ui.recherche import PanneauRecherche
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t, titre_technique
from mecaml.ui.widgets import PanneauConstats, Tableau, ZoneFigure, carte, nombre

__all__ = ["EcranModeles"]


class EcranModeles(Ecran):
    """Modèles à comparer, toujours face à une référence naïve, sur les mêmes plis de validation croisée."""

    NUMERO = 6
    CLE = "s6"

    def construire(self, corps: ttk.Frame) -> None:
        self.var_cible_connue = tk.BooleanVar(value=False)
        self._cle: Any = None
        cadre = carte(corps, t("ecran.s6.modeles.titre"))
        cadre.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        cadre.grid_columnconfigure(0, weight=1)
        self.liste = ListeChoix(
            cadre,
            offertes=self._offertes,
            colonnes=lambda: (),
            note=self._note,
            ordonnee=False,
            defauts=_defauts,
            au_changement=self.verifier,
        )
        self.liste.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(
            cadre,
            text=t("ecran.s5.cible_connue"),
            variable=self.var_cible_connue,
            command=self.liste.actualiser,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            cadre, text=t("ecran.s6.reference"), style="CarteDoux.TLabel", wraplength=820, justify="left"
        ).grid(row=2, column=0, sticky="w")
        self.bouton(cadre, "ecran.s6.proposer", self.proposer, row=3, column=0, sticky="w", pady=(8, 0))
        controles = carte(corps, t("ecran.s5.controles.titre"))
        controles.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        controles.grid_columnconfigure(0, weight=1)
        self.avis = ttk.Label(controles, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.avis.grid(row=0, column=0, sticky="w")
        self.constats = PanneauConstats(controles, hauteur=4)
        self.constats.grid(row=1, column=0, sticky="ew")
        self.bouton(
            corps,
            "ecran.s6.entrainer",
            self.entrainer,
            principal=True,
            condition=lambda: self.session.etat >= Etat.PRETRAITEMENT,
            row=2,
            column=0,
            sticky="w",
            pady=(0, 10),
        )
        self._carte_resultat(corps).grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.recherche = PanneauRecherche(corps, self)
        self.recherche.grid(row=4, column=0, sticky="ew")

    def _carte_resultat(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s6.resultat.titre"))
        cadre.grid_columnconfigure(0, weight=1)
        self.resume = ttk.Label(
            cadre, text=t("ecran.s6.pas_encore"), style="Carte.TLabel", wraplength=820, justify="left"
        )
        self.resume.grid(row=0, column=0, sticky="w", pady=(0, 6))
        colonnes = [
            ("modele", t("ecran.s6.col.modele"), 230, "w"),
            ("moyenne", t("ecran.s6.col.moyenne"), 90, "e"),
            ("ecart", t("ecran.s6.col.ecart"), 90, "e"),
            ("gain", t("ecran.s6.col.gain"), 110, "e"),
            ("plis", t("ecran.s6.col.plis"), 80, "e"),
            ("duree", t("ecran.s6.col.duree"), 80, "e"),
            ("statut", t("ecran.s6.col.statut"), 260, "w"),
        ]
        self.tableau = Tableau(cadre, colonnes, hauteur=6)
        self.tableau.grid(row=1, column=0, sticky="ew")
        self.figure = ZoneFigure(cadre, hauteur=280)
        self.figure.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.avertissements = ttk.Label(cadre, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.avertissements.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.bouton(
            cadre,
            "ecran.s6.continuer",
            lambda: self.app.afficher(7),
            condition=lambda: self.session.etat >= Etat.MODELES,
            row=4,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        return cadre

    # Catalogue offert
    def _offertes(self) -> Sequence[EntreeCatalogue]:
        spec = self.session.spec
        if spec is None:
            return ()
        entrees = train.modeles_disponibles(spec.probleme)
        if not self.var_cible_connue.get():
            entrees = tuple(e for e in entrees if e.id not in cible_passee.TECHNIQUES)
        return entrees

    def _note(self, choix: ChoixTechnique) -> str:
        """Ce qui limite un modèle : borne de lignes qui le coupe, ou export du script indisponible."""
        table = self.session.table
        coupee = technique_coupee(choix.id, table.n_lignes) if table is not None else None
        if coupee:
            return coupee
        return t("ecran.s6.sans_export") if choix.id in {e.id for e in OPTIONNELLES} else ""

    # Contrôle, entraînement
    def proposer(self) -> None:
        spec = self.session.spec
        self.liste.definir(train.proposer_modeles(spec.probleme))
        self.verifier()

    def _specification(self):
        return self.session.spec.avec(modeles=self.liste.choix)

    def verifier(self) -> None:
        """Rejoue l'audit avec les modèles choisis : cible passée non déclarée ou décomptée."""
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
        self.avis.configure(text="\n".join(verification.avis))
        self.constats.afficher(verification.constats, self.session.acquittes)
        self.mettre_a_jour_boutons()

    def entrainer(self) -> None:
        """Lance la validation croisée dans un fil annulable ; les résultats vont à la session."""
        try:
            spec = self._specification()
        except (ValueError, TypeError) as erreur:
            self.app.erreur(erreur)
            return
        self.app.executer(
            self.ctrl.travail_entrainer(spec), nom=t("ecran.s6.entrainement"), au_succes=self._entraine
        )

    def _entraine(self, session: Session) -> None:
        self.ctrl.adopter(session)
        self.rafraichir()

    # Affichage
    def rafraichir(self) -> None:
        spec = self.session.spec
        if spec is None or self.session.etat < Etat.PRETRAITEMENT:
            return
        cle = (self.session.table, spec.objectif(), spec.decoupage(), spec.pretraitement)
        if self.session.etat >= Etat.MODELES:
            self.var_cible_connue.set(any(cible_passee.declaree(c.parametres_dict()) for c in spec.modeles))
            self.liste.definir(spec.modeles)
            self._cle = cle
            self._montrer_resultat()
        elif self._cle != cle:
            self._cle = cle
            self.proposer()
        self.liste.actualiser()
        self.recherche.actualiser()
        self.mettre_a_jour_boutons()

    def _montrer_resultat(self) -> None:
        resultat = self.session.resultats
        lignes = [t("ecran.s6.metrique", metrique=libelles.metrique(resultat.metrique))]
        lignes.append(
            t(
                "ecran.s6.validation",
                validation=LIBELLES_VALIDATION.get(resultat.validation, resultat.validation),
                plis=len(resultat.plis),
            )
        )
        if resultat.classe_positive is not None:
            lignes.append(t("ecran.s6.positive", classe=resultat.classe_positive))
        if resultat.n_sans_cible:
            lignes.append(t("ecran.s6.sans_cible", n=resultat.n_sans_cible))
        self.resume.configure(text="\n".join(lignes))
        self.tableau.vider()
        for k, m in enumerate(resultat.classement()):
            self.tableau.ajouter(self._ligne(resultat, m), iid=str(k))
        self.figure.afficher(decision.figure_comparaison(resultat))
        avertissements = [
            f"{titre_technique(m.id)} : {a}" for m in resultat.modeles for a in m.avertissements[:2]
        ]
        self.avertissements.configure(text="\n".join(avertissements[:6]))

    def _ligne(self, resultat: train.Resultat, m: train.ResultatModele) -> tuple[str, ...]:
        gain = resultat.gain(m.id)
        return (
            titre_technique(m.id),
            nombre(resultat.valeur(m), 4),
            nombre(m.ecart_type(resultat.metrique), 4),
            "" if gain is None else nombre(gain, 4),
            f"{m.plis_valides(resultat.metrique)} / {len(resultat.plis)}",
            nombre(m.duree_s, 2),
            self._statut(m),
        )

    @staticmethod
    def _statut(m: train.ResultatModele) -> str:
        if m.erreur:
            return t("ecran.s6.echec", motif=m.erreur)
        return t("ecran.s6.statut.reference") if m.famille == "reference" else t("ecran.s6.statut.ajuste")

    def basculer(self, occupe: bool) -> None:
        super().basculer(occupe)
        self.liste.basculer(not occupe)
        self.recherche.basculer(not occupe and self.session.etat >= Etat.MODELES)


def _defauts(entree: EntreeCatalogue) -> dict[str, Any]:
    return {cible_passee.PARAMETRE: True} if entree.id in cible_passee.TECHNIQUES else {}
