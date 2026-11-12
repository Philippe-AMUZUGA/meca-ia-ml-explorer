"""MécaML : écran S7, diagnostic hors pli du modèle et décision motivée (seuil, coût, importances)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from mecaml.core import decision
from mecaml.core.decision import ErreurDecision
from mecaml.core.session import Etat, Session
from mecaml.ui.controleur import Vue
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t, titre_technique
from mecaml.ui.widgets import Planche, carte, champ

__all__ = ["EcranDiagnostic"]

ONGLETS = ("diagnostic", "seuil", "lecture", "importances", "courbe", "garde")


class EcranDiagnostic(Ecran):
    """Lire les prédictions sur des lignes que le modèle n'a pas vues, puis retenir un modèle et le dire."""

    NUMERO = 7
    CLE = "s7"

    def construire(self, corps: ttk.Frame) -> None:
        self.var_modele = tk.StringVar()
        self.var_seuil = tk.StringVar()
        self.var_cout_fp = tk.StringVar(value="1")
        self.var_cout_fn = tk.StringVar(value="1")
        self.var_justification = tk.StringVar()
        self._ids: dict[str, str] = {}
        self._cle: Any = None
        cadre = carte(corps, t("ecran.s7.modele.titre"))
        cadre.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        cadre.grid_columnconfigure(1, weight=1)
        self.liste_modele = ttk.Combobox(cadre, textvariable=self.var_modele, state="readonly")
        self.liste_modele.bind("<<ComboboxSelected>>", lambda _e: self.diagnostiquer())
        champ(cadre, 0, t("ecran.s7.champ_modele"), self.liste_modele)
        self.garde = ttk.Label(cadre, text="", style="CarteAlerte.TLabel", wraplength=820, justify="left")
        self.garde.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.onglets = ttk.Notebook(corps)
        self.onglets.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.planches: dict[str, Planche] = {}
        for nom in ONGLETS:
            onglet = ttk.Frame(self.onglets, style="Carte.TFrame", padding=10)
            onglet.grid_columnconfigure(0, weight=1)
            self.onglets.add(onglet, text=t(f"ecran.s7.onglet.{nom}"))
            self.planches[nom] = Planche(onglet)
            self._equiper(onglet, nom)
        self._carte_decision(corps).grid(row=2, column=0, sticky="ew")

    def _equiper(self, onglet: ttk.Frame, nom: str) -> None:
        """Commandes propres à chaque onglet ; la planche de lecture se pose dessous."""
        ligne = ttk.Frame(onglet, style="Carte.TFrame")
        ligne.grid(row=0, column=0, sticky="w", pady=(0, 6))
        if nom == "seuil":
            self._commandes_seuil(ligne)
        elif nom in ("lecture", "importances", "courbe"):
            self.bouton(ligne, f"ecran.s7.calculer.{nom}", lambda n=nom: self.calculer(n), row=0, column=0)
        self.planches[nom].grid(row=1, column=0, sticky="ew")

    def _commandes_seuil(self, ligne: ttk.Frame) -> None:
        ttk.Label(ligne, text=t("ecran.s7.seuil"), style="CarteDoux.TLabel").grid(row=0, column=0)
        ttk.Entry(ligne, textvariable=self.var_seuil, width=7).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(ligne, text=t("ecran.s3.cout_fp"), style="CarteDoux.TLabel").grid(row=0, column=2)
        ttk.Entry(ligne, textvariable=self.var_cout_fp, width=6).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(ligne, text=t("ecran.s3.cout_fn"), style="CarteDoux.TLabel").grid(row=0, column=4)
        ttk.Entry(ligne, textvariable=self.var_cout_fn, width=6).grid(row=0, column=5, padx=(4, 12))
        self.bouton(ligne, "ecran.s7.appliquer_seuil", self.diagnostiquer, row=0, column=6)
        self.bouton(ligne, "ecran.s7.seuil_cout", self.proposer_seuil, row=0, column=7, padx=8)

    def _carte_decision(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s7.decision.titre"))
        cadre.grid_columnconfigure(1, weight=1)
        champ(cadre, 0, t("ecran.s7.justification"), ttk.Entry(cadre, textvariable=self.var_justification))
        self.bouton(
            cadre,
            "ecran.s7.retenir",
            self.retenir,
            principal=True,
            condition=lambda: self.session.etat >= Etat.MODELES,
            row=1,
            column=1,
            sticky="w",
            pady=(8, 0),
        )
        return cadre

    # Modèle choisi
    def _id_choisi(self) -> str:
        return self._ids.get(self.var_modele.get(), "")

    def _seuil(self) -> float | None:
        texte = self.var_seuil.get().strip().replace(",", ".")
        return float(texte) if texte else None

    def _couts(self) -> tuple[float, float]:
        return float(self.var_cout_fp.get().replace(",", ".")), float(
            self.var_cout_fn.get().replace(",", ".")
        )

    # Calculs
    def diagnostiquer(self) -> None:
        """Diagnostic hors pli du modèle choisi, avec le seuil et les coûts saisis."""
        if not self._id_choisi() or self.app.occupe:
            return
        try:
            seuil, (cout_fp, cout_fn) = self._seuil(), self._couts()
        except ValueError:
            self.app.erreur(t("ecran.s7.nombre_invalide"))
            return
        resultat = self.session.resultats
        self.garde.configure(text="\n".join(decision.avertissements(resultat, self._id_choisi())))
        travail = self.ctrl.travail_diagnostic(self._id_choisi(), seuil, cout_fp, cout_fn)
        self.app.executer(travail, nom=t("ecran.s7.diagnostic"), au_succes=self._diagnostique)

    def _diagnostique(self, vue: Vue) -> None:
        self.ctrl.vues["diagnostic"] = vue
        self._montrer_diagnostic(vue)

    def _mises_en_garde(self) -> str:
        """Mises en garde du cœur sur le modèle choisi ; son refus de le retenir se lit ici, tel quel."""
        try:
            return "\n".join(decision.avertissements(self.session.resultats, self._id_choisi()))
        except ErreurDecision as erreur:
            return str(erreur)

    def _montrer_diagnostic(self, vue: Vue) -> None:
        figures = tuple(f for f in vue.figures if f[0] != "seuil")
        self.planches["diagnostic"].afficher(Vue(figures, vue.tableaux, vue.textes), self._titres())
        seuil = tuple(f for f in vue.figures if f[0] == "seuil")
        self.planches["seuil"].afficher(Vue(seuil), self._titres())

    def proposer_seuil(self) -> None:
        """Seuil de coût moyen minimal sur les prédictions hors pli, tel que le cœur le calcule."""
        vue = self.ctrl.vues.get("diagnostic")
        if vue is None or vue.seuil_cout is None:
            self.app.erreur(t("ecran.s7.pas_de_seuil"))
            return
        self.var_seuil.set(f"{vue.seuil_cout:.3f}")
        self.diagnostiquer()

    def calculer(self, nom: str) -> None:
        """Lecture du modèle, importances ou courbe d'apprentissage : un calcul plus long, dans un fil."""
        identifiant = self._id_choisi()
        travaux = {
            "lecture": self.ctrl.travail_lecture_du_modele,
            "importances": self.ctrl.travail_importances,
            "courbe": self.ctrl.travail_courbe_d_apprentissage,
        }
        self.app.executer(
            travaux[nom](identifiant),
            nom=t(f"ecran.s7.calculer.{nom}"),
            au_succes=lambda vue: self.planches[nom].afficher(vue, self._titres()),
        )

    def _titres(self) -> dict[str, str]:
        return {
            nom: t(f"ecran.s7.vue.{nom}")
            for nom in (
                "regression",
                "confusion",
                "roc_pr",
                "seuil",
                "resume",
                "coefficients",
                "impurete",
                "arbre",
                "importances",
                "courbe",
                "composantes",
                "poids",
                "plan",
                "profils",
                "effectifs",
                "anomalies",
                "anormales",
            )
        }

    def retenir(self) -> None:
        """Décision motivée : la session refuse un modèle non comparé ou une métrique différente."""
        try:
            seuil = self._seuil()
        except ValueError:
            self.app.erreur(t("ecran.s7.nombre_invalide"))
            return
        travail = self.ctrl.travail_decider(self._id_choisi(), seuil, self.var_justification.get())
        self.app.executer(travail, nom=t("ecran.s7.retention"), au_succes=self._retenu)

    def _retenu(self, session: Session) -> None:
        self.ctrl.adopter(session)
        self.app.suivant()

    # Affichage
    def rafraichir(self) -> None:
        session = self.session
        if session.etat < Etat.MODELES:
            return
        resultat = session.resultats
        if self._cle is not resultat:
            self._cle = resultat
            self._preparer_modeles(resultat)
            self.diagnostiquer()
        elif session.etat >= Etat.DIAGNOSTIC and session.decision is not None:
            self._selectionner(session.decision.modele_id)
        self.mettre_a_jour_boutons()

    def _preparer_modeles(self, resultat: Any) -> None:
        if resultat.spec.probleme.avec_cible:
            retenus = [m.id for m in decision.modeles_a_retenir(resultat)]
        else:
            retenus = [m.id for m in resultat.modeles if m.ok]
        self._ids = {titre_technique(i): i for i in retenus}
        self.liste_modele.configure(values=list(self._ids))
        self._selectionner(decision.proposer(resultat).modele_id)
        self.var_seuil.set("")

    def _selectionner(self, identifiant: str) -> None:
        for titre, i in self._ids.items():
            if i == identifiant:
                self.var_modele.set(titre)
