"""MécaML : écran S3, objectif du projet (cible, problème, rôles, structure, métrique), audit rejoué."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import pandas as pd

from mecaml.core import metrics, renvois
from mecaml.core.audit import proposer_structure
from mecaml.core.codegen import LIBELLES_VALIDATION, STRUCTURES_FR
from mecaml.core.guards import decimale
from mecaml.core.session import Etat, FixerObjectif, Session
from mecaml.core.types import (
    GRAINE_CONTRAT,
    Probleme,
    Role,
    SpecProjet,
    identifiant_metrique,
    validations_permises,
)
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import pedagogie, t
from mecaml.ui.widgets import (
    PARTS_DE_TEST,
    PanneauConstats,
    Tableau,
    carte,
    champ,
    lire_decimal,
    renvoi_livre,
)

__all__ = ["EcranObjectif"]

PROBLEMES = (Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE)
DEDUITE = ""  # validation déduite de la structure par le découpage


class EcranObjectif(Ecran):
    """Ce que l'on cherche à prédire, et ce que chaque colonne y joue."""

    NUMERO = 3
    CLE = "s3"

    def construire(self, corps: ttk.Frame) -> None:
        self.var_probleme = tk.StringVar(value=Probleme.REGRESSION.value)
        self.var_cible = tk.StringVar()
        self.var_role = tk.StringVar()
        self.var_structure = tk.StringVar()
        self.var_metrique = tk.StringVar()
        self.var_positive = tk.StringVar()
        self.var_validation = tk.StringVar()
        self.var_cout_fp = tk.StringVar(value="1")
        self.var_cout_fn = tk.StringVar(value="1")
        self.var_part = tk.StringVar(value=decimale(0.2, 2))
        self.var_plis = tk.StringVar(value="5")
        self.var_graine = tk.StringVar(value=str(GRAINE_CONTRAT))
        self.roles: dict[str, Role] = {}
        self._spec_proposee: SpecProjet | None = None
        self._cle_proposition: Any = None
        self._classes: list[Any] = []
        self._metriques: dict[str, str] = {}
        self._validations: dict[str, str] = {}
        self._carte_objectif(corps).grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._carte_roles(corps).grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._carte_constats(corps).grid(row=2, column=0, sticky="ew")

    # Construction
    def _carte_objectif(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s3.objectif.titre"))
        cadre.grid_columnconfigure(1, weight=1)
        radios = ttk.Frame(cadre, style="Carte.TFrame")
        for k, probleme in enumerate(PROBLEMES):
            ttk.Radiobutton(
                radios,
                text=t(f"probleme.{probleme.value}"),
                value=probleme.value,
                variable=self.var_probleme,
                command=self.changer_probleme,
            ).grid(row=0, column=k, padx=(0, 18))
        champ(cadre, 0, t("ecran.s3.probleme"), radios)
        self.liste_cible = ttk.Combobox(cadre, textvariable=self.var_cible, state="readonly")
        self.liste_cible.bind("<<ComboboxSelected>>", lambda _e: self.proposer(self.var_cible.get()))
        champ(cadre, 1, t("ecran.s3.cible"), self.liste_cible)
        self.liste_structure = ttk.Combobox(
            cadre, textvariable=self.var_structure, state="readonly", values=list(STRUCTURES_FR.values())
        )
        self.liste_structure.bind("<<ComboboxSelected>>", lambda _e: self._actualiser_validations())
        champ(cadre, 2, t("ecran.s3.structure"), self.liste_structure)
        self.liste_metrique = ttk.Combobox(cadre, textvariable=self.var_metrique, state="readonly")
        self.liste_metrique.bind("<<ComboboxSelected>>", lambda _e: self._actualiser_cout())
        champ(cadre, 3, t("ecran.s3.metrique"), self.liste_metrique)
        zone = ttk.Frame(cadre, style="Carte.TFrame")
        zone.grid(row=4, column=1, sticky="w")
        self.aide_metrique = ttk.Label(
            zone, text="", style="CarteDoux.TLabel", wraplength=640, justify="left"
        )
        self.aide_metrique.grid(row=0, column=0, sticky="w")
        self.cadre_cout = self._cadre_cout(zone)
        self.cadre_cout.grid(row=1, column=0, sticky="w")
        self.liste_positive = ttk.Combobox(cadre, textvariable=self.var_positive, state="readonly")
        champ(cadre, 5, t("ecran.s3.positive"), self.liste_positive)
        self.liste_validation = ttk.Combobox(cadre, textvariable=self.var_validation, state="readonly")
        champ(cadre, 6, t("ecran.s3.validation"), self.liste_validation)
        champ(
            cadre,
            7,
            t("ecran.s3.part_test"),
            ttk.Spinbox(cadre, textvariable=self.var_part, values=PARTS_DE_TEST, width=8),
        )
        champ(
            cadre,
            8,
            t("ecran.s3.plis"),
            ttk.Spinbox(cadre, textvariable=self.var_plis, from_=2, to=10, width=8),
        )
        champ(cadre, 9, t("ecran.s3.graine"), ttk.Entry(cadre, textvariable=self.var_graine, width=10))
        return cadre

    def _cadre_cout(self, parent: ttk.Frame) -> ttk.Frame:
        cadre = ttk.Frame(parent, style="Carte.TFrame")
        ttk.Label(cadre, text=t("ecran.s3.cout_fp"), style="CarteDoux.TLabel").grid(row=0, column=0)
        ttk.Entry(cadre, textvariable=self.var_cout_fp, width=6).grid(row=0, column=1, padx=(4, 14))
        ttk.Label(cadre, text=t("ecran.s3.cout_fn"), style="CarteDoux.TLabel").grid(row=0, column=2)
        ttk.Entry(cadre, textvariable=self.var_cout_fn, width=6).grid(row=0, column=3, padx=4)
        return cadre

    def _carte_roles(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s3.roles.titre"))
        cadre.grid_columnconfigure(0, weight=1)
        colonnes = [
            ("nom", t("ecran.s2.col.nom"), 220, "w"),
            ("role", t("ecran.s2.col.role"), 160, "w"),
            ("raison", t("ecran.s2.col.raison"), 480, "w"),
        ]
        self.tab_roles = Tableau(cadre, colonnes, hauteur=9)
        self.tab_roles.grid(row=0, column=0, columnspan=4, sticky="ew")
        ttk.Label(cadre, text=t("ecran.s3.role_choisi"), style="CarteDoux.TLabel").grid(
            row=1, column=0, sticky="e", pady=8
        )
        self.liste_role = ttk.Combobox(
            cadre,
            textvariable=self.var_role,
            state="readonly",
            width=24,
            values=[t(f"role.{r.value}") for r in Role if r is not Role.CIBLE],
        )
        self.liste_role.grid(row=1, column=1, padx=8)
        self.bouton(cadre, "ecran.s3.appliquer_role", self.appliquer_role, row=1, column=2)
        self.bouton(
            cadre,
            "ecran.s3.reproposer",
            lambda: self.proposer(self.var_cible.get() or None),
            row=1,
            column=3,
            padx=8,
        )
        return cadre

    def _carte_constats(self, parent: ttk.Frame) -> ttk.LabelFrame:
        cadre = carte(parent, t("ecran.s3.constats.titre"))
        cadre.grid_columnconfigure(0, weight=1)
        self.constats = PanneauConstats(cadre)
        self.constats.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.bouton(
            cadre, "ecran.s3.fixer", self.fixer, principal=True, row=1, column=0, sticky="w", pady=(10, 0)
        )
        self.bouton(
            cadre,
            "ecran.s2.acquitter_choix",
            lambda: self.app.acquitter(self.constats.codes_alertes()),
            condition=lambda: self.session.etat >= Etat.OBJECTIF,
            row=1,
            column=1,
            padx=8,
            pady=(10, 0),
        )
        self.bouton(
            cadre,
            "ecran.s3.continuer",
            lambda: self.app.afficher(4),
            condition=lambda: self.session.etat >= Etat.OBJECTIF,
            row=1,
            column=2,
            pady=(10, 0),
        )
        return cadre

    # Proposition du cœur
    def proposer(self, cible: str | None, sans_cible: bool = False) -> None:
        """Demande au cœur une spécification pour cette cible ; les champs se remplissent d'après elle."""
        if self.session.table is None:
            return
        travail = self.ctrl.travail_proposer(cible, sans_cible)
        self.app.executer(travail, nom=t("ecran.s3.proposition"), au_succes=self._proposition)

    def _proposition(self, spec: SpecProjet) -> None:
        self._spec_proposee = spec
        self._remplir_depuis(spec)
        self.mettre_a_jour_boutons()

    def changer_probleme(self) -> None:
        """Le problème change : la cible s'enlève ou se choisit ; métriques et validations suivent."""
        if self.var_probleme.get() == Probleme.SANS_CIBLE.value:
            self.proposer(None, sans_cible=True)
        elif self.roles and not any(r is Role.CIBLE for r in self.roles.values()):
            self.proposer(None)
        else:
            self._actualiser_metriques()
            self._actualiser_validations()

    # Champs, depuis la spécification
    def _remplir_depuis(self, spec: SpecProjet) -> None:
        table = self.session.table
        self.var_probleme.set(spec.probleme.value)
        self.liste_cible.configure(values=list(table.colonnes))
        self.var_cible.set(spec.cible or "")
        self.roles = dict(spec.roles)
        self.var_structure.set(STRUCTURES_FR[spec.structure])
        self._classes = _classes_de(table.donnees, spec.cible) if spec.probleme.avec_cible else []
        self._actualiser_metriques(spec.metrique)
        self.liste_positive.configure(values=[str(c) for c in self._classes])
        positive = spec.classe_positive
        self.var_positive.set("" if positive is None else str(positive))
        self.var_part.set(decimale(spec.part_test, 2))
        self.var_plis.set(str(spec.n_plis))
        self.var_graine.set(str(spec.graine))
        self._actualiser_validations(spec.validation)
        self._remplir_roles(table.colonnes)

    def _remplir_roles(self, colonnes: tuple[str, ...]) -> None:
        raisons = {p.nom: p.raison for p in self.ctrl.profils}
        self.tab_roles.vider()
        for nom in colonnes:
            role = self.roles.get(nom, Role.IGNOREE)
            self.tab_roles.ajouter((nom, t(f"role.{role.value}"), raisons.get(nom, "")), iid=nom)

    def _actualiser_metriques(self, choix: str | None = None) -> None:
        probleme = Probleme(self.var_probleme.get())
        offertes = [m for m in metrics.METRIQUES if probleme in m.problemes and m.scalaire]
        self._metriques = {m.libelle: m.id for m in offertes}
        self.liste_metrique.configure(values=list(self._metriques))
        voulue = identifiant_metrique(choix) if choix else None
        courant = next((m.libelle for m in offertes if m.id == voulue), None)
        if courant is None and offertes:
            courant = offertes[0].libelle
        self.var_metrique.set(courant or "")
        self.liste_positive.configure(state="readonly" if probleme is Probleme.CLASSIFICATION else "disabled")
        self._actualiser_cout()

    def _actualiser_cout(self) -> None:
        identifiant = self._metriques.get(self.var_metrique.get(), "")
        if identifiant == "cout":
            self.cadre_cout.grid()
        else:
            self.cadre_cout.grid_remove()
        pedagogie_metrique = pedagogie(f"metrique.{identifiant}") or {}
        phrases = [pedagogie_metrique[c] for c in ("quoi", "regarder") if c in pedagogie_metrique]
        livre = renvoi_livre(renvois.metrique(identifiant)) if identifiant in renvois.METRIQUES else ""
        if livre:
            voir = t("commun.voir_livre", renvoi=livre)
            phrases.append(voir[0].upper() + voir[1:] + ".")
        self.aide_metrique.configure(text=" ".join(phrases))

    def _actualiser_validations(self, choix: str | None = None) -> None:
        structure = next((s for s, nom in STRUCTURES_FR.items() if nom == self.var_structure.get()), None)
        probleme = Probleme(self.var_probleme.get())
        permises = sorted(validations_permises(structure, probleme)) if structure else []
        self._validations = {LIBELLES_VALIDATION[v]: v for v in permises}
        self.liste_validation.configure(values=[t("ecran.s3.validation_deduite"), *self._validations])
        courante = next((nom for nom, v in self._validations.items() if v == choix), None)
        self.var_validation.set(courante or t("ecran.s3.validation_deduite"))

    # Rôles
    def appliquer_role(self) -> None:
        """Donne le rôle choisi aux colonnes sélectionnées."""
        voulu = next((r for r in Role if t(f"role.{r.value}") == self.var_role.get()), None)
        if voulu is None:
            return
        for nom in self.tab_roles.selection():
            if self.roles.get(nom) is not Role.CIBLE:
                self.roles[nom] = voulu
        self._remplir_roles(self.session.table.colonnes)
        self._actualiser_structure()

    def _actualiser_structure(self) -> None:
        """La structure suit les rôles posés (groupe, date) : celle que le cœur déduit de ces rôles."""
        structure = proposer_structure(tuple(self.roles.items()))
        self.var_structure.set(STRUCTURES_FR[structure])
        self._actualiser_validations()

    # Objectif
    def construire_spec(self) -> SpecProjet:
        """Spécification d'après les champs ; le cœur refuse ce qui est incohérent (`ValueError`)."""
        probleme = Probleme(self.var_probleme.get())
        structure = next(s for s, nom in STRUCTURES_FR.items() if nom == self.var_structure.get())
        roles = dict(self.roles)
        cible = self.var_cible.get() or None
        if probleme.avec_cible and cible:
            roles[cible] = Role.CIBLE
        elif not probleme.avec_cible:
            roles = {c: r for c, r in roles.items() if r is not Role.CIBLE}
        return SpecProjet(
            probleme=probleme,
            cible=cible if probleme.avec_cible else None,
            roles=tuple(roles.items()),
            metrique=self._metrique_choisie(),
            structure=structure,
            part_test=lire_decimal(self.var_part.get()),
            n_plis=int(self.var_plis.get()),
            graine=int(self.var_graine.get()),
            validation=self._validations.get(self.var_validation.get()),
            classe_positive=self._positive_choisie(probleme),
        )

    def _metrique_choisie(self) -> str:
        identifiant = self._metriques.get(self.var_metrique.get(), "")
        if identifiant == "cout":
            return f"cout:fp={float(self.var_cout_fp.get()):g},fn={float(self.var_cout_fn.get()):g}"
        return identifiant

    def _positive_choisie(self, probleme: Probleme) -> Any:
        if probleme is not Probleme.CLASSIFICATION or len(self._classes) != 2:
            return None
        return next((c for c in self._classes if str(c) == self.var_positive.get()), None)

    def fixer(self) -> None:
        """Envoie l'objectif à la session : l'audit est rejoué avec la cible et les rôles choisis."""
        try:
            spec = self.construire_spec()
        except (ValueError, TypeError, StopIteration) as erreur:
            self.app.erreur(erreur if str(erreur) else t("ecran.s3.spec_invalide"))
            return
        self.app.executer(
            self.ctrl.travail_actions(FixerObjectif(spec)), nom=t("ecran.s3.audit"), au_succes=self._fixe
        )

    def _fixe(self, session: Session) -> None:
        self.ctrl.adopter(session)
        self.rafraichir()

    # Affichage
    def rafraichir(self) -> None:
        table = self.session.table
        if table is None:
            return
        if self.session.spec is not None and self.session.etat >= Etat.OBJECTIF:
            self._remplir_depuis(self.session.spec)
            self._spec_proposee = self.session.spec
        elif self._cle_proposition is not table:
            self._cle_proposition = table
            self.proposer(None)
        visible = self.session.etat >= Etat.OBJECTIF
        self.constats.afficher(self.session.constats if visible else (), self.session.acquittes)
        self.mettre_a_jour_boutons()


def _classes_de(donnees: pd.DataFrame, cible: str | None) -> list[Any]:
    if cible is None or cible not in donnees.columns:
        return []
    return pd.unique(donnees[cible].dropna()).tolist()
