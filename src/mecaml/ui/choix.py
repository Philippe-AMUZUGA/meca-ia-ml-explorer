"""
MécaML : liste modifiable de techniques choisies (prétraitement de S5, modèles de S6) et fenêtre de réglage.

Le catalogue vient du cœur (`catalog`) ; ce module ne décide de rien : il montre les entrées offertes, laisse
poser des paramètres dans les bornes de leur grille et rend des `ChoixTechnique`, que la spécification et la
session acceptent ou refusent.
"""

from __future__ import annotations

import ast
import tkinter as tk
from collections.abc import Callable, Sequence
from tkinter import ttk
from typing import Any

from mecaml.core import catalog, hyperparametres
from mecaml.core.catalog.entree import EntreeCatalogue
from mecaml.core.catalog.vote import COMPOSITIONS
from mecaml.core.types import ChoixTechnique
from mecaml.ui.textes import pedagogie, t, titre_technique
from mecaml.ui.widgets import Tableau, nombre, renvoi_livre

__all__ = ["DialogueTechnique", "ListeChoix", "decrire_parametres"]


def decrire_parametres(choix: ChoixTechnique) -> str:
    """Paramètres d'un choix, « libellé=valeur » séparés par des virgules."""
    reglages = {h.nom: h for h in hyperparametres.schema(choix.id)}
    morceaux = []
    for nom, valeur in choix.parametres_dict().items():
        if nom in reglages:
            texte = hyperparametres.texte_valeur(reglages[nom], valeur)
            morceaux.append(f"{hyperparametres.libelle(choix.id, nom)}={texte}")
        else:
            morceaux.append(f"{nom}={_texte(valeur)}")
    return ", ".join(morceaux)


def _texte(valeur: Any) -> str:
    if isinstance(valeur, float):
        return nombre(valeur, 4)
    if isinstance(valeur, tuple):
        return "(" + ", ".join(_texte(v) for v in valeur) + ")"
    return str(valeur)


def _lire(texte: str) -> Any:
    """Valeur écrite dans un champ : un littéral Python (nombre, texte, tuple), sinon le texte."""
    try:
        return ast.literal_eval(texte)
    except (ValueError, SyntaxError):
        return texte


class DialogueTechnique(tk.Toplevel):
    """Réglage d'une technique : réglages principaux, réglages avancés repliés, colonnes visées.

    Un modèle qui déclare ses réglages (`core.hyperparametres`) les montre avec libellé, plage permise,
    valeur par défaut et aide d'une ligne ; une valeur hors plage est refusée avec un message. Une technique
    sans schéma garde les paramètres de sa grille bornée.
    """

    def __init__(
        self,
        parent: tk.Misc,
        entree: EntreeCatalogue,
        choix: ChoixTechnique | None,
        colonnes: Sequence[str],
        au_valider: Callable[[ChoixTechnique], None],
        defauts: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(titre_technique(entree.id))
        self.transient(parent.winfo_toplevel())
        self.entree = entree
        self._base = choix.parametres_dict() if choix else dict(defauts or {})
        self._au_valider = au_valider
        self.reglages = hyperparametres.schema(entree.id)
        self.champs: dict[str, tk.StringVar] = {}
        self._listes: dict[str, tuple[tk.Listbox, list[str]]] = {}
        cadre = ttk.Frame(self, padding=14)
        cadre.grid(row=0, column=0, sticky="nsew")
        ttk.Label(cadre, text=titre_technique(entree.id), style="CarteGras.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ligne = self._poser_reglages(cadre, 1) if self.reglages else self._poser_grille(cadre, 1)
        self.liste = tk.Listbox(
            cadre, selectmode="extended", height=min(8, max(3, len(colonnes))), exportselection=False
        )
        self._colonnes = tuple(colonnes)
        for c in colonnes:
            self.liste.insert("end", c)
        for k, c in enumerate(colonnes):
            if choix and c in choix.colonnes:
                self.liste.selection_set(k)
        if colonnes:  # un modèle lit toutes les variables : pas de colonnes à choisir
            ttk.Label(cadre, text=t("choix.colonnes")).grid(row=ligne, column=0, sticky="nw", pady=(8, 0))
            self.liste.grid(row=ligne, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        self.erreur = ttk.Label(cadre, text="", style="Bloquant.TLabel", wraplength=520)
        self.erreur.grid(row=ligne + 1, column=0, columnspan=3, sticky="w")
        boutons = ttk.Frame(cadre)
        boutons.grid(row=ligne + 2, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(boutons, text=t("choix.valider"), style="Accent.TButton", command=self.valider).pack(
            side="left"
        )
        ttk.Button(boutons, text=t("choix.annuler"), command=self.destroy).pack(side="left", padx=(8, 0))
        if self.reglages:
            ttk.Button(boutons, text=t("choix.defauts"), command=self.retablir_defauts).pack(
                side="left", padx=(8, 0)
            )

    # Réglages déclarés
    def _poser_reglages(self, cadre: ttk.Frame, ligne: int) -> int:
        principaux = [h for h in self.reglages if not h.avance]
        avances = [h for h in self.reglages if h.avance]
        ligne = self._bloc(cadre, principaux, ligne)
        if avances:
            self._cadre_avance = ttk.Frame(cadre)
            self._bloc(self._cadre_avance, avances, 0)
            self._ouvert = tk.BooleanVar(value=any(h.nom in self._base for h in avances))
            self._bouton_avance = ttk.Button(cadre, command=self._basculer_avance)
            self._bouton_avance.grid(row=ligne, column=0, sticky="w", pady=(8, 0))
            self._cadre_avance.grid(row=ligne + 1, column=0, columnspan=3, sticky="ew")
            self._basculer_avance(inverser=False)
            ligne += 2
        return ligne

    def _basculer_avance(self, inverser: bool = True) -> None:
        if inverser:
            self._ouvert.set(not self._ouvert.get())
        if self._ouvert.get():
            self._cadre_avance.grid()
        else:
            self._cadre_avance.grid_remove()
        self._bouton_avance.configure(text=("▾ " if self._ouvert.get() else "▸ ") + t("choix.avance"))

    def _bloc(self, cadre: ttk.Frame, reglages: Sequence[hyperparametres.Hyperparametre], ligne: int) -> int:
        ident = self.entree.id
        for h in reglages:
            ttk.Label(cadre, text=hyperparametres.libelle(ident, h.nom)).grid(
                row=ligne, column=0, sticky="w", pady=(6, 0)
            )
            variable = tk.StringVar(value=hyperparametres.texte_valeur(h, self._base.get(h.nom, h.defaut)))
            self.champs[h.nom] = variable
            self._champ(cadre, h, variable).grid(row=ligne, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
            defaut = hyperparametres.texte_valeur(h, h.defaut)
            plage = t("choix.plage", bornes=hyperparametres.texte_bornes(h), valeur=defaut)
            ttk.Label(cadre, text=plage, style="Doux.TLabel").grid(
                row=ligne, column=2, sticky="w", pady=(6, 0)
            )
            ttk.Label(cadre, text=hyperparametres.aide(ident, h.nom), style="Doux.TLabel").grid(
                row=ligne + 1, column=0, columnspan=3, sticky="w"
            )
            ligne += 2
        return ligne

    def _champ(
        self, cadre: ttk.Frame, h: hyperparametres.Hyperparametre, variable: tk.StringVar
    ) -> tk.Widget:
        if h.genre is hyperparametres.Genre.BOOLEEN:
            valeurs = [hyperparametres.texte_valeur(h, True), hyperparametres.texte_valeur(h, False)]
            return ttk.Combobox(cadre, textvariable=variable, values=valeurs, state="readonly", width=22)
        if h.genre is hyperparametres.Genre.CHOIX:
            valeurs = [str(c) for c in h.choix] + [hyperparametres.texte_valeur(h, v) for v, _ in h.speciaux]
            return ttk.Combobox(cadre, textvariable=variable, values=valeurs, state="readonly", width=22)
        if h.genre is hyperparametres.Genre.LISTE:
            return self._liste_de_modeles(cadre, h, variable)
        return ttk.Entry(cadre, textvariable=variable, width=24)

    def _liste_de_modeles(
        self, cadre: ttk.Frame, h: hyperparametres.Hyperparametre, variable: tk.StringVar
    ) -> tk.Listbox:
        ids = _candidats(self.entree)
        boite = tk.Listbox(
            cadre, selectmode="multiple", height=min(6, len(ids)), exportselection=False, width=34
        )
        actuels = {i.strip() for i in variable.get().split(",")}
        for k, i in enumerate(ids):
            boite.insert("end", titre_technique(i))
            if i in actuels:
                boite.selection_set(k)

        self._listes[h.nom] = (boite, ids)
        return boite

    def retablir_defauts(self) -> None:
        """Remet chaque réglage à sa valeur par défaut."""
        for h in self.reglages:
            self.champs[h.nom].set(hyperparametres.texte_valeur(h, h.defaut))
            if h.nom in self._listes:
                boite, ids = self._listes[h.nom]
                boite.selection_clear(0, "end")
                for k, i in enumerate(ids):
                    if i in h.defaut:
                        boite.selection_set(k)

    # Grille bornée des techniques sans schéma
    def _poser_grille(self, cadre: ttk.Frame, ligne: int) -> int:
        for nom, candidats in self.entree.grille_bornee:
            ttk.Label(cadre, text=nom).grid(row=ligne, column=0, sticky="w", pady=3)
            variable = tk.StringVar(value=_texte(self._base[nom]) if nom in self._base else "")
            ttk.Combobox(cadre, textvariable=variable, values=[_texte(c) for c in candidats], width=26).grid(
                row=ligne, column=1, columnspan=2, sticky="ew", padx=(8, 0)
            )
            self.champs[nom] = variable
            ligne += 1
        return ligne

    def construire_choix(self) -> ChoixTechnique:
        """Choix d'après les champs ; un champ vide ou égal au défaut garde la valeur du modèle."""
        parametres = {k: v for k, v in self._base.items() if k not in self.champs}
        for nom, (boite, ids) in self._listes.items():  # la sélection de la liste fait foi
            self.champs[nom].set(", ".join(ids[k] for k in boite.curselection()))
        if self.reglages:
            saisies = {k: v.get() for k, v in self.champs.items()}
            parametres.update(hyperparametres.valider(self.entree.id, saisies))
        else:
            for nom, variable in self.champs.items():
                texte = variable.get().strip()
                if texte:
                    parametres[nom] = _lire(texte)
        visees = tuple(self._colonnes[k] for k in self.liste.curselection())
        return ChoixTechnique.creer(self.entree.id, colonnes=visees, **parametres)

    def destroy(self) -> None:
        """Ferme la fenêtre et lâche les variables Tk dans le fil principal, non au ramasse-miettes."""
        self.champs.clear()
        self._listes.clear()
        super().destroy()

    def valider(self) -> None:
        try:
            choix = self.construire_choix()
        except (TypeError, ValueError) as erreur:
            self.erreur.configure(text=str(erreur))
            return
        self._au_valider(choix)
        self.destroy()


def _candidats(entree: EntreeCatalogue) -> list[str]:
    """Modèles qu'un vote ou un empilement peut combiner : même problème, aucune composition."""
    return [e.id for e in catalog.entrees(famille=entree.famille) if e.id not in COMPOSITIONS]


class ListeChoix(ttk.Frame):
    """Tableau de techniques choisies, avec ajout, réglage, retrait et, si l'ordre compte, déplacement."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        offertes: Callable[[], Sequence[EntreeCatalogue]],
        colonnes: Callable[[], Sequence[str]],
        note: Callable[[ChoixTechnique], str] = lambda _c: "",
        ordonnee: bool = True,
        defauts: Callable[[EntreeCatalogue], dict[str, Any]] = lambda _e: {},
        au_changement: Callable[[], None] = lambda: None,
        hauteur: int = 7,
    ) -> None:
        super().__init__(parent, style="Carte.TFrame")
        self._offertes, self._colonnes_de, self._note = offertes, colonnes, note
        self._ordonnee, self._au_changement, self._defauts = ordonnee, au_changement, defauts
        self._par_titre: dict[str, EntreeCatalogue] = {}
        self._choix: list[ChoixTechnique] = []
        self.grid_columnconfigure(0, weight=1)
        self.tableau = Tableau(
            self,
            [
                ("rang", "#", 36, "center"),
                ("technique", t("choix.technique"), 330, "w"),
                ("colonnes", t("choix.colonnes_court"), 170, "w"),
                ("parametres", t("choix.parametres"), 200, "w"),
                ("note", t("choix.note"), 200, "w"),
            ],
            hauteur=hauteur,
            selection="browse",
        )
        self.tableau.grid(row=0, column=0, sticky="ew")
        self.detail = ttk.Label(self, text="", style="CarteDoux.TLabel", wraplength=820, justify="left")
        self.detail.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.tableau.lier_selection(lambda _s: self._montrer_detail())
        self.tableau.arbre.bind("<Double-1>", lambda _e: self.regler())
        self.boutons = self._barre()
        self.boutons.grid(row=2, column=0, sticky="w", pady=(8, 0))

    def _barre(self) -> ttk.Frame:
        barre = ttk.Frame(self, style="Carte.TFrame")
        self.var_ajout = tk.StringVar()
        self.liste_ajout = ttk.Combobox(barre, textvariable=self.var_ajout, state="readonly", width=34)
        self.liste_ajout.grid(row=0, column=0, padx=(0, 6))
        self._boutons = [
            ttk.Button(barre, text=t("choix.ajouter"), command=self.ajouter),
            ttk.Button(barre, text=t("choix.regler"), command=self.regler),
            ttk.Button(barre, text=t("choix.retirer"), command=self.retirer),
        ]
        if self._ordonnee:
            self._boutons += [
                ttk.Button(barre, text=t("choix.monter"), command=lambda: self.deplacer(-1)),
                ttk.Button(barre, text=t("choix.descendre"), command=lambda: self.deplacer(1)),
            ]
        for k, bouton in enumerate(self._boutons, start=1):
            bouton.grid(row=0, column=k, padx=3)
        return barre

    # Contenu
    def definir(self, choix: Sequence[ChoixTechnique]) -> None:
        self._choix = list(choix)
        self.actualiser()

    @property
    def choix(self) -> tuple[ChoixTechnique, ...]:
        return tuple(self._choix)

    def actualiser(self) -> None:
        offertes = {titre_technique(e.id): e for e in self._offertes()}
        self._par_titre = offertes
        self.liste_ajout.configure(values=sorted(offertes))
        self.tableau.vider()
        for k, c in enumerate(self._choix):
            self.tableau.ajouter(
                (k + 1, titre_technique(c.id), ", ".join(c.colonnes), decrire_parametres(c), self._note(c)),
                iid=str(k),
            )

    def basculer(self, actif: bool) -> None:
        etat = "normal" if actif else "disabled"
        for bouton in self._boutons:
            bouton.configure(state=etat)
        self.liste_ajout.configure(state="readonly" if actif else "disabled")

    def _selection(self) -> int | None:
        choisi = self.tableau.selection()
        return int(choisi[0]) if choisi else None

    def _montrer_detail(self) -> None:
        k = self._selection()
        if k is None:
            return
        entree = catalog.entree(self._choix[k].id)
        morceaux = [titre_technique(entree.id), t("choix.cout", cout=t(f"cout.{entree.cout.value}"))]
        livre = renvoi_livre(entree.ref_livre)
        if livre:
            morceaux.append(t("commun.voir_livre", renvoi=livre))
        texte = " | ".join(morceaux)
        pedago = pedagogie(entree.texte_id)
        if pedago:
            texte += "\n" + "\n".join(pedago[c] for c in ("quoi", "pourquoi", "regarder") if c in pedago)
        self.detail.configure(text=texte)

    # Actions
    def ajouter(self) -> None:
        entree = self._par_titre.get(self.var_ajout.get())
        if entree is None:
            return
        self.ouvrir_reglage(entree, None, None)

    def regler(self) -> None:
        k = self._selection()
        if k is not None:
            self.ouvrir_reglage(catalog.entree(self._choix[k].id), self._choix[k], k)

    def ouvrir_reglage(self, entree: EntreeCatalogue, choix: ChoixTechnique | None, rang: int | None) -> None:
        """Fenêtre de réglage d'une technique ajoutée ou existante."""

        def valider(nouveau: ChoixTechnique) -> None:
            if rang is None:
                self._choix.append(nouveau)
            else:
                self._choix[rang] = nouveau
            self.actualiser()
            self._au_changement()

        DialogueTechnique(self, entree, choix, self._colonnes_de(), valider, self._defauts(entree))

    def retirer(self) -> None:
        k = self._selection()
        if k is not None:
            del self._choix[k]
            self.actualiser()
            self._au_changement()

    def deplacer(self, sens: int) -> None:
        k = self._selection()
        if k is None or not 0 <= k + sens < len(self._choix):
            return
        self._choix[k], self._choix[k + sens] = self._choix[k + sens], self._choix[k]
        self.actualiser()
        self.tableau.arbre.selection_set(str(k + sens))
        self._au_changement()

    def ajouter_choix(self, choix: ChoixTechnique) -> None:
        """Ajoute un choix déjà construit (sans passer par la fenêtre de réglage)."""
        self._choix.append(choix)
        self.actualiser()
        self._au_changement()
