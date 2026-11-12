"""
MécaML : contrôleur de l'interface, sans aucune dépendance à Tk.

Le contrôleur garde la session courante et ce que l'interface doit encore connaître (le chargement, le
découpage, les profils de colonnes). Chaque méthode `travail_*` rend une fonction que l'on passe à une
`Tache` : elle ne touche qu'à des valeurs gelées (la session est immuable, la table partagée en lecture)
et rend sa valeur ; c'est l'interface qui l'adopte (`adopter_*`) une fois la tâche finie, dans le fil de Tk.

Aucune règle ne vit ici : chaque refus, chaque proposition et chaque calcul vient du cœur (session,
audit, découpage, apprentissage, évaluation, rapport). Ce module enchaîne seulement des appels.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mecaml import cli
from mecaml.core import audit, decision, evaluate, explain, preprocess, report, split, train, unsupervised
from mecaml.core import io as fichiers
from mecaml.core.explain import ErreurExplication
from mecaml.core.session import (
    Acquitter,
    Action,
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Etat,
    Revenir,
    Session,
    Tester,
)
from mecaml.core.tasks import Annulation, Progression
from mecaml.core.types import ChoixTechnique, Constat, Probleme, SpecProjet, Table

__all__ = ["Chargee", "Controleur", "Verification", "Vue"]

Avancer = Callable[[Progression], None]


@dataclass(frozen=True)
class Chargee:
    """Résultat de la lecture des fichiers : la session en S1 et ce que l'on en dit."""

    session: Session
    chargement: fichiers.Chargement
    chargement_test: fichiers.Chargement | None = None
    octets: int = 0  # mémoire de la table, mesurée dans le fil de lecture (longue sur une grande table)


@dataclass(frozen=True)
class Audit:
    """Session en S2 et profil des colonnes."""

    session: Session
    profils: tuple[audit.ProfilColonne, ...]


@dataclass(frozen=True)
class Scelle:
    """Session en S4 et découpage déclaré."""

    session: Session
    decoupage: split.Decoupage | None


@dataclass(frozen=True)
class Verification:
    """Constats de l'audit rejoué sur la spécification complète, et avertissements de prétraitement."""

    constats: tuple[Constat, ...]
    avis: tuple[str, ...] = ()


@dataclass(frozen=True)
class Vue:
    """Ce qu'un écran de lecture affiche : figures, tableaux et textes déjà calculés par le cœur."""

    figures: tuple[tuple[str, Any], ...] = ()
    tableaux: tuple[tuple[str, pd.DataFrame], ...] = ()
    textes: tuple[tuple[str, str], ...] = ()
    seuil_cout: float | None = None


@dataclass
class Controleur:
    """Session courante et état annexe de l'interface."""

    session: Session = field(default_factory=lambda: Session(auditeur=audit.auditer))
    chargement: fichiers.Chargement | None = None
    chargement_test: fichiers.Chargement | None = None
    octets: int = 0
    profils: tuple[audit.ProfilColonne, ...] = ()
    decoupage: split.Decoupage | None = None
    vues: dict[str, Vue] = field(default_factory=dict)
    recherches: list[train.ResultatRecherche] = field(default_factory=list)
    export: report.Export | None = None

    # Lecture
    @property
    def etat(self) -> Etat:
        return self.session.etat

    @property
    def table(self) -> Table | None:
        return self.session.table

    # Adoption des résultats (fil de Tk)
    def adopter(self, session: Session) -> None:
        """Retient la nouvelle session ; ce qui dépend d'un état invalidé est oublié."""
        if session.resultats is not self.session.resultats:
            self.recherches.clear()  # une recherche se lit sur les plis de la comparaison qu'elle accompagne
        self.session = session
        if session.etat < Etat.DONNEES:
            self.chargement = self.chargement_test = None
            self.profils = ()
        if session.etat < Etat.DECOUPE:
            self.decoupage = None
        if session.etat < Etat.DIAGNOSTIC:
            self.vues.pop("diagnostic", None)
        if session.etat < Etat.TESTE:
            self.vues.pop("test", None)
        if session.etat < Etat.EXPORTE:
            self.export = None

    def adopter_chargee(self, chargee: Chargee) -> None:
        self.chargement, self.chargement_test = chargee.chargement, chargee.chargement_test
        self.octets = chargee.octets
        self.profils = ()
        self.decoupage = None
        self.vues.clear()
        self.adopter(chargee.session)

    def adopter_audit(self, resultat: Audit) -> None:
        self.profils = resultat.profils
        self.adopter(resultat.session)

    def adopter_scelle(self, resultat: Scelle) -> None:
        self.adopter(resultat.session)
        self.decoupage = resultat.decoupage

    # Actions rapides (sans audit, donc sans fil)
    def acquitter(self, codes: frozenset[str]) -> None:
        self.adopter(self.session.appliquer(Acquitter(codes)))

    # Travaux de fond : chaque méthode rend une fonction (annulation, avancer) -> valeur
    def travail_charger(
        self,
        chemin: str | Path,
        feuille: str | None = None,
        chemin_test: str | Path | None = None,
        feuille_test: str | None = None,
    ) -> Callable[[Annulation, Avancer], Chargee]:
        """Lit le fichier (et le jeu de test à part, s'il y en a un) : nouveau projet en S1."""
        base = self.session

        def travail(annulation: Annulation, avancer: Avancer) -> Chargee:
            def lire(cible: str | Path, nom: str | None, etape: str) -> fichiers.Chargement:
                def suivre(fraction: float, message: str) -> None:
                    avancer(Progression(etape, int(fraction * 100), 100, message))

                return fichiers.charger(cible, feuille=nom, progression=suivre, annuler=annulation)

            principal = lire(chemin, feuille, "Lecture des données")
            test = (
                lire(chemin_test or chemin, feuille_test, "Lecture du test")
                if (chemin_test or feuille_test)
                else None
            )
            neuve = base.appliquer(Revenir(Etat.VIDE)) if base.etat > Etat.VIDE else base
            session = neuve.appliquer(Charger(principal.table))
            return Chargee(session, principal, test, principal.table.taille_octets())

        return travail

    def travail_auditer(self) -> Callable[[Annulation, Avancer], Audit]:
        """S1 vers S2 : audit sans cible et profil des colonnes."""
        base = self.session

        def travail(annulation: Annulation, avancer: Avancer) -> Audit:
            avancer(Progression("Audit", 0, 2, "Audit des données"))
            session = base.appliquer(Auditer())
            avancer(Progression("Audit", 1, 2, "Profil des colonnes"))
            return Audit(session, audit.profiler(session.table))

        return travail

    def travail_proposer(
        self, cible: str | None, sans_cible: bool = False
    ) -> Callable[[Annulation, Avancer], SpecProjet]:
        """Spécification proposée par le cœur pour cette cible : rôles, problème, structure, métrique."""
        table = self.session.table

        def travail(annulation: Annulation, avancer: Avancer) -> SpecProjet:
            avancer(Progression("Proposition", 0, 1, "Proposition des rôles"))
            projet = cli.proposer_projet(table, cible=cible, sans_cible=sans_cible)
            return SpecProjet.depuis_dict(projet["projet"])

        return travail

    def travail_actions(self, *actions: Action) -> Callable[[Annulation, Avancer], Session]:
        """Applique des actions de session à la suite ; chacune peut rejouer l'audit (donc un fil)."""
        base = self.session

        def travail(annulation: Annulation, avancer: Avancer) -> Session:
            session = base
            for action in actions:
                avancer(Progression("Étape", 0, 1, type(action).__name__))
                session = session.appliquer(action)
            return session

        return travail

    def travail_sceller(
        self, spec: SpecProjet, methode: split.Methode | None
    ) -> Callable[[Annulation, Avancer], Scelle]:
        """S3 vers S4 : le test est mis sous scellé selon la méthode, puis le coffre va à la session."""
        base, table, test = self.session, self.session.table, self.chargement_test

        def travail(annulation: Annulation, avancer: Avancer) -> Scelle:
            avancer(Progression("Découpage", 0, 2, "Mise sous scellé du test"))
            if not spec.probleme.avec_cible:
                return Scelle(base.appliquer(Decouper(None, spec=spec)), None)
            if test is not None:
                decoupage, coffre = split.sceller_fichier(table, test.table, spec)
            else:
                decoupage, coffre = split.sceller(table, spec, methode=methode)
            avancer(Progression("Découpage", 1, 2, "Audit du découpage"))
            return Scelle(base.appliquer(Decouper(coffre, spec=spec)), decoupage)

        return travail

    def travail_proposer_pretraitement(self) -> Callable[[Annulation, Avancer], tuple[ChoixTechnique, ...]]:
        """Prétraitement proposé sur les lignes d'apprentissage seules (jamais le test)."""
        spec, table, coffre = self.session.spec, self.session.table, self.session.coffre

        def travail(annulation: Annulation, avancer: Avancer) -> tuple[ChoixTechnique, ...]:
            return preprocess.proposer_pretraitement(spec, _apprentissage(table, coffre))

        return travail

    def travail_verifier(self, spec: SpecProjet) -> Callable[[Annulation, Avancer], Verification]:
        """Rejoue l'audit sur la spécification complète : cible passée et découpage se lisent d'avance."""
        table, coffre = self.session.table, self.session.coffre

        def travail(annulation: Annulation, avancer: Avancer) -> Verification:
            avancer(Progression("Contrôle", 0, 1, "Contrôle de la spécification"))
            constats = audit.auditer(table, spec, coffre)
            avis = preprocess.controles_pretraitement(spec, _apprentissage(table, coffre))
            return Verification(tuple(constats), tuple(avis))

        return travail

    def travail_entrainer(self, spec: SpecProjet) -> Callable[[Annulation, Avancer], Session]:
        """S5 vers S6 : validation croisée de chaque modèle, puis résultats confiés à la session."""
        base, table, coffre = self.session, self.session.table, self.session.coffre

        def travail(annulation: Annulation, avancer: Avancer) -> Session:
            resultat = train.entrainer(spec, table, coffre, annulation=annulation, progression=avancer)
            return base.appliquer(Entrainer(resultat, spec=resultat.spec))

        return travail

    def travail_recherche(
        self, id_modele: str, mode: str, essais: int
    ) -> Callable[[Annulation, Avancer], train.ResultatRecherche]:
        """Recherche d'hyperparamètres d'un modèle comparé, sur les plis de la comparaison (test scellé)."""
        spec, table, coffre = self.session.spec, self.session.table, self.session.coffre

        def travail(annulation: Annulation, avancer: Avancer) -> train.ResultatRecherche:
            choix = next(c for c in spec.modeles if c.id == id_modele)
            return train.rechercher(
                spec,
                choix,
                table,
                coffre,
                mode=mode,
                n_iter=essais,
                annulation=annulation,
                progression=avancer,
            )

        return travail

    # Lecture des résultats
    def travail_diagnostic(
        self, id_modele: str, seuil: float | None = None, cout_fp: float = 1.0, cout_fn: float = 1.0
    ) -> Callable[[Annulation, Avancer], Vue]:
        """Diagnostic hors pli d'un modèle comparé : figures et chiffres, ou lecture sans cible."""
        resultat, table = self.session.resultats, self.session.table

        def travail(annulation: Annulation, avancer: Avancer) -> Vue:
            avancer(Progression("Diagnostic", 0, 1, "Lecture des prédictions hors pli"))
            if not resultat.spec.probleme.avec_cible:
                return _vue_sans_cible(unsupervised.analyser(resultat, id_modele, table), table.donnees)
            diag = evaluate.diagnostiquer_validation(resultat, id_modele, seuil=seuil)
            vue = _vue_diagnostic(diag)
            if not diag.binaire:
                return vue
            proposition = decision.seuil_minimisant_le_cout(
                resultat, id_modele, cout_fp=cout_fp, cout_fn=cout_fn
            )
            figure = evaluate.figure_seuil(proposition.courbe, seuil, proposition.seuil)
            return Vue(
                (*vue.figures, ("seuil", figure)), vue.tableaux, vue.textes, seuil_cout=proposition.seuil
            )

        return travail

    def travail_lecture_du_modele(self, id_modele: str) -> Callable[[Annulation, Avancer], Vue]:
        """Ce que dit le modèle : coefficients, importances par impureté ou règles de l'arbre, s'il en a."""
        resultat, table, coffre = self.session.resultats, self.session.table, self.session.coffre
        spec = resultat.spec

        def travail(annulation: Annulation, avancer: Avancer) -> Vue:
            avancer(Progression("Lecture du modèle", 0, 1, "Ajustement sur tout l'apprentissage"))
            choix = next(c for c in spec.modeles if c.id == id_modele)
            ajuste = train.ajuster_final(spec, choix, table, coffre)
            tableaux: list[tuple[str, pd.DataFrame]] = []
            figures: list[tuple[str, Any]] = []
            textes: list[tuple[str, str]] = []
            coefficients = explain.coefficients(ajuste)
            if coefficients is not None:
                tableaux.append(("coefficients", coefficients))
            try:
                impuretes = explain.importance_par_impurete(ajuste)
                tableaux.append(("impurete", impuretes))
                figures.append(("impurete", explain.figure_importances(impuretes)))
            except ErreurExplication:
                pass
            with contextlib.suppress(ErreurExplication):
                textes.append(("arbre", explain.arbre_texte(ajuste)))
            return Vue(tuple(figures), tuple(tableaux), tuple(textes))

        return travail

    def travail_importances(self, id_modele: str) -> Callable[[Annulation, Avancer], Vue]:
        """Importance par permutation, mesurée pli par pli sur les lignes de validation."""
        resultat, table, coffre = self.session.resultats, self.session.table, self.session.coffre

        def travail(annulation: Annulation, avancer: Avancer) -> Vue:
            avancer(Progression("Importances", 0, 1, "Importance par permutation"))
            tableau = explain.importances_validation(
                resultat, id_modele, table, coffre, annulation=annulation
            )
            return Vue((("importances", explain.figure_importances(tableau)),), (("importances", tableau),))

        return travail

    def travail_courbe_d_apprentissage(self, id_modele: str) -> Callable[[Annulation, Avancer], Vue]:
        resultat, table, coffre = self.session.resultats, self.session.table, self.session.coffre

        def travail(annulation: Annulation, avancer: Avancer) -> Vue:
            avancer(Progression("Courbe", 0, 1, "Courbe d'apprentissage"))
            courbe = evaluate.courbe_d_apprentissage(resultat, id_modele, table, coffre)
            figure = evaluate.figure_courbe_apprentissage(courbe)
            return Vue((("courbe", figure),), (("courbe", courbe.moyennes()),))

        return travail

    def travail_decider(
        self, id_modele: str, seuil: float | None, justification: str
    ) -> Callable[[Annulation, Avancer], Session]:
        """S6 vers S7 : décision motivée sur un modèle comparé."""
        base, resultat = self.session, self.session.resultats

        def travail(annulation: Annulation, avancer: Avancer) -> Session:
            choix = decision.decider(resultat, id_modele, seuil=seuil, justification=justification)
            return base.appliquer(Decider(choix))

        return travail

    def travail_tester(self) -> Callable[[Annulation, Avancer], Session]:
        """S7 vers S8 : le coffre s'ouvre une fois ; le modèle retenu est ajusté puis évalué sur le test."""
        base = self.session

        def travail(annulation: Annulation, avancer: Avancer) -> Session:
            avancer(Progression("Test", 0, 2, "Ajustement du modèle retenu"))
            evaluer = evaluate.preparer_test(base.spec, base.table, base.coffre, base.decision)
            avancer(Progression("Test", 1, 2, "Évaluation sur le jeu de test"))
            return base.appliquer(Tester(evaluer))

        return travail

    def vue_du_test(self) -> Vue:
        """Figures et tableaux du résultat de test déjà obtenu."""
        resultat = self.session.resultat_test
        diag = resultat.diagnostic
        vue = _vue_diagnostic(diag) if diag is not None else Vue()
        tableaux = [("scores", resultat.tableau()), *vue.tableaux]
        figures = list(vue.figures)
        if resultat.importances is not None:
            tableaux.append(("importances", resultat.importances))
            figures.append(("importances", explain.figure_importances(resultat.importances)))
        return Vue(tuple(figures), tuple(tableaux), vue.textes)

    def travail_exporter(
        self, dossier: str | Path, formats: Sequence[str]
    ) -> Callable[[Annulation, Avancer], tuple[Session, report.Export]]:
        """S8 (ou S7 sans cible) vers S9 : rapport, classeur, modèle, script et projet dans le dossier."""
        base = self.session
        options = {
            "decoupage": self.decoupage,
            "chargement": self.chargement,
            "formats": tuple(formats),
            "recherches": tuple(self.recherches),
            "table_test": None if self.chargement_test is None else self.chargement_test.table,
        }

        def travail(annulation: Annulation, avancer: Avancer) -> tuple[Session, report.Export]:
            avancer(Progression("Export", 0, 1, "Écriture des fichiers"))
            Path(dossier).mkdir(
                parents=True, exist_ok=True
            )  # le dossier est celui que l'utilisateur a choisi
            return report.exporter_et_clore(base, dossier, **options)

        return travail

    def travail_revenir(self, etat: Etat) -> Callable[[Annulation, Avancer], Session]:
        return self.travail_actions(Revenir(etat))


def _apprentissage(table: Table, coffre: Any) -> pd.DataFrame:
    """Lignes d'apprentissage du coffre ; la table entière sans cible, où il n'y a pas de test."""
    return table.donnees if coffre is None else coffre.apprentissage()


def _vue_diagnostic(diag: evaluate.Diagnostic) -> Vue:
    """Figures et chiffres d'un diagnostic, avec ou sans cible binaire."""
    if diag.probleme is Probleme.REGRESSION:
        figures: list[tuple[str, Any]] = [("regression", evaluate.figure_regression(diag))]
    else:
        figures = [("confusion", evaluate.figure_confusion(diag))]
        if diag.binaire:
            figures.append(("roc_pr", evaluate.figure_roc_pr(diag)))
    resume = pd.DataFrame({"mesure": list(diag.resume()), "valeur": list(diag.resume().values())})
    return Vue(tuple(figures), (("resume", resume),))


def _vue_sans_cible(analyse: Any, donnees: pd.DataFrame) -> Vue:
    """Lecture d'un modèle sans cible : composantes, groupes ou anomalies."""
    if isinstance(analyse, unsupervised.AnalyseComposantes):
        poids = analyse.poids.reset_index()
        return Vue((("composantes", unsupervised.figure_composantes(analyse)),), (("poids", poids),))
    if isinstance(analyse, unsupervised.AnalyseRegroupement):
        plan = unsupervised.figure_plan(analyse.plan, analyse.etiquettes, analyse.modele_id)
        effectifs = analyse.effectifs.rename("effectif").reset_index()
        figures = (("plan", plan), ("profils", unsupervised.figure_profils(analyse)))
        return Vue(figures, (("effectifs", effectifs), ("profils", analyse.profils.reset_index())))
    plan = unsupervised.figure_plan(analyse.plan, analyse.etiquettes, analyse.modele_id, anomalies=True)
    figures = (("anomalies", unsupervised.figure_anomalies(analyse)), ("plan", plan))
    anormales = analyse.plus_anormales(donnees).reset_index()
    return Vue(figures, (("anormales", anormales),))
