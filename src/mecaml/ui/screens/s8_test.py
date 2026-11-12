"""MécaML : écran S8, ouverture unique du jeu de test scellé et lecture des scores sur ce jeu."""

from __future__ import annotations

from tkinter import ttk

from mecaml.core import libelles
from mecaml.core.guards import nombre_lisible
from mecaml.core.session import Etat, Session, Tester
from mecaml.ui.screens.base import Ecran
from mecaml.ui.textes import t, titre_technique
from mecaml.ui.widgets import Planche, carte, nombre

__all__ = ["EcranTest"]


def _evaluation_vide(lignes: object) -> None:
    """Évaluation sans effet : sert seulement à interroger la session (`peut`), jamais exécutée."""


class EcranTest(Ecran):
    """Le test s'ouvre une fois, avec une décision en main ; ensuite tout score de test est dit optimiste."""

    NUMERO = 8
    CLE = "s8"

    def construire(self, corps: ttk.Frame) -> None:
        cadre = carte(corps, t("ecran.s8.scelle.titre"))
        cadre.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        cadre.grid_columnconfigure(0, weight=1)
        self.etat = ttk.Label(cadre, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.etat.grid(row=0, column=0, sticky="w")
        self.bouton(
            cadre,
            "ecran.s8.ouvrir",
            self.ouvrir,
            principal=True,
            condition=self._peut_ouvrir,
            row=1,
            column=0,
            sticky="w",
            pady=(10, 0),
        )
        resultat = carte(corps, t("ecran.s8.resultat.titre"))
        resultat.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        resultat.grid_columnconfigure(0, weight=1)
        self.lecture = ttk.Label(resultat, text="", style="Carte.TLabel", wraplength=820, justify="left")
        self.lecture.grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.planche = Planche(resultat)
        self.planche.grid(row=1, column=0, sticky="ew")
        self.bouton(
            resultat,
            "ecran.s8.exporter",
            lambda: self.app.afficher(9),
            condition=self._peut_exporter,
            row=2,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

    def _peut_ouvrir(self) -> bool:
        """La session dit si le coffre peut s'ouvrir : décision prise, test jamais ouvert, avec cible."""
        return self.session.peut(Tester(_evaluation_vide))

    def _peut_exporter(self) -> bool:
        return self.session.etat >= Etat.TESTE or (
            self.session.sans_cible and self.session.etat >= Etat.DIAGNOSTIC
        )

    def ouvrir(self) -> None:
        """Ouvre le coffre : confirmation, puis ajustement du modèle retenu et évaluation, dans un fil."""
        if not self.app.confirmer(t("ecran.s8.confirmation")):
            return
        self.app.executer(self.ctrl.travail_tester(), nom=t("ecran.s8.test"), au_succes=self._teste)

    def _teste(self, session: Session) -> None:
        self.ctrl.adopter(session)
        self.rafraichir()

    # Affichage
    def rafraichir(self) -> None:
        session = self.session
        if session.spec is None or session.etat < Etat.DIAGNOSTIC:
            return
        if session.sans_cible:
            self.etat.configure(text=t("ecran.s8.sans_objet"))
            self.lecture.configure(text="")
            self.planche.vider()
        elif session.etat < Etat.TESTE:
            self._avant(session)
        else:
            self._apres(session)
        self.mettre_a_jour_boutons()

    def _avant(self, session: Session) -> None:
        coffre = session.coffre
        self.etat.configure(
            text=t(
                "ecran.s8.avant",
                test=nombre_lisible(coffre.n_test),
                modele=titre_technique(session.decision.modele_id),
                ouvertures=coffre.ouvertures,
            )
        )
        self.lecture.configure(text="")
        self.planche.vider()

    def _apres(self, session: Session) -> None:
        coffre, test = session.coffre, session.resultat_test
        message = t("ecran.s8.ouvert", ouvertures=coffre.ouvertures)
        if session.score_test_optimiste:
            message += "\n" + t("ecran.s8.optimiste")
        self.etat.configure(text=message)
        modele = session.resultats.modele(session.decision.modele_id)
        lignes = [
            t(
                "ecran.s8.scores",
                metrique=libelles.metrique_courte(test.metrique),
                validation=nombre(session.resultats.valeur(modele), 4),
                test=nombre(test.valeur(), 4),
                lignes=nombre_lisible(test.n_lignes),
            )
        ]
        if test.n_sans_cible:
            lignes.append(t("ecran.s8.sans_cible", n=test.n_sans_cible))
        lignes += [f"{titre_technique(session.decision.modele_id)} : {a}" for a in test.avertissements[:3]]
        self.lecture.configure(text="\n".join(lignes))
        vue = self.ctrl.vue_du_test()
        self.ctrl.vues["test"] = vue
        titres = {
            n: t(f"ecran.s7.vue.{n}") for n in ("regression", "confusion", "roc_pr", "importances", "resume")
        }
        titres["scores"] = t("ecran.s8.vue.scores")
        self.planche.afficher(vue, titres)
