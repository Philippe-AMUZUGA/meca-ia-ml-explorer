"""Données synthétiques minuscules et parcours type partagés par les tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mecaml.core.catalog import Cout, EntreeCatalogue
from mecaml.core.session import (
    Acquitter,
    Action,
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Etat,
    Exporter,
    FixerObjectif,
    Pretraiter,
    Revenir,
    Session,
    Tester,
    auditeur_neutre,
)
from mecaml.core.types import (
    ChoixTechnique,
    CoffreTest,
    Constat,
    Decision,
    Probleme,
    Role,
    SpecProjet,
    Structure,
    Table,
)

N_LIGNES = 60
REGLE = Callable[[Table, SpecProjet | None, CoffreTest | None], Sequence[Constat]]


def fabriquer_dataframe(n: int = N_LIGNES, graine: int = 0) -> pd.DataFrame:
    """Six machines suivies sur dix cycles : identifiant, groupe, ordre, mesures, deux cibles."""
    rng = np.random.default_rng(graine)
    machine = np.repeat([f"M{i}" for i in range(1, 7)], n // 6)
    cycle = np.tile(np.arange(1, n // 6 + 1), 6)
    temperature = rng.normal(60.0, 5.0, n)
    effort = rng.normal(12.0, 2.0, n)
    y_reg = 0.3 * temperature + 1.5 * effort + rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "id_piece": np.arange(1, n + 1),
            "machine": machine,
            "cycle": cycle,
            "temperature": temperature,
            "effort": effort,
            "materiau": rng.choice(["acier", "alu", "inox"], n),
            "y_reg": y_reg,
            "y_cls": np.where(y_reg > np.median(y_reg), "non_conforme", "conforme"),
        }
    )


ROLES_COMMUNS: tuple[tuple[str, Role], ...] = (
    ("id_piece", Role.IDENTIFIANT),
    ("machine", Role.GROUPE),
    ("cycle", Role.DATE),
    ("temperature", Role.NUMERIQUE),
    ("effort", Role.NUMERIQUE),
    ("materiau", Role.CATEGORIELLE),
)
MODELES_REGRESSION = (
    ChoixTechnique.creer("reference.dummy_regression", strategy="mean"),
    ChoixTechnique.creer("regression.ridge", alpha=1.0),
)
MODELES_SANS_CIBLE = (ChoixTechnique.creer("sans_cible.kmeans", n_clusters=3),)


def spec_regression(**changements: object) -> SpecProjet:
    """Régression de `y_reg`, métrique RMSE, lignes indépendantes."""
    spec = SpecProjet(
        probleme=Probleme.REGRESSION,
        cible="y_reg",
        roles=(*ROLES_COMMUNS, ("y_reg", Role.CIBLE), ("y_cls", Role.IGNOREE)),
        metrique="rmse",
    )
    return spec.avec(**changements) if changements else spec


def spec_sans_cible() -> SpecProjet:
    """Regroupement sur les mesures, sans cible."""
    return SpecProjet(
        probleme=Probleme.SANS_CIBLE,
        roles=(*ROLES_COMMUNS, ("y_reg", Role.NUMERIQUE), ("y_cls", Role.IGNOREE)),
        metrique="silhouette",
        structure=Structure.INDEPENDANTE,
    )


def coffre_sur(table: Table, decoupage: tuple[Any, ...] | None = None, pas: int = 5) -> CoffreTest:
    """Coffre dont le test est une ligne sur `pas`, pour le découpage donné (régression par défaut)."""
    decoupage = spec_regression().decoupage() if decoupage is None else decoupage
    return CoffreTest.par_positions(table, np.arange(0, table.n_lignes, pas), decoupage=decoupage)


@dataclass
class AuditeurEspion:
    """Auditeur de test : rend des constats programmés et note chaque appel."""

    regle: REGLE = auditeur_neutre
    appels: list[SpecProjet | None] = field(default_factory=list)
    coffres: list[CoffreTest | None] = field(default_factory=list)

    def __call__(self, table: Table, spec: SpecProjet | None, coffre: CoffreTest | None) -> Sequence[Constat]:
        self.appels.append(spec)
        self.coffres.append(coffre)
        return self.regle(table, spec, coffre)


CHAINE: tuple[type[Action], ...] = (
    Charger,
    Auditer,
    FixerObjectif,
    Decouper,
    Pretraiter,
    Entrainer,
    Decider,
    Tester,
    Exporter,
)


class Parcours:
    """Construit des actions valides et mène une session jusqu'à un état donné."""

    def __init__(self, table: Table, spec: SpecProjet, auditeur: Callable | None = None) -> None:
        self.table = table
        self.spec = spec
        self.auditeur = auditeur or auditeur_neutre
        self.modeles = MODELES_REGRESSION if spec.probleme.avec_cible else MODELES_SANS_CIBLE

    def nouvelle(self) -> Session:
        """Session vide munie de l'auditeur du parcours."""
        return Session(auditeur=self.auditeur)

    def action_valide(self, type_action: type[Action], session: Session) -> Action:
        """Action au contenu valide pour la session ; seul l'état peut la faire refuser."""
        spec = session.spec or self.spec
        table = session.table or self.table
        modele_choisi = (session.spec.modeles if session.spec and session.spec.modeles else self.modeles)[0]
        fabriques: dict[type[Action], Callable[[], Action]] = {
            Charger: lambda: Charger(self.table),
            Auditer: Auditer,
            Acquitter: lambda: Acquitter(frozenset()),
            FixerObjectif: lambda: FixerObjectif(self.spec),
            Decouper: lambda: Decouper(
                coffre_sur(table, spec.decoupage()) if spec.probleme.avec_cible else None
            ),
            Pretraiter: lambda: Pretraiter(
                spec.avec(pretraitement=(ChoixTechnique.creer("manquants.mediane"),))
            ),
            Entrainer: lambda: Entrainer(
                {"reference.dummy_regression": 1.0}, spec=spec.avec(modeles=self.modeles)
            ),
            Decider: lambda: Decider(
                Decision(modele_choisi.id, metrique=spec.metrique or "silhouette", score_validation=1.0)
            ),
            Tester: lambda: Tester(lambda lignes: {"n_lignes": len(lignes)}),
            Exporter: lambda: Exporter(("rapport.html",)),
            Revenir: lambda: Revenir(Etat.VIDE),
        }
        return fabriques[type_action]()

    def jusqua(self, etat: Etat, depart: Session | None = None) -> Session:
        """Session menée jusqu'à `etat` par la chaîne type (S8 sauté sans cible)."""
        s = depart or self.nouvelle()
        for type_action in CHAINE:
            if s.etat >= etat:
                break
            if type_action.produit is not None and type_action.produit <= s.etat:
                continue
            if type_action is Tester and not self.spec.probleme.avec_cible:
                continue
            s = s.appliquer(self.action_valide(type_action, s))
        assert s.etat == etat, f"état {etat} inatteignable par la chaîne type"
        return s


# ============================================================
# FIXTURES
# ============================================================
@pytest.fixture
def df_petit() -> pd.DataFrame:
    return fabriquer_dataframe()


@pytest.fixture
def table_petite(df_petit: pd.DataFrame) -> Table:
    return Table(df_petit, nom="synthetique.csv")


@pytest.fixture
def spec_reg() -> SpecProjet:
    return spec_regression()


@pytest.fixture
def spec_sc() -> SpecProjet:
    return spec_sans_cible()


@pytest.fixture
def modeles_reg() -> tuple[ChoixTechnique, ...]:
    return MODELES_REGRESSION


@pytest.fixture
def coffre_de() -> Callable[..., CoffreTest]:
    """`coffre_de(table, decoupage=None)` : test d'une ligne sur cinq."""
    return coffre_sur


@pytest.fixture
def espion() -> Callable[..., AuditeurEspion]:
    """Fabrique d'auditeurs espions : `espion(regle)`."""

    def fabriquer(regle: REGLE | None = None) -> AuditeurEspion:
        return AuditeurEspion(regle or auditeur_neutre)

    return fabriquer


@pytest.fixture
def fabrique_parcours(table_petite: Table) -> Callable[..., Parcours]:
    """`fabrique_parcours(spec, auditeur=None)` : parcours sur la table synthétique."""

    def fabriquer(spec: SpecProjet, auditeur: Callable | None = None) -> Parcours:
        return Parcours(table_petite, spec, auditeur)

    return fabriquer


@pytest.fixture
def parcours(table_petite: Table) -> Parcours:
    return Parcours(table_petite, spec_regression())


@pytest.fixture
def parcours_sans_cible(table_petite: Table) -> Parcours:
    return Parcours(table_petite, spec_sans_cible())


@pytest.fixture
def entree_dummy() -> EntreeCatalogue:
    """Entrée d'exemple : référence naïve de régression."""

    def fabrique(graine: int, **parametres: object) -> object:
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(**parametres)

    return EntreeCatalogue(
        id="reference.dummy_regression",
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=fabrique,
        grille_bornee=(("strategy", ("mean", "median")),),
        cout=Cout.LEGER,
        ref_livre="C02",
        texte_id="reference.dummy_regression",
    )
