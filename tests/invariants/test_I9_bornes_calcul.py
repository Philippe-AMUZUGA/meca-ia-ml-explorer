"""I9, bornes de calcul : techniques coûteuses coupées, grilles et paramètres bornés, budget, annulation.

Les bornes de lecture (200 000 lignes, 500 colonnes, fichiers) relèvent de l'entrée des données ; ici, les
bornes du calcul : SVC, SVR, SVM à une classe et exclusion d'une ligne coupés au delà de 20 000 lignes,
grilles à 50 combinaisons au plus, nombre d'arbres et de passes bornés, temps total borné, arrêt demandé
pris en compte entre deux (modèle, pli).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core import train
from mecaml.core.catalog import GRILLE_MAX_COMBINAISONS, REGISTRE, Cout, entree
from mecaml.core.catalog._outils import LIMITES
from mecaml.core.tasks import Annulation, ErreurAnnulation
from mecaml.core.train import LIGNES_MAX_LOURD, ErreurBorne, ErreurBudget
from mecaml.core.types import ChoixTechnique, CoffreTest, Table

CT = ChoixTechnique.creer
LOURDS = ["classification.svc", "regression.svr", "sans_cible.svm_une_classe"]


@pytest.mark.parametrize("identifiant", LOURDS)
def test_technique_lourde_coupee_au_dela_de_la_borne(identifiant: str) -> None:
    e = entree(identifiant)
    assert e.cout is Cout.LOURD
    probleme = next(iter(e.problemes))
    base = {"regression": aides.spec_regression, "classification": aides.spec_classification}.get(
        probleme.value, aides.spec_sans_cible
    )
    spec = base(modeles=(CT(identifiant),))
    train.verifier_bornes(spec, LIGNES_MAX_LOURD)
    with pytest.raises(ErreurBorne) as erreur:
        train.verifier_bornes(spec, LIGNES_MAX_LOURD + 1)
    message = str(erreur.value)
    assert identifiant in message and "20 000" in message and "plus légère" in message  # actionnable


def test_exclusion_d_une_ligne_coupee_au_dela_de_la_borne() -> None:
    spec = aides.spec_regression(modeles=(CT("regression.ridge"),), validation="validation.exclusion_unique")
    train.verifier_bornes(spec, LIGNES_MAX_LOURD)
    with pytest.raises(ErreurBorne, match=r"validation.exclusion_unique"):
        train.verifier_bornes(spec, LIGNES_MAX_LOURD + 1)


def test_les_techniques_legeres_ne_sont_pas_coupees() -> None:
    legeres = [
        e.id for e in REGISTRE if e.cout is not Cout.LOURD and e.famille in ("regression", "classification")
    ]
    assert len(legeres) >= 20
    for identifiant in legeres:
        spec = aides.spec_regression(modeles=(CT(identifiant),))
        train.verifier_bornes(spec, 200_000)


def test_le_refus_precede_tout_calcul() -> None:
    n = 26_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": "x", "y": rng.choice(["p", "q"], n)}
    )
    table = Table(df, "grand")
    spec = aides.spec_classification(modeles=(CT("classification.svc"),), classe_positive="q")
    coffre = CoffreTest.par_positions(table, np.arange(0, n, 5), decoupage=spec.decoupage())
    debut = time.perf_counter()
    with pytest.raises(ErreurBorne):
        train.entrainer(spec, table, coffre)
    assert time.perf_counter() - debut < 2.0, "le refus ne doit pas attendre un ajustement"


@pytest.mark.lent
def test_svc_a_la_borne_reste_dans_un_temps_raisonnable() -> None:
    """Mesure de la borne : un SVC sur 20 000 lignes s'ajuste en quelques secondes."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(size=(LIGNES_MAX_LOURD, 6))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.5, LIGNES_MAX_LOURD) > 0).astype(int)
    modele = make_pipeline(StandardScaler(), entree("classification.svc").construire(0))
    debut = time.perf_counter()
    modele.fit(x, y)
    duree = time.perf_counter() - debut
    print(f"SVC a {LIGNES_MAX_LOURD} lignes : {duree:.2f} s")
    assert duree < 30.0


def test_grilles_et_parametres_bornes() -> None:
    assert all(e.combinaisons <= GRILLE_MAX_COMBINAISONS for e in REGISTRE)
    assert GRILLE_MAX_COMBINAISONS == 50
    for nom, borne in LIMITES.items():
        assert borne > 0, nom
    with pytest.raises(ValueError, match="n_estimators"):
        entree("regression.foret").construire(0, n_estimators=LIMITES["n_estimators"] + 1)
    with pytest.raises(ValueError, match="max_iter"):
        entree("regression.reseau_neurones").construire(0, max_iter=LIMITES["max_iter"] + 1)
    with pytest.raises(ValueError, match="hors bornes"):
        entree("classification.reseau_neurones").construire(0, hidden_layer_sizes=(512,))


def test_recherche_aleatoire_bornee_a_cinquante_tirages() -> None:
    table = Table(aides.jeu_regression(), "regression")
    spec = aides.spec_regression()
    coffre = aides.coffre_par_pas(table, spec)
    with pytest.raises(ErreurBorne, match="de 1 à 50"):
        train.rechercher(spec, CT("regression.ridge"), table, coffre, mode="aleatoire", n_iter=51)


def test_budget_de_temps_borne_l_entrainement_et_la_recherche() -> None:
    table = Table(aides.jeu_regression(), "regression")
    spec = aides.spec_regression(modeles=(CT("regression.ridge"), CT("regression.foret", n_estimators=20)))
    coffre = aides.coffre_par_pas(table, spec)
    with pytest.raises(ErreurBudget, match="moins de modèles"):
        train.entrainer(spec, table, coffre, budget_s=0.0)
    with pytest.raises(ErreurBudget):
        train.rechercher(spec, CT("regression.ridge"), table, coffre, budget_s=0.0)
    assert train.BUDGET_PAR_DEFAUT_S == 300.0


def test_l_arret_demande_est_pris_en_compte_a_la_prochaine_unite_de_travail() -> None:
    table = Table(aides.jeu_regression(), "regression")
    spec = aides.spec_regression(modeles=(CT("regression.ridge"), CT("regression.foret", n_estimators=20)))
    coffre = aides.coffre_par_pas(table, spec)
    annulation = Annulation()
    unites = []

    def avancer(progression) -> None:
        unites.append(progression.fait)
        if progression.fait == 4:
            annulation.demander()

    with pytest.raises(ErreurAnnulation):
        train.entrainer(spec, table, coffre, annulation=annulation, progression=avancer)
    assert unites == [1, 2, 3, 4], "aucune unité (modèle, pli) ne commence après la demande d'arrêt"


def test_l_entrainement_ne_bloque_pas_l_appelant() -> None:
    """L'entraînement lancé dans un fil rend la main aussitôt ; l'état se lit pendant le calcul."""
    table = Table(aides.jeu_regression(), "regression")
    spec = aides.spec_regression(modeles=(CT("regression.foret", n_estimators=200),))
    coffre = aides.coffre_par_pas(table, spec)
    debut = time.perf_counter()
    tache = train.lancer_entrainement(spec, table, coffre)
    rendu = time.perf_counter() - debut
    assert rendu < 0.5, "le lancement doit rendre la main presque immédiatement"
    etats = {tache.etat.value}
    while not tache.terminee:
        etats.add(tache.etat.value)
        time.sleep(0.005)
    assert tache.attendre(60) and "terminee" in {tache.etat.value}
    assert tache.resultat().modele("regression.foret").ok
    assert "en_cours" in etats or tache.terminee
