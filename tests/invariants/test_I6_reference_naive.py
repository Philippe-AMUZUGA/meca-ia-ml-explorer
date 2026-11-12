"""I6, référence naïve : toute comparaison avec cible contient la moyenne ou la classe majoritaire.

Un score n'a de sens que lu contre la référence : l'entraînement l'ajoute d'office, la session refuse une
comparaison qui n'en a pas, et sur des données sans aucun signal aucun modèle ne la bat nettement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aides_apprentissage as aides
from mecaml.core import train
from mecaml.core.session import (
    Auditer,
    Charger,
    Decouper,
    Entrainer,
    FixerObjectif,
    Pretraiter,
    Session,
    TransitionRefusee,
)
from mecaml.core.types import ChoixTechnique, Probleme, Table

CT = ChoixTechnique.creer
MODELES = {
    Probleme.REGRESSION: (CT("regression.ridge"), CT("regression.arbre", max_depth=3)),
    Probleme.CLASSIFICATION: (CT("classification.logistique"), CT("classification.arbre", max_depth=3)),
}


def _projet(probleme: Probleme):
    if probleme is Probleme.REGRESSION:
        table = Table(aides.jeu_regression(), "regression")
        spec = aides.spec_regression(modeles=MODELES[probleme])
    else:
        table = Table(aides.jeu_classification(), "classification")
        spec = aides.spec_classification(modeles=MODELES[probleme])
    return table, spec, aides.coffre_par_pas(table, spec)


@pytest.mark.parametrize("probleme", [Probleme.REGRESSION, Probleme.CLASSIFICATION])
def test_la_reference_est_ajoutee_en_tete_et_notee_comme_les_autres(probleme: Probleme) -> None:
    table, spec, coffre = _projet(probleme)
    assert not any(c.famille == "reference" for c in spec.modeles)
    res = train.entrainer(spec, table, coffre)
    assert res.spec.modeles[0].famille == "reference"
    reference = res.reference()
    assert reference is not None and reference.ok and reference.id == res.spec.modeles[0].id
    # la référence passe par les mêmes plis et les mêmes métriques que les autres modèles
    for m in res.modeles:
        assert set(m.scores_plis) >= set(reference.scores_plis) - {"roc_auc", "precision_moyenne"}
        assert len(m.scores_plis[res.metrique]) == len(res.plis)
    # chaque modèle a un gain défini contre elle ; le tableau le porte
    assert all(res.gain(m.id) is not None for m in res.modeles)
    assert res.tableau()["gain_vs_reference"].notna().all()


def test_une_reference_fournie_n_est_pas_doublee() -> None:
    table, spec, coffre = _projet(Probleme.REGRESSION)
    choisie = CT("reference.dummy_regression", strategy="median")
    res = train.entrainer(spec.avec(modeles=(choisie, *spec.modeles)), table, coffre)
    references = [m for m in res.modeles if m.famille == "reference"]
    assert len(references) == 1 and references[0].choix == choisie


def test_comparer_la_seule_reference_est_refuse() -> None:
    table, spec, coffre = _projet(Probleme.REGRESSION)
    with pytest.raises(ValueError, match="Aucun modèle à comparer"):
        train.entrainer(spec.avec(modeles=(CT("reference.dummy_regression"),)), table, coffre)


def test_sans_cible_aucune_reference_n_est_exigee() -> None:
    table = Table(aides.jeu_sans_cible(), "amas")
    res = train.entrainer(aides.spec_sans_cible(modeles=(CT("sans_cible.kmeans", n_clusters=2),)), table)
    assert res.reference() is None and [m.id for m in res.modeles] == ["sans_cible.kmeans"]


def test_la_session_refuse_une_comparaison_sans_reference_et_accepte_celle_de_l_entrainement() -> None:
    table, spec, coffre = _projet(Probleme.REGRESSION)
    s = Session().appliquer(Charger(table)).appliquer(Auditer())
    s = s.appliquer(FixerObjectif(spec)).appliquer(Decouper(coffre)).appliquer(Pretraiter(spec))
    res = train.entrainer(spec, table, coffre)
    with pytest.raises(TransitionRefusee, match="référence naïve"):
        s.appliquer(Entrainer(res, spec=spec))  # la spécification initiale n'a pas la référence
    s = s.appliquer(Entrainer(res, spec=res.spec))
    assert s.spec.modeles[0].famille == "reference"


def test_sur_du_bruit_pur_aucun_modele_ne_bat_la_reference() -> None:
    """La référence protège de l'illusion : une cible sans lien avec les variables ne donne aucun gain net."""
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "d": rng.normal(size=n),
            "c": rng.choice(["x", "y", "z"], n),
            "y": rng.normal(size=n),
        }
    )
    table = Table(df, "bruit")
    spec = aides.spec_regression(modeles=(CT("regression.ridge"), CT("regression.foret", n_estimators=30)))
    res = train.entrainer(spec, table, aides.coffre_par_pas(table, spec))
    reference = res.valeur(res.reference())
    assert reference == pytest.approx(1.0, abs=0.15)  # écart type de la cible
    for id_modele in ("regression.ridge", "regression.foret"):
        assert res.gain(id_modele) < 0.05, "un modèle ne doit pas sembler meilleur que le hasard sur du bruit"
    # sur la même base, avec un vrai signal, le gain est net
    signal = aides.jeu_regression(manquants=False)
    t2 = Table(signal, "signal")
    s2 = aides.spec_regression(modeles=(CT("regression.ridge"),))
    r2 = train.entrainer(s2, t2, aides.coffre_par_pas(t2, s2))
    assert r2.gain("regression.ridge") > 1.0


def test_la_classification_sans_signal_ne_depasse_pas_la_reference_sur_l_exactitude_equilibree() -> None:
    rng = np.random.default_rng(6)
    n = 400
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.choice(["x", "y", "z"], n),
            "y": rng.choice(["conforme", "defaut"], n, p=[0.7, 0.3]),
        }
    )
    table = Table(df, "bruit")
    spec = aides.spec_classification(
        modeles=(CT("classification.logistique"),), metrique="exactitude_equilibree"
    )
    res = train.entrainer(spec, table, aides.coffre_par_pas(table, spec))
    assert res.valeur(res.reference()) == pytest.approx(0.5)
    assert res.valeur(res.modele("classification.logistique")) < 0.6
