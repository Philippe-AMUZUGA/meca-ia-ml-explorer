"""Libellés français des identifiants montrés dans le rapport, les figures et les lignes d'état."""

from __future__ import annotations

import re

from mecaml.core import catalog, libelles, metrics

IDENTIFIANT = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


def test_chaque_technique_du_catalogue_a_un_titre_francais() -> None:
    for entree in catalog.entrees():
        titre = libelles.technique(entree.id)
        assert titre != entree.id and not IDENTIFIANT.match(titre), entree.id


def test_un_identifiant_inconnu_se_rend_tel_quel() -> None:
    assert libelles.technique("famille.inconnue") == "famille.inconnue"
    assert libelles.metrique("metrique_inconnue") == "metrique_inconnue"
    assert libelles.metrique_courte("metrique_inconnue") == "metrique_inconnue"
    assert libelles.famille("inconnue") == "inconnue"


def test_la_ponctuation_double_prend_une_espace_insecable() -> None:
    assert libelles.technique("reference.dummy_regression").startswith("Référence naïve :")


def test_chaque_metrique_a_un_libelle_et_un_nom_court() -> None:
    for m in metrics.METRIQUES:
        assert libelles.metrique(m.id) == m.libelle
        court = libelles.metrique_courte(m.id)
        assert court and len(court) <= len(m.libelle) and m.id != court


def test_les_sigles_servent_de_nom_court_et_les_parametres_se_lisent() -> None:
    assert libelles.metrique_courte("rmse") == "RMSE" and libelles.metrique_courte("r2") == "R²"
    assert libelles.metrique("cout:fp=1,fn=10").endswith("(fp = 1, fn = 10)")


def test_chaque_famille_du_catalogue_a_un_nom_francais() -> None:
    for entree in catalog.entrees():
        assert libelles.famille(entree.famille) != entree.famille, entree.famille
