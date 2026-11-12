"""Liste des techniques du livre : module fidèle au tableau, identifiants valides, écart avec la source."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

from mecaml.core.livre import RE_IDENTIFIANT, TECHNIQUES_LIVRE, est_identifiant, technique_du_livre

RACINE = Path(__file__).resolve().parents[2]
TABLEAU = RACINE / "tests" / "data" / "techniques_livre.md"
GENERATEUR = RACINE / "tests" / "outils" / "generer_livre.py"
# Liste du livre telle qu'elle est publiée, quand la variable d'environnement MECAML_LISTE_LIVRE la désigne.
LISTE_PUBLIEE = os.environ.get("MECAML_LISTE_LIVRE")


def _generateur():
    spec = importlib.util.spec_from_file_location("generer_livre", GENERATEUR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_le_module_genere_est_fidele_au_tableau() -> None:
    lignes = _generateur().lire_tableau(TABLEAU.read_text(encoding="utf-8"))
    assert len(lignes) == len(TECHNIQUES_LIVRE) == 63
    for identifiant, chapitre, section, libelle, hors in lignes:
        technique = TECHNIQUES_LIVRE[identifiant]
        assert (technique.chapitre, technique.section, technique.libelle, technique.hors_application) == (
            chapitre,
            section,
            libelle,
            hors,
        ), identifiant


def test_identifiants_bien_formes_chapitre_et_section_coherents() -> None:
    for identifiant, technique in TECHNIQUES_LIVRE.items():
        assert RE_IDENTIFIANT.match(identifiant) and technique.id == identifiant
        chapitre, section, _ = identifiant[1:].split("-")
        assert technique.section.startswith(f"{technique.chapitre}."), identifiant
        if "sans fiche" in technique.libelle:
            # sans fiche : chapitre et section où le livre traite la technique
            assert technique.libelle.count("sans fiche") == 1, identifiant
        else:
            assert int(chapitre) == technique.chapitre, identifiant
            assert technique.section == f"{technique.chapitre}.{section}", identifiant
        assert technique.libelle and not technique.libelle.endswith("]")


def test_est_identifiant_et_technique_du_livre() -> None:
    assert est_identifiant("T08-2-1") and not est_identifiant("C08") and not est_identifiant("T99-9-9")
    assert not est_identifiant(None) and not est_identifiant(8)
    assert "forêt" in technique_du_livre("T08-2-1").libelle.lower()
    with pytest.raises(KeyError, match="Renvoi au livre inconnu"):
        technique_du_livre("C08")


def test_la_liste_ne_contient_aucun_ancien_numero_de_chapitre() -> None:
    assert not any(re.fullmatch(r"C\d{2}", identifiant) for identifiant in TECHNIQUES_LIVRE)


@pytest.mark.skipif(
    not LISTE_PUBLIEE or not Path(LISTE_PUBLIEE).is_file(),
    reason="MECAML_LISTE_LIVRE ne désigne aucune liste publiée",
)
def test_la_copie_du_depot_est_identique_a_la_liste_publiee() -> None:
    """Quand la liste du livre change, recopier la liste publiée dans `tests/data`, puis relancer le
    générateur ; ce test, lancé avec MECAML_LISTE_LIVRE, signale tout écart."""
    ligne = re.compile(r"^\|\s*T\d{2}-\d+-\d+\s*\|")
    publiee = [
        x.strip() for x in Path(LISTE_PUBLIEE).read_text(encoding="utf-8").splitlines() if ligne.match(x)
    ]
    copie = [x.strip() for x in TABLEAU.read_text(encoding="utf-8").splitlines() if ligne.match(x)]
    assert publiee == copie
