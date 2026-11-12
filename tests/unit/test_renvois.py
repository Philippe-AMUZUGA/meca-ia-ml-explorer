"""Table des renvois : chaque renvoi désigne une fiche du livre, sinon la section qui couvre la technique."""

from __future__ import annotations

from mecaml.core import renvois
from mecaml.core.livre import TECHNIQUES_LIVRE

SANS_FICHE = {i for i, t in TECHNIQUES_LIVRE.items() if "sans fiche" in t.libelle}


def _cibles() -> dict[str, str]:
    cibles = {f"technique {k}": v for k, v in renvois.TECHNIQUES.items()}
    cibles.update({f"métrique {k}": v for k, v in renvois.METRIQUES.items()})
    cibles.update({f"thème {k}": v for k, v in renvois.THEMES.items() if v is not None})
    return cibles


def test_la_liste_compte_les_fiches_et_les_techniques_sans_fiche() -> None:
    assert len(TECHNIQUES_LIVRE) == 63 and len(SANS_FICHE) == 25
    assert "T17-2-1" in TECHNIQUES_LIVRE and "T17-2-1" not in SANS_FICHE


def test_chaque_renvoi_existe_dans_la_liste_du_livre() -> None:
    for quoi, identifiant in _cibles().items():
        assert identifiant in TECHNIQUES_LIVRE, f"{quoi} : {identifiant}"


def test_un_renvoi_sans_fiche_donne_la_section_qui_couvre_la_technique() -> None:
    for quoi, identifiant in _cibles().items():
        if identifiant in SANS_FICHE:
            technique = TECHNIQUES_LIVRE[identifiant]
            assert technique.section.startswith(f"{technique.chapitre}."), quoi
            assert "sous-section" in technique.libelle, quoi


def test_la_selection_de_variables_renvoie_a_la_fiche_qui_la_traite() -> None:
    for cle in ("selection.variance", "selection.k_meilleures", "selection.rfecv"):
        assert renvois.technique(cle) == "T11-2-1" and "T11-2-1" not in SANS_FICHE
