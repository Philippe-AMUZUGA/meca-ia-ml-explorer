"""Textes en français : aucune clé manquante, aucune clé inutilisée, chaque renvoi existe dans le livre."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from mecaml.core import catalog, metrics, renvois
from mecaml.core.livre import TECHNIQUES_LIVRE
from mecaml.ui import textes
from mecaml.ui.textes import cles, pedagogie, t, titre_technique
from mecaml.ui.widgets import renvoi_livre

DOSSIER = Path(textes.__file__).resolve().parents[2] / "textes"
CHAMPS = ("quoi", "regarder")
ECRANS = ("accueil", *(f"s{k}" for k in range(1, 10)))
TIRETS = re.compile(f"[{chr(0x2013)}{chr(0x2014)}]|-{{2}}")
MOTS_MAX_PAR_PHRASE = 20


def _textes_attendus() -> dict[str, str]:
    """Clé de texte de chaque technique et de chaque métrique, avec son titre attendu."""
    titres = {e.texte_id: titre_technique(e.id) for e in catalog.REGISTRE}
    titres.update({m.texte_id: m.libelle for m in metrics.METRIQUES})
    return titres


def _source(nom: str) -> dict:
    return tomllib.loads((DOSSIER / nom).read_text(encoding="utf-8"))


def test_le_dossier_des_textes_ne_contient_que_les_fichiers_attendus() -> None:
    assert sorted(p.name for p in DOSSIER.iterdir() if p.is_file()) == [
        "__init__.py",
        "etapes_fr.toml",
        "interface_fr.toml",
    ]


def test_les_titres_des_techniques_et_des_metriques_viennent_du_coeur_seul() -> None:
    """Le fichier pédagogique ne répète aucun titre : un seul endroit à relire."""
    assert all("titre" not in table for table in _source("etapes_fr.toml").values())


def test_aucune_cle_manquante_pour_les_techniques_et_les_metriques() -> None:
    attendues = set(_textes_attendus())
    assert len(attendues) == 92  # 75 techniques et 17 métriques
    assert sorted(attendues - set(_source("etapes_fr.toml"))) == []


def test_aucune_cle_inutilisee_dans_les_textes_pedagogiques() -> None:
    assert sorted(set(_source("etapes_fr.toml")) - set(_textes_attendus())) == []


def test_chaque_texte_a_ses_deux_champs_et_eux_seuls() -> None:
    for cle, table in _source("etapes_fr.toml").items():
        assert tuple(table) == CHAMPS, cle
        assert all(isinstance(v, str) and v.strip() == v and v for v in table.values()), cle


def test_chaque_champ_est_une_phrase_courte_qui_finit_par_un_point() -> None:
    for cle in _textes_attendus():
        for champ in CHAMPS:
            phrase = pedagogie(cle)[champ]
            assert (phrase[0].isupper() or phrase[0].isdigit()) and phrase.endswith("."), f"{cle}.{champ}"
            assert len(phrase.split()) <= MOTS_MAX_PAR_PHRASE, f"{cle}.{champ} : phrase trop longue"


def test_aucun_champ_n_est_repete_d_une_technique_a_l_autre_sauf_variantes() -> None:
    """Deux techniques distinctes ne portent pas la même phrase « quoi »."""
    quoi = [pedagogie(c)["quoi"] for c in _textes_attendus()]
    assert len(quoi) == len(set(quoi))


@pytest.mark.parametrize("nom", ["etapes_fr.toml", "interface_fr.toml"])
def test_la_charte_du_texte_est_respectee(nom: str) -> None:
    source = (DOSSIER / nom).read_text(encoding="utf-8")
    assert not TIRETS.search(source), "tiret interdit"
    assert "\r" not in source
    assert "..." not in source.replace("{sha}...", "")


def test_la_typographie_francaise_des_textes_pedagogiques() -> None:
    for cle in _textes_attendus():
        for champ in CHAMPS:
            texte = pedagogie(cle)[champ]
            assert not re.search(r" [:;!?»]", texte), f"{cle}.{champ}"
            assert "« " not in texte


def test_chaque_renvoi_existe_dans_la_table_du_livre() -> None:
    """Le renvoi de chaque technique, métrique et thème est un identifiant de la liste du livre."""
    renvois_attendus = {e.texte_id: renvois.technique(e.id) for e in catalog.REGISTRE}
    renvois_attendus.update({m.texte_id: renvois.metrique(m.id) for m in metrics.METRIQUES})
    for cle, identifiant in renvois_attendus.items():
        fiche = TECHNIQUES_LIVRE.get(identifiant)
        assert fiche is not None, f"{cle} : {identifiant} absent de la liste du livre"
        assert not fiche.hors_application, f"{cle} : {identifiant} hors application"
    for theme, identifiant in renvois.THEMES.items():
        assert identifiant is None or identifiant in TECHNIQUES_LIVRE, theme


def test_le_renvoi_affiche_donne_le_chapitre_la_section_et_l_identifiant() -> None:
    texte = renvoi_livre("T08-2-1")
    assert texte == "chapitre 8, section 8.2 (T08-2-1)"
    assert t("commun.voir_livre", renvoi=texte) == "Livre Méca-IA, chapitre 8, section 8.2 (T08-2-1)"
    assert renvoi_livre(None) == "" and renvoi_livre("T99-9-9") == ""
    for entree in catalog.entrees():
        affiche = renvoi_livre(entree.ref_livre)
        sans_fiche = "sans fiche" in TECHNIQUES_LIVRE[entree.ref_livre].libelle if entree.ref_livre else False
        motif = r"chapitre \d+, section \d+\.\d+" + ("" if sans_fiche else r" \(T\d{2}-\d+-\d+\)")
        assert re.fullmatch(motif, affiche), entree.id


def test_une_technique_sans_fiche_n_affiche_pas_d_identifiant_que_le_livre_n_imprime_pas() -> None:
    sans_fiche = next(i for i, x in TECHNIQUES_LIVRE.items() if "sans fiche" in x.libelle)
    assert "(T" not in renvoi_livre(sans_fiche)
    assert renvoi_livre(sans_fiche).startswith("chapitre ")


def test_la_licence_et_le_credit_sont_celles_du_depot() -> None:
    assert "MIT" in t("pied.credit", version="1.0.0")
    assert "AGPL" not in "".join(t(c) for c in cles())


def test_les_deux_techniques_de_cible_passee_disent_la_condition_et_le_refus() -> None:
    for cle in ("reference.persistance", "chronologique.cible_decalee"):
        corps = " ".join(pedagogie(cle).values())
        assert re.search(r"connue", corps), cle
        assert re.search(r"compte à rebours|décompte", corps), cle
