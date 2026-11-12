"""Garde-fous : bornes, extensions, techniques coupées, confinement des écritures, contrôle de la cible."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mecaml.core.guards import (
    COLONNES_MAX,
    LIGNES_LECTURE_EN_FIL,
    LIGNES_MAX,
    LIGNES_MAX_LENTES,
    ErreurAnnulee,
    ErreurDonnees,
    chemin_dans_dossier,
    controle_cible,
    decimale,
    detecter_probleme,
    extension_acceptee,
    lecture_en_fil_requise,
    lignes_sans_cible,
    nombre_lisible,
    renvoi,
    resumer_cible,
    technique_coupee,
    verifier_bornes,
    verifier_taille_fichier,
)
from mecaml.core.types import ChoixTechnique, Niveau, Probleme, Role


def codes(constats) -> dict[str, Niveau]:
    return {c.code: c.niveau for c in constats}


# ============================================================
# MISE EN FORME
# ============================================================
def test_nombre_lisible_et_decimale() -> None:
    assert nombre_lisible(200_000) == "200\u202f000" and nombre_lisible(7) == "7"
    assert decimale(3.39, 2) == "3,39" and decimale(12.0, 0) == "12" and decimale(0.5) == "0,5"


# ============================================================
# EXTENSIONS ET TAILLES
# ============================================================
@pytest.mark.parametrize("nom", ["a.xlsx", "A.XLSX", "dossier/b.csv", "c.Csv"])
def test_extensions_acceptees(nom: str) -> None:
    assert extension_acceptee(nom) in (".xlsx", ".csv")


@pytest.mark.parametrize(
    "nom", ["a.xls", "a.xlsm", "a.pkl", "a.joblib", "a.json", "a.exe", "a", "a.xlsx.bak"]
)
def test_extensions_refusees_avec_conseil(nom: str) -> None:
    with pytest.raises(ErreurDonnees, match=r"\.xlsx.*\.csv.*tableur"):
        extension_acceptee(nom)


def test_taille_de_fichier() -> None:
    verifier_taille_fichier(1)
    with pytest.raises(ErreurDonnees, match="vide"):
        verifier_taille_fichier(0, "« t.csv »")
    with pytest.raises(ErreurDonnees, match=r"Mio.*limite.*échantillonner"):
        verifier_taille_fichier(600 * 2**20)


# ============================================================
# BORNES (I9)
# ============================================================
def test_les_bornes_de_la_feuille_de_route() -> None:
    assert (LIGNES_MAX, COLONNES_MAX, LIGNES_LECTURE_EN_FIL, LIGNES_MAX_LENTES) == (
        200_000,
        500,
        50_000,
        20_000,
    )


@pytest.mark.parametrize(("n_lignes", "n_colonnes"), [(1, 2), (200_000, 2), (10, 500), (200_000, 500)])
def test_bornes_incluses_acceptees(n_lignes: int, n_colonnes: int) -> None:
    verifier_bornes(n_lignes, n_colonnes)


def test_trop_de_lignes_message_actionnable() -> None:
    with pytest.raises(ErreurDonnees) as e:
        verifier_bornes(200_001, 5)
    message = str(e.value)
    assert "200\u202f001 lignes" in message and "200\u202f000" in message and "Échantillonner" in message


def test_trop_de_colonnes_message_actionnable() -> None:
    with pytest.raises(ErreurDonnees) as e:
        verifier_bornes(10, 501)
    message = str(e.value)
    assert "501 colonnes" in message and "limite de 500" in message and "Retirer les colonnes" in message


def test_jeu_inexploitable() -> None:
    with pytest.raises(ErreurDonnees, match="aucune ligne"):
        verifier_bornes(0, 5)
    with pytest.raises(ErreurDonnees, match="qu'une colonne"):
        verifier_bornes(10, 1)


def test_lecture_en_fil() -> None:
    assert not lecture_en_fil_requise(50_000) and lecture_en_fil_requise(50_001)


@pytest.mark.parametrize(
    "id_technique",
    [
        "classification.svc",
        "regression.svr",
        "sans_cible.svm_une_classe",
        "validation.exclusion_unique",
    ],
)
def test_techniques_coupees_au_dela_de_20000_lignes(id_technique: str) -> None:
    assert technique_coupee(id_technique, 20_000) is None
    message = technique_coupee(id_technique, 20_001)
    assert (
        message is not None
        and id_technique in message
        and "20\u202f000" in message
        and "échantillon" in message
    )


@pytest.mark.parametrize(
    "id_technique", ["regression.ridge", "classification.random_forest", "reference.dummy"]
)
def test_autres_techniques_jamais_coupees(id_technique: str) -> None:
    assert technique_coupee(id_technique, 10_000_000) is None


# ============================================================
# CONFINEMENT DES ÉCRITURES (I10)
# ============================================================
def test_chemin_dans_le_dossier_choisi(tmp_path: Path) -> None:
    assert chemin_dans_dossier(tmp_path, "rapport.html") == (tmp_path / "rapport.html").resolve()
    (tmp_path / "sortie").mkdir()
    assert chemin_dans_dossier(tmp_path, "sortie/modele.joblib").parent == (tmp_path / "sortie").resolve()


@pytest.mark.parametrize(
    "nom",
    [
        "../hors.html",
        "a/../../hors.html",
        "..\\hors.html",
        "/etc/passwd",
        "\\\\serveur\\partage\\f.txt",
        "C:\\Windows\\f.txt",
        "C:f.txt",
        "f.txt:flux",
        "rapport?.html",
        "CON.txt",
        "aux",
        "fin.",
        "",
        "   ",
    ],
)
def test_chemin_hors_du_dossier_refuse(tmp_path: Path, nom: str) -> None:
    with pytest.raises(ErreurDonnees):
        chemin_dans_dossier(tmp_path, nom)


def test_dossier_de_sortie_inexistant(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match="n'existe pas"):
        chemin_dans_dossier(tmp_path / "absent", "f.html")


def test_lien_symbolique_sortant_refuse(tmp_path: Path) -> None:
    dehors = tmp_path / "dehors"
    dehors.mkdir()
    dedans = tmp_path / "dedans"
    dedans.mkdir()
    try:
        os.symlink(dehors, dedans / "lien", target_is_directory=True)
    except (OSError, NotImplementedError):
        # Sous Windows sans droit de lien symbolique : une jonction, qui sort du dossier de la même façon.
        sortie = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dedans / "lien"), str(dehors)], capture_output=True, check=False
        )
        if sys.platform != "win32" or sortie.returncode != 0:
            pytest.skip("création de lien ou de jonction non permise sur ce poste")
    with pytest.raises(ErreurDonnees, match="lien symbolique"):
        chemin_dans_dossier(dedans, "lien/f.txt")


# ============================================================
# CIBLE (I7)
# ============================================================
def classes(effectifs: dict[str, int]) -> pd.Series:
    return pd.Series([k for k, n in effectifs.items() for _ in range(n)])


def test_resume_de_la_cible() -> None:
    r = resumer_cible(pd.Series([1.0, 2.0, np.nan, 2.0, 3.0]))
    assert (r.n_total, r.n_valides, r.n_manquantes, r.n_classes) == (5, 4, 1, 3)
    assert r.numerique and r.entiere and r.plus_petite_effectif == 1
    assert not resumer_cible(pd.Series([1.5, 2.0])).entiere
    texte = resumer_cible(classes({"a": 5, "b": 2}))
    assert not texte.numerique and (texte.plus_petite_nom, texte.plus_petite_effectif) == ("b", 2)


@pytest.mark.parametrize(
    ("serie", "attendu"),
    [
        (pd.Series(["a", "b", "a", "c"]), Probleme.CLASSIFICATION),
        (pd.Series([True, False, True]), Probleme.CLASSIFICATION),
        (pd.Series([0, 1, 0, 1, 1]), Probleme.CLASSIFICATION),
        (pd.Series([1, 2, 3, 4, 5, 1, 2, 3]), Probleme.CLASSIFICATION),
        (pd.Series(np.linspace(0.0, 1.0, 40)), Probleme.REGRESSION),
        (pd.Series(np.arange(100)), Probleme.REGRESSION),
        (pd.Series([np.nan, np.nan]), Probleme.CLASSIFICATION),
    ],
)
def test_detection_du_probleme(serie: pd.Series, attendu: Probleme) -> None:
    probleme, raison = detecter_probleme(serie)
    assert probleme is attendu and raison


def test_lignes_sans_cible() -> None:
    assert lignes_sans_cible(pd.DataFrame({"y": [1.0, np.nan, 3.0, np.nan]}), "y") == 2


def test_cible_saine_aucun_constat() -> None:
    y = classes({"a": 40, "b": 60})
    assert controle_cible(y, "y", Probleme.CLASSIFICATION) == ()
    assert controle_cible(pd.Series(np.linspace(0.0, 9.0, 50)), "y", Probleme.REGRESSION) == ()


def test_cible_vide_bloquante() -> None:
    assert codes(controle_cible(pd.Series([np.nan] * 5), "y", Probleme.REGRESSION)) == {
        "I7.cible_vide": Niveau.BLOQUANT
    }


def test_lignes_sans_cible_comptees_dans_le_message() -> None:
    y = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 6.0])
    (c,) = controle_cible(y, "y", Probleme.REGRESSION)
    assert c.code == "I7.cible_manquante" and c.niveau is Niveau.ALERTE
    assert "2 ligne(s)" in c.message and "33,3" in c.message and "retirées" in c.message and c.action


def test_cible_continue_en_classification_bloquante() -> None:
    c = codes(controle_cible(pd.Series(np.linspace(0.0, 1.0, 60)), "y", Probleme.CLASSIFICATION))
    assert c == {"I7.cible_continue": Niveau.BLOQUANT}
    entiers = pd.Series(np.arange(200))
    assert codes(controle_cible(entiers, "y", Probleme.CLASSIFICATION)) == {
        "I7.cible_continue": Niveau.BLOQUANT
    }


def test_cible_entiere_a_peu_de_valeurs_distinctes_en_classification_est_admise() -> None:
    assert controle_cible(pd.Series(np.arange(30) % 3), "y", Probleme.CLASSIFICATION) == ()


def test_cible_entiere_nombreuse_en_classification_alerte() -> None:
    c = codes(controle_cible(pd.Series(np.arange(300) % 30), "y", Probleme.CLASSIFICATION))
    assert c == {"I7.cible_entiere_nombreuse": Niveau.ALERTE}


def test_cible_textuelle_en_regression_bloquante() -> None:
    c = codes(controle_cible(classes({"a": 30, "b": 30}), "y", Probleme.REGRESSION))
    assert c == {"I7.cible_textuelle": Niveau.BLOQUANT}


def test_cible_binaire_en_regression_alerte() -> None:
    c = codes(controle_cible(pd.Series([0, 1] * 30), "y", Probleme.REGRESSION))
    assert c == {"I7.cible_binaire_en_regression": Niveau.ALERTE}


def test_une_seule_classe_bloquante() -> None:
    c = codes(controle_cible(classes({"a": 50}), "y", Probleme.CLASSIFICATION))
    assert c == {"I7.moins_de_deux_classes": Niveau.BLOQUANT}


def test_classe_plus_petite_que_les_plis_bloquante_avec_action_chiffree() -> None:
    (c,) = controle_cible(classes({"a": 50, "b": 3}), "y", Probleme.CLASSIFICATION, n_plis=5)
    assert c.code == "I7.classe_plus_petite_que_les_plis" and c.niveau is Niveau.BLOQUANT
    assert "«\u00a0b\u00a0»" in c.message and "3 ligne(s)" in c.message and "5 plis" in c.message
    assert "3 au plus" in c.action
    ok = controle_cible(classes({"a": 50, "b": 5}), "y", Probleme.CLASSIFICATION, n_plis=5, part_test=0.0001)
    assert "I7.classe_plus_petite_que_les_plis" not in codes(ok)


def test_classe_qui_passera_sous_le_nombre_de_plis_apres_mise_a_part_du_test() -> None:
    c = codes(
        controle_cible(classes({"a": 90, "b": 6}), "y", Probleme.CLASSIFICATION, n_plis=5, part_test=0.3)
    )
    assert c["I7.classe_trop_petite_apres_decoupage"] is Niveau.ALERTE
    apres = controle_cible(
        classes({"a": 90, "b": 4}),
        "y",
        Probleme.CLASSIFICATION,
        n_plis=5,
        part_test=0.3,
        apres_decoupage=True,
    )
    assert codes(apres)["I7.classe_plus_petite_que_les_plis"] is Niveau.BLOQUANT


def test_classe_rare_alerte_et_conseil_de_metrique() -> None:
    cs = controle_cible(classes({"0": 966, "1": 34}), "Panne", Probleme.CLASSIFICATION)
    (c,) = cs
    assert c.code == "I7.classe_rare" and c.niveau is Niveau.ALERTE
    assert "3,40" in c.message and "rappel" in c.action and "stratification" in c.action


def test_classe_a_5_pour_cent_n_est_pas_rare() -> None:
    assert controle_cible(classes({"0": 950, "1": 50}), "y", Probleme.CLASSIFICATION) == ()


@pytest.mark.parametrize("nom", ["../hors.html", "a/../../hors.html", "..\\hors.html", "/tmp/x", "C:\\x.txt"])
def test_chemin_qui_sort_du_dossier_est_refuse_avant_toute_resolution(tmp_path: Path, nom: str) -> None:
    """Premier rempart, sur le nom seul : le message conseille un simple nom de fichier."""
    with pytest.raises(ErreurDonnees, match=r"simple nom de fichier"):
        chemin_dans_dossier(tmp_path, nom)


def test_la_coupure_suit_le_cout_du_registre_pour_les_70_techniques() -> None:
    """Un seul critère : `EntreeCatalogue.coupee` (coût LOURD au delà de `LIGNES_MAX_LENTES`)."""
    from mecaml.core.catalog import REGISTRE
    from mecaml.core.types import LIGNES_MAX_LENTES

    assert LIGNES_MAX_LENTES == 20_000
    coupees = {e.id for e in REGISTRE if technique_coupee(e.id, LIGNES_MAX_LENTES + 1) is not None}
    assert coupees == {e.id for e in REGISTRE if e.coupee(LIGNES_MAX_LENTES + 1)} != set()
    assert all(technique_coupee(e.id, LIGNES_MAX_LENTES) is None for e in REGISTRE)


def test_technique_inconnue_du_registre_jamais_coupee() -> None:
    assert technique_coupee("classification.inconnue", 10_000_000) is None


def test_technique_coupee_suit_le_registre_fourni_et_donne_le_seul_message_de_refus() -> None:
    """Un registre fourni l'emporte ; `train.verifier_bornes` et `split` rendent ce même message."""
    from mecaml.core.catalog import Cout, EntreeCatalogue
    from mecaml.core.split import ErreurDecoupage, plis_validation
    from mecaml.core.train import ErreurBorne
    from mecaml.core.train import verifier_bornes as verifier_bornes_calcul
    from mecaml.core.types import SpecProjet

    lourde = EntreeCatalogue(
        id="regression.perso",
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=lambda graine, **parametres: None,
        grille_bornee=(),
        cout=Cout.LOURD,
        ref_livre="T05-1-1",
        texte_id="regression.perso",
    )
    assert technique_coupee("regression.perso", 30_000) is None  # absente du registre du projet
    message = technique_coupee("regression.perso", 30_000, [lourde])
    assert message is not None and "regression.perso" in message and "30 000" in message
    spec = SpecProjet(
        Probleme.REGRESSION,
        "y",
        (("x", Role.NUMERIQUE), ("y", Role.CIBLE)),
        "rmse",
        modeles=(ChoixTechnique.creer("regression.perso"),),
    )
    with pytest.raises(ErreurBorne) as erreur:
        verifier_bornes_calcul(spec, 30_000, [lourde])
    assert str(erreur.value) == message
    # la validation par exclusion d'une ligne : même critère, même message
    n = LIGNES_MAX_LENTES + 1
    donnees = pd.DataFrame({"x": np.zeros(n), "y": np.zeros(n)})
    exclusion = spec.avec(modeles=(), validation="validation.exclusion_unique")
    with pytest.raises(ErreurDecoupage) as refus:
        plis_validation(donnees, exclusion)
    assert str(refus.value) == technique_coupee("validation.exclusion_unique", n)


def test_une_seule_exception_d_annulation_pour_la_lecture_et_les_calculs() -> None:
    from mecaml.core.tasks import ErreurAnnulation

    assert ErreurAnnulee is ErreurAnnulation


def test_les_renvois_sont_des_identifiants_du_livre() -> None:
    from mecaml.core.livre import TECHNIQUES_LIVRE, est_identifiant
    from mecaml.core.renvois import THEMES

    for theme, identifiant in THEMES.items():
        assert identifiant is None or (
            est_identifiant(identifiant) and not TECHNIQUES_LIVRE[identifiant].hors_application
        ), theme
        assert renvoi(theme) == identifiant  # `guards.renvoi` lit la table unique, sans table propre
    assert renvoi("fuite") == "T02-1-1" and renvoi("personnel") is None
    with pytest.raises(KeyError, match="Renvoi au livre manquant"):
        renvoi("inconnu")


def test_chaque_constat_de_cible_renvoie_au_livre() -> None:
    from mecaml.core.livre import est_identifiant

    cas = [
        (pd.Series([np.nan] * 5), Probleme.REGRESSION),
        (pd.Series(np.linspace(0.0, 1.0, 60)), Probleme.CLASSIFICATION),
        (classes({"a": 30, "b": 30}), Probleme.REGRESSION),
        (pd.Series([0, 1] * 30), Probleme.REGRESSION),
        (classes({"a": 50}), Probleme.CLASSIFICATION),
        (classes({"a": 50, "b": 3}), Probleme.CLASSIFICATION),
        (classes({"0": 966, "1": 34}), Probleme.CLASSIFICATION),
        (pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 6.0]), Probleme.REGRESSION),
    ]
    for serie, probleme in cas:
        for c in controle_cible(serie, "y", probleme):
            assert est_identifiant(c.ref_livre), c.code
