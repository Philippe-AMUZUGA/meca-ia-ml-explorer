"""Chargement des fichiers .xlsx et .csv : formats, encodages, séparateurs, refus, progression."""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from mecaml.core.guards import ErreurAnnulee, ErreurDonnees
from mecaml.core.io import ATTR_NOTES, charger, empreinte_fichier, feuilles_challenge, lister_feuilles, sonder

EXEMPLES = Path(__file__).resolve().parents[2] / "examples"


# ============================================================
# OUTILS
# ============================================================
def ecrire_xlsx(
    chemin: Path, lignes: list[list], feuille: str = "Sheet1", autres: dict | None = None
) -> Path:
    """Classeur de test : `lignes` dans la première feuille, `autres` {nom: lignes} dans les suivantes."""
    classeur = Workbook()
    ws = classeur.active
    ws.title = feuille
    for ligne in lignes:
        ws.append(ligne)
    for nom, contenu in (autres or {}).items():
        suivante = classeur.create_sheet(nom)
        for ligne in contenu:
            suivante.append(ligne)
    classeur.save(chemin)
    return chemin


def reecrire_zip(
    source: Path, destination: Path, modifier: Callable[[str, bytes], bytes], ajouts=None
) -> Path:
    """Copie un classeur en modifiant le contenu de ses parties, et en ajoutant des parties."""
    with zipfile.ZipFile(source) as lu, zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as ecrit:
        for info in lu.infolist():
            ecrit.writestr(info.filename, modifier(info.filename, lu.read(info.filename)))
        for nom, contenu in (ajouts or {}).items():
            ecrit.writestr(nom, contenu)
    return destination


def csv_octets(chemin: Path, texte: str, encodage: str = "utf-8") -> Path:
    chemin.write_bytes(texte.encode(encodage))
    return chemin


BASE = "a,b,c\n1,2.5,x\n3,4.5,y\n5,6.5,z\n"


# ============================================================
# CSV : ENCODAGE, SÉPARATEUR, DÉCIMALE
# ============================================================
def test_csv_simple(tmp_path: Path) -> None:
    r = charger(csv_octets(tmp_path / "t.csv", BASE))
    df = r.table.donnees
    assert r.table.forme == (3, 3) and list(df.columns) == ["a", "b", "c"]
    assert df["b"].tolist() == [2.5, 4.5, 6.5] and df["c"].tolist() == ["x", "y", "z"]
    assert (r.encodage, r.separateur, r.decimal) == ("utf-8", ",", ".")
    assert r.table.nom == "t.csv" and r.table.feuille is None and r.table.est_intacte()


def test_csv_bom_utf8_ne_pollue_pas_le_premier_titre(tmp_path: Path) -> None:
    r = charger(csv_octets(tmp_path / "t.csv", BASE, "utf-8-sig"))
    assert r.encodage == "utf-8-sig" and list(r.table.donnees.columns) == ["a", "b", "c"]


def test_csv_windows_1252(tmp_path: Path) -> None:
    r = charger(
        csv_octets(tmp_path / "t.csv", "mat;force\nacier 42CrMo4;12,5\nalliage léger;9,25\n", "cp1252")
    )
    assert r.encodage == "cp1252"
    assert r.table.donnees["mat"].tolist() == ["acier 42CrMo4", "alliage léger"]


def test_csv_octets_hors_cp1252_lus_en_latin1(tmp_path: Path) -> None:
    contenu = b"a;b\n\x81;1\n\x8d;2\n\x8f;3\n"
    chemin = tmp_path / "t.csv"
    chemin.write_bytes(contenu)
    r = charger(chemin)
    assert r.encodage == "latin-1" and r.table.n_lignes == 3


def test_csv_utf16_avec_bom(tmp_path: Path) -> None:
    chemin = tmp_path / "t.csv"
    chemin.write_bytes(BASE.encode("utf-16"))
    r = charger(chemin)
    assert r.encodage == "utf-16" and r.table.forme == (3, 3)


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_csv_separateurs_detectes(tmp_path: Path, sep: str) -> None:
    texte = BASE.replace(",", sep)
    r = charger(csv_octets(tmp_path / "t.csv", texte))
    assert r.separateur == sep and r.table.forme == (3, 3)


def test_csv_virgule_decimale_avec_point_virgule(tmp_path: Path) -> None:
    texte = "force;temp\n12,5;300\n13,25;310\n11,75;295\n14,5;305\n"
    r = charger(csv_octets(tmp_path / "t.csv", texte))
    assert r.decimal == "," and r.table.donnees["force"].tolist() == [12.5, 13.25, 11.75, 14.5]


def test_csv_texte_entre_guillemets_avec_separateur(tmp_path: Path) -> None:
    texte = 'nom,valeur\n"acier, doux",1\n"alliage, léger",2\n'
    r = charger(csv_octets(tmp_path / "t.csv", texte))
    assert r.table.donnees["nom"].tolist() == ["acier, doux", "alliage, léger"]


def test_csv_octets_nuls_refuses(tmp_path: Path) -> None:
    chemin = tmp_path / "t.csv"
    chemin.write_bytes(b"a,b\n1,2\x00\n3,4\n")
    with pytest.raises(ErreurDonnees, match="octets nuls"):
        charger(chemin)


def test_csv_vide_refuse(tmp_path: Path) -> None:
    chemin = tmp_path / "t.csv"
    chemin.write_bytes(b"")
    with pytest.raises(ErreurDonnees, match="vide"):
        charger(chemin)


def test_csv_en_tete_seul_refuse(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match="aucune ligne"):
        charger(csv_octets(tmp_path / "t.csv", "a,b,c\n"))


def test_csv_lignes_irregulieres_refusees_avec_conseil(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match=r"mal formé.*même nombre de champs"):
        charger(csv_octets(tmp_path / "t.csv", "a,b\n1,2\n3,4,5,6\n"))


def test_csv_une_seule_colonne_refusee_avec_conseil_separateur(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match=r"qu'une colonne.*séparateur"):
        charger(csv_octets(tmp_path / "t.csv", "a\n1\n2\n"))


def test_csv_titres_doubles_et_vides_corriges(tmp_path: Path) -> None:
    r = charger(csv_octets(tmp_path / "t.csv", "a,a,,b\n1,2,3,4\n5,6,7,8\n"))
    assert list(r.table.donnees.columns) == ["a", "a.1", "Unnamed: 2", "b"]
    assert any("double" in n for n in r.notes) and any("sans titre" in n for n in r.notes)


def test_csv_dates_iso_converties(tmp_path: Path) -> None:
    r = charger(csv_octets(tmp_path / "t.csv", "jour,x\n2026-01-05,1\n2026-01-06,2\n2026-01-07,3\n"))
    assert pd.api.types.is_datetime64_any_dtype(r.table.donnees["jour"])
    assert any("dates" in n for n in r.notes)


def test_csv_dates_non_iso_laissees_en_texte(tmp_path: Path) -> None:
    r = charger(csv_octets(tmp_path / "t.csv", "jour,x\n05/01/2026,1\n06/01/2026,2\n"))
    assert not pd.api.types.is_datetime64_any_dtype(r.table.donnees["jour"])


def test_csv_lignes_vides_gardent_leur_position_comme_read_csv(tmp_path: Path) -> None:
    chemin = csv_octets(tmp_path / "t.csv", "a,b\n1,2\n,\n3,4\n,\n")
    r = charger(chemin)
    assert r.table.n_lignes == 4 and any("2 ligne(s) sans aucune valeur" in n for n in r.notes)
    assert r.table.donnees.equals(pd.read_csv(chemin))


def test_csv_les_remarques_sont_rangees_dans_les_attributs(tmp_path: Path) -> None:
    r = charger(csv_octets(tmp_path / "t.csv", "a,a\n1,2\n3,4\n"))
    assert r.table.donnees.attrs[ATTR_NOTES] == r.notes and r.notes


# ============================================================
# FORMATS ET FICHIERS REFUSÉS
# ============================================================
@pytest.mark.parametrize(
    "nom", ["t.xls", "t.xlsm", "t.xlsb", "t.ods", "t.json", "t.parquet", "t.pkl", "t.joblib", "t"]
)
def test_extensions_refusees(tmp_path: Path, nom: str) -> None:
    chemin = tmp_path / nom
    chemin.write_bytes(b"contenu")
    with pytest.raises(ErreurDonnees, match=r"seuls les fichiers \.xlsx ou \.csv"):
        charger(chemin)


def test_fichier_introuvable(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match="introuvable"):
        charger(tmp_path / "absent.csv")


def test_dossier_refuse(tmp_path: Path) -> None:
    (tmp_path / "d.csv").mkdir()
    with pytest.raises(ErreurDonnees, match="n'est pas un fichier"):
        charger(tmp_path / "d.csv")


def test_fichier_vide_refuse(tmp_path: Path) -> None:
    chemin = tmp_path / "t.xlsx"
    chemin.write_bytes(b"")
    with pytest.raises(ErreurDonnees, match="vide"):
        charger(chemin)


def test_texte_renomme_en_xlsx_refuse(tmp_path: Path) -> None:
    chemin = tmp_path / "t.xlsx"
    chemin.write_text("pas un classeur", encoding="utf-8")
    with pytest.raises(ErreurDonnees, match=r"pas un classeur .xlsx valide"):
        charger(chemin)


def test_classeur_tronque_refuse(tmp_path: Path) -> None:
    complet = ecrire_xlsx(tmp_path / "ok.xlsx", [["a", "b"], [1, 2], [3, 4]])
    tronque = tmp_path / "t.xlsx"
    tronque.write_bytes(complet.read_bytes()[:200])
    with pytest.raises(ErreurDonnees, match=r"pas un classeur .xlsx valide"):
        charger(tronque)


def test_ancien_format_ou_mot_de_passe_refuse(tmp_path: Path) -> None:
    chemin = tmp_path / "t.xlsx"
    chemin.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)
    with pytest.raises(ErreurDonnees, match=r"\.xls.*mot de passe"):
        charger(chemin)


def test_classeur_a_macros_refuse(tmp_path: Path) -> None:
    ok = ecrire_xlsx(tmp_path / "ok.xlsx", [["a", "b"], [1, 2], [3, 4]])
    avec_macros = reecrire_zip(ok, tmp_path / "m.xlsx", lambda _n, c: c, {"xl/vbaProject.bin": b"MACRO"})
    with pytest.raises(ErreurDonnees, match="macros"):
        charger(avec_macros)


def test_classeur_a_entites_xml_refuse(tmp_path: Path) -> None:
    ok = ecrire_xlsx(tmp_path / "ok.xlsx", [["a", "b"], [1, 2], [3, 4]])

    def piegee(nom: str, contenu: bytes) -> bytes:
        if nom == "xl/workbook.xml":
            return b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e "boum">]>' + contenu
        return contenu

    with pytest.raises(ErreurDonnees, match="entités"):
        charger(reecrire_zip(ok, tmp_path / "e.xlsx", piegee))


# ============================================================
# XLSX : FEUILLES, TITRES, TYPES
# ============================================================
def test_xlsx_aller_retour_comme_read_excel(tmp_path: Path) -> None:
    lignes = [["force", "materiau", "ok"], [1.5, "acier", True], [2.5, "alu", False], [None, "inox", True]]
    chemin = ecrire_xlsx(tmp_path / "t.xlsx", lignes)
    mien = charger(chemin).table.donnees
    pandas = pd.read_excel(chemin)
    assert list(mien.columns) == list(pandas.columns)
    assert mien["force"].isna().sum() == 1 and mien["materiau"].tolist() == ["acier", "alu", "inox"]
    assert [str(t) for t in mien.dtypes] == [str(t) for t in pandas.dtypes]


@pytest.mark.parametrize(
    "fichier",
    [
        "MecaIA-Traitement_Acier.xlsx",
        "MecaIA-Controle_Conformite.xlsx",
        "MecaIA-TimeSeries_RUL_extrait.xlsx",
        "MecaIA-Anomalies_Pompe.xlsx",
    ],
)
def test_jeux_d_exemple_identiques_a_read_excel(fichier: str) -> None:
    mien = charger(EXEMPLES / fichier).table.donnees
    pandas = pd.read_excel(EXEMPLES / fichier)
    assert mien.equals(pandas)
    assert [str(t) for t in mien.dtypes] == [str(t) for t in pandas.dtypes]


def test_xlsx_feuilles_et_choix(tmp_path: Path) -> None:
    chemin = ecrire_xlsx(
        tmp_path / "t.xlsx", [["a", "b"], [1, 2], [3, 4]], "premiere", {"seconde": [["c", "d"], [5, 6]]}
    )
    assert lister_feuilles(chemin) == ("premiere", "seconde")
    assert charger(chemin).table.feuille == "premiere"
    r = charger(chemin, feuille="seconde")
    assert r.table.feuille == "seconde" and list(r.table.donnees.columns) == ["c", "d"]
    assert r.feuilles == ("premiere", "seconde")


def test_xlsx_feuille_inconnue_liste_les_feuilles(tmp_path: Path) -> None:
    chemin = ecrire_xlsx(tmp_path / "t.xlsx", [["a", "b"], [1, 2]], "premiere")
    with pytest.raises(ErreurDonnees, match=r"absente du classeur.*premiere"):
        charger(chemin, feuille="inconnue")


def test_xlsx_feuille_masquee_ignoree(tmp_path: Path) -> None:
    classeur = Workbook()
    classeur.active.title = "cachee"
    classeur.active.append(["x", "y"])
    classeur.active.sheet_state = "hidden"
    visible = classeur.create_sheet("visible")
    visible.append(["a", "b"])
    visible.append([1, 2])
    chemin = tmp_path / "t.xlsx"
    classeur.save(chemin)
    assert lister_feuilles(chemin) == ("visible",)
    assert charger(chemin).table.feuille == "visible"


def test_xlsx_feuille_vide_refusee(tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match="feuille choisie est vide"):
        charger(ecrire_xlsx(tmp_path / "t.xlsx", []))


def test_xlsx_titres_vides_et_doubles(tmp_path: Path) -> None:
    chemin = ecrire_xlsx(tmp_path / "t.xlsx", [[None, "a", "a"], [1, 2, 3], [4, 5, 6]])
    r = charger(chemin)
    assert list(r.table.donnees.columns) == ["Unnamed: 0", "a", "a.1"]
    assert len(r.notes) == 2


def test_xlsx_titres_numeriques_deviennent_du_texte(tmp_path: Path) -> None:
    r = charger(ecrire_xlsx(tmp_path / "t.xlsx", [[1, 2.5], [3, 4], [5, 6]]))
    assert list(r.table.donnees.columns) == ["1", "2.5"]


def test_xlsx_lignes_vides_gardent_leur_position_comme_read_excel(tmp_path: Path) -> None:
    lignes = [["a", "b", "vide"], [1, 2, None], [None, None, None], [3, 4, None], [None, None, None]]
    chemin = ecrire_xlsx(tmp_path / "t.xlsx", lignes)
    r = charger(chemin)
    df = r.table.donnees
    assert df.shape == (3, 3) and str(df["vide"].dtype) == "float64" and df["vide"].isna().all()
    assert df["a"].isna().tolist() == [False, True, False] and df.equals(pd.read_excel(chemin))
    assert any("1 ligne(s) sans aucune valeur" in n for n in r.notes)
    assert any("en fin de feuille" in n for n in r.notes)


def test_xlsx_cellules_en_erreur_lues_comme_manquantes(tmp_path: Path) -> None:
    r = charger(ecrire_xlsx(tmp_path / "t.xlsx", [["a", "b"], [1, 2], ["#DIV/0!", 4], [5, "#N/A"]]))
    df = r.table.donnees
    assert df["a"].isna().tolist() == [False, True, False] and df["b"].isna().tolist() == [False, False, True]
    assert pd.api.types.is_numeric_dtype(df["a"]) and any("erreur de tableur" in n for n in r.notes)


def test_xlsx_dates_en_cellules_et_en_texte_iso(tmp_path: Path) -> None:
    from datetime import datetime

    lignes = [["jour", "texte"], [datetime(2026, 1, 5), "2026-01-05"], [datetime(2026, 1, 6), "2026-01-06"]]
    df = charger(ecrire_xlsx(tmp_path / "t.xlsx", lignes)).table.donnees
    assert pd.api.types.is_datetime64_any_dtype(df["jour"]) and pd.api.types.is_datetime64_any_dtype(
        df["texte"]
    )


def test_xlsx_nombres_ecrits_en_texte_restent_du_texte(tmp_path: Path) -> None:
    r = charger(ecrire_xlsx(tmp_path / "t.xlsx", [["a", "b"], ["12,5", 1], ["13,5", 2]]))
    assert not pd.api.types.is_numeric_dtype(r.table.donnees["a"])


# ============================================================
# XLSX : FORMULES LUES EN VALEURS
# ============================================================
def classeur_a_formules(tmp_path: Path, avec_cache: bool) -> Path:
    chemin = ecrire_xlsx(
        tmp_path / "f.xlsx", [["a", "b", "c"], [1, 2, "=A2+B2"], [3, 4, "=A3+B3"], [5, 6, "=A4+B4"]]
    )
    if not avec_cache:
        return chemin

    def valeurs_en_cache(nom: str, contenu: bytes) -> bytes:
        if nom == "xl/worksheets/sheet1.xml":
            return contenu.replace(b"<v></v>", b"<v>99</v>")
        return contenu

    return reecrire_zip(chemin, tmp_path / "f_cache.xlsx", valeurs_en_cache)


def test_formules_avec_valeur_en_cache_lues_en_valeurs(tmp_path: Path) -> None:
    r = charger(classeur_a_formules(tmp_path, avec_cache=True))
    assert r.formules == 3 and r.formules_sans_valeur == 0
    assert r.table.donnees["c"].tolist() == [99, 99, 99]  # la valeur du tableur, jamais A+B recalculé


def test_formules_sans_valeur_lues_comme_manquantes_et_signalees(tmp_path: Path) -> None:
    r = charger(classeur_a_formules(tmp_path, avec_cache=False))
    assert r.formules == 3 and r.formules_sans_valeur == 3
    assert r.table.donnees["c"].isna().all() and any("formule(s) sans valeur" in n for n in r.notes)


# ============================================================
# SONDE, PROGRESSION, ANNULATION, MODE CHALLENGE
# ============================================================
def test_sonde(tmp_path: Path) -> None:
    chemin = ecrire_xlsx(tmp_path / "t.xlsx", [["a", "b"], *([[1, 2]] * 40)], "feuille_un")
    s = sonder(chemin)
    assert s.extension == ".xlsx" and s.feuilles == ("feuille_un",) and s.lignes_estimees == 41
    assert sonder(csv_octets(tmp_path / "t.csv", BASE)).feuilles == ()


def test_progression_croissante_jusqu_a_un(tmp_path: Path) -> None:
    lignes = [["a", "b"], *([[1, 2]] * 12_000)]
    appels: list[tuple[float, str]] = []
    charger(ecrire_xlsx(tmp_path / "t.xlsx", lignes), progression=lambda f, m: appels.append((f, m)))
    fractions = [f for f, _ in appels]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert len(appels) >= 4 and all(0.0 <= f <= 1.0 for f in fractions)


def test_annulation_interrompt_la_lecture(tmp_path: Path) -> None:
    lignes = [["a", "b"], *([[1, 2]] * 30_000)]
    chemin = ecrire_xlsx(tmp_path / "t.xlsx", lignes)
    appels = {"n": 0}

    def annuler() -> bool:
        appels["n"] += 1
        return appels["n"] > 3

    with pytest.raises(ErreurAnnulee):
        charger(chemin, annuler=annuler)


def test_annulation_d_un_csv(tmp_path: Path) -> None:
    chemin = csv_octets(tmp_path / "t.csv", BASE)
    with pytest.raises(ErreurAnnulee):
        charger(chemin, annuler=lambda: True)


def test_feuilles_challenge() -> None:
    assert feuilles_challenge(("Apprentissage", "Test")) == ("Apprentissage", "Test")
    assert feuilles_challenge(("train", "test", "notes")) == ("train", "test")
    assert feuilles_challenge(("données", "test")) is None
    assert feuilles_challenge(()) is None


def test_chargement_sans_effet_de_bord_sur_le_fichier(tmp_path: Path) -> None:
    source = shutil.copyfile(EXEMPLES / "MecaIA-Traitement_Acier.xlsx", tmp_path / "acier.xlsx")
    avant = (source.stat().st_size, source.stat().st_mtime_ns, source.read_bytes())
    charger(source)
    assert (source.stat().st_size, source.stat().st_mtime_ns, source.read_bytes()) == avant
    assert sorted(p.name for p in tmp_path.iterdir()) == ["acier.xlsx"]


def test_donnees_numeriques_conservees_exactement(tmp_path: Path) -> None:
    valeurs = np.random.default_rng(3).normal(size=(50, 3))
    lignes = [["x", "y", "z"], *valeurs.tolist()]
    df = charger(ecrire_xlsx(tmp_path / "t.xlsx", lignes)).table.donnees
    assert np.allclose(df.to_numpy(), valeurs, rtol=1e-12, atol=0.0)


def test_empreinte_du_fichier_pour_la_liste_blanche_du_depot(tmp_path: Path) -> None:
    import hashlib

    chemin = shutil.copyfile(EXEMPLES / "MecaIA-Traitement_Acier.xlsx", tmp_path / "acier.xlsx")
    attendue = hashlib.sha256(chemin.read_bytes()).hexdigest()
    assert charger(chemin).sha256 == attendue == empreinte_fichier(chemin)
    chemin.write_bytes(chemin.read_bytes() + b"x")
    assert empreinte_fichier(chemin) != attendue
    with pytest.raises(ErreurDonnees, match=r"introuvable|n'a pas pu être lu"):
        empreinte_fichier(tmp_path / "absent.xlsx")


# ============================================================
# ANNULATION COMMUNE AVEC LES TÂCHES, AVERTISSEMENTS PAR FIL, LIGNES SANS ÉTAT PARTAGÉ
# ============================================================
def test_chargement_dans_une_tache_annulee_n_est_pas_un_echec(tmp_path: Path) -> None:
    from mecaml.core.tasks import EtatTache, Tache

    chemin = ecrire_xlsx(tmp_path / "gros.xlsx", [["a", "b"], *([[1, 2]] * 30_000)])

    def calcul(annulation, _avancer):
        return charger(
            chemin, annuler=annulation, progression=lambda f, m: annulation.demander() if f > 0.1 else None
        )

    tache = Tache(calcul, nom="chargement").demarrer()
    assert tache.attendre(60)
    assert tache.etat is EtatTache.ANNULEE
    with pytest.raises(ErreurAnnulee):
        tache.resultat()


def test_chargement_dans_une_tache_non_annulee_rend_la_table(tmp_path: Path) -> None:
    from mecaml.core.tasks import EtatTache, Tache

    chemin = ecrire_xlsx(tmp_path / "petit.xlsx", [["a", "b"], [1, 2], [3, 4]])
    tache = Tache(lambda annulation, _avancer: charger(chemin, annuler=annulation), nom="petit").demarrer()
    assert tache.attendre(30) and tache.etat is EtatTache.TERMINEE
    assert tache.resultat().table.forme == (2, 2)


def test_les_avertissements_du_lecteur_sont_rassembles_dans_les_remarques(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings

    import openpyxl

    vrai = openpyxl.load_workbook

    def bavard(*args, **kwargs):
        warnings.warn("feuille sans style par défaut", UserWarning, stacklevel=2)
        return vrai(*args, **kwargs)

    monkeypatch.setattr(openpyxl, "load_workbook", bavard)
    r = charger(ecrire_xlsx(tmp_path / "t.xlsx", [["a", "b"], [1, 2], [3, 4]]))
    assert any("feuille sans style par défaut" in n and "UserWarning" in n for n in r.notes)


def test_le_filtre_ajoute_est_idempotent_et_aucun_etat_n_est_restaure() -> None:
    import warnings

    from mecaml.core.io import _avertissements_de_lecture

    with _avertissements_de_lecture():
        pass
    avant = len(warnings.filters)
    for _ in range(5):
        with _avertissements_de_lecture():
            pass
    assert len(warnings.filters) == avant


def test_deux_taches_simultanees_ne_melangent_pas_leurs_avertissements(tmp_path: Path) -> None:
    """Un chargement et un calcul qui avertit tournent ensemble : chaque fil ne voit que les siens."""
    import threading
    import warnings

    from mecaml.core.tasks import Tache, collecter_avertissements

    chemin = ecrire_xlsx(tmp_path / "t.xlsx", [["a", "b"], *([[1, 2]] * 20_000)])
    depart = threading.Barrier(2)

    def lecture(annulation, _avancer) -> list[str]:
        depart.wait(timeout=30)
        return list(charger(chemin, annuler=annulation).notes)

    def calcul(_annulation, _avancer) -> list[str]:
        depart.wait(timeout=30)
        warnings.filterwarnings(
            "always", message="calcul-"
        )  # le filtre « error » des tests ne doit pas lever ici
        with collecter_avertissements() as siens:
            for i in range(200):
                warnings.warn(f"calcul-{i}", UserWarning, stacklevel=1)
        return list(siens)

    t1, t2 = Tache(lecture, nom="lecture").demarrer(), Tache(calcul, nom="calcul").demarrer()
    assert t1.attendre(60) and t2.attendre(60)
    assert not any("calcul-" in n for n in t1.resultat())
    assert len(t2.resultat()) == 200 and all(a.startswith("UserWarning : calcul-") for a in t2.resultat())


def test_lecture_sans_etat_partage_deux_fichiers_differents_dans_deux_taches(tmp_path: Path) -> None:
    from mecaml.core.tasks import Tache

    a = ecrire_xlsx(tmp_path / "a.xlsx", [["x", "y"], *([[1, 2]] * 500)])
    b = csv_octets(tmp_path / "b.csv", "u;v;w\n" + "1,5;2;3\n" * 700)
    ta = Tache(lambda an, _av: charger(a, annuler=an).table.forme, nom="a").demarrer()
    tb = Tache(lambda an, _av: charger(b, annuler=an).table.forme, nom="b").demarrer()
    assert ta.attendre(30) and tb.attendre(30)
    assert (ta.resultat(), tb.resultat()) == ((500, 2), (700, 3))
