"""Sorties du projet : rapport HTML hors ligne, classeur, modèle, projet rejouable, garde-fous (I10)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

import aides_apprentissage as jeux
import aides_reproductibilite as aides
from mecaml.core import codegen, evaluate, libelles, report, split, train
from mecaml.core.guards import ErreurDonnees
from mecaml.core.session import Etat, Revenir, Session, Tester
from mecaml.core.types import ChoixTechnique, Constat, Niveau, SpecProjet

CT = ChoixTechnique.creer
RIDGE = CT("regression.ridge")
ARBRE = CT("regression.arbre", max_depth=3)
LOGISTIQUE = CT("classification.logistique")
ARBRE_CLS = CT("classification.arbre", max_depth=3)
MAINTENANT = datetime(2026, 10, 2, 9, 30)
ENV = {"python": "3.12.0", "numpy": "2.4.6", "pandas": "3.0.5", "scipy": "1.18.0", "scikit-learn": "1.9.0"}
OPTIONS = {"maintenant": MAINTENANT, "env": ENV}


def _mener(dossier: Path, jeu: pd.DataFrame, fabrique, **options: Any) -> aides.Projet:
    dossier.mkdir(parents=True, exist_ok=True)
    return aides.mener_le_projet(aides.ecrire_csv(jeu, dossier / "jeu.csv"), fabrique, **options)


@dataclasses.dataclass
class Cas:
    projet: aides.Projet
    session: Session


def _cas(dossier: Path, jeu: pd.DataFrame, fabrique, importances: bool = True, **options: Any) -> Cas:
    projet = _mener(dossier, jeu, fabrique, **options)
    return Cas(projet, aides.mener_la_session(projet, importances=importances))


@pytest.fixture(scope="module")
def regression(tmp_path_factory: pytest.TempPathFactory) -> Cas:
    return _cas(
        tmp_path_factory.mktemp("rapport_regression"),
        jeux.jeu_regression(n=120),
        lambda t: jeux.spec_regression(modeles=(RIDGE, ARBRE)),
    )


@pytest.fixture(scope="module")
def classification(tmp_path_factory: pytest.TempPathFactory) -> Cas:
    return _cas(
        tmp_path_factory.mktemp("rapport_classification"),
        jeux.jeu_classification(n=300),
        lambda t: jeux.spec_classification(modeles=(LOGISTIQUE, ARBRE_CLS)),
        decider=lambda res: __import__("mecaml.core.decision", fromlist=["x"]).decider(
            res, "classification.logistique", seuil=0.35
        ),
    )


@pytest.fixture(scope="module")
def serie(tmp_path_factory: pytest.TempPathFactory) -> Cas:
    return _cas(tmp_path_factory.mktemp("rapport_serie"), aides.jeu_serie_lacunaire(), aides.spec_lacunaire)


@pytest.fixture(scope="module")
def persistance(tmp_path_factory: pytest.TempPathFactory) -> Cas:
    return _cas(
        tmp_path_factory.mktemp("rapport_persistance"), jeux.jeu_serie_mesuree(), aides.spec_persistance
    )


@pytest.fixture(scope="module")
def pannes(tmp_path_factory: pytest.TempPathFactory) -> Cas:
    return _cas(
        tmp_path_factory.mktemp("rapport_pannes"), aides.jeu_pannes(), aides.spec_classification_groupee
    )


@pytest.fixture(scope="module")
def sans_cible(tmp_path_factory: pytest.TempPathFactory) -> Cas:
    return _cas(
        tmp_path_factory.mktemp("rapport_sans_cible"),
        jeux.jeu_sans_cible(n=150),
        lambda t: jeux.spec_sans_cible(
            modeles=(
                CT("sans_cible.kmeans", n_clusters=2),
                CT("sans_cible.acp", n_components=2),
                CT("sans_cible.foret_isolement", contamination=0.05),
            )
        ),
        decider=lambda res: __import__("mecaml.core.decision", fromlist=["x"]).decider(
            res, "sans_cible.kmeans"
        ),
    )


# ============================================================
# LECTURE DU HTML
# ============================================================
ATTRIBUTS_A_ADRESSE = {
    "src",
    "href",
    "data",
    "action",
    "poster",
    "srcset",
    "formaction",
    "cite",
    "background",
}


class Collecteur(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.balises: Counter[str] = Counter()
        self.adresses: list[tuple[str, str, str]] = []
        self.evenements: list[tuple[str, str]] = []
        self.style = ""
        self._dans_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.balises[tag] += 1
        self._dans_style = tag == "style"
        for cle, valeur in attrs:
            if cle in ATTRIBUTS_A_ADRESSE and valeur:
                self.adresses.append((tag, cle, valeur))
            if cle.startswith("on"):
                self.evenements.append((tag, cle))

    def handle_endtag(self, tag: str) -> None:
        self._dans_style = False

    def handle_data(self, data: str) -> None:
        if self._dans_style:
            self.style += data


def lire(html: str) -> Collecteur:
    collecteur = Collecteur()
    collecteur.feed(html)
    collecteur.close()
    return collecteur


def _html_neuf(cas: Cas, **options: Any) -> str:
    return report.rapport_html(cas.session, **{**OPTIONS, "chargement": cas.projet.chargement, **options})


_RAPPORTS: dict[int, tuple[Cas, str]] = {}


def _html(cas: Cas, **options: Any) -> str:
    """Rapport du cas ; sans option, calculé une seule fois (le cas est gardé : son identité reste unique)."""
    if options:
        return _html_neuf(cas, **options)
    if id(cas) not in _RAPPORTS:
        _RAPPORTS[id(cas)] = (cas, _html_neuf(cas))
    return _RAPPORTS[id(cas)][1]


# ============================================================
# RAPPORT HTML : OUVERT SANS RÉSEAU
# ============================================================
@pytest.mark.parametrize(
    "nom", ["regression", "classification", "serie", "persistance", "pannes", "sans_cible"]
)
def test_le_rapport_s_ouvre_sans_reseau(nom: str, request: pytest.FixtureRequest) -> None:
    cas = request.getfixturevalue(nom)
    html = _html(cas)
    collecteur = lire(html)
    assert (
        html.startswith("<!DOCTYPE html>") and '<html lang="fr">' in html and '<meta charset="utf-8">' in html
    )
    # aucune ressource extérieure : pas de lien vers l'extérieur, pas de script, pas de feuille de style liée
    externes = [a for a in collecteur.adresses if not a[2].startswith("#")]
    assert externes == []
    for balise in ("script", "link", "iframe", "img", "object", "embed", "form", "base"):
        assert collecteur.balises[balise] == 0, balise
    assert collecteur.evenements == []
    assert "@import" not in collecteur.style and "url(" not in collecteur.style
    assert not re.search(r'http-equiv\s*=\s*"?refresh', html, re.I)
    assert (
        collecteur.balises["svg"] >= 1 and collecteur.balises["style"] >= 1
    )  # style de la page et des figures


def test_le_rapport_contient_les_sections_et_se_range_par_ancres(regression: Cas) -> None:
    html = _html(regression)
    collecteur = lire(html)
    ancres = {valeur[1:] for _, _, valeur in collecteur.adresses if valeur.startswith("#")}
    identifiants = set(re.findall(r'<section id="([a-z_]+)"', html))
    assert {
        "resume",
        "donnees",
        "audit",
        "decoupage",
        "pretraitement",
        "modeles",
        "decision",
        "test",
    } <= identifiants
    assert {"reproductibilite", "limites", "renvois"} <= identifiants
    assert ancres <= identifiants | {"resume"} and ancres >= {
        "resume",
        "donnees",
        "test",
    }  # le sommaire pointe sur les sections
    assert "Aucune donnée n'a quitté cet ordinateur" in html
    assert "2026" in html and "09:30" in html


def test_le_rapport_est_identique_quand_les_entrees_le_sont(regression: Cas) -> None:
    assert _html_neuf(regression) == _html_neuf(regression)


def test_le_rapport_echappe_les_noms_de_colonnes_et_les_messages(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 120
    piege = "<img src=x onerror=alert(1)>"
    jeu = pd.DataFrame({piege: rng.normal(size=n), "b": rng.normal(size=n)})
    jeu["y"] = 0.5 * jeu[piege] + rng.normal(0, 1.0, n)  # une relation réelle, sans fuite par la cible
    from mecaml.core.types import Probleme, Role

    def spec(_: Any) -> SpecProjet:
        return SpecProjet(
            probleme=Probleme.REGRESSION,
            cible="y",
            roles=((piege, Role.NUMERIQUE), ("b", Role.NUMERIQUE), ("y", Role.CIBLE)),
            metrique="rmse",
            pretraitement=(CT("transformation.standard"),),
            modeles=(RIDGE,),
        )

    def auditeur(table: Any, spec_: Any, coffre: Any) -> tuple[Constat, ...]:
        return (Constat(Niveau.ALERTE, "ESSAI.balise", "Une <b>alerte</b> & un <script>", "L'acquitter."),)

    projet = _mener(tmp_path, jeu, spec)
    session = aides.mener_la_session(projet, auditeur=auditeur)
    html = report.rapport_html(session, **OPTIONS)
    collecteur = lire(html)
    assert (
        collecteur.balises["img"] == 0 and collecteur.balises["script"] == 0 and collecteur.balises["b"] == 0
    )
    assert (
        "&lt;img src=x onerror=alert(1)&gt;" in html
        and "Une &lt;b&gt;alerte&lt;/b&gt; &amp; un &lt;script&gt;" in html
    )


# ============================================================
# CE QUE LE RAPPORT DIT
# ============================================================
def test_le_rapport_cite_les_mesures_de_l_application(regression: Cas, classification: Cas) -> None:
    html = _html(regression)
    assert "visée 20,0 %, obtenue 20,0 %" in html and "5 sur 5" in html and "plis aléatoires, 5 plis" in html
    assert "validation.kfold" not in html and "regression.ridge" not in html
    assert "Lignes de test sans cible, retirées" in html and "Ouvertures du jeu de test" in html
    assert "T04-2-1" in html  # la technique de prétraitement renvoie à sa section du livre
    html = _html(classification)
    assert "fixée par l'utilisateur en S3" in html and "Classe positive" in html
    assert (
        "Matrice de confusion" in html
        and "Courbes ROC et précision rappel" in html
        and "Scores selon le seuil" in html
    )
    assert "Seuil de décision" in html and "0,350" in html
    assert "Mêmes figures sur le test" in html


def test_le_rapport_dit_la_part_de_test_obtenue_et_non_la_part_visee(tmp_path: Path) -> None:
    """Des unités entières écartent la part obtenue de la part visée : le rapport écrit celle du coffre."""
    cas = _cas(
        tmp_path, aides.jeu_pannes(), lambda t: aides.spec_classification_groupee(t).avec(part_test=0.27)
    )
    coffre, spec = cas.session.coffre, cas.session.spec
    obtenue = split.part_obtenue(coffre)
    assert abs(obtenue - spec.part_test) > 0.005  # les unités entières écartent les deux parts
    voulu = f"visée {report._n(100 * spec.part_test, 1)} %, obtenue {report._n(100 * obtenue, 1)} %"
    assert voulu in _html(cas)


def test_le_rapport_liste_les_alertes_acquittees(serie: Cas) -> None:
    assert "I7.cible_manquante" in serie.session.acquittes
    html = _html(serie)
    assert "Alertes acquittées par l'utilisateur (1)" in html and "I7.cible_manquante" in html
    assert "acquittée" in html and "d'apprentissage sans cible ont été retirées de la validation" in html


def test_le_rapport_ecrit_l_hypothese_de_la_cible_du_pas_precedent(persistance: Cas, regression: Cas) -> None:
    html = _html(persistance)
    assert "la cible du pas précédent est connue au moment de la prévision" in html
    assert "cible_connue = vrai pour chronologique.cible_decalee, reference.persistance" in html
    assert "ne repère qu'un compte à rebours régulier" in html
    assert "cible du pas précédent" not in _html(regression)


def test_le_rapport_dit_le_score_de_test_optimiste_quand_le_test_etait_deja_vu(tmp_path: Path) -> None:
    projet = _mener(tmp_path, jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE,)))
    session = aides.mener_la_session(projet)
    assert not session.test_optimiste and "Score de test optimiste" not in report.rapport_html(
        session, **OPTIONS
    )
    session = session.appliquer(Revenir(Etat.DIAGNOSTIC))  # un retour en S7, puis le test est rouvert
    evaluer = evaluate.preparer_test(
        projet.spec, projet.table, projet.coffre, projet.decision, importances=False
    )
    session = session.appliquer(Tester(evaluer))
    assert session.test_optimiste and session.coffre.ouvertures == 2
    html = report.rapport_html(session, **OPTIONS)
    assert "Score de test optimiste" in html and "test avait déjà été ouvert" in html
    assert "Ouvertures du jeu de test" in html


def test_le_rapport_dit_que_la_recherche_d_hyperparametres_est_optimiste(regression: Cas) -> None:
    recherche = train.rechercher(
        regression.projet.spec,
        CT("regression.ridge"),
        regression.projet.table,
        regression.projet.coffre,
        mode="grille",
    )
    html = report.rapport_html(
        regression.session,
        **{**OPTIONS, "recherches": (recherche,), "chargement": regression.projet.chargement},
    )
    assert "Recherche d'hyperparamètres : Régression de crête (Ridge)" in html
    assert "son score est optimiste" in html
    assert "Recherche d'hyperparamètres" not in _html(regression)


def test_les_tableaux_d_explication_ont_des_en_tetes_en_francais(regression: Cas, sans_cible: Cas) -> None:
    html = _html(regression)
    explication = html.split('<section id="explication">')[1].split("</section>")[0]
    for entete in ("Variable", "Baisse moyenne du score", "Écart type", "Coefficient"):
        assert f"<th>{entete}</th>" in explication or f"<th class=num>{entete}</th>" in explication
    for brut in ("ecart_type", "plis_valides", ">moyenne<"):
        assert brut not in explication
    lecture = _html(sans_cible).split('<section id="lecture">')[1].split("</section>")[0]
    assert "Effectif" in lecture and "effectif<" not in lecture
    assert "plis_valides" not in html  # le test n'a qu'un seul jeu de lignes : pas de plis à compter


def test_le_rapport_dit_les_limites_et_le_contenu_du_modele_exporte(regression: Cas) -> None:
    html = _html(regression)
    assert "joblib.load exécute du code" in html and "historique et contexte" in html
    assert "relancer le script ne le rend pas neuf" in html
    assert "script_projet.py" in html and "mecaml.cli rejouer" in html


def test_le_rapport_sans_cible_lit_le_modele_retenu(sans_cible: Cas) -> None:
    html = _html(sans_cible)
    assert "Lecture du modèle sans cible" in html
    assert "Un projet sans cible n'a pas de jeu de test" in html and 'id="test"' not in html
    assert "Effectifs" in html and "Un regroupement existe toujours" in html
    assert lire(html).balises["svg"] >= 2


def test_le_rapport_d_une_classification_groupee_dit_que_la_base_de_test_n_est_pas_du_depot(
    pannes: Cas,
) -> None:
    html = _html(pannes)
    assert (
        "unités entières, classes réparties" in html
        and "unites_stratifiees" not in html
        and "ne s'écrit pas dans la forme de base de test" in html
    )
    assert "Base de test déclarée" not in html


def test_le_rapport_ne_contient_aucune_ligne_du_jeu_hors_les_anomalies(regression: Cas) -> None:
    html = _html(regression)
    valeurs_de_la_cible = {
        f"{v:.4f}".replace(".", ",") for v in regression.projet.table.donnees["y"].head(20)
    }
    assert not any(f">{v}<" in html for v in valeurs_de_la_cible)


# ============================================================
# EXPORT : FICHIERS, EMPREINTES, GARDE-FOUS
# ============================================================
def test_l_export_ecrit_cinq_fichiers_dans_le_dossier_choisi_et_rien_ailleurs(
    regression: Cas, tmp_path: Path
) -> None:
    dossier = tmp_path / "sortie"
    dossier.mkdir()
    avant = sorted(p.name for p in tmp_path.iterdir())
    export = report.exporter(regression.session, dossier, chargement=regression.projet.chargement, **OPTIONS)
    assert sorted(p.name for p in dossier.iterdir()) == sorted(report.NOMS.values())
    assert sorted(p.name for p in tmp_path.iterdir()) == [*avant]  # rien à côté du dossier
    assert export.dossier == dossier.resolve() and set(export.fichiers) == set(report.FORMATS)
    assert export.liste() == tuple(p.name for p in export.fichiers.values())
    for nom, empreinte in export.empreintes.items():
        assert hashlib.sha256((dossier / nom).read_bytes()).hexdigest() == empreinte
    assert all(c.isascii() for c in "".join(export.liste()))


def test_l_export_rejoue_les_memes_octets_pour_le_texte(regression: Cas, tmp_path: Path) -> None:
    for nom in ("a", "b"):
        (tmp_path / nom).mkdir()
        report.exporter(
            regression.session, tmp_path / nom, chargement=regression.projet.chargement, **OPTIONS
        )
    for fichier in ("rapport.html", "script_projet.py", "projet.json"):
        assert (tmp_path / "a" / fichier).read_bytes() == (tmp_path / "b" / fichier).read_bytes(), fichier


def test_l_export_peut_se_limiter_a_certains_fichiers(regression: Cas, tmp_path: Path) -> None:
    export = report.exporter(regression.session, tmp_path, formats=("projet", "script"), **OPTIONS)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["projet.json", "script_projet.py"]
    assert set(export.fichiers) == {"projet", "script"}
    with pytest.raises(codegen.ErreurExport, match="Format\\(s\\) inconnu\\(s\\)"):
        report.exporter(regression.session, tmp_path, formats=("pdf",), **OPTIONS)


def test_l_export_exige_un_dossier_existant_et_n_en_cree_pas(regression: Cas, tmp_path: Path) -> None:
    with pytest.raises(ErreurDonnees, match="n'existe pas"):
        report.exporter(regression.session, tmp_path / "absent", **OPTIONS)
    assert not (tmp_path / "absent").exists()


def test_un_export_qui_echoue_ne_laisse_aucun_fichier(
    regression: Cas, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def echec(*_: Any, **__: Any) -> None:
        raise OSError("disque plein")

    monkeypatch.setattr(report.joblib, "dump", echec)
    with pytest.raises(OSError, match="disque plein"):
        report.exporter(regression.session, tmp_path, **OPTIONS)
    assert list(tmp_path.iterdir()) == []


def test_l_export_se_fait_depuis_S8_ou_S7_sans_cible_et_pas_avant(
    regression: Cas, sans_cible: Cas, tmp_path: Path
) -> None:
    projet = _mener(
        tmp_path / "s7", jeux.jeu_regression(n=120), lambda t: jeux.spec_regression(modeles=(RIDGE,))
    )
    en_s7 = aides.mener_la_session(projet, jusqu_au_test=False)
    assert en_s7.etat is Etat.DIAGNOSTIC
    with pytest.raises(codegen.ErreurExport, match="évaluer le modèle choisi sur le test"):
        report.exporter(en_s7, tmp_path, **OPTIONS)
    assert sans_cible.session.etat is Etat.DIAGNOSTIC
    assert report.exporter(sans_cible.session, tmp_path, **OPTIONS).dossier == tmp_path.resolve()
    with pytest.raises(codegen.ErreurExport, match="Rien à exporter"):
        report.exporter(Session(), tmp_path, **OPTIONS)
    assert list(tmp_path.glob("*.xlsx")) and not list(tmp_path.glob("*.png"))


def test_l_export_refuse_une_alerte_non_acquittee_ou_un_constat_bloquant(
    regression: Cas, tmp_path: Path
) -> None:
    session = regression.session
    alerte = Constat(Niveau.ALERTE, "ESSAI.alerte", "Une alerte.", "L'acquitter.")
    bloquant = Constat(Niveau.BLOQUANT, "ESSAI.bloquant", "Un blocage.", "Le lever.")
    avec_alerte = dataclasses.replace(session, constats=(*session.constats, alerte))
    with pytest.raises(codegen.ErreurExport, match=r"ESSAI\.alerte"):
        report.exporter(avec_alerte, tmp_path, **OPTIONS)
    avec_blocage = dataclasses.replace(session, constats=(*session.constats, bloquant))
    with pytest.raises(codegen.ErreurExport, match=r"ESSAI\.bloquant"):
        report.exporter(avec_blocage, tmp_path, **OPTIONS)
    assert list(tmp_path.iterdir()) == []


def test_l_export_tient_la_ceinture_sur_la_cible_passee(regression: Cas, tmp_path: Path) -> None:
    session = regression.session
    non_declaree = session.spec.avec(modeles=(*session.spec.modeles, CT("reference.persistance")))
    falsifiee = dataclasses.replace(session, spec=non_declaree)  # une interface qui contournerait les règles
    with pytest.raises(codegen.ErreurExport, match="cible_connue=True"):
        report.exporter(falsifiee, tmp_path, **OPTIONS)
    with pytest.raises(codegen.ErreurExport, match="cible_connue=True"):
        report.rapport_html(falsifiee, **OPTIONS)
    assert list(tmp_path.iterdir()) == []


def test_l_export_refuse_un_fichier_different_du_fichier_publie(regression: Cas, tmp_path: Path) -> None:
    chargement = regression.projet.chargement
    export = report.exporter(
        regression.session,
        tmp_path,
        chargement=chargement,
        sha256_publie=chargement.sha256.upper(),
        **OPTIONS,
    )
    assert "identique à l'empreinte publiée" in (tmp_path / "rapport.html").read_text(encoding="utf-8")
    assert export.fichiers
    autre = tmp_path / "autre"
    autre.mkdir()
    with pytest.raises(codegen.ErreurExport, match="n'est pas le fichier publié"):
        report.exporter(regression.session, autre, chargement=chargement, sha256_publie="0" * 64, **OPTIONS)
    with pytest.raises(codegen.ErreurExport, match="transmettre le `Chargement`"):
        report.exporter(regression.session, autre, sha256_publie="0" * 64, **OPTIONS)
    assert list(autre.iterdir()) == []


def test_l_export_clos_la_session_en_S9(regression: Cas, tmp_path: Path) -> None:
    session, export = report.exporter_et_clore(regression.session, tmp_path, **OPTIONS)
    assert session.etat is Etat.EXPORTE and session.export is export
    nouvel, _ = report.exporter_et_clore(session, tmp_path, **OPTIONS)  # réexporter remplace les fichiers
    assert nouvel.etat is Etat.EXPORTE
    revenu = session.appliquer(Revenir(Etat.VIDE))
    assert revenu.etat is Etat.VIDE


def test_les_options_inconnues_sont_refusees(regression: Cas) -> None:
    with pytest.raises(TypeError, match="Option\\(s\\) inconnue\\(s\\)"):
        report.rapport_html(regression.session, couleur="bleu")


# ============================================================
# CLASSEUR, MODÈLE, PROJET
# ============================================================
def test_le_classeur_reprend_les_chiffres_du_rapport(regression: Cas, tmp_path: Path) -> None:
    report.exporter(
        regression.session, tmp_path, formats=("xlsx",), chargement=regression.projet.chargement, **OPTIONS
    )
    classeur = load_workbook(tmp_path / "rapport.xlsx")
    assert classeur.sheetnames == [
        "Résumé",
        "Constats",
        "Modèles",
        "Plis",
        "Test",
        "Importances",
        "Paramètres",
    ]
    resume = {ligne[0].value: ligne[1].value for ligne in classeur["Résumé"].iter_rows(min_row=2)}
    test = regression.session.resultat_test
    retenu = regression.session.decision.modele_id
    assert resume["Modèle retenu"] == libelles.technique(retenu) and resume["Identifiant du modèle"] == retenu
    assert resume["Graine"] == 42
    assert resume["Score de test"] == pytest.approx(test.valeur()) and isinstance(
        resume["Score de test"], float
    )
    modeles = [[c.value for c in ligne] for ligne in classeur["Modèles"].iter_rows(min_row=2)]
    assert len(modeles) == len(regression.session.resultats.modeles) and all(
        isinstance(m[2], float) for m in modeles
    )
    plis = list(classeur["Plis"].iter_rows(min_row=2, values_only=True))
    assert len(plis) == sum(
        len(v) for m in regression.session.resultats.modeles for v in m.scores_plis.values()
    )
    scores_test = dict(classeur["Test"].iter_rows(min_row=2, values_only=True))
    assert scores_test[libelles.metrique("rmse")] == pytest.approx(test.scores["rmse"])
    assert classeur["Résumé"]["A1"].font.bold and classeur["Résumé"].freeze_panes == "A2"
    assert [c.value for c in classeur["Importances"][1]] == [
        "Variable",
        "Baisse moyenne du score",
        "Écart type",
    ]


def test_le_classeur_ne_fait_jamais_d_un_texte_une_formule(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 120
    jeu = pd.DataFrame({"=1+1": rng.normal(size=n), "b": rng.normal(size=n)})
    jeu["y"] = 0.5 * jeu["=1+1"] + rng.normal(0, 1.0, n)  # une relation réelle, sans fuite par la cible
    from mecaml.core.types import Probleme, Role

    def spec(_: Any) -> SpecProjet:
        return SpecProjet(
            probleme=Probleme.REGRESSION,
            cible="y",
            roles=(("=1+1", Role.NUMERIQUE), ("b", Role.NUMERIQUE), ("y", Role.CIBLE)),
            metrique="rmse",
            pretraitement=(CT("transformation.standard"),),
            modeles=(RIDGE,),
        )

    cas = _cas(tmp_path / "jeu", jeu, spec, importances=False)
    (tmp_path / "s").mkdir()
    report.exporter(cas.session, tmp_path / "s", formats=("xlsx",), **OPTIONS)
    feuille = load_workbook(tmp_path / "s" / "rapport.xlsx")["Paramètres"]
    cellule = feuille["B2"]
    assert cellule.value.startswith("=1+1=numerique") and cellule.data_type == "s"  # du texte, non évalué


def test_une_valeur_indefinie_laisse_la_cellule_vide() -> None:
    from openpyxl import Workbook

    feuille = Workbook().active
    report._poser(feuille, 1, 1, float("nan"))
    report._poser(feuille, 1, 2, np.float64(2.5))
    report._poser(feuille, 1, 3, np.int64(3))
    report._poser(feuille, 1, 4, "=SOMME(A1:A2)")
    assert feuille["A1"].value is None and feuille["B1"].value == 2.5 and feuille["C1"].value == 3
    assert feuille["D1"].data_type == "s"


def test_le_modele_exporte_predit_comme_le_modele_evalue(regression: Cas, serie: Cas, tmp_path: Path) -> None:
    for cas in (regression, serie):
        dossier = tmp_path / ("reg" if cas is regression else "ser")
        dossier.mkdir()
        report.exporter(cas.session, dossier, formats=("joblib",), **OPTIONS)
        relu = joblib.load(
            dossier / "modele.joblib"
        )  # le test relit ses propres fichiers ; l'application, jamais
        evalue = cas.session.resultat_test.modele.pipeline
        lignes = cas.projet.table.donnees.iloc[:40]
        x = lignes[list(evalue.feature_names_in_)]
        assert np.array_equal(relu.predict(x), evalue.predict(x))
    ct = relu[0].transformers_
    historiques = [t[1][1].historique_ for t in ct if t[0].startswith("chronologique")]
    assert historiques and len(historiques[0]) > 0  # le modèle embarque des lignes du jeu, le rapport le dit


def test_le_projet_exporte_se_relit_en_specification_identique(
    regression: Cas, pannes: Cas, tmp_path: Path
) -> None:
    for cas, base_test in ((regression, True), (pannes, False)):
        dossier = tmp_path / ("a" if base_test else "b")
        dossier.mkdir()
        report.exporter(
            cas.session, dossier, formats=("projet",), chargement=cas.projet.chargement, **OPTIONS
        )
        projet = json.loads((dossier / "projet.json").read_text(encoding="utf-8"))
        assert projet["format"] == report.FORMAT_DU_PROJET == 1
        assert SpecProjet.depuis_dict(projet["projet"]) == cas.session.spec
        assert (
            projet["donnees"]["fichier"] == "jeu.csv"
            and projet["donnees"]["sha256"] == cas.projet.chargement.sha256
        )
        attendus = projet["attendus"]
        assert attendus["empreinte_donnees"] == cas.projet.table.empreinte
        assert attendus["empreinte_test"] == cas.projet.decoupage.empreinte_sha256
        assert attendus["empreinte_plis"] == codegen.empreinte_plis(cas.session.resultats.plis)
        assert attendus["test"]["empreinte_predictions"] == cas.session.resultat_test.empreinte_predictions
        assert ("base_test" in projet["decoupage"]) is base_test
        if base_test:
            assert split.Decoupage.depuis_base_test(projet["decoupage"]["base_test"]) == cas.projet.decoupage
        assert projet["decision"]["modele"] == cas.session.decision.modele_id


def test_le_projet_ecrit_les_valeurs_indefinies_en_null_et_les_alertes_acquittees(
    serie: Cas, tmp_path: Path
) -> None:
    report.exporter(serie.session, tmp_path, formats=("projet",), **OPTIONS)
    brut = (tmp_path / "projet.json").read_text(encoding="utf-8")
    assert "NaN" not in brut and "Infinity" not in brut
    projet = json.loads(brut)
    assert projet["acquittements"] == sorted(serie.session.acquittes) == ["I7.cible_manquante"]
    assert projet["environnement"] == ENV


def test_le_sans_cible_exporte_sans_test(sans_cible: Cas, tmp_path: Path) -> None:
    report.exporter(sans_cible.session, tmp_path, chargement=sans_cible.projet.chargement, **OPTIONS)
    projet = json.loads((tmp_path / "projet.json").read_text(encoding="utf-8"))
    assert (
        projet["decoupage"] is None
        and "test" not in projet["attendus"]
        and "empreinte_test" not in projet["attendus"]
    )
    classeur = load_workbook(tmp_path / "rapport.xlsx")
    assert "Test" not in classeur.sheetnames
    assert joblib.load(tmp_path / "modele.joblib")[-1].__class__.__name__ == "KMeans"


def test_un_resultat_de_test_etranger_est_refuse(regression: Cas, tmp_path: Path) -> None:
    falsifiee = dataclasses.replace(regression.session, resultat_test={"rmse": 1.0})
    with pytest.raises(codegen.ErreurExport, match=r"evaluate\.preparer_test"):
        report.exporter(falsifiee, tmp_path, **OPTIONS)
    sans_resultat = dataclasses.replace(regression.session, resultats=object())
    with pytest.raises(codegen.ErreurExport, match="comparaison des modèles"):
        report.exporter(sans_resultat, tmp_path, **OPTIONS)
    assert isinstance(regression.session.resultat_test, evaluate.ResultatTest)
