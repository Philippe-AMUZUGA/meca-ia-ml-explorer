"""Textes de l'interface : clés employées, clés orphelines, typographie, titres des techniques."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from mecaml.core import catalog, metrics
from mecaml.core.report import FORMATS
from mecaml.core.types import Niveau, Probleme, Role
from mecaml.ui import textes
from mecaml.ui.textes import cles, existe, pedagogie, t, titre_technique

UI = Path(textes.__file__).resolve().parents[1]
ECRANS = ("accueil", *(f"s{k}" for k in range(1, 10)))
VUES = (
    "regression", "confusion", "roc_pr", "seuil", "resume", "coefficients", "impurete", "arbre",
    "importances", "courbe", "composantes", "poids", "plan", "profils", "effectifs", "anomalies", "anormales",
)  # fmt: skip
PREFIXE_AIDE = "{prefixe}.{rubrique}"  # les trois repères de chaque écran, vérifiés à part


def _sources() -> list[Path]:
    return sorted(p for p in UI.rglob("*.py"))


SECTIONS = ("ecran", "aide", "rail", "commun", "pied", "niveau", "role", "famille", "probleme", "cout")
SECTIONS += ("constats", "choix", "app")
FORME_DE_CLE = re.compile(rf"^({'|'.join(SECTIONS)})\.[a-z0-9_.]+$")


def _cles_employees() -> tuple[set[str], set[str]]:
    """(clés écrites en toutes lettres, motifs de clés composées d'un champ `{...}`)."""
    litterales: set[str] = set()
    motifs: set[str] = set()
    for source in _sources():
        arbre = ast.parse(source.read_text(encoding="utf-8"))
        morceaux = {id(v) for n in ast.walk(arbre) if isinstance(n, ast.JoinedStr) for v in n.values}
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Constant) and isinstance(noeud.value, str) and id(noeud) not in morceaux:
                if FORME_DE_CLE.match(noeud.value):
                    litterales.add(noeud.value)
            elif isinstance(noeud, ast.Call) and isinstance(noeud.func, ast.Name) and noeud.func.id == "t":
                premier = noeud.args[0] if noeud.args else None
                if isinstance(premier, ast.JoinedStr):
                    motifs.add(ast.unparse(premier)[2:-1].replace("{{", "{"))
    return litterales, motifs


def _regex(motif: str) -> re.Pattern[str]:
    return re.compile("^" + re.sub(r"\\\{[^}]*\\\}", "[a-z0-9_]+", re.escape(motif)) + "$")


def test_le_fichier_de_textes_se_lit_et_n_est_pas_vide() -> None:
    assert len(cles()) > 300
    assert all(isinstance(t(c), str) and t(c) for c in cles() if "{" not in t(c))


def test_chaque_cle_ecrite_en_toutes_lettres_a_son_texte() -> None:
    litterales, _ = _cles_employees()
    assert litterales
    assert sorted(c for c in litterales if not existe(c)) == []


def test_chaque_cle_composee_a_au_moins_un_texte() -> None:
    _, motifs = _cles_employees()
    assert motifs
    for motif in motifs - {PREFIXE_AIDE}:
        assert any(_regex(motif).match(c) for c in cles()), f"aucun texte pour {motif}"


def test_chaque_ecran_a_pour_titre_le_nom_de_son_onglet_sans_bandeau_d_aide() -> None:
    """Un titre, pas de bandeau : le titre d'un écran est la clé de son onglet."""
    assert [e for e in ECRANS if not existe(f"rail.{e}")] == []
    assert [
        c
        for c in cles()
        if c.startswith("aide.")
        or ".aide." in c
        or (c.endswith(".titre") and c.count(".") == 2 and c.startswith("ecran.s"))
    ] == []


def test_les_enumerations_du_coeur_sont_toutes_nommees() -> None:
    attendues = [f"role.{r.value}" for r in Role] + [f"niveau.{n.value}" for n in Niveau]
    attendues += [f"probleme.{p.value}" for p in Probleme] + [
        f"cout.{c}" for c in ("leger", "moyen", "lourd")
    ]
    attendues += [f"ecran.s9.format.{f}" for f in FORMATS] + [f"ecran.s7.vue.{v}" for v in VUES]
    attendues += [
        f"ecran.s7.onglet.{o}" for o in ("diagnostic", "seuil", "lecture", "importances", "courbe", "garde")
    ]
    attendues += [f"ecran.s7.calculer.{o}" for o in ("lecture", "importances", "courbe")]
    attendues += [f"rail.s{k}" for k in range(1, 10)] + ["rail.accueil"]
    assert sorted(c for c in attendues if not existe(c)) == []


def test_aucune_cle_orpheline() -> None:
    """Toute clé du fichier est employée quelque part (les titres de techniques sont lus par identifiant)."""
    litterales, motifs = _cles_employees()
    regexes = [_regex(m) for m in motifs]
    orphelines = [
        c
        for c in sorted(cles())
        if not c.startswith("technique.") and c not in litterales and not any(r.match(c) for r in regexes)
    ]
    assert orphelines == []


def test_chaque_technique_du_catalogue_a_son_titre() -> None:
    sans_titre = [e.id for e in catalog.entrees() if not existe(f"technique.{e.id}")]
    assert sans_titre == []
    assert len({titre_technique(e.id) for e in catalog.entrees()}) == len(catalog.entrees())


def test_aucun_titre_pour_une_technique_inconnue() -> None:
    inconnues = sorted(c.removeprefix("technique.") for c in cles() if c.startswith("technique."))
    registre = {e.id for e in catalog.entrees()}
    assert [i for i in inconnues if i not in registre] == []
    assert titre_technique("famille.inconnue") == "famille.inconnue"


def test_chaque_metrique_a_un_libelle_du_coeur() -> None:
    assert all(m.libelle for m in metrics.METRIQUES)


def test_la_cle_absente_leve_une_erreur_explicite() -> None:
    with pytest.raises(KeyError, match=r"interface_fr\.toml"):
        t("ecran.inexistant.titre")


def test_la_typographie_francaise_est_posee_a_la_lecture() -> None:
    """Espace insécable avant « : ; ! ? » et dans les guillemets français ; aucun tiret interdit."""
    for cle in cles():
        texte = t(cle)
        assert not re.search(r" [:;!?»]", texte), f"{cle} : espace ordinaire avant une ponctuation haute"
        assert "« " not in texte, cle
        assert not re.search(f"[{chr(0x2013)}{chr(0x2014)}]|-{{2}}", texte), f"{cle} : tiret interdit"
    assert " " in t("ecran.s1.forme", nom="a", lignes="1", colonnes=2)


def test_les_champs_remplis_ne_laissent_aucune_accolade() -> None:
    assert "{" not in t("ecran.s4.effectifs", app="10", test="2")
    with pytest.raises(KeyError):
        t("ecran.s4.effectifs", app="10")


def test_les_textes_pedagogiques_sont_facultatifs() -> None:
    """Sans fichier `textes/etapes_fr.toml`, l'interface ne montre que les titres ; avec lui, ses textes."""
    assert pedagogie("regression.ridge") is None or set(pedagogie("regression.ridge")) <= {
        "titre", "quoi", "pourquoi", "regarder"
    }  # fmt: skip
