"""
MécaML : sorties du projet (S9), écrites dans le dossier choisi et nulle part ailleurs.

Cinq fichiers, aux noms fixes et sans accent :

- `rapport.html` : un seul fichier, ouvert sans réseau (style et figures en ligne, aucune ressource
  extérieure) ; - `rapport.xlsx` : les mêmes chiffres en classeur, pour les reprendre dans un tableur ; -
  `modele.joblib` : le pipeline ajusté sur tout l'apprentissage. Il embarque des lignes du jeu quand le
  projet construit des variables temporelles (`historique_` et `contexte`), et `joblib.load` exécute du code
  : ne charger que ses propres fichiers. L'application ne charge jamais de modèle sérialisé ; -
  `script_projet.py` : le script scikit-learn équivalent (voir `codegen`) ; - `projet.json` : le projet
  rejouable par `mecaml.cli` (spécification, découpage, décision, scores attendus).

Garde-fous de l'export : la session doit être en S8 (S7 sans cible), sans constat bloquant ni alerte à
acquitter ; la cible passée doit être déclarée connue et ne pas décompter (ceinture de
`codegen.verifier_ceinture`) ; en mode challenge, l'empreinte du fichier lu doit être celle du fichier
publié. Toute écriture passe par `guards.chemin_dans_dossier` ; un export qui échoue ne laisse aucun de ses
fichiers.

Le rapport dit ce qui rend un score fragile : test déjà ouvert (score optimiste), recherche
d'hyperparamètres non imbriquée, classe positive proposée mais non confirmée, plis sur lesquels une métrique
n'est pas définie, lignes sans cible retirées, cible du pas précédent déclarée connue.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import io
import json
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from mecaml import __version__
from mecaml.core import (
    cible_passee,
    codegen,
    evaluate,
    explain,
    libelles,
    livre,
    metrics,
    renvois,
    split,
    train,
    unsupervised,
)
from mecaml.core import (
    decision as decision_du_projet,
)
from mecaml.core.codegen import ErreurExport
from mecaml.core.guards import chemin_dans_dossier, decimale, nombre_lisible
from mecaml.core.session import Etat, Session
from mecaml.core.types import Niveau, Probleme, SpecProjet, Table

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from mecaml.core.io import Chargement

__all__ = [
    "FORMATS",
    "NOMS",
    "ErreurExport",
    "Export",
    "classeur_xlsx",
    "exporter",
    "exporter_et_clore",
    "projet_json",
    "rapport_html",
]

FORMATS = ("html", "xlsx", "joblib", "script", "projet")
NOMS = {
    "html": "rapport.html",
    "xlsx": "rapport.xlsx",
    "joblib": "modele.joblib",
    "script": codegen.NOM_DU_SCRIPT,
    "projet": "projet.json",
}
FORMAT_DU_PROJET = 1
NIVEAUX = {Niveau.BLOQUANT: "Bloquant", Niveau.ALERTE: "Alerte", Niveau.INFO: "Information"}
IMPORTANCES_MAX = 15
LIBELLES_EXPLICATION = {
    "variable": "Variable",
    "moyenne": "Baisse moyenne du score",
    "ecart_type": "Écart type",
    "plis_valides": "Plis valides",
    "coefficient": "Coefficient",
    "importance": "Importance",
}
LIBELLES_SANS_CIBLE = {
    "groupe": "Groupe",
    "effectif": "Effectif",
    "position": "Ligne",
    "score_anomalie": "Score d'anomalie",
    "variable": "Variable",
}


# ============================================================
# CE QU'ON REÇOIT, CE QU'ON REND
# ============================================================
@dataclass(frozen=True)
class Export:
    """Fichiers écrits : chemins, empreintes SHA-256, avertissements. C'est ce que reçoit `Exporter`."""

    dossier: Path
    fichiers: Mapping[str, Path]
    empreintes: Mapping[str, str]
    avertissements: tuple[str, ...] = ()
    elements: tuple[str, ...] = field(default=())

    def liste(self) -> tuple[str, ...]:
        """Noms des fichiers écrits."""
        return tuple(p.name for p in self.fichiers.values())


@dataclass
class _Projet:
    """Tout ce que les sorties lisent, rassemblé et vérifié une fois."""

    session: Session
    spec: SpecProjet
    table: Table
    resultat: train.Resultat
    decision: Any
    test: evaluate.ResultatTest | None
    decoupage: split.Decoupage | None
    chargement: Chargement | None
    sha256_publie: str | None
    recherches: Sequence[train.ResultatRecherche]
    maintenant: datetime
    env: Mapping[str, str]
    importances: pd.DataFrame | None = None
    modele: train.ModeleAjuste | None = None
    notes: list[str] = field(default_factory=list)
    table_test: Table | None = None  # jeu de test fourni à part, sinon None

    @property
    def avec_cible(self) -> bool:
        return self.spec.probleme.avec_cible


def _rassembler(
    session: Session,
    *,
    decoupage: split.Decoupage | None,
    chargement: Chargement | None,
    sha256_publie: str | None,
    recherches: Sequence[train.ResultatRecherche],
    maintenant: datetime | None,
    env: Mapping[str, str] | None,
    table_test: Table | None = None,
) -> _Projet:
    """Vérifie l'état de la session et les garde-fous, puis rassemble ce qui s'écrit."""
    spec, table = session.spec, session.table
    if spec is None or table is None:
        raise ErreurExport("Rien à exporter : charger les données et mener le projet jusqu'à S8 d'abord.")
    attendus = (Etat.DIAGNOSTIC, Etat.EXPORTE) if not spec.probleme.avec_cible else (Etat.TESTE, Etat.EXPORTE)
    if session.etat not in attendus or (session.etat is Etat.DIAGNOSTIC and spec.probleme.avec_cible):
        etape = (
            "décider du modèle en S7"
            if not spec.probleme.avec_cible
            else "évaluer le modèle choisi sur le test (S8)"
        )
        raise ErreurExport(f"Export impossible en S{int(session.etat)} : {etape} d'abord.")
    if not isinstance(session.resultats, train.Resultat) or session.decision is None:
        raise ErreurExport("Export impossible : la comparaison des modèles (S6) ou la décision (S7) manque.")
    if session.bloquants() or session.alertes_en_attente():
        codes = [c.code for c in (*session.bloquants(), *session.alertes_en_attente())][:3]
        raise ErreurExport(
            f"Export impossible : constats à lever ou alertes à acquitter d'abord ({', '.join(codes)})."
        )
    codegen.verifier_ceinture(spec, table, session.coffre)
    if sha256_publie is not None:
        _comparer_au_fichier_publie(chargement, sha256_publie)
    test = session.resultat_test
    if spec.probleme.avec_cible and not isinstance(test, evaluate.ResultatTest):
        raise ErreurExport(
            "Export impossible : le résultat du test n'est pas celui de `evaluate.preparer_test`."
        )
    if spec.probleme.avec_cible and decoupage is None:
        decoupage = codegen.retrouver_decoupage(table, spec, session.coffre)
    projet = _Projet(
        session,
        spec,
        table,
        session.resultats,
        session.decision,
        test,
        decoupage,
        chargement,
        sha256_publie,
        tuple(recherches),
        maintenant or datetime.now(),
        env or codegen.environnement(),
    )
    projet.table_test = table_test
    return projet


def _comparer_au_fichier_publie(chargement: Chargement | None, publie: str) -> None:
    """Mode challenge : le fichier lu doit être le fichier publié, octet pour octet."""
    if chargement is None:
        raise ErreurExport("Pour comparer au fichier publié, transmettre le `Chargement` du fichier lu.")
    if chargement.sha256.lower() != publie.lower():
        raise ErreurExport(
            f"Le fichier lu ({chargement.sha256[:12]}...) n'est pas le fichier publié ({publie[:12]}...) : "
            "recharger le fichier du dépôt d'exercices tel quel, sans l'ouvrir ni l'enregistrer dans "
            "un tableur."
        )


# ============================================================
# FORMATS DE NOMBRES ET DE TEXTE
# ============================================================
def _e(valeur: Any) -> str:
    """Texte échappé pour le contenu d'un élément (jamais pour un attribut : aucun n'est calculé)."""
    return html.escape(str(valeur), quote=False)


def _n(valeur: Any, chiffres: int = 4) -> str:
    """Nombre à la française ; un nombre indéfini se dit « indéfini »."""
    if valeur is None:
        return "-"
    if isinstance(valeur, (bool, np.bool_)):
        return "oui" if valeur else "non"
    if isinstance(valeur, (int, np.integer)):
        return nombre_lisible(int(valeur)).replace(" ", " ")
    if isinstance(valeur, (float, np.floating)):
        return "indéfini" if not math.isfinite(valeur) else decimale(float(valeur), chiffres)
    return str(valeur)


def _json_sur(valeur: Any) -> Any:
    """Valeur écrite en JSON : types numpy convertis, nan écrit null."""
    if isinstance(valeur, Mapping):
        return {str(k): _json_sur(v) for k, v in valeur.items()}
    if isinstance(valeur, (list, tuple, set, frozenset)):
        return [
            _json_sur(v)
            for v in (sorted(valeur, key=str) if isinstance(valeur, (set, frozenset)) else valeur)
        ]
    if isinstance(valeur, np.ndarray):
        return [_json_sur(v) for v in valeur.tolist()]
    if isinstance(valeur, (np.floating, float)):
        return None if not math.isfinite(valeur) else float(valeur)
    if isinstance(valeur, np.integer):
        return int(valeur)
    if isinstance(valeur, np.bool_):
        return bool(valeur)
    return valeur


def _sha256(donnees: bytes) -> str:
    return hashlib.sha256(donnees).hexdigest()


# ============================================================
# FIGURES EN LIGNE
# ============================================================
def _svg(figure: Figure) -> str:
    """Figure en SVG (texte) sans date ni identifiant aléatoire : deux exports identiques le restent."""
    import matplotlib

    tampon = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": "mecaml", "svg.fonttype": "path"}):
        figure.savefig(tampon, format="svg", metadata={"Date": None})
    texte = tampon.getvalue()
    return texte[texte.index("<svg") :]


def _figure(figure: Figure, legende: str) -> str:
    return f"<figure>{_svg(figure)}<figcaption>{_e(legende)}</figcaption></figure>"


# ============================================================
# BRIQUES HTML
# ============================================================
COULEURS = {
    "texte": "#1b1b1b",
    "fond": "#fff",
    "filet": "#d0d4d9",
    "accent": "#1f4e79",
    "doux": "#f4f6f8",
    "alerte": "#9c5700",
    "bloquant": "#a4262c",
}
CSS = Template("""
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 system-ui,"Segoe UI",Roboto,Arial,sans-serif;color:$texte;
background:$fond}
header,main,footer{max-width:1000px;margin:0 auto;padding:0 20px}
header{padding-top:28px;padding-bottom:14px;border-bottom:3px solid $accent}
h1{font-size:1.7rem;margin:0 0 4px;color:$accent}
h2{font-size:1.25rem;margin:2rem 0 .6rem;padding-bottom:.25rem;border-bottom:1px solid $filet;
color:$accent}
h3{font-size:1.02rem;margin:1.2rem 0 .4rem}
p,li{max-width:80ch}
nav ol{columns:2;margin:.8rem 0;padding-left:1.2rem}
a{color:$accent}
table{border-collapse:collapse;margin:.6rem 0 1rem;font-size:.92rem}
th,td{border:1px solid $filet;padding:4px 9px;text-align:left;vertical-align:top}
th{background:$doux}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
dl.faits{display:grid;grid-template-columns:max-content 1fr;gap:3px 18px;margin:.6rem 0}
dl.faits dt{font-weight:600}
dl.faits dd{margin:0}
figure{margin:1rem 0;text-align:center}
figure svg{max-width:100%;height:auto}
figcaption{font-size:.85rem;color:#555}
.note{border-left:4px solid $accent;background:$doux;padding:.5rem .9rem;margin:.8rem 0}
.alerte{border-left-color:$alerte}
.bloquant{border-left-color:$bloquant}
.etiquette{display:inline-block;padding:0 7px;border-radius:9px;font-size:.78rem;
border:1px solid $filet}
.etiquette.alerte{color:$alerte;border-color:$alerte}
.etiquette.bloquant{color:$bloquant;border-color:$bloquant}
code,pre{font-family:Consolas,"Cascadia Mono",monospace;font-size:.88rem}
pre{background:$doux;padding:.6rem .8rem;overflow-x:auto;border:1px solid $filet}
footer{padding-top:14px;padding-bottom:30px;color:#555;font-size:.85rem;
border-top:1px solid $filet;margin-top:2rem}
@media print{nav{display:none}h2{break-after:avoid}figure,table{break-inside:avoid}}
""").substitute(COULEURS)

PAGE = Template(
    """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$titre</title>
<style>$style</style>
</head>
<body>
<header>
<h1>$titre</h1>
<p>$sous_titre</p>
</header>
<main>
<nav aria-label="Sommaire"><ol>
$sommaire
</ol></nav>
$corps
</main>
<footer>$pied</footer>
</body>
</html>
"""
)


def _table(
    entetes: Sequence[str], lignes: Iterable[Sequence[Any]], *, numeriques: Collection[int] = ()
) -> str:
    tete = "".join(
        f"<th{' class=num' if k in numeriques else ''}>{_e(h)}</th>" for k, h in enumerate(entetes)
    )
    corps = []
    for ligne in lignes:
        cellules = "".join(
            f"<td{' class=num' if k in numeriques else ''}>{c if isinstance(c, _Brut) else _e(c)}</td>"
            for k, c in enumerate(ligne)
        )
        corps.append(f"<tr>{cellules}</tr>")
    return f"<table><thead><tr>{tete}</tr></thead><tbody>{''.join(corps)}</tbody></table>"


class _Brut(str):
    """Texte HTML déjà sûr, que `_table` n'échappe pas une seconde fois."""


def _faits(paires: Iterable[tuple[str, Any]]) -> str:
    lignes = "".join(f"<dt>{_e(k)}</dt><dd>{v if isinstance(v, _Brut) else _e(v)}</dd>" for k, v in paires)
    return f'<dl class="faits">{lignes}</dl>'


def _note(texte: str, niveau: str = "") -> str:
    return f'<p class="note {niveau}">{_e(texte)}</p>'


def _section(identifiant: str, titre: str, contenu: str) -> tuple[str, str, str]:
    return identifiant, titre, f'<section id="{identifiant}"><h2>{_e(titre)}</h2>\n{contenu}\n</section>'


def _frame(
    df: pd.DataFrame,
    formats: Mapping[str, int] | None = None,
    maximum: int | None = None,
    libelles: Mapping[str, str] | None = None,
) -> str:
    """Table pandas en HTML : nombres à la française, colonnes numériques à droite.

    `libelles` : en-têtes en français pour les colonnes qu'il nomme (les autres gardent leur nom).
    """
    formats = formats or {}
    libelles = libelles or {}
    lignes = df.head(maximum) if maximum else df
    numeriques = [k for k, c in enumerate(df.columns) if pd.api.types.is_numeric_dtype(df[c])]
    rendu = [
        [
            _n(v, formats.get(str(c), 4)) if isinstance(v, (float, np.floating, int, np.integer)) else v
            for c, v in zip(df.columns, ligne, strict=True)
        ]
        for ligne in lignes.itertuples(index=False)
    ]
    return _table([libelles.get(str(c), str(c)) for c in df.columns], rendu, numeriques=numeriques)


# ============================================================
# SECTIONS DU RAPPORT
# ============================================================
def _majuscule(texte: str) -> str:
    return texte[:1].upper() + texte[1:]


def _minuscule(texte: str) -> str:
    return texte[:1].lower() + texte[1:]


def _lib_probleme(p: Probleme) -> str:
    return codegen.PROBLEMES_FR[p]


def _section_resume(r: _Projet) -> tuple[str, str, str]:
    s, d = r.spec, r.decision
    faits: list[tuple[str, Any]] = [
        ("Fichier", f"{r.table.nom} ({_n(r.table.n_lignes)} lignes, {r.table.n_colonnes} colonnes)"),
        ("Problème", _lib_probleme(s.probleme) + (f", cible « {s.cible} »" if s.cible else "")),
        ("Métrique du projet", libelles.metrique(s.metrique) if s.metrique else "silhouette"),
        ("Modèle retenu", libelles.technique(d.modele_id)),
        ("Score de validation", f"{_n(d.score_validation)} (moyenne des plis, métrique du projet)"),
    ]
    if r.test is not None:
        optimiste = (
            " : score de test optimiste, le test a déjà été ouvert" if r.session.test_optimiste else ""
        )
        faits.append(("Score de test", f"{_n(r.test.valeur())} sur {_n(r.test.n_lignes)} lignes{optimiste}"))
    if d.seuil is not None:
        faits.append(("Seuil de décision", _n(d.seuil, 3)))
    faits.append(("Graine unique", s.graine))
    contenu = _faits(faits)
    if r.session.test_optimiste:
        contenu += _note(
            "Score de test optimiste : le jeu de test avait déjà été ouvert, ou les mêmes données ont "
            "déjà servi de test dans cette session. Il a pu guider des choix : ne pas le présenter comme "
            "une mesure indépendante.",
            "alerte",
        )
    return _section("resume", "Résumé", contenu)


def _section_donnees(r: _Projet) -> tuple[str, str, str]:
    t, s = r.table, r.spec
    faits: list[tuple[str, Any]] = [
        ("Fichier", t.nom + (f", feuille « {t.feuille} »" if t.feuille else "")),
        ("Dimensions", f"{_n(t.n_lignes)} lignes, {t.n_colonnes} colonnes"),
        ("Empreinte de la table", _Brut(f"<code>{_e(t.empreinte)}</code>")),
    ]
    if r.chargement is not None:
        faits.append(("Empreinte SHA-256 du fichier", _Brut(f"<code>{_e(r.chargement.sha256)}</code>")))
        if r.sha256_publie:
            faits.append(("Fichier publié", "identique à l'empreinte publiée"))
    contenu = _faits(faits)
    if r.chargement is not None and r.chargement.notes:
        contenu += (
            "<h3>Remarques de lecture</h3><ul>"
            + "".join(f"<li>{_e(n)}</li>" for n in r.chargement.notes)
            + "</ul>"
        )
    roles = [(nom, role.value) for nom, role in s.roles]
    contenu += "<h3>Rôle de chaque colonne</h3>" + _table(["Colonne", "Rôle"], roles)
    return _section("donnees", "Données", contenu)


def _section_audit(r: _Projet) -> tuple[str, str, str]:
    constats = r.session.constats
    if not constats:
        return _section("audit", "Audit", _note("Aucun constat."))
    acquittes = r.session.acquittes
    lignes = []
    for c in sorted(constats, key=lambda c: (list(Niveau).index(c.niveau), c.code)):
        etat = "acquittée" if c.code in acquittes else ""
        etiquette = _Brut(
            f'<span class="etiquette {c.niveau.value}">{_e(NIVEAUX[c.niveau])}</span> {_e(etat)}'
        )
        lignes.append(
            [etiquette, _Brut(f"<code>{_e(c.code)}</code>"), c.message, c.action, c.ref_livre or ""]
        )
    contenu = _table(["Niveau", "Code", "Constat", "Action", "Livre"], lignes)
    if acquittes:
        contenu += (
            f"<p>Alertes acquittées par l'utilisateur ({len(acquittes)}) : "
            + ", ".join(f"<code>{_e(code)}</code>" for code in sorted(acquittes))
            + ". Une alerte acquittée reste une réserve sur le résultat : la relire avant de conclure.</p>"
        )
    return _section("audit", "Audit des données", contenu)


def _section_decoupage(r: _Projet) -> tuple[str, str, str]:
    if not r.avec_cible:
        return _section(
            "decoupage", "Découpage", _note("Un projet sans cible n'a pas de jeu de test mis de côté.")
        )
    s, coffre, d = r.spec, r.session.coffre, r.decoupage
    faits: list[tuple[str, Any]] = [
        ("Méthode", _majuscule(codegen.LIBELLES_DECOUPAGE[d.methode])),
        ("Structure des lignes", codegen.STRUCTURES_FR[s.structure]),
        ("Lignes d'apprentissage et de test", f"{_n(coffre.n_apprentissage)} et {_n(coffre.n_test)}"),
    ]
    if r.table_test is not None:
        faits.append(("Fichier de test fourni à part", r.table_test.nom))
    if d.methode is not split.Methode.FICHIER:
        faits.append(
            (
                "Part de test",
                f"visée {_n(100 * s.part_test, 1)} %, obtenue {_n(100 * split.part_obtenue(coffre), 1)} %",
            )
        )
    faits += [
        ("Graine", s.graine),
        ("Empreinte des positions de test", _Brut(f"<code>{_e(d.empreinte_sha256)}</code>")),
        ("Ouvertures du jeu de test", coffre.ouvertures),
    ]
    contenu = _faits(faits)
    if d.conforme_contrat:
        contenu += (
            "<h3>Base de test déclarée (dépôt d'exercices)</h3><pre>"
            + _e(json.dumps(d.vers_base_test(), indent=2, ensure_ascii=False))
            + "</pre>"
        )
    else:
        contenu += _note(
            f"La méthode « {codegen.LIBELLES_DECOUPAGE[d.methode]} » ne s'écrit pas dans la forme de base "
            "de test du dépôt d'exercices : la méthode et la graine ci-dessus suffisent à la refaire avec "
            "l'application ou avec le script."
        )
    if s.validation is not None or r.resultat.validation:
        validation = _minuscule(libelles.technique(r.resultat.validation))
        contenu += f"<p>Validation croisée : {_e(validation)}, {s.n_plis} plis.</p>"
    return _section("decoupage", "Découpage apprentissage et test", contenu)


def _technique(choix: Any) -> list[Any]:
    try:
        reference = renvois.technique(choix.id)
        fiche = livre.technique_du_livre(reference)
        renvoi = f"{fiche.id} : {fiche.libelle} (chapitre {fiche.chapitre}, section {fiche.section})"
    except KeyError:
        renvoi = ""
    parametres = ", ".join(f"{k} = {v}" for k, v in choix.parametres) or "défauts de la technique"
    colonnes = ", ".join(choix.colonnes) or "toutes les colonnes concernées"
    return [libelles.technique(choix.id), parametres, colonnes, renvoi]


def _section_pretraitement(r: _Projet) -> tuple[str, str, str]:
    s = r.spec
    if not s.pretraitement:
        return _section("pretraitement", "Prétraitement", _note("Aucun prétraitement n'a été choisi."))
    contenu = (
        "<p>Chaque étape est ajustée dans les plis de la validation croisée, jamais sur le jeu entier.</p>"
        + _table(["Technique", "Paramètres", "Colonnes", "Livre"], [_technique(c) for c in s.pretraitement])
    )
    declarees = sorted({c.id for c in cible_passee.choix_concernes(s)})
    if declarees:
        contenu += _note(
            "Hypothèse déclarée par le projet : la cible du pas précédent est connue au moment de la "
            f"prévision (cible_connue = vrai pour {', '.join(declarees)}). Le contrôle automatique ne "
            "repère qu'un compte à rebours régulier : cette déclaration engage le projet.",
            "alerte",
        )
    return _section("pretraitement", "Prétraitement", contenu)


def _lignes_modeles(r: _Projet) -> tuple[list[str], list[list[Any]]]:
    tableau = r.resultat.tableau()
    entetes = [
        "Modèle",
        "Famille",
        "Moyenne",
        "Écart type",
        "Gain sur la référence",
        "Plis valides",
        "Durée (s)",
        "Statut",
    ]
    lignes = [
        [
            libelles.technique(m.modele),
            libelles.famille(m.famille),
            _n(m.moyenne),
            _n(m.ecart_type),
            _n(m.gain_vs_reference),
            f"{m.plis_valides} sur {len(r.resultat.plis)}" if r.avec_cible else "-",
            _n(m.duree_s, 2),
            m.statut,
        ]
        for m in tableau.itertuples(index=False)
    ]
    return entetes, lignes


def _section_modeles(r: _Projet) -> tuple[str, str, str]:
    res = r.resultat
    entetes, lignes = _lignes_modeles(r)
    contenu = (
        f"<p>Métrique du projet : {_e(libelles.metrique(res.metrique))}. "
        "Les écarts se lisent contre la référence naïve.</p>"
    )
    contenu += _table(entetes, lignes, numeriques={2, 3, 4, 6})
    contenu += _figure(
        decision_du_projet.figure_comparaison(res),
        "Score moyen de chaque modèle, avec la dispersion entre plis.",
    )
    if res.classe_positive is not None:
        source = {"specification": "fixée par l'utilisateur en S3"}.get(
            res.classe_positive_source or "",
            f"proposée d'après « {res.classe_positive_source} », non confirmée",
        )
        contenu += f"<p>Classe positive : <code>{_e(res.classe_positive)}</code> ({_e(source)}).</p>"
    if res.n_sans_cible:
        contenu += (
            f"<p>{_n(res.n_sans_cible)} ligne(s) d'apprentissage sans cible ont été retirées de la "
            "validation ; elles complètent les séries quand des variables temporelles sont construites.</p>"
        )
    for recherche in r.recherches:
        contenu += _recherche(recherche)
    return _section("modeles", "Comparaison des modèles", contenu)


def _recherche(recherche: train.ResultatRecherche) -> str:
    classement = recherche.classement()[:5]
    lignes = [
        [
            ", ".join(f"{k} = {v}" for k, v in m.choix.parametres) or "défauts",
            _n(m.moyenne(recherche.metrique)),
        ]
        for m in classement
    ]
    return (
        f"<h3>Recherche d'hyperparamètres : {_e(libelles.technique(recherche.choix_initial.id))}</h3>"
        + _table(["Paramètres", libelles.metrique_courte(recherche.metrique)], lignes, numeriques={1})
        + _note(
            "Le meilleur essai est choisi sur les mêmes plis que ceux qui le notent : son score est "
            "optimiste. Seul le jeu de test, ouvert une fois, donne une mesure honnête.",
            "alerte",
        )
    )


def _figures_du_diagnostic(diag: evaluate.Diagnostic, courbe_seuil: bool) -> list[str]:
    origine = "hors pli" if diag.origine == "validation" else "sur le test"
    figures = []
    try:
        if diag.probleme is Probleme.REGRESSION:
            figures.append(_figure(evaluate.figure_regression(diag), f"Résidus {origine}."))
        else:
            figures.append(_figure(evaluate.figure_confusion(diag), f"Matrice de confusion {origine}."))
            if diag.binaire:
                figures.append(
                    _figure(evaluate.figure_roc_pr(diag), f"Courbes ROC et précision rappel {origine}.")
                )
                if (
                    courbe_seuil
                    and diag.scores is not None
                    and diag.scores.min() >= 0.0
                    and diag.scores.max() <= 1.0
                ):
                    figures.append(
                        _figure(
                            evaluate.figure_seuil(diag.courbe_seuil(), diag.seuil), "Scores selon le seuil."
                        )
                    )
    except evaluate.ErreurEvaluation as erreur:
        figures.append(_note(f"Figure non tracée : {erreur}"))
    return figures


def _section_decision(r: _Projet) -> tuple[str, str, str]:
    d, res = r.decision, r.resultat
    contenu = _faits(
        [
            ("Modèle retenu", libelles.technique(d.modele_id)),
            ("Métrique", libelles.metrique(d.metrique)),
            ("Score de validation", _n(d.score_validation)),
            ("Seuil", _n(d.seuil, 3) if d.seuil is not None else "aucun (étiquette prédite par le modèle)"),
            ("Justification", d.justification or "-"),
        ]
    )
    try:
        mises_en_garde = decision_du_projet.avertissements(res, d.modele_id)
    except decision_du_projet.ErreurDecision:
        mises_en_garde = ()
    contenu += "".join(_note(m, "alerte") for m in mises_en_garde)
    if r.avec_cible:
        diag = evaluate.diagnostiquer_validation(res, d.modele_id, seuil=d.seuil)
        contenu += "<h3>Diagnostic hors pli</h3>" + "".join(_figures_du_diagnostic(diag, courbe_seuil=True))
    return _section("decision", "Décision et diagnostic", contenu)


def _section_test(r: _Projet) -> tuple[str, str, str]:
    t = r.test
    lignes = [[libelles.metrique(k), _n(v)] for k, v in t.scores.items()]
    contenu = _faits(
        [
            ("Lignes évaluées", _n(t.n_lignes)),
            ("Lignes de test sans cible, retirées", _n(t.n_sans_cible)),
            ("Ouvertures du jeu de test", r.session.coffre.ouvertures),
            ("Empreinte des prédictions", _Brut(f"<code>{_e(t.empreinte_predictions)}</code>")),
        ]
    )
    contenu += _table(["Métrique", "Valeur sur le test"], lignes, numeriques={1})
    if t.seuil is not None:
        contenu += (
            f"<p>Les étiquettes prédites appliquent le seuil {_e(_n(t.seuil, 3))} "
            "sur le score de la classe positive.</p>"
        )
    if r.session.test_optimiste:
        contenu += _note("Score de test optimiste : le test avait déjà été ouvert.", "alerte")
    if t.diagnostic is not None:
        contenu += "<h3>Mêmes figures sur le test</h3>" + "".join(
            _figures_du_diagnostic(t.diagnostic, courbe_seuil=False)
        )
    return _section("test", "Évaluation sur le test", contenu)


def _section_explication(r: _Projet) -> tuple[str, str, str] | None:
    contenu = ""
    if r.importances is not None and not r.importances.empty:
        figure = explain.figure_importances(
            r.importances.head(IMPORTANCES_MAX), "Importance par permutation (test)"
        )
        contenu += _figure(figure, "Baisse du score quand la variable est mélangée, sur les lignes de test.")
        contenu += _frame(
            r.importances.head(IMPORTANCES_MAX).drop(columns="plis_valides", errors="ignore"),
            libelles=LIBELLES_EXPLICATION,
        )
    if r.modele is not None:
        for titre, calcul in (
            ("Importance par impureté", lambda: explain.importance_par_impurete(r.modele)),
            ("Coefficients", lambda: explain.coefficients(r.modele)),
        ):
            try:
                tableau = calcul()
            except explain.ErreurExplication:
                continue
            if tableau is not None:
                contenu += f"<h3>{titre}</h3>" + _frame(
                    tableau.head(IMPORTANCES_MAX), libelles=LIBELLES_EXPLICATION
                )
        with contextlib.suppress(explain.ErreurExplication):
            contenu += "<h3>Règles de l'arbre</h3><pre>" + _e(explain.arbre_texte(r.modele)) + "</pre>"
    if not contenu:
        return None
    contenu += _note(
        "Une importance dit ce dont le modèle se sert, non ce qui cause la cible ; deux variables "
        "corrélées se partagent l'importance, et une variable très importante peut cacher une fuite à "
        "vérifier."
    )
    return _section("explication", "Ce qui explique le modèle", contenu)


def _section_sans_cible(r: _Projet) -> tuple[str, str, str]:
    contenu = ""
    try:
        analyse = unsupervised.analyser(r.resultat, r.decision.modele_id, r.table)
    except unsupervised.ErreurAnalyse as erreur:
        return _section("lecture", "Lecture du modèle", _note(str(erreur)))
    if isinstance(analyse, unsupervised.AnalyseComposantes):
        contenu += _figure(unsupervised.figure_composantes(analyse), "Variance expliquée par composante.")
        contenu += "<h3>Poids des variables</h3>" + _frame(
            analyse.poids.rename_axis("variable").reset_index(), libelles=LIBELLES_SANS_CIBLE
        )
    elif isinstance(analyse, unsupervised.AnalyseRegroupement):
        contenu += _figure(
            unsupervised.figure_plan(analyse.plan, analyse.etiquettes, analyse.modele_id),
            "Lignes dans le plan des deux premières composantes.",
        )
        contenu += _figure(
            unsupervised.figure_profils(analyse), "Valeur moyenne de chaque variable par groupe."
        )
        contenu += "<h3>Effectifs</h3>" + _frame(
            analyse.effectifs.rename_axis("groupe").reset_index(), libelles=LIBELLES_SANS_CIBLE
        )
    else:
        contenu += _figure(unsupervised.figure_anomalies(analyse), "Score d'anomalie de chaque ligne.")
        contenu += _figure(
            unsupervised.figure_plan(analyse.plan, analyse.etiquettes, analyse.modele_id, True),
            "Lignes dans le plan des deux premières composantes.",
        )
        contenu += _faits([("Part déclarée anormale", f"{_n(100 * analyse.taux, 2)} %")])
        contenu += "<h3>Lignes les plus anormales</h3>" + _frame(
            analyse.plus_anormales(r.table.donnees[list(r.spec.variables())]),
            maximum=10,
            libelles={k: v for k, v in LIBELLES_SANS_CIBLE.items() if k in ("position", "score_anomalie")},
        )
    contenu += _note(
        "Un regroupement existe toujours, et une anomalie n'est pas une panne : ce sont des pistes à "
        "confirmer avec l'expérience du terrain, non des preuves."
    )
    return _section("lecture", "Lecture du modèle sans cible", contenu)


def _section_reproductibilite(r: _Projet, fichiers: Mapping[str, str]) -> tuple[str, str, str]:
    env = ", ".join(f"{k} {v}" for k, v in r.env.items())
    contenu = _faits(
        [
            ("Application", f"MécaML Explorer {__version__}"),
            ("Graine unique", r.spec.graine),
            ("Environnement", env),
            (
                "Plis de validation (empreinte)",
                _Brut(f"<code>{_e(codegen.empreinte_plis(r.resultat.plis))}</code>"),
            )
            if r.avec_cible
            else ("Plis", "sans objet"),
        ]
    )
    contenu += (
        "<p>Le script <code>"
        + _e(fichiers.get("script", NOMS["script"]))
        + "</code> refait la lecture, le découpage, "
        "la validation croisée et l'évaluation avec scikit-learn seul et vérifie les empreintes ci-dessus "
        "avant de rendre ses scores. <code>projet.json</code> se rejoue avec <code>python -m mecaml.cli "
        "rejouer</code>.</p>"
    )
    if r.modele is not None or r.avec_cible:
        contenu += _note(
            "Le fichier modele.joblib embarque, avec les variables temporelles, des lignes du jeu "
            "(historique et contexte). joblib.load exécute du code : ne charger que ses propres fichiers.",
        )
    return _section("reproductibilite", "Reproductibilité", contenu)


def _section_limites() -> tuple[str, str, str]:
    contenu = (
        "<ul>"
        "<li>Le score de test vient d'une seule ouverture du jeu de test ; relancer le script ne le rend "
        "pas neuf.</li>"
        "<li>Une validation croisée mesure le modèle sur les données dont elle dispose : si les futures "
        "données sortent de leur domaine (autre machine, autre matériau, autre plage), le score ne tient "
        "plus.</li>"
        "<li>Les importances et les groupes décrivent le modèle et les données, non des causes.</li>"
        "<li>Les variables à valeurs manquantes, les unités mal définies et les étiquettes mal posées se "
        "paient toujours : relire les alertes acquittées.</li>"
        "</ul>"
    )
    return _section("limites", "Limites", contenu)


def _section_renvois(r: _Projet) -> tuple[str, str, str]:
    techniques = {c.id: c for c in (*r.spec.pretraitement, *r.spec.modeles)}
    lignes = [_technique(c) for c in techniques.values()]
    metrique, _ = metrics.analyser(r.resultat.metrique)
    try:
        fiche = livre.technique_du_livre(renvois.metrique(metrique.id))
        lignes.append(
            [
                libelles.metrique(r.resultat.metrique),
                "métrique du projet",
                "",
                f"{fiche.id} : {fiche.libelle}",
            ]
        )
    except KeyError:
        pass
    return _section(
        "renvois", "Où le livre l'explique", _table(["Technique", "Paramètres", "Colonnes", "Livre"], lignes)
    )


# ============================================================
# RAPPORT HTML
# ============================================================
def _sections(r: _Projet, fichiers: Mapping[str, str]) -> list[tuple[str, str, str]]:
    sections = [_section_resume(r), _section_donnees(r), _section_audit(r), _section_decoupage(r)]
    sections += [_section_pretraitement(r), _section_modeles(r), _section_decision(r)]
    if r.avec_cible:
        sections.append(_section_test(r))
        explication = _section_explication(r)
        if explication is not None:
            sections.append(explication)
    else:
        sections.append(_section_sans_cible(r))
    sections += [_section_reproductibilite(r, fichiers), _section_limites(), _section_renvois(r)]
    return sections


def rapport_html(session: Session, **options: Any) -> str:
    """Rapport en un seul document HTML : style et figures en ligne, aucune ressource extérieure.

    Mêmes options et mêmes refus que `exporter` (découpage, chargement, fichier publié, recherches, date).
    """
    r = _rassembler(session, **_options_de_lecture(options))
    _preparer_explications(r)
    return _document_html(r, NOMS)


def _document_html(r: _Projet, fichiers: Mapping[str, str]) -> str:
    sections = _sections(r, fichiers)
    sommaire = "\n".join(f'<li><a href="#{i}">{_e(titre)}</a></li>' for i, titre, _ in sections)
    quand = r.maintenant.strftime("%d/%m/%Y à %H:%M")
    return PAGE.substitute(
        titre=_e(f"Rapport : {r.table.nom}"),
        sous_titre=_e(
            f"{_lib_probleme(r.spec.probleme)}, modèle retenu : {libelles.technique(r.decision.modele_id)}"
        ),
        style=CSS,
        sommaire=sommaire,
        corps="\n".join(contenu for _, _, contenu in sections),
        pied=_e(
            f"Rapport engendré par MécaML Explorer {__version__} le {quand}. "
            "Aucune donnée n'a quitté cet ordinateur."
        ),
    )


# ============================================================
# CLASSEUR XLSX
# ============================================================
def _poser(feuille: Any, ligne: int, colonne: int, valeur: Any) -> None:
    """Écrit une cellule : nombre en nombre, texte en texte (jamais une formule), indéfini en vide."""
    cellule = feuille.cell(row=ligne, column=colonne)
    if isinstance(valeur, (np.integer,)):
        valeur = int(valeur)
    elif isinstance(valeur, (np.floating,)):
        valeur = float(valeur)
    elif isinstance(valeur, np.bool_):
        valeur = bool(valeur)
    if isinstance(valeur, float) and not math.isfinite(valeur):
        return
    cellule.value = valeur
    if isinstance(valeur, str) and valeur[:1] == "=":
        cellule.data_type = "s"


def _ecrire(feuille: Any, entetes: Sequence[str], lignes: Iterable[Sequence[Any]]) -> None:
    from openpyxl.styles import Font

    for k, h in enumerate(entetes, start=1):
        _poser(feuille, 1, k, h)
        feuille.cell(row=1, column=k).font = Font(bold=True)
    largeurs = [len(str(h)) for h in entetes]
    for i, ligne in enumerate(lignes, start=2):
        for k, v in enumerate(ligne, start=1):
            _poser(feuille, i, k, v)
            largeurs[k - 1] = max(largeurs[k - 1], min(len(str(v)), 60))
    for k, largeur in enumerate(largeurs, start=1):
        feuille.column_dimensions[feuille.cell(row=1, column=k).column_letter].width = largeur + 2
    feuille.freeze_panes = "A2"


def classeur_xlsx(session: Session, **options: Any) -> Any:
    """Classeur `openpyxl` des chiffres du rapport : résumé, constats, modèles, plis, test, paramètres."""
    r = _rassembler(session, **_options_de_lecture(options))
    _preparer_explications(r)
    return _classeur(r)


def _classeur(r: _Projet) -> Any:
    from openpyxl import Workbook

    classeur = Workbook()
    resume = classeur.active
    resume.title = "Résumé"
    s, d = r.spec, r.decision
    lignes = [
        ("Fichier", r.table.nom),
        ("Problème", _lib_probleme(s.probleme)),
        ("Cible", s.cible or ""),
        ("Métrique", libelles.metrique(s.metrique) if s.metrique else "silhouette"),
        ("Identifiant de la métrique", s.metrique or "silhouette"),
        ("Modèle retenu", libelles.technique(d.modele_id)),
        ("Identifiant du modèle", d.modele_id),
        ("Score de validation", float(d.score_validation)),
        ("Seuil", d.seuil if d.seuil is not None else ""),
        ("Graine", s.graine),
        ("Application", f"MécaML Explorer {__version__}"),
    ]
    if r.test is not None:
        lignes += [
            ("Score de test", float(r.test.valeur())),
            ("Score de test optimiste", "oui" if r.session.test_optimiste else "non"),
        ]
    _ecrire(resume, ["Clé", "Valeur"], lignes)
    _ecrire(
        classeur.create_sheet("Constats"),
        ["Niveau", "Code", "Constat", "Action", "Acquittée"],
        [
            [
                NIVEAUX[c.niveau],
                c.code,
                c.message,
                c.action,
                "oui" if c.code in r.session.acquittes else "non",
            ]
            for c in r.session.constats
        ],
    )
    entetes, _ = _lignes_modeles(r)
    tableau = r.resultat.tableau()
    _ecrire(
        classeur.create_sheet("Modèles"),
        entetes,
        [
            [libelles.technique(m.modele), libelles.famille(m.famille), *m[2:]]
            for m in tableau.itertuples(index=False)
        ],
    )
    _ecrire(classeur.create_sheet("Plis"), ["Modèle", "Métrique", "Pli", "Valeur"], _lignes_plis(r.resultat))
    if r.test is not None:
        _ecrire(
            classeur.create_sheet("Test"),
            ["Métrique", "Valeur"],
            [[libelles.metrique(k), v] for k, v in r.test.scores.items()],
        )
        if r.importances is not None:
            visibles = r.importances.drop(columns="plis_valides", errors="ignore")
            _ecrire(
                classeur.create_sheet("Importances"),
                [LIBELLES_EXPLICATION.get(c, c) for c in visibles.columns],
                visibles.itertuples(index=False),
            )
    _ecrire(classeur.create_sheet("Paramètres"), ["Clé", "Valeur"], _lignes_parametres(r))
    return classeur


def _lignes_plis(resultat: train.Resultat) -> list[list[Any]]:
    return [
        [libelles.technique(m.id), libelles.metrique(metrique), k, valeur]
        for m in resultat.modeles
        for metrique, valeurs in m.scores_plis.items()
        for k, valeur in enumerate(valeurs, start=1)
    ]


def _lignes_parametres(r: _Projet) -> list[list[Any]]:
    s = r.spec
    lignes: list[list[Any]] = [["roles", ", ".join(f"{n}={role.value}" for n, role in s.roles)]]
    for c in (*s.pretraitement, *s.modeles):
        lignes.append(
            [c.id, ", ".join(f"{k}={v}" for k, v in c.parametres) or "défauts"]
        )  # identifiants stables
    return lignes


# ============================================================
# PROJET REJOUABLE
# ============================================================
def projet_json(session: Session, **options: Any) -> str:
    """Projet rejouable : spécification, données, découpage, décision, acquittements, scores attendus."""
    return _projet_json(_rassembler(session, **_options_de_lecture(options)))


def _projet_json(r: _Projet) -> str:
    s, d, res = r.spec, r.decision, r.resultat
    attendus: dict[str, Any] = {
        "empreinte_donnees": r.table.empreinte,
        "validation": {m.id: {k: list(v) for k, v in m.scores_plis.items()} for m in res.modeles},
    }
    donnees: dict[str, Any] = {
        "fichier": r.table.nom,
        "feuille": r.table.feuille,
        "fichier_test": None if r.table_test is None else r.table_test.nom,
        "feuille_test": None if r.table_test is None else r.table_test.feuille,
    }
    if r.chargement is not None:
        donnees["sha256"] = r.chargement.sha256
    if r.avec_cible:
        attendus["empreinte_test"] = r.decoupage.empreinte_sha256
        attendus["empreinte_plis"] = codegen.empreinte_plis(res.plis)
        attendus["test"] = {
            "scores": dict(r.test.scores),
            "n_lignes": r.test.n_lignes,
            "n_sans_cible": r.test.n_sans_cible,
            "empreinte_predictions": r.test.empreinte_predictions,
        }
    decoupage = None
    if r.decoupage is not None:
        decoupage = {"methode": r.decoupage.methode.value}
        if r.decoupage.conforme_contrat:
            decoupage["base_test"] = r.decoupage.vers_base_test()
    contenu = {
        "format": FORMAT_DU_PROJET,
        "application": {"nom": "MécaML Explorer", "version": __version__},
        "donnees": donnees,
        "projet": s.vers_dict(),
        "decoupage": decoupage,
        "acquittements": sorted(r.session.acquittes),
        "decision": {"modele": d.modele_id, "seuil": d.seuil, "justification": d.justification},
        "attendus": attendus,
        "environnement": dict(r.env),
    }
    return json.dumps(_json_sur(contenu), indent=2, ensure_ascii=False) + "\n"


# ============================================================
# EXPORT
# ============================================================
def _options_de_lecture(options: Mapping[str, Any]) -> dict[str, Any]:
    """Options communes de rassemblement, avec leurs défauts."""
    permises = {"decoupage", "chargement", "sha256_publie", "recherches", "maintenant", "env", "table_test"}
    inconnues = sorted(set(options) - permises)
    if inconnues:
        raise TypeError(f"Option(s) inconnue(s) : {inconnues}.")
    return {
        "decoupage": options.get("decoupage"),
        "chargement": options.get("chargement"),
        "sha256_publie": options.get("sha256_publie"),
        "recherches": options.get("recherches", ()),
        "maintenant": options.get("maintenant"),
        "env": options.get("env"),
        "table_test": options.get("table_test"),
    }


def _modele_a_ecrire(r: _Projet) -> Any:
    """Pipeline ajusté sur tout l'apprentissage : celui évalué par le test, sinon un ajustement identique."""
    if r.test is not None and r.test.modele is not None:
        return r.test.modele.pipeline
    if r.avec_cible:
        choix = next(c for c in r.spec.modeles if c.id == r.decision.modele_id)
        return train.ajuster_final(r.spec, choix, r.table, r.session.coffre).pipeline
    return r.resultat.modele(r.decision.modele_id).pipeline


def _preparer_explications(r: _Projet) -> None:
    """Importances du test (déjà calculées en S8) et modèle ajusté, pour le rapport."""
    if r.test is not None:
        r.importances = r.test.importances
        r.modele = r.test.modele


def exporter(
    session: Session,
    dossier: str | Path,
    *,
    decoupage: split.Decoupage | None = None,
    chargement: Chargement | None = None,
    sha256_publie: str | None = None,
    recherches: Sequence[train.ResultatRecherche] = (),
    formats: Collection[str] = FORMATS,
    maintenant: datetime | None = None,
    env: Mapping[str, str] | None = None,
    table_test: Table | None = None,
) -> Export:
    """Écrit rapport, classeur, modèle, script et projet dans `dossier` ; rend l'`Export` pour `Exporter`.

    Refus (`ErreurExport`) avant toute écriture : session pas en S8 (S7 sans cible), constat bloquant ou
    alerte en attente, cible passée non déclarée ou qui décompte, fichier lu différent du fichier publié
    (`sha256_publie`). Si une écriture échoue, les fichiers déjà écrits par cet appel sont retirés. Les
    fichiers de même nom d'un export précédent sont remplacés.
    """
    inconnus = sorted(set(formats) - set(FORMATS))
    if inconnus:
        raise ErreurExport(f"Format(s) inconnu(s) : {inconnus} ; permis : {list(FORMATS)}.")
    r = _rassembler(
        session,
        decoupage=decoupage,
        chargement=chargement,
        sha256_publie=sha256_publie,
        recherches=recherches,
        maintenant=maintenant,
        env=env,
        table_test=table_test,
    )
    _preparer_explications(r)
    dossier = Path(dossier)
    contenus = _contenus(r, formats)
    ecrits: dict[str, Path] = {}
    empreintes: dict[str, str] = {}
    try:
        for cle, (nom, ecrire) in contenus.items():
            chemin = chemin_dans_dossier(dossier, nom)
            empreintes[nom] = ecrire(chemin)
            ecrits[cle] = chemin
    except Exception:
        for chemin in ecrits.values():
            chemin.unlink(missing_ok=True)
        raise
    return Export(dossier.resolve(), ecrits, empreintes, tuple(r.notes), tuple(sorted(formats)))


def _contenus(r: _Projet, formats: Collection[str]) -> dict[str, tuple[str, Callable[[Path], str]]]:
    """Pour chaque fichier demandé, son nom et la fonction qui l'écrit et rend son empreinte."""
    noms = {cle: NOMS[cle] for cle in FORMATS if cle in formats}
    script = None
    if "script" in formats:
        fichier_test = None if r.table_test is None else codegen.lecture_du_fichier(r.table_test)
        script = codegen.generer_script(
            r.spec,
            r.table,
            r.session.coffre,
            decoupage=r.decoupage,
            decision=r.decision,
            chargement=r.chargement,
            fichier_test=fichier_test,
            env=r.env,
        )
    ecritures: dict[str, tuple[str, Callable[[Path], str]]] = {}
    if "script" in formats:
        ecritures["script"] = (noms["script"], lambda p: _ecrire_texte(p, script))
    if "projet" in formats:
        ecritures["projet"] = (noms["projet"], lambda p: _ecrire_texte(p, _projet_json(r)))
    if "html" in formats:
        ecritures["html"] = (noms["html"], lambda p: _ecrire_texte(p, _document_html(r, noms)))
    if "xlsx" in formats:
        ecritures["xlsx"] = (noms["xlsx"], lambda p: _ecrire_classeur(p, _classeur(r)))
    if "joblib" in formats:
        ecritures["joblib"] = (noms["joblib"], lambda p: _ecrire_modele(p, _modele_a_ecrire(r)))
    return ecritures


def _ecrire_texte(chemin: Path, texte: str) -> str:
    donnees = texte.encode("utf-8")
    chemin.write_bytes(donnees)
    return _sha256(donnees)


def _ecrire_classeur(chemin: Path, classeur: Any) -> str:
    classeur.save(chemin)
    return _sha256(chemin.read_bytes())


def _ecrire_modele(chemin: Path, pipeline: Any) -> str:
    joblib.dump(pipeline, chemin)
    return _sha256(chemin.read_bytes())


def exporter_et_clore(session: Session, dossier: str | Path, **options: Any) -> tuple[Session, Export]:
    """`exporter`, puis l'action `Exporter` (S9) ; rend la nouvelle session et l'export."""
    from mecaml.core.session import Exporter

    export = exporter(session, dossier, **options)
    return session.appliquer(Exporter(export)), export
