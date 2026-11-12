"""
MécaML en ligne de commande : rejoue un projet sans interface, par les mêmes actions de session que
l'application.

    python -m mecaml.cli proposer FICHIER [-c CIBLE] [-s PROJET.json]   projet proposé d'après les données
    python -m mecaml.cli auditer FICHIER [-p PROJET.json]               constats d'audit, niveau par niveau
    python -m mecaml.cli rejouer PROJET.json [-d FICHIER] [-o DOSSIER]  S0 à S9 puis export

Le projet est un fichier JSON (`projet.json`, écrit par l'export) : spécification, données, découpage,
décision et alertes acquittées. `rejouer` mène la session de S0 à S9 avec les garde-fous de l'application :
une alerte non acquittée par le projet arrête le rejeu (l'option `-a` acquitte toutes les alertes, à ne
choisir qu'en connaissance de cause), et le jeu de test n'est ouvert qu'une fois. Avec `-v`, les scores
recalculés sont comparés aux scores attendus que le projet a gardés (écart toléré : 1e-9) : un rejeu qui
diffère est un échec.

Codes de retour : 0 réussi, 1 projet refusé ou différent de l'attendu (le message dit quoi corriger), 2 usage.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from mecaml import __version__
from mecaml.core import audit, decision, evaluate, metrics, preprocess, report, split, train
from mecaml.core import io as fichiers
from mecaml.core.codegen import ErreurExport, empreinte_plis
from mecaml.core.guards import ErreurDonnees, decimale
from mecaml.core.session import (
    Acquitter,
    Auditer,
    Charger,
    Decider,
    Decouper,
    Entrainer,
    Exporter,
    FixerObjectif,
    Pretraiter,
    Session,
    Tester,
    TransitionRefusee,
)
from mecaml.core.tasks import erreur_etrangere, message_utilisateur
from mecaml.core.types import Probleme, SpecProjet, Table

__all__ = [
    "ErreurProjet",
    "Execution",
    "lire_projet",
    "principal",
    "proposer_projet",
    "rejouer",
    "verifier_attendus",
]

TOLERANCE = 1e-9
FORMAT_DU_PROJET = report.FORMAT_DU_PROJET


class ErreurProjet(ValueError):
    """Fichier de projet illisible, incomplet ou incohérent ; le message dit quoi corriger."""


@dataclass
class Execution:
    """Ce que le rejeu a produit : la session finale, le chargement, le découpage, l'export."""

    session: Session
    chargement: fichiers.Chargement
    decoupage: split.Decoupage | None
    export: report.Export | None = None


def _option(nom: str) -> str:
    return "-" * 2 + nom  # écrit ainsi pour que ce fichier reste sans double trait d'union


# ============================================================
# LECTURE DU PROJET
# ============================================================
def lire_projet(chemin: str | Path) -> dict[str, Any]:
    """Projet JSON : lu, puis contrôlé (format, rubriques obligatoires)."""
    chemin = Path(chemin)
    try:
        projet = json.loads(chemin.read_text(encoding="utf-8"))
    except OSError as erreur:
        raise ErreurProjet(f"Projet illisible : {chemin} ({erreur.strerror}).") from erreur
    except json.JSONDecodeError as erreur:
        raise ErreurProjet(
            f"Le projet {chemin.name} n'est pas du JSON valide (ligne {erreur.lineno})."
        ) from erreur
    if not isinstance(projet, dict) or projet.get("format") != FORMAT_DU_PROJET:
        raise ErreurProjet(f"Le projet {chemin.name} n'est pas au format {FORMAT_DU_PROJET} de MécaML.")
    manquantes = [k for k in ("donnees", "projet") if k not in projet]
    if manquantes:
        raise ErreurProjet(f"Le projet {chemin.name} n'a pas de rubrique {manquantes}.")
    return projet


# ============================================================
# REJEU
# ============================================================
def _dire(avancer: Callable[[str], None] | None, message: str) -> None:
    if avancer is not None:
        avancer(message)


def _chemin_des_donnees(donnees: Mapping[str, Any], base: Path, remplacement: Path | None) -> Path:
    if remplacement is not None:
        return Path(remplacement)
    fichier = donnees.get("fichier")
    if not fichier:
        raise ErreurProjet(
            "Le projet ne dit pas quel fichier lire : donner le fichier de données (option -d)."
        )
    chemin = base / fichier
    if not chemin.is_file():
        raise ErreurProjet(
            f"Fichier introuvable : {chemin}. Le projet cherche son classeur à côté du projet.json ; "
            "s'il est ailleurs, le désigner avec l'option -d (par exemple -d le_classeur.xlsx)."
        )
    return chemin


class _Scelle(NamedTuple):
    """Ce que donne la mise sous scellé : la spécification au découpage déclaré, le découpage, le coffre."""

    spec: SpecProjet
    decoupage: split.Decoupage | None
    coffre: Any
    table_test: Table | None = None  # jeu de test fourni à part


def _sceller(
    projet: Mapping[str, Any], table: Table, spec: SpecProjet, chemin_test: Path | None, base: Path
) -> _Scelle:
    """Jeu de test mis sous scellé comme le projet le déclare : base de test, fichier à part, ou méthode."""
    if not spec.probleme.avec_cible:
        return _Scelle(spec, None, None)
    declaration = projet.get("decoupage") or {}
    donnees = projet["donnees"]
    if chemin_test is not None or donnees.get("fichier_test"):
        cible = chemin_test or base / donnees["fichier_test"]
        test = fichiers.charger(cible, feuille=donnees.get("feuille_test")).table
        decoupage, coffre = split.sceller_fichier(table, test, spec)
        return _Scelle(spec, decoupage, coffre, test)
    if "base_test" in declaration:
        return _Scelle(*split.sceller_base_test(table, spec, declaration["base_test"]))
    decoupage, coffre = split.sceller(table, spec, methode=declaration.get("methode"))
    return _Scelle(spec, decoupage, coffre)


def _acquitter(session: Session, projet: Mapping[str, Any], tout: bool) -> Session:
    """Acquitte les alertes nommées par le projet ; les autres arrêtent le rejeu (sauf tout acquitter)."""
    nommees = frozenset(projet.get("acquittements") or ())
    presentes = {c.code: c for c in session.alertes_en_attente()}
    voulues = frozenset(presentes) if tout else nommees & frozenset(presentes)
    if voulues:
        session = session.appliquer(Acquitter(voulues))
    restantes = session.alertes_en_attente()
    if restantes:
        details = " ; ".join(f"{c.code} : {c.message}" for c in restantes[:3])
        raise ErreurProjet(
            f"{len(restantes)} alerte(s) à acquitter avant de poursuivre : {details}. Les relire, puis les "
            f"lister dans la rubrique « acquittements » du projet (ou l'option {_option('tout-acquitter')} "
            "en connaissance de cause)."
        )
    return session


def _decision(projet: Mapping[str, Any], resultat: train.Resultat) -> Any:
    voulue = projet.get("decision") or {}
    if not voulue.get("modele"):
        return decision.proposer(resultat)
    return decision.decider(
        resultat, voulue["modele"], seuil=voulue.get("seuil"), justification=voulue.get("justification") or ""
    )


def rejouer(
    projet: Mapping[str, Any],
    base: str | Path,
    *,
    donnees: str | Path | None = None,
    fichier_test: str | Path | None = None,
    sortie: str | Path | None = None,
    tout_acquitter: bool = False,
    importances: bool = True,
    avancer: Callable[[str], None] | None = None,
) -> Execution:
    """Rejoue un projet de S0 à S8, puis exporte dans `sortie` s'il est donné (S9).

    `base` : dossier du fichier de projet, d'où se lit un fichier de données donné par son seul nom. Toute
    étape refusée par la session lève l'erreur de son motif ; une alerte que le projet n'acquitte pas arrête
    le rejeu.
    """
    base = Path(base)
    spec = SpecProjet.depuis_dict(projet["projet"])
    _dire(avancer, "Lecture des données")
    chargement = fichiers.charger(
        _chemin_des_donnees(projet["donnees"], base, donnees and Path(donnees)),
        feuille=projet["donnees"].get("feuille"),
    )
    table = chargement.table
    session = Session(auditeur=audit.auditer).appliquer(Charger(table)).appliquer(Auditer())
    _dire(avancer, "Objectif et découpage")
    session = session.appliquer(FixerObjectif(spec))
    spec, decoupage, coffre, table_test = _sceller(
        projet, table, spec, fichier_test and Path(fichier_test), base
    )
    session = _acquitter(session.appliquer(Decouper(coffre, spec=spec)), projet, tout_acquitter)
    _dire(avancer, "Prétraitement et entraînement")
    session = session.appliquer(Pretraiter(spec))
    resultat = train.entrainer(spec, table, coffre)
    session = session.appliquer(Entrainer(resultat, spec=resultat.spec))
    choix = _decision(projet, resultat)
    session = session.appliquer(Decider(choix))
    if spec.probleme.avec_cible:
        _dire(avancer, "Ouverture du jeu de test")
        evaluer = evaluate.preparer_test(resultat.spec, table, coffre, choix, importances=importances)
        session = session.appliquer(Tester(evaluer))
    execution = Execution(session, chargement, decoupage)
    if sortie is not None:
        _dire(avancer, "Export")
        Path(sortie).mkdir(parents=True, exist_ok=True)  # le dossier est celui que l'utilisateur a choisi
        export = report.exporter(
            session,
            sortie,
            decoupage=decoupage,
            chargement=chargement,
            sha256_publie=projet["donnees"].get("sha256_publie"),
            table_test=table_test,
        )
        execution.session = session.appliquer(Exporter(export))
        execution.export = export
    return execution


# ============================================================
# COMPARAISON AUX SCORES ATTENDUS
# ============================================================
def _ecart(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0 if (a is None or math.isnan(a)) and (b is None or math.isnan(b)) else math.inf
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return math.inf if math.isnan(a) or math.isnan(b) else abs(a - b)


def verifier_attendus(
    execution: Execution, attendus: Mapping[str, Any], tolerance: float = TOLERANCE
) -> list[str]:
    """Écarts entre le rejeu et les scores attendus du projet (liste vide : le rejeu reproduit le projet)."""
    session = execution.session
    resultat: train.Resultat = session.resultats
    ecarts: list[str] = []

    def comparer(nom: str, voulu: Any, obtenu: Any) -> None:
        if voulu != obtenu:
            ecarts.append(f"{nom} : attendu {str(voulu)[:16]}..., obtenu {str(obtenu)[:16]}...")

    comparer("empreinte des données", attendus.get("empreinte_donnees"), session.table.empreinte)
    if session.spec.probleme.avec_cible:
        comparer("empreinte du test", attendus.get("empreinte_test"), execution.decoupage.empreinte_sha256)
        comparer("empreinte des plis", attendus.get("empreinte_plis"), empreinte_plis(resultat.plis))
    for modele in resultat.modeles:
        voulu = (attendus.get("validation") or {}).get(modele.id)
        if voulu is None:
            ecarts.append(f"modèle {modele.id} absent des scores attendus")
            continue
        for metrique, valeurs in modele.scores_plis.items():
            for k, (v, w) in enumerate(zip(valeurs, voulu.get(metrique, []), strict=False), start=1):
                if _ecart(v, w) > tolerance:
                    ecarts.append(f"{modele.id}, {metrique}, pli {k} : attendu {w}, obtenu {v}")
    test = session.resultat_test
    if test is not None and "test" in attendus:
        comparer(
            "empreinte des prédictions",
            attendus["test"].get("empreinte_predictions"),
            test.empreinte_predictions,
        )
        for metrique, v in test.scores.items():
            w = attendus["test"]["scores"].get(metrique)
            if _ecart(v, w) > tolerance:
                ecarts.append(f"test, {metrique} : attendu {w}, obtenu {v}")
    return ecarts


# ============================================================
# PROJET PROPOSÉ D'APRÈS LES DONNÉES
# ============================================================
def proposer_projet(table: Table, cible: str | None = None, sans_cible: bool = False) -> dict[str, Any]:
    """Projet proposé : rôles, problème, structure, métrique, modèles, prétraitement ; à relire et confirmer.

    Le prétraitement se propose sur les lignes d'apprentissage seules, une fois le test mis sous scellé.
    """
    profils = audit.profiler(table)
    cible = None if sans_cible else cible or audit.proposer_cible(profils)
    roles = audit.proposer_roles(table, cible)
    structure = audit.proposer_structure(roles)
    probleme = Probleme.SANS_CIBLE if cible is None else audit.probleme_propose(table, cible)
    notes: list[str] = []
    classe_positive = None
    if probleme is Probleme.REGRESSION:
        metrique = "rmse"
    elif probleme is Probleme.CLASSIFICATION:
        classes = table.donnees[cible].dropna().nunique()
        metrique = "f1" if classes == 2 else "exactitude_equilibree"
        if classes == 2:
            classe_positive, origine = metrics.proposer_classe_positive(
                table.donnees[cible].dropna().to_numpy()
            )
            notes.append(f"classe positive proposée d'après « {origine} » : {classe_positive!r}, à confirmer")
    else:
        metrique = "silhouette"
    spec = SpecProjet(
        probleme=probleme,
        cible=cible,
        roles=roles,
        metrique=metrique,
        structure=structure,
        classe_positive=classe_positive,
        modeles=train.proposer_modeles(probleme, structure),
    )
    if probleme.avec_cible:
        coffre = _sceller({"donnees": {}}, table, spec, None, Path()).coffre
        apprentissage = coffre.apprentissage()
        plis = min(spec.n_plis, max(2, split.plis_au_plus(apprentissage, spec)))
        if plis < spec.n_plis:
            notes.append(
                f"nombre de plis ramené à {plis} : l'apprentissage compte trop peu d'unités ou de lignes"
            )
            spec = spec.avec(n_plis=plis)
        spec = spec.avec(pretraitement=preprocess.proposer_pretraitement(spec, apprentissage))
    else:
        spec = spec.avec(pretraitement=preprocess.proposer_pretraitement(spec, table.donnees))
    return {
        "format": FORMAT_DU_PROJET,
        "donnees": {"fichier": table.nom, "feuille": table.feuille},
        "projet": spec.vers_dict(),
        "decoupage": None,
        "acquittements": [],
        "decision": {"modele": None, "seuil": None, "justification": ""},
        "a_confirmer": notes,
    }


# ============================================================
# AFFICHAGE
# ============================================================
def _resume(execution: Execution) -> str:
    session = execution.session
    resultat: train.Resultat = session.resultats
    lignes = [f"Données : {session.table.nom}, {session.table.n_lignes} lignes", ""]
    lignes.append(f"Comparaison des modèles ({resultat.metrique}, {resultat.validation}) :")
    for m in resultat.classement():
        valeur = resultat.valeur(m)
        etat = (
            f"échec : {m.erreur}"
            if m.erreur
            else f"{decimale(valeur, 4)} +/- {decimale(m.ecart_type(resultat.metrique), 4)}"
        )
        lignes.append(f"  {m.id:<46} {etat}")
    choix = session.decision
    lignes += [
        "",
        f"Modèle retenu : {choix.modele_id}" + (f", seuil {choix.seuil}" if choix.seuil is not None else ""),
    ]
    if session.resultat_test is not None:
        test = session.resultat_test
        optimiste = " (score optimiste : test déjà vu)" if session.test_optimiste else ""
        lignes.append(f"Test ({test.n_lignes} lignes{optimiste}) :")
        lignes += [f"  {nom:<28} {decimale(v, 4)}" for nom, v in test.scores.items()]
    if execution.export is not None:
        lignes += ["", f"Export dans {execution.export.dossier} : {', '.join(execution.export.liste())}"]
    return "\n".join(lignes)


def _afficher_constats(table: Table, spec: SpecProjet | None) -> None:
    constats = audit.auditer(table, spec, None)
    for niveau in ("bloquant", "alerte", "info"):
        retenus = [c for c in constats if c.niveau.value == niveau]
        print(f"{niveau.upper()} ({len(retenus)})")
        for c in retenus:
            print(f"  {c.code} : {c.message}" + (f" -> {c.action}" if c.action else ""))


# ============================================================
# COMMANDES
# ============================================================
def _commande_rejouer(options: argparse.Namespace) -> int:
    chemin = Path(options.projet)
    projet = lire_projet(chemin)
    if options.verifier and not projet.get("attendus"):
        raise ErreurProjet(
            f"Le projet {chemin.name} n'a pas de scores attendus à comparer : vérifier le projet.json "
            "que l'export a écrit, qui les garde."
        )
    for note in projet.get("a_confirmer") or ():
        print(f"A confirmer avant de s'y fier : {note}", file=sys.stderr)
    execution = rejouer(
        projet,
        chemin.parent,
        donnees=options.donnees,
        fichier_test=options.test,
        sortie=options.sortie,
        tout_acquitter=options.tout_acquitter,
        importances=not options.sans_importances,
        avancer=lambda message: print(f"... {message}", file=sys.stderr),
    )
    print(_resume(execution))
    if options.verifier:
        ecarts = verifier_attendus(execution, projet.get("attendus") or {})
        if ecarts:
            print("\nLe rejeu diffère du projet :\n  " + "\n  ".join(ecarts[:12]), file=sys.stderr)
            return 1
        print("\nLe rejeu reproduit les scores attendus du projet.")
    return 0


def _commande_auditer(options: argparse.Namespace) -> int:
    table = fichiers.charger(options.fichier).table
    spec = None
    if options.projet:
        spec = SpecProjet.depuis_dict(lire_projet(options.projet)["projet"])
    _afficher_constats(table, spec)
    return 0


def _commande_proposer(options: argparse.Namespace) -> int:
    table = fichiers.charger(options.fichier).table
    projet = proposer_projet(table, options.cible, options.sans_cible)
    texte = json.dumps(projet, indent=2, ensure_ascii=False) + "\n"
    if options.sortie:
        Path(options.sortie).write_text(texte, encoding="utf-8", newline="\n")
        print(f"Projet proposé écrit dans {options.sortie} ; à relire avant de le rejouer.")
    else:
        print(texte)
    for note in projet["a_confirmer"]:
        print(f"A confirmer : {note}", file=sys.stderr)
    return 0


def _analyseur() -> argparse.ArgumentParser:
    analyseur = argparse.ArgumentParser(
        prog="mecaml.cli", description="MécaML en ligne de commande : rejoue un projet sans interface."
    )
    analyseur.add_argument(_option("version"), action="version", version=f"MécaML Explorer {__version__}")
    commandes = analyseur.add_subparsers(dest="commande", required=True)
    rejeu = commandes.add_parser("rejouer", help="rejoue un projet de S0 à S9")
    rejeu.add_argument("projet", help="fichier projet.json")
    rejeu.add_argument("-d", _option("donnees"), help="fichier de données (remplace celui du projet)")
    rejeu.add_argument("-t", _option("test"), help="jeu de test fourni à part (remplace celui du projet)")
    rejeu.add_argument(
        "-o", _option("sortie"), help="dossier de l'export (sans lui, le rejeu s'arrête en S8)"
    )
    rejeu.add_argument(
        "-v", _option("verifier"), action="store_true", help="comparer aux scores attendus du projet"
    )
    rejeu.add_argument(
        "-a", _option("tout-acquitter"), action="store_true", help="acquitter toutes les alertes"
    )
    rejeu.add_argument(
        _option("sans-importances"), action="store_true", help="ne pas calculer les importances du test"
    )
    rejeu.set_defaults(fonction=_commande_rejouer)
    audit_ = commandes.add_parser("auditer", help="affiche les constats d'audit d'un fichier")
    audit_.add_argument("fichier")
    audit_.add_argument("-p", _option("projet"), help="projet.json dont l'objectif cadre l'audit")
    audit_.set_defaults(fonction=_commande_auditer)
    propose = commandes.add_parser("proposer", help="propose un projet d'après les données")
    propose.add_argument("fichier")
    propose.add_argument("-c", _option("cible"), help="colonne à prédire (sinon proposée)")
    propose.add_argument(
        _option("sans-cible"), action="store_true", help="projet sans cible (regroupement, anomalies)"
    )
    propose.add_argument("-s", _option("sortie"), help="fichier où écrire le projet proposé")
    propose.set_defaults(fonction=_commande_proposer)
    return analyseur


def principal(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée : 0 réussi, 1 projet refusé ou différent de l'attendu, 2 usage."""
    for flux in (sys.stdout, sys.stderr):
        # accents et signes lisibles sous Windows ; un flux déjà détourné (tests, redirection) reste tel quel
        with contextlib.suppress(AttributeError, OSError, ValueError):
            flux.reconfigure(encoding="utf-8", errors="replace")
    options = _analyseur().parse_args(argv)
    try:
        return options.fonction(options)
    except (
        ErreurProjet,
        ErreurDonnees,
        ErreurExport,
        TransitionRefusee,
        train.ErreurEntrainement,
        ValueError,
        KeyError,
    ) as erreur:
        message = erreur.args[0] if isinstance(erreur, KeyError) and erreur.args else erreur
        if erreur_etrangere(erreur):
            message = message_utilisateur(erreur)
        print(f"Erreur : {message}", file=sys.stderr)
        return 1
    except Exception as erreur:
        if not erreur_etrangere(erreur):
            raise
        print(f"Erreur : {message_utilisateur(erreur)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(principal())
