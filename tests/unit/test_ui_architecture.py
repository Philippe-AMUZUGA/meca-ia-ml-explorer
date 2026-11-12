"""Architecture de l'interface : aucune règle de calcul dans `ui/`, aucun accès réseau, fonctions courtes."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mecaml.ui

UI = Path(mecaml.ui.__file__).resolve().parent
SOURCES = sorted(UI.rglob("*.py"))
LIGNES_MAX_D_UNE_FONCTION = 60

# Bibliothèques de calcul et d'accès au monde : l'interface passe par le cœur pour tout cela.
IMPORTS_INTERDITS = (
    "sklearn", "scipy", "joblib", "pickle", "shelve", "marshal", "dill", "cloudpickle",
    "socket", "ssl", "urllib", "http", "ftplib", "smtplib", "requests", "xmlrpc", "webbrowser",
    "subprocess", "multiprocessing", "importlib", "shutil", "tempfile",
)  # fmt: skip
# Appels qui seraient une règle de calcul ou un chargement dangereux.
APPELS_INTERDITS = (
    "fit", "fit_transform", "predict", "predict_proba", "transform", "score", "train_test_split",
    "cross_val_score", "cross_validate", "eval", "exec", "compile", "__import__", "read_pickle",
    "load", "loads",
)  # fmt: skip


def _arbre(chemin: Path) -> ast.Module:
    return ast.parse(chemin.read_text(encoding="utf-8"))


def _imports(arbre: ast.Module) -> set[str]:
    noms: set[str] = set()
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            noms.update(a.name for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom) and noeud.module:
            noms.add(noeud.module)
    return noms


@pytest.mark.parametrize("chemin", SOURCES, ids=lambda p: p.relative_to(UI).as_posix())
def test_aucun_import_de_calcul_ni_de_reseau(chemin: Path) -> None:
    fautifs = sorted(
        n for n in _imports(_arbre(chemin)) if any(n == i or n.startswith(f"{i}.") for i in IMPORTS_INTERDITS)
    )
    assert fautifs == [], f"{chemin.name} importe {fautifs} : passer par le cœur"


@pytest.mark.parametrize("chemin", SOURCES, ids=lambda p: p.relative_to(UI).as_posix())
def test_aucun_calcul_d_apprentissage_dans_l_interface(chemin: Path) -> None:
    appels = []
    for noeud in ast.walk(_arbre(chemin)):
        if isinstance(noeud, ast.Call):
            f = noeud.func
            nom = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if isinstance(f, ast.Attribute) and ast.unparse(f.value) in ("tomllib", "re"):
                continue  # textes de l'interface (fichier du paquet) et expressions régulières
            if nom in APPELS_INTERDITS:
                appels.append(f"{nom} (ligne {noeud.lineno})")
    assert appels == [], f"{chemin.name} calcule ou charge : {appels}"


@pytest.mark.parametrize("chemin", SOURCES, ids=lambda p: p.relative_to(UI).as_posix())
def test_fonctions_courtes(chemin: Path) -> None:
    longues = [
        f"{n.name} ({n.end_lineno - n.lineno + 1} lignes)"
        for n in ast.walk(_arbre(chemin))
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and n.end_lineno - n.lineno + 1 > LIGNES_MAX_D_UNE_FONCTION
    ]
    assert longues == []


@pytest.mark.parametrize("chemin", SOURCES, ids=lambda p: p.relative_to(UI).as_posix())
def test_aucune_boucle_d_attente_dans_l_interface(chemin: Path) -> None:
    """La boucle Tk n'attend jamais : ni `sleep`, ni fenêtre bloquante, ni `mainloop` hors de `app.py`."""
    interdits = []
    for noeud in ast.walk(_arbre(chemin)):
        if isinstance(noeud, ast.Call) and isinstance(noeud.func, ast.Attribute):
            if noeud.func.attr in ("sleep", "wait_window", "wait_variable", "wait_visibility", "grab_set"):
                interdits.append(f"{noeud.func.attr} (ligne {noeud.lineno})")
            if noeud.func.attr == "attendre":
                interdits.append(f"attendre (ligne {noeud.lineno})")
            if noeud.func.attr == "mainloop" and chemin.name != "app.py":
                interdits.append(f"mainloop (ligne {noeud.lineno})")
    assert interdits == []


def test_le_coeur_n_importe_pas_l_interface() -> None:
    coeur = UI.parent / "core"
    fautifs = [
        p.name
        for p in coeur.rglob("*.py")
        if any(n == "mecaml.ui" or n.startswith("mecaml.ui.") for n in _imports(_arbre(p)))
    ]
    assert fautifs == []


def test_l_interface_ne_lit_aucune_valeur_du_jeu_de_test() -> None:
    """Aucune lecture du coffre en dehors de l'action de la session : ni `ouvrir`, ni positions de test."""
    interdits = ("ouvrir", "lignes_test", "positions_test", "empreinte_test")
    trouvees = []
    for chemin in SOURCES:
        for noeud in ast.walk(_arbre(chemin)):
            suspect = isinstance(noeud, ast.Attribute) and noeud.attr in interdits
            if suspect and ast.unparse(noeud.value).endswith("coffre"):
                trouvees.append(f"{chemin.name}:{noeud.lineno} {noeud.attr}")
    assert trouvees == []
