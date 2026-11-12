"""Fenêtre et parcours partagés par les tests de fumée de l'interface."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import aides_interface as ai
from mecaml.core.session import Etat
from mecaml.ui.app import Application

EXEMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def app() -> Iterator[Application]:
    fenetre = ai.ouvrir_application()
    yield fenetre
    ai.fermer(fenetre)


@pytest.fixture(scope="module")
def instantanes(app: Application, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Parcours de l'exemple d'acier : l'état du contrôleur après chaque étape, export compris."""
    pris: dict[Etat, ai.Instantane] = {}
    ai.mener_jusqu_a(
        app,
        EXEMPLES / "MecaIA-Traitement_Acier.xlsx",
        Etat.TESTE,
        apres_etape=lambda etat: pris.__setitem__(etat, ai.prendre(app)),
    )
    ai.exporter(app, tmp_path_factory.mktemp("export_acier"))
    pris[Etat.EXPORTE] = ai.prendre(app)
    return pris
