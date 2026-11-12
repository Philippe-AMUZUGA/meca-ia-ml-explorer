"""Forêt d'isolement : détection d'anomalies par isolement aléatoire."""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import preparer
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.types import Probleme

SANS_CIBLE_FORET_ISOLEMENT = "sans_cible.foret_isolement"


def _fabrique(graine: int, **parametres: Any) -> Any:
    from sklearn.ensemble import IsolationForest

    return IsolationForest(
        **preparer(
            SANS_CIBLE_FORET_ISOLEMENT,
            {"n_estimators": 100, "contamination": "auto", "n_jobs": 1},
            parametres,
            graine,
        )
    )


ENTREES = (
    EntreeCatalogue(
        id=SANS_CIBLE_FORET_ISOLEMENT,
        problemes=frozenset({Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(("contamination", (0.01, 0.05, 0.1)), ("n_estimators", (50, 100, 200))),
        cout=Cout.MOYEN,
        ref_livre=ref(SANS_CIBLE_FORET_ISOLEMENT),
        texte_id=SANS_CIBLE_FORET_ISOLEMENT,
    ),
)
