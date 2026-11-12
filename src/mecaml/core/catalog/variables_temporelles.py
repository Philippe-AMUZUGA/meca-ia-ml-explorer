"""Variables temporelles : retards, différences, fenêtres glissantes, moyenne exponentielle, fréquences.

Le calcul est celui de `mecaml.core.features_ts` (par unité, sans le futur). Le transformeur reçoit
la colonne de date, la colonne d'unité et les colonnes à transformer : `preprocess` les lui transmet,
elles ne se choisissent pas ici. Réservé aux structures chronologiques.
"""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import exiger
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.features_ts import fabrique_variables_temporelles
from mecaml.core.types import Probleme

CHRONOLOGIQUE_VARIABLES_TEMPORELLES = "chronologique.variables_temporelles"


def _fabrique(graine: int, **parametres: Any) -> Any:
    exiger(CHRONOLOGIQUE_VARIABLES_TEMPORELLES, parametres, ("colonne_date",))
    return fabrique_variables_temporelles(graine, **parametres)


ENTREES = (
    EntreeCatalogue(
        id=CHRONOLOGIQUE_VARIABLES_TEMPORELLES,
        problemes=frozenset({Probleme.REGRESSION, Probleme.CLASSIFICATION, Probleme.SANS_CIBLE}),
        fabrique=_fabrique,
        grille_bornee=(
            ("retards", ((1,), (1, 2), (1, 2, 3))),
            ("fenetres", ((), (5,), (3, 5))),
            ("differences", ((), (1,))),
        ),
        cout=Cout.LEGER,
        ref_livre=ref(CHRONOLOGIQUE_VARIABLES_TEMPORELLES),
        texte_id=CHRONOLOGIQUE_VARIABLES_TEMPORELLES,
    ),
)
