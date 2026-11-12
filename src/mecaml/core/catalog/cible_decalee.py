"""Cible décalée (autorégression) : les valeurs passées de la cible deviennent des variables.

Seuls les retards sont permis : une moyenne glissante, une différence ou une moyenne exponentielle de la
cible contiendrait la valeur courante, donc la réponse. Les retards sont calculés par unité et sans le
futur, par le transformeur de `mecaml.core.features_ts`. Le choix n'est permis que si la valeur passée de
la cible est connue au moment de la prévision, ce qu'il déclare par `cible_connue=True` (sinon il est
refusé, voir `mecaml.core.cible_passee`) ; une cible qui décompte, comme une durée de vie restante, est
refusée même déclarée.
"""

from __future__ import annotations

from typing import Any

from mecaml.core.catalog._outils import exiger
from mecaml.core.catalog._refs import ref
from mecaml.core.catalog.entree import Cout, EntreeCatalogue
from mecaml.core.cible_passee import CIBLE_DECALEE, PARAMETRE, exiger_declaration
from mecaml.core.features_ts import fabrique_variables_temporelles
from mecaml.core.types import Probleme

CHRONOLOGIQUE_CIBLE_DECALEE = CIBLE_DECALEE
INTERDITS = ("fenetres", "differences", "demi_vies", "fenetres_frequence")


def _fabrique(graine: int, **parametres: Any) -> Any:
    exiger_declaration(CHRONOLOGIQUE_CIBLE_DECALEE, parametres)
    reglages = {nom: valeur for nom, valeur in parametres.items() if nom != PARAMETRE}
    exiger(CHRONOLOGIQUE_CIBLE_DECALEE, reglages, ("colonne_date", "colonnes"))
    interdits = [nom for nom in INTERDITS if reglages.get(nom)]
    if interdits:
        raise ValueError(
            f"{CHRONOLOGIQUE_CIBLE_DECALEE} : {', '.join(interdits)} inclurait la valeur courante "
            "de la cible, donc la réponse ; seuls les retards sont permis."
        )
    return fabrique_variables_temporelles(graine, **{"retards": (1,), **reglages})


ENTREES = (
    EntreeCatalogue(
        id=CHRONOLOGIQUE_CIBLE_DECALEE,
        problemes=frozenset({Probleme.REGRESSION}),
        fabrique=_fabrique,
        grille_bornee=(("retards", ((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4, 5))),),
        cout=Cout.LEGER,
        ref_livre=ref(CHRONOLOGIQUE_CIBLE_DECALEE),
        texte_id=CHRONOLOGIQUE_CIBLE_DECALEE,
    ),
)
