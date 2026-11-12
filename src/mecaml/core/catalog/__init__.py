"""
MécaML : registre explicite des techniques.

Chaque technique vit dans un module de ce paquet qui expose
`ENTREES: tuple[EntreeCatalogue, ...]`. Le registre est la liste de ces modules,
importés un par un en tête de ce fichier : aucune découverte automatique, pour que
l'exécutable autonome embarque exactement ce qui est listé ici.

Ajouter une technique :

1. écrire `catalog/<nom>.py`, qui importe `EntreeCatalogue` et `Cout` depuis
   `mecaml.core.catalog.entree` (jamais depuis ce paquet) et définit `ENTREES` ;
   scikit-learn s'importe dans la fabrique, jamais en tête ;
2. recenser son renvoi au livre dans la table unique `mecaml/core/renvois.py` (`TECHNIQUES`) ;
3. ajouter `<nom>` dans le bloc `from mecaml.core.catalog import (...)` ci-dessous,
   par ordre alphabétique, et dans `_MODULES` à la place voulue pour l'affichage.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType

from mecaml.core.catalog import (
    acp,
    adaboost,
    analyse_discriminante,
    arbre,
    arbre_texte,
    arbres_extremes,
    bagging,
    bayes_naif,
    cible_decalee,
    classification_hierarchique,
    courbe_apprentissage,
    courbe_validation,
    dbscan,
    discretisation,
    echelle_minmax,
    echelle_robuste,
    echelle_standard,
    ecretage_tukey,
    elastic_net,
    empilement,
    encodage_one_hot,
    encodage_ordinal,
    exclusion_unique,
    facteur_local,
    foret,
    foret_isolement,
    gradient_boosting,
    gradient_histogramme,
    importance_permutation,
    impurete,
    imputation_knn,
    imputation_simple,
    indicateur_manquants,
    kmeans,
    lasso,
    lineaire,
    logistique,
    melange_gaussien,
    persistance,
    plis_groupes,
    polynomiale,
    quantile,
    recherche_aleatoire,
    recherche_grille,
    reference_dummy,
    reseau_dense,
    reseau_neurones,
    ridge,
    selection_depuis_modele,
    selection_k_meilleures,
    selection_rfecv,
    selection_variance,
    serie_temporelle,
    svm,
    svm_une_classe,
    transformation_puissance,
    validation_croisee,
    variables_temporelles,
    voisins,
    vote,
)
from mecaml.core.catalog.entree import (
    FAMILLES,
    GRILLE_MAX_COMBINAISONS,
    Cout,
    EntreeCatalogue,
    Fabrique,
)
from mecaml.core.types import Probleme

__all__ = [
    "FAMILLES",
    "GRILLE_MAX_COMBINAISONS",
    "OPTIONNELLES",
    "REGISTRE",
    "Cout",
    "EntreeCatalogue",
    "Fabrique",
    "construire_registre",
    "entree",
    "entrees",
    "optionnelles_disponibles",
]

# Modules de techniques, dans l'ordre d'affichage : prétraitement, sélection, modèles, validation,
# techniques sans cible, explication.
_MODULES: tuple[ModuleType, ...] = (
    imputation_simple,
    imputation_knn,
    indicateur_manquants,
    encodage_one_hot,
    encodage_ordinal,
    echelle_standard,
    echelle_minmax,
    echelle_robuste,
    transformation_puissance,
    polynomiale,
    discretisation,
    ecretage_tukey,
    variables_temporelles,
    cible_decalee,
    selection_variance,
    selection_k_meilleures,
    selection_depuis_modele,
    selection_rfecv,
    lineaire,
    ridge,
    lasso,
    elastic_net,
    quantile,
    voisins,
    arbre,
    foret,
    arbres_extremes,
    bagging,
    gradient_histogramme,
    gradient_boosting,
    svm,
    reseau_neurones,
    logistique,
    bayes_naif,
    analyse_discriminante,
    adaboost,
    vote,
    empilement,
    reference_dummy,
    persistance,
    validation_croisee,
    plis_groupes,
    serie_temporelle,
    exclusion_unique,
    recherche_grille,
    recherche_aleatoire,
    courbe_apprentissage,
    courbe_validation,
    acp,
    kmeans,
    melange_gaussien,
    classification_hierarchique,
    dbscan,
    foret_isolement,
    facteur_local,
    svm_une_classe,
    importance_permutation,
    impurete,
    arbre_texte,
)


def construire_registre(modules: Iterable[ModuleType]) -> tuple[EntreeCatalogue, ...]:
    """Assemble les `ENTREES` des modules listés ; refuse les identifiants en double."""
    registre: list[EntreeCatalogue] = []
    vus: set[str] = set()
    for module in modules:
        entrees_module = getattr(module, "ENTREES", None)
        if not isinstance(entrees_module, tuple):
            raise TypeError(f"Registre : le module {module.__name__} n'expose pas ENTREES (tuple).")
        for e in entrees_module:
            if not isinstance(e, EntreeCatalogue):
                raise TypeError(f"Registre : entrée invalide dans {module.__name__}.")
            if e.id in vus:
                raise ValueError(f"Registre : identifiant en double {e.id!r}.")
            vus.add(e.id)
            registre.append(e)
    return tuple(registre)


REGISTRE: tuple[EntreeCatalogue, ...] = construire_registre(_MODULES)


# Techniques facultatives, hors du registre de base : jamais requises, utilisables seulement si leur
# bibliothèque est installée (réseaux denses Keras).
OPTIONNELLES: tuple[EntreeCatalogue, ...] = construire_registre((reseau_dense,))


def optionnelles_disponibles() -> tuple[EntreeCatalogue, ...]:
    """Techniques facultatives que cet environnement peut exécuter : vide sans Keras."""
    return OPTIONNELLES if reseau_dense.keras_present() else ()


def entrees(
    registre: Iterable[EntreeCatalogue] | None = None,
    *,
    probleme: Probleme | None = None,
    famille: str | None = None,
) -> tuple[EntreeCatalogue, ...]:
    """Entrées filtrées par problème et famille : registre de base, puis techniques facultatives."""
    source = (*REGISTRE, *optionnelles_disponibles()) if registre is None else tuple(registre)
    return tuple(
        e
        for e in source
        if (probleme is None or probleme in e.problemes) and (famille is None or e.famille == famille)
    )


def entree(id: str, registre: Iterable[EntreeCatalogue] | None = None) -> EntreeCatalogue:
    """Entrée d'identifiant donné ; `KeyError` explicite sinon."""
    for e in (*REGISTRE, *optionnelles_disponibles()) if registre is None else registre:
        if e.id == id:
            return e
    if registre is None and any(e.id == id for e in OPTIONNELLES):
        raise KeyError(f"{id} demande Keras, absent de cet environnement : l'installer ou retirer ce modèle.")
    raise KeyError(f"Technique inconnue du registre : {id!r}.")
