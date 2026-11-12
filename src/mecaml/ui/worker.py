"""
MécaML : pont entre le fil de calcul du cœur et la boucle Tk.

Tout calcul tourne dans une `Tache` du cœur (un fil, annulable). La boucle Tk n'attend jamais : le
`Pilote` interroge la tâche toutes les 60 ms par `after`, met à jour la barre de progression, puis rend
le résultat ou l'erreur à l'interface, toujours dans le fil de Tk.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from typing import Any

from mecaml.core.session import TransitionRefusee
from mecaml.core.tasks import EtatTache, Fonction, Progression, Tache, erreur_etrangere, message_utilisateur

__all__ = ["INTERVALLE_MS", "Pilote", "message_d_erreur"]

INTERVALLE_MS = 60
journal = logging.getLogger(__name__)


def message_d_erreur(erreur: BaseException, etape: str | None = None) -> str:
    """Texte affiché pour une erreur : le motif du cœur, ou un message français si une bibliothèque a levé."""
    if isinstance(erreur, TransitionRefusee):
        return erreur.motif[:1].upper() + erreur.motif[1:]
    if erreur_etrangere(erreur):
        return message_utilisateur(erreur, etape)
    texte = str(erreur).strip()
    return texte or type(erreur).__name__


class Pilote:
    """Suit une tâche à la fois et rend la main à l'interface à sa fin."""

    def __init__(
        self,
        racine: tk.Misc,
        sur_progression: Callable[[float, str], None],
        sur_occupation: Callable[[bool], None],
    ) -> None:
        self._racine = racine
        self._sur_progression = sur_progression
        self._sur_occupation = sur_occupation
        self.tache: Tache | None = None
        self._au_succes: Callable[[Any], None] | None = None
        self._en_echec: Callable[[BaseException], None] | None = None
        self._annulee: Callable[[], None] | None = None
        self._horloge: str | None = None

    @property
    def occupe(self) -> bool:
        return self.tache is not None

    def lancer(
        self,
        fonction: Fonction[Any],
        *,
        nom: str,
        au_succes: Callable[[Any], None],
        en_echec: Callable[[BaseException], None],
        si_annulee: Callable[[], None] | None = None,
    ) -> Tache:
        """Démarre le calcul dans un fil ; `au_succes(valeur)` ou `en_echec(erreur)` s'exécutent dans Tk."""
        if self.tache is not None:
            raise RuntimeError("Un calcul est déjà en cours.")
        self._au_succes, self._en_echec, self._annulee = au_succes, en_echec, si_annulee
        self.tache = Tache(fonction, nom=nom).demarrer()
        self._sur_occupation(True)
        self._sur_progression(0.0, nom)
        self._horloge = self._racine.after(INTERVALLE_MS, self._sonder)
        return self.tache

    def annuler(self) -> None:
        """Demande l'arrêt du calcul ; le fil s'arrête entre deux unités de travail."""
        if self.tache is not None:
            self.tache.annuler()

    def _sonder(self) -> None:
        tache = self.tache
        if tache is None:
            return
        avancement: Progression | None = tache.progression()
        if avancement is not None:
            message = avancement.message or avancement.etape
            self._sur_progression(avancement.fraction, message)
        if not tache.terminee:
            self._horloge = self._racine.after(INTERVALLE_MS, self._sonder)
            return
        self._terminer(tache)

    def _terminer(self, tache: Tache) -> None:
        self.tache = None
        self._horloge = None
        self._sur_occupation(False)
        au_succes, en_echec, annulee = self._au_succes, self._en_echec, self._annulee
        if tache.etat is EtatTache.TERMINEE and au_succes is not None:
            self._sur_progression(0.0, "")
            au_succes(tache.resultat())
        elif tache.etat is EtatTache.ANNULEE:
            self._sur_progression(0.0, "")
            if annulee is not None:
                annulee()
        elif en_echec is not None:
            journal.info("calcul en échec : %s", tache.trace)
            self._sur_progression(0.0, "")
            en_echec(tache.erreur or RuntimeError("Calcul interrompu."))
