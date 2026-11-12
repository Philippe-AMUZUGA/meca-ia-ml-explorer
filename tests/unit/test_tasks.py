"""Calcul en fil séparé : cycle de vie, annulation entre deux unités, erreurs gardées, avancement."""

from __future__ import annotations

import threading
import time
import warnings

import pytest

from mecaml.core.tasks import (
    Annulation,
    ErreurAnnulation,
    ErreurAnnulee,
    EtatTache,
    Progression,
    Tache,
    collecter_avertissements,
    erreur_etrangere,
    message_utilisateur,
    precharger_scikit_learn,
)


def _compter(n: int, pas_s: float = 0.0):
    def calcul(annulation: Annulation, avancer):
        for k in range(1, n + 1):
            annulation.verifier()
            avancer(Progression("boucle", k, n, f"unité {k}"))
            time.sleep(pas_s)
        return n

    return calcul


def test_cycle_de_vie_nominal() -> None:
    tache = Tache(_compter(5), nom="essai")
    assert tache.etat is EtatTache.PRETE and not tache.terminee
    assert tache.demarrer() is tache
    assert tache.attendre(10) is True
    assert tache.etat is EtatTache.TERMINEE and tache.terminee
    assert tache.resultat() == 5
    assert tache.erreur is None


def test_une_tache_ne_se_lance_qu_une_fois() -> None:
    tache = Tache(_compter(1)).demarrer()
    tache.attendre(10)
    with pytest.raises(RuntimeError, match="déjà lancée"):
        tache.demarrer()


def test_avancement_dernier_et_evenements_videes() -> None:
    tache = Tache(_compter(4)).demarrer()
    tache.attendre(10)
    derniere = tache.progression()
    assert derniere == Progression("boucle", 4, 4, "unité 4")
    assert derniere.fraction == 1.0
    evenements = tache.evenements()
    assert [e.fait for e in evenements] == [1, 2, 3, 4]
    assert tache.evenements() == []  # la liste est vidée à chaque lecture
    assert Progression("x", 3, 0).fraction == 0.0
    assert Progression("x", 9, 4).fraction == 1.0


def test_annulation_entre_deux_unites() -> None:
    """La demande d'arrêt est prise en compte à la prochaine vérification, jamais au milieu d'une unité."""
    fait = []
    pret = threading.Event()

    def calcul(annulation: Annulation, avancer):
        for k in range(1000):
            annulation.verifier()
            fait.append(k)
            pret.set()
            time.sleep(0.01)
        return "fini"

    tache = Tache(calcul, nom="long").demarrer()
    assert pret.wait(5)
    tache.annuler()
    assert tache.attendre(5) is True
    assert tache.etat is EtatTache.ANNULEE
    assert len(fait) < 1000
    assert isinstance(tache.erreur, ErreurAnnulation)
    with pytest.raises(ErreurAnnulation):
        tache.resultat()


def test_annulation_avant_le_premier_pas() -> None:
    jeton = Annulation()
    assert not jeton.demandee
    jeton.demander()
    assert jeton.demandee
    with pytest.raises(ErreurAnnulation, match="annulé"):
        jeton.verifier()


def test_erreur_du_calcul_gardee_et_relue() -> None:
    def calcul(annulation: Annulation, avancer):
        raise ValueError("données incohérentes")

    tache = Tache(calcul).demarrer()
    tache.attendre(5)
    assert tache.etat is EtatTache.ECHEC and tache.terminee
    assert isinstance(tache.erreur, ValueError)
    assert "ValueError" in tache.trace and "données incohérentes" in tache.trace
    with pytest.raises(ValueError, match="incohérentes"):
        tache.resultat()


def test_resultat_avant_la_fin_est_refuse() -> None:
    liberer = threading.Event()

    def calcul(annulation: Annulation, avancer):
        liberer.wait(5)
        return 1

    tache = Tache(calcul).demarrer()
    assert tache.etat is EtatTache.EN_COURS
    with pytest.raises(RuntimeError, match="non terminée"):
        tache.resultat()
    liberer.set()
    tache.attendre(5)
    assert tache.resultat() == 1


def test_le_fil_est_un_demon_nomme() -> None:
    """Fermer l'application ne reste pas suspendu à un calcul en cours."""
    tache = Tache(_compter(1), nom="nomme")
    assert tache._fil.daemon and tache._fil.name == "mecaml-nomme"


def test_precharger_scikit_learn() -> None:
    tache = precharger_scikit_learn()
    assert tache.attendre(60) is True
    assert tache.etat is EtatTache.TERMINEE
    assert [e.fait for e in tache.evenements()] == [1, 2, 3, 4]


# ============================================================
# UN SEUL MÉCANISME D'ANNULATION POUR TOUT LE CŒUR
# ============================================================
def test_le_jeton_est_aussi_un_appelable_pour_la_lecture_des_fichiers() -> None:
    jeton = Annulation()
    assert jeton() is False and not jeton.demandee
    jeton.demander()
    assert jeton() is True and jeton.demandee


def test_un_chargement_annule_dans_une_tache_est_note_annule_non_echec(tmp_path) -> None:
    from mecaml.core import guards, io

    csv = tmp_path / "jeu.csv"
    csv.write_text("a;b\n" + "\n".join(f"{k};{2 * k}" for k in range(200)), encoding="utf-8")
    tache = Tache(lambda annulation, avancer: io.charger(csv, annuler=annulation), nom="lecture")
    tache.annuler()  # l'arrêt est demandé avant la première vérification de la lecture
    tache.demarrer()
    assert tache.attendre(10)
    assert tache.etat is EtatTache.ANNULEE, tache.trace
    assert isinstance(tache.erreur, ErreurAnnulation)
    assert ErreurAnnulee is ErreurAnnulation is guards.ErreurAnnulee  # une seule classe d'annulation


def test_un_appelable_simple_arrete_aussi_la_lecture_et_la_tache_est_annulee(tmp_path) -> None:
    from mecaml.core import io

    csv = tmp_path / "jeu.csv"
    csv.write_text("a;b\n" + "\n".join(f"{k};{2 * k}" for k in range(200)), encoding="utf-8")
    tache = Tache(lambda annulation, avancer: io.charger(csv, annuler=lambda: True), nom="lecture")
    tache.demarrer()
    assert tache.attendre(10)
    assert tache.etat is EtatTache.ANNULEE, tache.trace
    assert type(tache.erreur) is ErreurAnnulation


# ============================================================
# AVERTISSEMENTS : RASSEMBLÉS FIL PAR FIL, SANS ÉTAT GLOBAL À RESTAURER
# ============================================================
def _emettre(message: str) -> None:
    """Avertissement comme en émet scikit-learn : attribué à un module `sklearn`."""
    warnings.warn_explicit(message, UserWarning, "x.py", 1, module="sklearn.fictif")


def test_les_collecteurs_de_deux_fils_ne_se_melangent_pas() -> None:
    barriere = threading.Barrier(2)
    resultats: dict[str, list[str]] = {}

    def travail(nom: str) -> None:
        with collecter_avertissements() as captes:
            barriere.wait(10)
            for k in range(25):
                _emettre(f"{nom} {k}")
                time.sleep(0.001)
        resultats[nom] = list(captes)

    fils = [threading.Thread(target=travail, args=(nom,)) for nom in ("A", "B")]
    for fil in fils:
        fil.start()
    for fil in fils:
        fil.join(30)
    for nom in ("A", "B"):
        assert resultats[nom] == [f"UserWarning : {nom} {k}" for k in range(25)]


def test_collecter_n_ajoute_aucun_filtre_a_chaque_appel_et_rend_la_main() -> None:
    with collecter_avertissements():
        pass
    filtres = list(warnings.filters)
    for _ in range(5):
        with collecter_avertissements() as captes:
            _emettre("encore")
        assert captes == ["UserWarning : encore"]
    assert list(warnings.filters) == filtres, "les filtres globaux ont bougé"
    # hors d'un collecteur, l'avertissement n'est pas avalé
    with pytest.warns(UserWarning, match="visible"):
        _emettre("visible")


def test_les_collecteurs_s_imbriquent() -> None:
    with collecter_avertissements() as externe:
        _emettre("avant")
        with collecter_avertissements() as interne:
            _emettre("dedans")
        _emettre("apres")
    assert interne == ["UserWarning : dedans"]
    assert externe == ["UserWarning : avant", "UserWarning : apres"]


def test_un_chargement_et_un_entrainement_simultanes_sans_interference(tmp_path) -> None:
    """Deux tâches à la fois : la lecture ne reçoit aucun avertissement, l'entraînement garde les siens."""
    import aides_apprentissage as aides
    from mecaml.core import io, train
    from mecaml.core.types import ChoixTechnique, Table

    csv = tmp_path / "gros.csv"
    csv.write_text("a;b\n" + "\n".join(f"{k};{k % 7}" for k in range(20000)), encoding="utf-8")
    df = aides.jeu_regression()
    table = Table(df, "regression")
    spec = aides.spec_regression(modeles=(ChoixTechnique.creer("regression.reseau_neurones", max_iter=3),))
    coffre = aides.coffre_par_pas(table, spec)
    lecture = Tache(lambda annulation, avancer: io.charger(csv, annuler=annulation), nom="lecture")
    entrainement = train.lancer_entrainement(spec, table, coffre)
    lecture.demarrer()
    assert lecture.attendre(60) and entrainement.attendre(60)
    assert lecture.etat is EtatTache.TERMINEE and entrainement.etat is EtatTache.TERMINEE
    reseau = entrainement.resultat().modele("regression.reseau_neurones")
    assert reseau.ok and any("ConvergenceWarning" in a for a in reseau.avertissements)
    assert lecture.resultat().table.n_lignes == 20000


# ============================================================
# ERREURS : CELLES DU CŒUR DISENT LE MOTIF, CELLES D'UNE BIBLIOTHÈQUE SE DISENT EN FRANÇAIS
# ============================================================
def _erreur_de_scikit_learn() -> BaseException:
    import numpy as np
    from sklearn.metrics import confusion_matrix

    vrai = np.array([0, 1, 1, 0])
    predit = np.array([0, 1, None, 0], dtype=object)  # le tableau d'objets d'avant la correction
    try:
        confusion_matrix(vrai, predit)
    except Exception as erreur:
        return erreur
    raise AssertionError("scikit-learn aurait dû refuser ce tableau")


def test_une_erreur_de_scikit_learn_est_etrangere_et_se_dit_en_francais() -> None:
    erreur = _erreur_de_scikit_learn()
    assert erreur_etrangere(erreur)
    texte = message_utilisateur(erreur, "Diagnostic")
    assert "Diagnostic" in texte and type(erreur).__name__ in texte
    assert "colonne cible" in texte
    assert not any(mot in texte.lower() for mot in ("classification metrics", "targets", "unknown"))
    assert "le calcul" in message_utilisateur(erreur)


def test_une_erreur_du_coeur_n_est_pas_etrangere() -> None:
    from mecaml.core import evaluate
    from mecaml.core.train import ErreurEntrainement

    try:
        raise ErreurEntrainement("motif en français")
    except ErreurEntrainement as erreur:
        assert not erreur_etrangere(erreur)  # une classe du paquet
    try:
        evaluate.predire_au_seuil([0.1], ("a", "b", "c"), "a", 0.5)
    except ValueError as erreur:  # une ValueError nue, levée par le cœur : son texte est français
        assert not erreur_etrangere(erreur)
    try:
        raise ValueError("levée par un module extérieur au paquet")
    except ValueError as erreur:
        assert erreur_etrangere(erreur)
