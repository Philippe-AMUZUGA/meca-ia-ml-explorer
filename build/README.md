# Empaquetage de MécaML Explorer

Ce dossier réunit ce qu'il faut pour livrer l'application de trois façons. Seules la spécification, les scripts et cette procédure sont versionnés : les dossiers `dist/` et `build/travail/` produits par une construction sont ignorés.

## 1. Lancer depuis les sources

`Lancer_MecaML.bat`, à la racine du dépôt, cherche Python 3.12 (le lanceur `py`, puis `python`), vérifie que les bibliothèques sont là et ouvre la fenêtre, sans rien installer. Sans Python 3.12, il dit quoi installer. Sans bibliothèque, l'application dit lesquelles manquent et la commande qui les installe.

```bat
Lancer_MecaML.bat
python -m mecaml
```

L'option `-t` ouvre la fenêtre, mène le jeu d'acier fourni en exemple jusqu'à l'export, puis la ferme : c'est l'essai de bon fonctionnement. Le code de retour est 0 s'il a réussi. `-s dossier` garde l'export au lieu de l'effacer, `-V` donne la version.

## 2. Installer le paquet

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install .
mecaml
```

La roue contient les textes (`mecaml/textes/*.toml`) et les icônes. Les jeux d'exemple restent dans le dépôt : pour les voir à l'accueil, lancer `mecaml` depuis le dossier du dépôt.

`python build/essai_environnement_propre.py` rejoue cette installation dans un environnement virtuel temporaire, sans réseau, depuis une copie du dépôt, puis lance `mecaml -t`. Il vérifie que le paquet, ses textes et ses icônes viennent bien de l'environnement neuf. Limite : l'environnement voit les bibliothèques déjà installées sur le poste, au lieu de les réinstaller.

## 3. Exécutable autonome (PyInstaller)

Un seul fichier, `dist/MecaML.exe`, qui n'a besoin ni de Python ni de pip. La spécification est `build/mecaml.spec` : elle emporte les textes TOML, les icônes, les jeux et le projet d'exemple de `examples/`, et le pilote Tk de matplotlib.

```bat
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pip install pyinstaller
py -3.12 build\construire.py
```

La seconde commande est la seule qui demande le réseau. `build/construire.py` rend le dossier `src` visible à PyInstaller (le paquet n'a pas besoin d'être installé pour que ses modules soient listés), construit l'exécutable, mesure sa taille, puis lance `MecaML.exe -t` : un code de retour 0 et un `projet.json` exporté disent que l'exécutable fonctionne seul. Un exécutable de cette sorte pèse plusieurs dizaines de mégaoctets : au delà de 5 Mo, il ne s'ajoute pas au dépôt et se livre en archive séparée.

## 4. Jeux d'exemple

`python build/preparer_exemples.py <dossier du dépôt du livre>` relit les quatre jeux dans le dépôt du livre, refuse tout fichier dont l'empreinte SHA-256 diffère de celle de son `manifest.json`, et écrit `examples/empreintes.json`. Avec `-p`, il refait aussi le projet exporté de `examples/projet_exemple/`.
