# Jeux d'exemple

Quatre petits classeurs pour essayer l'application et pour les tests. Le total reste sous 1 Mo.

| Fichier | Problème | Lignes × colonnes | Cible | Origine |
| :- | :- | :- | :- | :- |
| `MecaIA-Traitement_Acier.xlsx` | régression | 150 × 6 | `resistance_traction_MPa` | Iris (Fisher, 1936), UCI Machine Learning Repository, CC BY 4.0, transformé en paramètres de traitement thermique |
| `MecaIA-Controle_Conformite.xlsx` | classification binaire | 569 × 32 | `etat_piece` | Breast Cancer Wisconsin Diagnostic (UCI, CC BY 4.0), renommé en descripteurs de défaut, identifiant `id_piece` ajouté |
| `MecaIA-TimeSeries_RUL_extrait.xlsx` | régression chronologique et groupée | 3 881 × 10 | `Duree_Vie_Restante_Cycles` | jeu construit : vingt premières machines de la simulation de cent machines suivies jusqu'à la défaillance |
| `MecaIA-Anomalies_Pompe.xlsx` | sans cible (anomalies) | 276 × 4 | aucune | jeu construit : cycles de charge d'une pompe de procédé |

Chaque classeur se charge tel quel : l'application propose les rôles des colonnes (identifiant, date, groupe, variable, cible)
et signale ce qu'elle trouve, par exemple `id_piece` comme identifiant ou `ID_Machine` comme groupe à découper par unités entières.

Les empreintes SHA-256 de ces quatre classeurs et celles des jeux du dépôt du livre dont ils viennent sont dans `empreintes.json` ; le script `build/preparer_exemples.py` les refait en vérifiant chaque empreinte contre le `manifest.json` du dépôt du livre. Le jeu de séries chronologiques est un extrait : les machines 1 à 20 du jeu complet.

## Projet exporté d'exemple

`projet_exemple/` est l'export du jeu `MecaIA-Traitement_Acier.xlsx` mené jusqu'au test : `projet.json` (le projet rejouable, avec les scores attendus) et `rapport.xlsx` (le classeur des chiffres). Pour le rejouer sans l'interface, depuis la racine du dépôt :

```bat
python -m mecaml.cli rejouer examples\projet_exemple\projet.json -d examples\MecaIA-Traitement_Acier.xlsx -v
```

Ajouter `-o dossier` écrit aussi le rapport HTML, le modèle et le script Python équivalent.
