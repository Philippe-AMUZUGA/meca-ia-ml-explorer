"""
MécaML : cœur de l'application, sans aucune dépendance d'interface.

Le cœur charge, audite, découpe, prétraite, entraîne, évalue et exporte. Il
n'importe ni tkinter ni `mecaml.ui` (vérifié par un test d'architecture), et
scikit-learn n'y est importé qu'à l'usage, dans les fonctions.
"""
