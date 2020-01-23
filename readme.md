*Ce projet a été réalisé dans le cadre de la formation [Data Analyst](https://openclassrooms.com/fr/paths/65-data-analyst), sur la plateforme OpenClassrooms.*

### Introduction

Dans le cadre de ce projet, j'ai réalisé une analyse des données d'une plateforme de e-commerce fictive afin de déterminer de nouvelles stratégies marketing. Pour ce faire, j'ai d'abord restructuré et nettoyé les données issues du SI et les ai analysées selon trois axes.

J'ai étudié l'évolution du chiffre d'affaires au cours du temps afin de déterminer des effets saisonniers et d'éventuels produits régulateurs.

J'ai analysé la distribution des catégories de produits et la proportion des ventes de chaque produit (courbe de Lorenz, loi des 80-20).

Enfin, j'ai examiné les corrélations entre l'âge, le sexe, les dépenses, les fréquences d'achats et le panier des clients. Cela m'a permis d'établir trois profils types de clients détaillés et de suggérer des stratégies de ventes.

<br>

:arrow_forward: [Slides de présentation](présentation.pdf)

:notebook: [Notebook : Nettoyage des données](/2_preparation_de_donnees.ipynb)

:notebook: [Notebook : Analyse des données](/4_analyse_de_donnees.ipynb)

<br>

### Organisation des sources

3 dossiers :
- `1_dataset` : les données initiales du projet
- `3_clean_dataset` : les données nettoyées du projet 
- `graphiques` : l'ensemble des graphiques du projet, numérotés dans l'ordre dans lesquelles ils sont présentés lors de la soutenance.
    
2 jupyter notebooks :
- `2_preparation_de_donnees.ipynb` : récupère les données de `1_dataset`, les nettoie, et enregistre le résultat dans `3_clean_dataset`
- `4_analyse_de_donnees.ipynb` : analyse les données qu'il y a dans `3_clean_dataset`

*Une version exporté de ces notebooks au format `.py` est également disponible.*

- un fichier `functions.py` qui contient des fonctions utilisées par les notebooks.

-----------------------------------------------------------------------------------

Pour lancer les notebooks :
- lancer la commande `jupyter-notebook` dans une console positionné dans le dossier parent.

Pour lancer les scripts Python :
- lancer la commande `python {nom_du_script}` dans une console.
