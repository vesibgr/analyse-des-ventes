Ce projet a été réalisé dans le cadre de la formation [Data Analyst](https://openclassrooms.com/fr/paths/65-data-analyst), sur la plateforme OpenClassrooms.

Voir le fichier [présentation.pdf](présentation.pdf), utilisé comme support lors de la soutenance.

## Sujet

Analyse des ventes d'une entreprise fictive de vente de livres.

## Organisation des sources

Le dossier contient :

*3 dossiers*
- `1_dataset` : les données initiales du projet
- `3_clean_dataset` : les données nettoyées du projet 
- `graphiques` : l'ensemble des graphiques du projet, numérotés dans l'ordre dans lesquelles ils sont présentés lors de la soutenance.
    
2 *jupyter notebooks*
- `2_preparation_de_donnees.ipynb` : récupère les données de `1_dataset`, les nettoie, et enregistre le résultat dans `3_clean_dataset`
- `4_analyse_de_donnees.ipynb` : analyse les données qu'il y a dans `3_clean_dataset`

*Une version exporté de ces notebooks au format `.py` est également disponible.*

- un fichier `functions.py` qui contient des fonctions utilisées par les notebooks.

-----------------------------------------------------------------------------------

Pour lancer les notebooks :
- lancer la commande `jupyter-notebook` dans une console positionné dans le dossier parent.

Pour lancer les scripts Python :
- lancer la commande `python {nom_du_script}` dans une console.
