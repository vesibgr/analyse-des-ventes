Ce projet a été réalisé dans le cadre de la formation [Data Analyst](https://openclassrooms.com/fr/paths/65-data-analyst), sur la plateforme OpenClassrooms.

## Scénario

> Vous êtes data analyst d'une grande chaîne de librairie, fraîchement embauché depuis une semaine ! Vous avez fait connaissance avec vos collègues, votre nouveau bureau, mais surtout, la machine à café high-tech ! Mais revenons à votre mission : il est temps de mettre les mains dans le cambouis ! Le service Informatique vous a donné l’accès à la base de données des ventes. À vous de vous familiariser avec les données, et de les analyser. Votre manager souhaite que vous réalisiez une présentation pour vous "faire la main". Comme vous l'avez appris dans vos recherches avant de postuler, votre entreprise, "Rester livres" s'est d'abord développée dans une grande ville de France, avec plusieurs magasins, jusqu'à décider d'ouvrir une boutique en ligne. Son approche de la vente de livres en ligne, basée sur des algorithmes de recommandation, lui a valu un franc succès !

<br />

:arrow_forward: [Voir ma présentation des résultats](présentation.pdf).

---------------------------------------------------------------------------------

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
