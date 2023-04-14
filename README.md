# Learning-captioning-model
Service d'étiquetage d'image qui peut être entrainé sur un ensemble d'image étiqueté.

## Description
Le conteneur learning-captioning-model permet d’entraîner un modèle de reconnaissance d’image pour décrire, avec des mots et des phrases, le contenu de l’image. Nous extrayons ensuite les mots clés de la phrase générée à partir d'un deuxième modèle de classification « Zero-Shot ». La génération des phrases et des mots clés est donc exécutée par deux modèles distincts.
 
Le premier est un modèle de « Image Captioning », qui prend en entrée une image et retourne une phrase qui décrit son contenu. C'est un modèle que nous avons entraîné nous-mêmes. Nous l'avons entraîné avec la base de données Flickr30k. Pour l'entraîner, nous nous assurons d’abord d’associer chaque description à son image respective. Une fois que toutes les descriptions sont accessibles, nous notons tous les mots qui reviennent plus de 10 fois dans l’ensemble des descriptions. Nous associons ensuite un index numérique à chaque mot afin de faciliter la prédiction du prochain mot dans les phrases que nous générerons. Nous définissons ensuite un ensemble d'entraînement, un ensemble de tests et un ensemble de validation. Flickr a déjà défini ces ensembles pour nous. Nous rajoutons tout simplement les images ajoutées par Le Devoir dans l'ensemble d'entraînement et de tests. Nous nous assurons de garder un ratio de 9 photos dans l’ensemble d’entraînement pour chaque photo dans l’ensemble de tests.
 
Notre modèle est un modèle à trois couches. En entrée, nous donnons l'image et sa description. À la deuxième couche, nous passons la sortie de la première couche à une fonction d’activation RELU. Finalement, dans la troisième couche, nous cherchons à prédire le prochain mot dans la phrase avec une fonction d’activation SoftMax. La dernière couche n'est pas entrainable. Elle est tout simplement définie à partir d'un dictionnaire de 6 milliards (avec les « embeddings » en 50 dimensions) de mots anglais. Le modèle est ensuite compilé avec l'optimiseur « adam » et une fonction de perte d'entropie croisée catégorielle.
 
#### Spécificités
Notre modèle a 2 547 768 paramètres et 8 418 neurones (1,050 + 2048 + 1,050 + 256 + 256 + 256 + 5352 = 8,418).

Figure 1 : Sortie de la fonction « model.summary() » de notre modèle
![Figure 1](https://github.com/WilliamYn/learning-captioning-model/blob/main/figures/Picture1.png)

### Explication des choix

#### Nombre de couches 
Nous avons choisi d’utiliser un modèle à neuf couches qui peuvent être regroupées en trois couches, car nous n'avons pas vu de grandes différences dans les résultats lorsque nous ajoutions des couches. Le modèle était plus lent à entraîner et plus lourd avec plus de couches. 

#### Fonction d'activation RELU
Nous avons choisi la fonction d'activation RELU pour plusieurs raisons. Premièrement, ce n'est pas une fonction linéaire, ce qui améliore grandement la précision des réseaux de neurones. Deuxièmement, c'est une fonction très rapide à calculer et puisque nous avons beaucoup de données, nous mettons beaucoup d'emphase sur la rapidité d'entraînement du modèle. Troisièmement, nous sommes conscients du risque d'avoir des neurones morts avec cette fonction d'activation, mais nous avons baissé le taux d'apprentissage pour pallier ce problème.

#### Fonction d'activation SoftMax
Nous avons opté pour une fonction d'activation SoftMax, car c'est la meilleure et la plus simple pour faire de la classification multiclasse. Dans notre cas, nous cherchons à prédire le prochain mot de la phrase. Nous considérons comme classes tous les mots qui apparaissent au moins 10 fois dans les descriptions. Nous cherchons ensuite à déterminer quel est le mot qui a le plus de chances d'être le prochain à apparaître.

#### Entrainement
Finalement, nous avons le fichier data/textFiles/inputs.txt pour changer le nombre de « epochs » ou la taille de batch. Ces paramètres sont ceux qui nous donnent les meilleurs résultats dans le plus court laps de temps. Le modèle actuellement chargé a été testé avec 15 « epochs » et une taille de batch de 5. Il a pris environ 21 heures à entraîner sur un processeur i7 8750H. Pour entraîner le modèle, il faut avoir au moins deux images. Pour commencer à avoir de bons résultats, il faudrait téléverser au moins 10 000 photos. 


## Ensemble de donnée
Pour télécharger Flickr30k : Pour télécharger les images de Flickr30k, allez sur ce site : https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset. Créez un compte gratuit et téléchargez les images. Déplacez ensuite les images dans le dossier data/Images.

## Utilisation
Pour trouver les mots clés, nous le faisons de la même façon que le modèle pré-entraîné. (https://github.com/WilliamYn/pretrained-captioning)
Nous utilisons CLIP, un modèle fait par OpenAI qui permet de faire de la classification Zero-Shot avec des mots et une image. Ainsi, avec ce modèle, nous pouvons présenter une image avec une multitude de mots, puis le modèle assignera à chaque mot un pourcentage qui indiquera la portion de l’image occupée par ce mot. Les mots que nous utilisons viennent de la description générée par le modèle. 

Finalement, nous retournons dans un objet JSON tous les mots avec leur pourcentage et la description générée. 

### Notes
Il est à noter que l'entraînement sur de nouvelles images ne se fait pas automatiquement et instantanément. Une pipeline Github est disponible pour déclencher le réentraînement du modèle à partir des nouvelles images et descriptions ajoutées depuis l’interface.

## Comparaison des deux modèles de génération de description d’images
Pour comparer le modèle de génération d’images pré-entraîné au modèle entraîné que nous avons entraîné, nous générons les descriptions sur un ensemble de validation prédéfini. Nous avons choisi l’ensemble de validation défini par Flickr30k. Nous générons le score WER (« Word Error Rate ») entre la description de base et la description générée par notre modèle. Nous prenons ensuite la moyenne des scores WER pour chaque modèle. Un plus petit score WER représente un résultat plus précis.

## Utilisation
Il y a 3 routes exposés par ce service.
GET /
POST /caption
POST /learning-model

### /
Route GET qui retourne un message de type Hello World

### /caption
Route POST qui permet d'envoyer une image avec un caption pour entrainer le modèle.

#### Input JSON
```
{
    "image": "image au format base64",
    "caption": "phrase décrivant le contenu de l'image",
    "fileType": "extension de l'image (jpeg, png, ou autre)"
}
```

### /learning-model
Route POST qui permet de générer des étiquettes et des captions pour une image.


#### Input JSON
```
{
    "image": "image convertit en base64"
}
```
#### Output JSON
```
{
    "tags": [
        ["tag1", 0.38],
        ["tag2", 0.12],
        ...,
        ["tagX", 0.01]
    ],
    "captions": ["the best caption"]
    "english_captions": [
        "The sentence the tags were generated from.",
        ...,
        "Another sentence the tags were generated from."
    ]
}
```
