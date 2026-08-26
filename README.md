# Project OCTOPUS – Posidonia oceanica Semantic Segmentation

Projet de segmentation sémantique d'images sous-marines pour la détection de *Posidonia oceanica*.

Ce travail a été réalisé au GREAH – Université Le Havre Normandie dans le cadre du projet OCTOPUS. Il vise à comparer plusieurs architectures de deep learning pour identifier automatiquement les zones de *Posidonia oceanica* dans des images sous-marines.

## Objectifs

- Segmenter automatiquement les herbiers de *Posidonia oceanica*.
- Comparer des architectures convolutives et fondées sur les Transformers.
- Étudier l'influence des fonctions de perte, de la résolution d'entrée et des prétraitements.
- Évaluer la robustesse des modèles par validation croisée stratifiée.
- Étudier la calibration du seuil de décision.
- Analyser les probabilités et l'incertitude des prédictions.
- Comparer les temps d'inférence des différentes architectures.
- Étudier une stratégie d'ensemble par vote souple.
- Préparer une future utilisation des modèles sur une plateforme AUV.

## Jeu de données

Le jeu de données utilisé au cours du stage a été fourni par le GREAH pour les expérimentations de segmentation de *Posidonia oceanica*.

Les travaux de Martin-Abadal et al. décrivent des campagnes d'acquisition réalisées par le groupe Systems, Robotics & Vision (SRV) de l'Universitat de les Illes Balears (UIB), notamment à Palma Bay, Cala Blava et Valldemossa, à Majorque. Les acquisitions ont été réalisées avec le Turbot AUV, un véhicule de type SPARUS II, équipé de caméras orientées vers le fond marin.

Une version publique des données associées à ces travaux est disponible sur Zenodo :

https://doi.org/10.5281/zenodo.4526867

Le jeu utilisé dans cette étude contient 470 couples image–masque :

- 376 images pour le développement, utilisées pour l'entraînement et la validation ;
- 94 images pour le test indépendant.

L'organisation des données utilisée dans ce projet est la suivante :

```text
Dataset/
├── training/
│   ├── images/
│   └── gt_images/
└── testing/
    ├── images/
    └── gt_images/
```

Les données brutes ne sont pas dupliquées dans ce dépôt. Elles doivent être obtenues auprès de la source d'origine ou des responsables des données, puis organisées selon cette structure.

> **Note :** le jeu de 470 images utilisé pendant le stage a été fourni dans cette organisation par le GREAH. Le lien Zenodo correspond à une version publique des données associées aux travaux de Martin-Abadal et al. L'équivalence exacte entre les deux jeux doit encore être confirmée.

## Références du jeu de données

Martin-Abadal, M., Guerrero-Font, E., Bonin-Font, F. & Gonzalez-Cid, Y. (2018).  
**Deep Semantic Segmentation in an AUV for Online Posidonia Oceanica Meadows Identification.**  
IEEE Access, 6, 60956–60967.  
https://doi.org/10.1109/ACCESS.2018.2875412

Dataset associé :

Martin-Abadal, M., Guerrero-Font, E., Bonin-Font, F. & Gonzalez-Cid, Y.  
**Underwater images containing Posidonia Oceanica.**  
Zenodo.  
https://doi.org/10.5281/zenodo.4526867

## Architectures étudiées

Les architectures considérées au cours du projet sont :

- U-Net – exploration préliminaire ;
- VGG16-FCN8 – modèle convolutif de référence ;
- SegFormer-B0 ;
- SegFormer-B1 ;
- SegFormer-B2 – modèle principal ;
- ensemble VGG16-FCN8 + SegFormer-B0/B1/B2 par vote souple.

## Méthodologie

Le protocole expérimental comprend plusieurs étapes :

1. préparation et harmonisation des images et des masques ;
2. analyse de la distribution des données ;
3. sélection progressive de la configuration de SegFormer-B2 ;
4. validation croisée stratifiée à cinq folds ;
5. évaluation sur un ensemble de test indépendant ;
6. calibration du seuil de décision ;
7. analyse quantitative et qualitative des prédictions ;
8. analyse des probabilités et de l'incertitude ;
9. évaluation d'un modèle d'ensemble ;
10. comparaison des temps d'inférence.

La configuration finale de SegFormer-B2 utilise notamment :

- une fonction de perte CE+Dice ;
- une résolution de `352 × 480` pixels ;
- un batch size de `8` ;
- un learning rate de `1e-4` ;
- aucun prétraitement externe supplémentaire.

## Principaux résultats

Sur la validation croisée stratifiée à cinq folds, SegFormer-B2 obtient les meilleures performances globales parmi les modèles individuels étudiés, avec notamment :

- mIoU moyen : **94.03 % ± 0.11 %** ;
- Dice moyen : **96.63 % ± 0.07 %**.

Les variantes SegFormer-B0 et SegFormer-B1 offrent cependant des temps d'inférence plus faibles.

Le modèle d'ensemble améliore légèrement les performances de SegFormer-B2, mais au prix de l'utilisation simultanée de plusieurs réseaux.

## Application Web

Une application Web développée avec Flask permet d'exécuter les modèles et de visualiser :

- les prédictions de segmentation ;
- les faux positifs et faux négatifs ;
- les cartes de probabilités ;
- les cartes d'incertitude ;
- les principales métriques d'évaluation ;
- les résultats de plusieurs modèles et du modèle d'ensemble.

Le code correspondant se trouve dans :

```text
Application_web/
```

## Structure du dépôt

La structure principale du projet est actuellement :

```text
Posidonia_Oceanica_Detection_Project/
├── Application_web/
├── Dataset/
├── implementation_segformerb_tutoriel.ipynb
├── implementation_vgg16_fcn8.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

Le dossier `Dataset/` peut être conservé localement tout en étant exclu du suivi Git à l'aide du fichier `.gitignore`.

## Environnement logiciel

Le projet a été développé et évalué dans l'environnement principal suivant :

- Windows 11
- Python 3.12.10
- PyTorch 2.12.0+cpu
- Flask 3.1.3
- exécution CPU, sans CUDA

Les principales bibliothèques utilisées comprennent également :

- Transformers
- Torchvision
- NumPy
- OpenCV
- Matplotlib
- Pillow
- scikit-learn

Les versions exactes des dépendances doivent être enregistrées dans le fichier `requirements.txt` afin de permettre la reproduction de l'environnement.

## Reproductibilité

### 1. Cloner le dépôt

```bash
git clone https://github.com/rosalie0309/Project_OCTOPUS
cd Posidonia_Oceanica_Detection_Project
```

### 2. Créer un environnement virtuel

Sous Windows :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Sous Linux ou macOS :

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Préparer le jeu de données

Les données doivent être placées selon la structure suivante :

```text
Dataset/
├── training/
│   ├── images/
│   └── gt_images/
└── testing/
    ├── images/
    └── gt_images/
```

### 5. Exécuter les notebooks

Les notebooks actuellement présents dans le dépôt sont :

```text
implementation_vgg16_fcn8.ipynb
implementation_segformerb_tutoriel.ipynb
```

Ils contiennent les principales étapes d'entraînement et d'évaluation des modèles correspondants.

### 6. Lancer l'application Web

Depuis la racine du projet :

```bash
cd Application_web
python app.py
```

L'application est ensuite accessible à l'adresse locale indiquée dans le terminal.
Pour les modèles ils faut vous rassurer qu'ils sont dans un dossier ```models``` qui doit se trouver au même niveau hiérarchique que ```Dataset```

## Génération du fichier requirements

Afin de conserver les versions exactes de l'environnement utilisé pour les expérimentations, le fichier peut être généré depuis l'environnement Python d'origine avec :

```bash
pip freeze > requirements.txt
```

Pour vérifier les versions principales :

```bash
python -c "import torch, transformers, torchvision, numpy, cv2, matplotlib, flask, sklearn; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__); print('Torchvision:', torchvision.__version__); print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('Matplotlib:', matplotlib.__version__); print('Flask:', flask.__version__); print('scikit-learn:', sklearn.__version__)"
```

## Technologies

- Python
- PyTorch
- Transformers
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Flask

## Contexte

Ce projet a été réalisé dans le cadre d'un stage de Master 2 Informatique – parcours Exploration Informatique des Données et Décisionnel, en collaboration avec le GREAH – Université Le Havre Normandie.

Le travail s'inscrit dans le projet OCTOPUS, consacré notamment à l'utilisation de la robotique mobile et de l'intelligence artificielle pour la surveillance des environnements maritimes et côtiers.
