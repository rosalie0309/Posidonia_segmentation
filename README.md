# Project OCTOPUS – Posidonia oceanica Semantic Segmentation

Projet de segmentation sémantique d'images sous-marines pour la
détection de *Posidonia oceanica*.

Ce travail a été réalisé au GREAH – Université Le Havre Normandie dans
le cadre du projet OCTOPUS. Il compare plusieurs architectures de deep
learning, notamment VGG16-FCN8, U-Net et SegFormer, pour identifier les
zones de *Posidonia oceanica* dans des images sous-marines.

## Objectifs

- Segmenter automatiquement les herbiers de *Posidonia oceanica*.
- Comparer des architectures convolutives et Transformer.
- Étudier l'influence des fonctions de perte, de la résolution et des
  prétraitements.
- Évaluer la robustesse des modèles par validation croisée.
- Étudier la calibration du seuil de décision et l'incertitude.
- Comparer les temps d'inférence.
- Préparer une future utilisation sur une plateforme AUV.

## Dataset

Le jeu de données utilisé dans ce projet a été fourni par le GREAH et
est issu de campagnes d'acquisition réalisées par le groupe
Systems, Robotics & Vision (SRV) de l'Universitat de les Illes Balears
(UIB).

Les travaux de Martin-Abadal et al. décrivent des acquisitions
sous-marines de *Posidonia oceanica* réalisées à Majorque et
l'utilisation du Turbot AUV, un véhicule de type SPARUS II.

Une version publique des données de *Posidonia oceanica* mise à
disposition par les auteurs est disponible sur Zenodo :

https://doi.org/10.5281/zenodo.4526867

Le jeu utilisé dans cette étude contient 470 couples image–masque :

- 376 images pour le développement (entraînement et validation) ;
- 94 images pour le test indépendant.

Organisation utilisée dans ce projet :

Dataset/
├── training/
│   ├── images/
│   └── gt_images/
└── testing/
    ├── images/
    └── gt_images/

Les données brutes ne sont pas dupliquées dans ce dépôt. Elles doivent
être obtenues auprès de la source indiquée ci-dessus et organisées selon
cette structure.

## Référence du jeu de données

Martin-Abadal, M., Guerrero-Font, E., Bonin-Font, F. &
Gonzalez-Cid, Y. (2018).
Deep Semantic Segmentation in an AUV for Online Posidonia Oceanica
Meadows Identification.
IEEE Access, 6, 60956–60967.
https://doi.org/10.1109/ACCESS.2018.2875412

Dataset:
Martin-Abadal, M., Guerrero-Font, E., Bonin-Font, F. &
Gonzalez-Cid, Y.
Underwater images containing Posidonia Oceanica.
Zenodo.
https://doi.org/10.5281/zenodo.4526867

## Modèles étudiés

- U-Net
- VGG16-FCN8
- SegFormer-B0
- SegFormer-B1
- SegFormer-B2
- Ensemble par vote souple

## Fonctionnalités

- Entraînement et évaluation des modèles
- Validation croisée stratifiée
- Calibration du seuil de décision
- Analyse des probabilités et de l'incertitude
- Comparaison des temps d'inférence
- Application Web Flask pour la visualisation des résultats

## Technologies

- Python
- PyTorch
- Transformers
- OpenCV
- NumPy
- Matplotlib
- Flask
