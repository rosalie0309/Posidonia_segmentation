# Projet de Stage : Posidonia oceanica Semantic Segmentation

Projet de segmentation sémantique d'images sous-marines pour la détection de *Posidonia oceanica*.

Ce travail a été réalisé au GREAH de l'Université Le Havre Normandie dans le cadre de mon stage de fin d'études. Il vise à comparer plusieurs architectures de deep learning pour identifier automatiquement les zones de *Posidonia oceanica* dans des images sous-marines.

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

Le jeu de données utilisé au cours du stage a été fourni par l'Universitat de les Illes
Balears (UIB). Les images ont été acquises dans trois zones de Majorque qui sont **Valldemossa**,
**Palma Bay** et **Cala Blava** à l'aide de plateformes robotiques sous-marines équipées de
caméras orientées vers le fond marin.

Il contient **470 couples image–masque** en RGB, de résolution 360 × 480, annotés pixel
par pixel :

| Sous-ensemble | Images | Usage |
|---|---:|---|
| Développement | 376 | entraînement et validation |
| Test | 94 | évaluation finale, jamais utilisé pour la sélection |

Cette répartition en deux dossiers distincts était déjà constituée à la réception des
données ; elle n'a pas été construite dans le cadre de ce travail.

**Organisation des données :**

```text
Dataset/
├── training/
│   ├── images/
│   └── gt_images/
└── testing/
    ├── images/
    └── gt_images/
```

Les données ne sont pas redistribuées dans ce dépôt. Elles doivent être obtenues auprès
des responsables des données, puis organisées selon cette arborescence.

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

La structure principale du projet est la suivante :

```text
Posidonia_segmentation/
├── Application_web/      application Flask (app.py, pipeline_utils.py, templates/)
├── segformer/            fichiers des expériences SegFormer (b0, b1, b2)
├── models/               à créer en local : poids .pth et fichiers *_metadata.json
├── Dataset/              à créer en local : données
├── implementation_segformerb_tutoriel.ipynb
├── implementation_vgg16_fcn8.ipynb
├── statistical_analysis_segformer.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

Les dossiers `models/` et `Dataset/` sont conservés localement et exclus du suivi Git : les poids (`.pth`) dépassent la limite de taille de GitHub, et les données ne sont pas redistribuées.

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
git clone https://github.com/rosalie0309/Posidonia_segmentation
cd Posidonia_segmentation
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
├── testing/
│   ├── images/
│   └── gt_images/
├── fold_csvs/                 (fold0_val.csv à fold4_val.csv)
└── val_split_stratified.csv
```

Les dossiers `training/` et `testing/` suffisent pour segmenter des images. Les fichiers `fold_csvs/` et `val_split_stratified.csv` sont nécessaires à la page d'évaluation (validation croisée).

### 5. Exécuter les notebooks

Les notebooks actuellement présents dans le dépôt sont :

```text
implementation_vgg16_fcn8.ipynb
implementation_segformerb_tutoriel.ipynb
```

Ils contiennent les principales étapes d'entraînement et d'évaluation des modèles correspondants.

### Fichiers de métadonnées JSON

Le dépôt contient également plusieurs fichiers `*_metadata.json`
générés lors des différentes expérimentations.

Ces fichiers enregistrent notamment :

- la configuration utilisée pour l'entraînement ;
- la résolution d'entrée ;
- la fonction de perte ;
- le taux d'apprentissage et la taille de lot ;
- l'époque ou l'itération correspondant au meilleur checkpoint ;
- les métriques de validation et de test ;
- le seuil de décision calibré sur l'ensemble de validation ;
- les informations nécessaires à la comparaison des différentes
  configurations expérimentales.

Ils sont conservés dans le dépôt afin de fournir une trace des
expériences réalisées et de faciliter l'analyse des résultats sans
devoir réentraîner systématiquement les modèles.

Certains fichiers JSON correspondent à des expériences réalisées avant
les dernières évolutions du notebook et peuvent donc ne pas contenir
tous les champs présents dans les versions les plus récentes, par
exemple les métriques de validation calculées explicitement au seuil
fixe de `0.5`.

Ces fichiers ne doivent pas être considérés comme des paramètres à
modifier manuellement. Lors d'une nouvelle exécution complète des
notebooks, les fichiers de métadonnées correspondants sont générés à
partir des résultats effectivement obtenus.

### 6. Lancer l'application Web

Placer d'abord les modèles dans un dossier `models/` à la racine du projet, au même niveau que `Application_web/` et `Dataset/`. Chaque modèle comprend deux fichiers : le poids (`nom_best.pth`) et ses métadonnées (`nom_metadata.json`). Les `.pth` ne sont pas dans le dépôt et doivent être obtenus auprès de l'auteure ou alors vous pouvez refaire l'entraînement en utilisant les notebooks afin d'obtenir les modèles que vous pouvez évaluer. Les fichiers `*_metadata.json` fournis dans le dépôt se trouvent déjà dans `models/` : il suffit d'y ajouter les poids `.pth` correspondants.
Puis, depuis la racine du projet :

```bash
cd Application_web
python app.py
```

L'application est ensuite accessible à l'adresse locale indiquée dans le terminal (par défaut `http://127.0.0.1:5000`).

Deux variables d'environnement optionnelles permettent d'adapter le lancement :

| Variable | Rôle | Valeur par défaut |
|---|---|---|
| `MODELS_DIR` | dossier contenant les modèles | `models/` à la racine du projet |
| `FLASK_HOST` | adresse d'écoute du serveur | `127.0.0.1` |

Au premier chargement d'un modèle SegFormer, une connexion Internet est nécessaire : l'encodeur pré-entraîné `nvidia/mit-b0`, `b1` ou `b2` est téléchargé depuis Hugging Face avant d'être remplacé par les poids du checkpoint.

### 7. Lancer l'application avec Docker (optionnel)

L'application peut être exécutée dans un conteneur Docker. Cette procédure a été testée sous Linux (GitHub Codespaces). Les poids du modèle et les données ne sont pas copiés dans l'image : ils sont branchés par des volumes.

Construire l'image, depuis la racine du projet :

```bash
docker build -t octopus-posidonia .
```

Placer au moins un modèle (fichier `.pth` et son fichier `_metadata.json`) dans `models/`, puis lancer le conteneur :

```bash
docker run --rm -p 5000:5000 \
  -v "$PWD/models:/app/models" \
  -v hf_cache:/hf_cache \
  octopus-posidonia
```

L'application est ensuite accessible sur `http://localhost:5000`. Pour afficher les faux positifs, les faux négatifs et les métriques, ajouter aussi le jeu de données avec `-v "$PWD/Dataset:/app/Dataset"`.

Le fichier `requirements-docker.txt` contient uniquement les dépendances nécessaires à l'application, et PyTorch est installé en version CPU depuis le `Dockerfile`.

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

