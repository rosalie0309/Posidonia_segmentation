"""
app.py - Application web de segmentation Posidonia oceanica
Lancement : python app.py
"""

from pipeline_utils import (
    discover_pipelines,
    format_loss,
    format_preprocess,
    format_metric_value,
    get_main_metrics,
)

import os
import io
import base64
import time 
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import csv
import json
import cv2
import gc

from torchvision.models import vgg16, VGG16_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------
# CONFIG (s'adapte aux chemins)
# ----------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / "predictions" / "models"
TESTING_IMGS = BASE_DIR / "Dataset" / "testing" / "images"
TESTING_MSKS = BASE_DIR / "Dataset" / "testing" / "gt_images"
TRAINING_IMGS = BASE_DIR / "Dataset" / "training" / "images"
TRAINING_MSKS = BASE_DIR / "Dataset" / "training" / "gt_images"
# Dossier contenant fold0_val.csv, ..., fold4_val.csv
VGG_FOLDS_DIR = BASE_DIR / "Dataset" / "fold_csvs"

IMG_H, IMG_W = 352, 480
N_CLASSES    = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On fixe le seuil de décision à 0.5 pour la prédiction finale et l'évaluation des métriques
DECISION_THRESHOLD = 0.5
DIRICHLET_UNCERTAINTY_THRESHOLD = 0.5

THRESHOLD_MODE_DEFAULT = "fixed"
UNCERTAINTY_MODE_DEFAULT = "standard"

# Nettoyer la CACHE pour éviter que la RAM monte progressivement 
def clear_model_cache():
    """
    Libère les modèles conservés en mémoire.
    """

    global MODELS

    MODELS.clear()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[CACHE] Modèles libérés de la mémoire.")


# Détection automatique de tous les .pth dans models/
def make_model_label(pth_path):
    """
    Construit un nom lisible et explicite à partir du nom du checkpoint.
    Exemple :
      - fcn8vgg16_fold5_best.pth -> FCN-8s VGG16 fold5
      - 7A_segformerb2_ce_best.pth -> SegFormer-B2
    """
    fname = pth_path.stem.lower()

    if 'segformer' in fname:
        variant = next((v for v in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'] if v in fname), None)
        label = f"SegFormer-{variant.upper()}" if variant else f"SegFormer ({pth_path.stem})"

    elif 'unet' in fname:
        label = "U-Net"

    elif 'fcn' in fname or 'vgg' in fname:
        fold = None

        if 'fold' in fname:
            after = fname.split('fold', 1)[1]
            digits = ""

            for ch in after:
                if ch.isdigit():
                    digits += ch
                else:
                    break

            if digits:
                fold = digits

        if fold is not None:
            label = f"FCN-8s VGG16 fold{fold}"
        else:
            label = "FCN-8s VGG16"

    else:
        label = pth_path.stem

    return label

def checkpoint_experiment_stem(pth_path):
    """
    Retourne le nom de l'expérience associé à un checkpoint.

    Exemples :
      model.pth             -> model
      model_best.pth        -> model
      model_bestval.pth     -> model
      model_resume.pth      -> model
    """
    stem = Path(pth_path).stem

    for suffix in ("_bestval", "_best", "_resume"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    return stem

def checkpoint_priority(pth_path):
    """
    Priorité de sélection si plusieurs checkpoints existent
    pour la même expérience.
    """
    stem = Path(pth_path).stem.lower()

    if stem.endswith("_bestval"):
        return 3

    if stem.endswith("_best"):
        return 2

    if stem.endswith("_resume"):
        return 1

    return 0

def checkpoint_type(pth_path):
    stem = Path(pth_path).stem.lower()

    if stem.endswith("_bestval"):
        return "best_validation"

    if stem.endswith("_best"):
        return "best"

    if stem.endswith("_resume"):
        return "resume"

    return "final"

def build_models_config(models_dir):
    """
    Détecte les checkpoints disponibles.

    Si plusieurs checkpoints correspondent à la même expérience,
    la priorité est :
        bestval > best > resume > final
    """
    selected_checkpoints = {}

    for pth_path in sorted(models_dir.glob("*.pth")):
        experiment_stem = checkpoint_experiment_stem(pth_path)

        current_path = selected_checkpoints.get(experiment_stem)

        if (
            current_path is None
            or checkpoint_priority(pth_path) > checkpoint_priority(current_path)
        ):
            selected_checkpoints[experiment_stem] = pth_path

    config = {}

    for experiment_stem, pth_path in sorted(selected_checkpoints.items()):
        label = make_model_label(pth_path)

        if label in config:
            label = f"{label} ({experiment_stem})"

        config[label] = pth_path

        print(
            f"[MODEL] {label} -> {pth_path.name} "
            f"(type={checkpoint_type(pth_path)})"
        )

    return config


MODELS_CONFIG = build_models_config(MODELS_DIR)
print(f"Modèles détectés : {list(MODELS_CONFIG.keys())}")


def load_model_thresholds(models_dir):
    """
    Charge le seuil sélectionné sur validation pour chaque checkpoint retenu.
    """
    thresholds = {}

    for label, pth_path in MODELS_CONFIG.items():
        experiment_stem = checkpoint_experiment_stem(pth_path)

        expected_json_name = f"{experiment_stem}_metadata.json"
        matched_json = models_dir / expected_json_name

        if not matched_json.exists():
            candidates = [
                json_path
                for json_path in models_dir.glob("*.json")
                if json_path.stem.replace("_metadata", "") == experiment_stem
            ]

            matched_json = candidates[0] if candidates else None

        if matched_json is None or not matched_json.exists():
            print(
                f"[WARN] JSON introuvable pour {pth_path.name}. "
                f"Fichier attendu : {expected_json_name}"
            )
            thresholds[label] = DECISION_THRESHOLD
            continue

        try:
            with open(matched_json, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            # Seuil choisi sur validation uniquement
            threshold = metadata.get("val_threshold")

            # Compatibilité avec l'ancien format VGG16
            if threshold is None:
                threshold = (
                    metadata
                    .get("mix_metrics", {})
                    .get("threshold")
                )

            if threshold is None:
                print(
                    f"[WARN] Aucun seuil de validation dans "
                    f"{matched_json.name}; utilisation de 0.5."
                )
                threshold = DECISION_THRESHOLD

            thresholds[label] = round(float(threshold), 3)

            print(
                f"[THRESHOLD] {label} = {thresholds[label]:.3f} "
                f"depuis {matched_json.name}"
            )

        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(
                f"[WARN] JSON non lisible : {matched_json} — {exc}"
            )
            thresholds[label] = DECISION_THRESHOLD

    return thresholds


MODEL_THRESHOLDS = load_model_thresholds(MODELS_DIR)
print(f"Seuils chargés : {MODEL_THRESHOLDS}")

COLORS = {
    'TP': np.array([0,   200,   0], dtype=np.uint8),
    'FP': np.array([220,   0,   0], dtype=np.uint8),
    'FN': np.array([0,    80, 220], dtype=np.uint8),
}
ALPHA = 0.55

# -----------------------------------------------------
# ARCHITECTURES
# -----------------------------------------------------
# === U-NET COMPLET ===
# On laisse cette classe juste au cas où 
# Mais en réalité nous n'utilisons plus cette architecture dans nos analyse
class DoubleConv(nn.Module):
    """
    Bloc de double convolution : (Conv -> BN -> ReLU) x 2

    Structure pédagogique standard de U-Net (Ronneberger 2015).

    Args:
        in_channels : nombre de canaux d'entrée
        out_channels : nombre de canaux de sortie
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Première convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Deuxième convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
# === BLOC D'ENCODAGE : DOWN ===

class Down(nn.Module):
    """
    Bloc d'encodage : MaxPool puis DoubleConv.

    Réduit la résolution spatiale par 2 et applique une double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# === BLOC DE DÉCODAGE : UP ===

class Up(nn.Module):
    """
    Bloc de décodage : Upsample, concaténation avec skip, puis DoubleConv.

    Args:
        in_channels : nombre de canaux d'entrée (avant concaténation)
        out_channels : nombre de canaux de sortie
        use_transpose : si True, utilise ConvTranspose2d (apprenable)
                        sinon, utilise un upsample bilinéaire (plus léger)
    """
    def __init__(self, in_channels, out_channels, use_transpose=True):
        super().__init__()

        if use_transpose:
            # Convolution transposée (apprenable, plus de paramètres)
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Upsample bilinéaire (pas d'apprentissage, plus léger)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x_decoder, x_skip):
        """
        Args:
            x_decoder : tenseur du décodeur (à upsampler)
            x_skip : tenseur de la skip connection (depuis l'encodeur)
        """
        # Upsampling
        x_decoder = self.up(x_decoder)

        # Si les dimensions ne matchent pas exactement (peut arriver à cause des arrondis), on pad
        diff_y = x_skip.size(2) - x_decoder.size(2)
        diff_x = x_skip.size(3) - x_decoder.size(3)
        x_decoder = F.pad(x_decoder, [diff_x // 2, diff_x - diff_x // 2,
                                        diff_y // 2, diff_y - diff_y // 2])

        # Concaténation des canaux (le secret de U-Net)
        x = torch.cat([x_skip, x_decoder], dim=1)

        # Double convolution
        return self.conv(x)


class UNet(nn.Module):
    """
    Architecture U-Net pour segmentation sémantique.

    Args:
        in_channels : nombre de canaux d'entrée (3 pour RGB)
        n_classes : nombre de classes de sortie (2 pour binaire fond/posidonie)
        base_channels : nombre de canaux au premier niveau (default 64)
        use_transpose : si True, utilise ConvTranspose2d pour l'upsampling

    Référence : Ronneberger, Fischer, Brox (2015).
    """
    def __init__(self, in_channels=3, n_classes=2, base_channels=64, use_transpose=True):
        super().__init__()
        self.in_channels   = in_channels
        self.n_classes     = n_classes
        self.base_channels = base_channels


        self.enc1      = DoubleConv(in_channels, base_channels)
        self.enc2      = Down(base_channels,     base_channels * 2)
        self.enc3      = Down(base_channels * 2, base_channels * 4)
        self.enc4      = Down(base_channels * 4, base_channels * 8)
        self.bottleneck= Down(base_channels * 8, base_channels * 16)
        self.dec4      = Up(base_channels * 16, base_channels * 8,  use_transpose)
        self.dec3      = Up(base_channels * 8,  base_channels * 4,  use_transpose)
        self.dec2      = Up(base_channels * 4,  base_channels * 2,  use_transpose)
        self.dec1      = Up(base_channels * 2,  base_channels,      use_transpose)
        self.out_conv  = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Normalisation manuelle (non sauvegardée dans le .pth)
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x    = (x - mean) / std
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)



class FCN8VGG16(nn.Module):
    """
    Encodeur : VGG16 jusqu'à pool5, puis conv6 (7×7) et conv7 (1×1)
               converties depuis FC, avec dropout 0.5 — article II-A.
    Décodeur : skip connections depuis pool3 et pool4 (FCN-8).
               F.interpolate(..., size=skip.shape) = transposed conv bilinéaire,
               gère proprement les dimensions non-multiples de 32.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg      = vgg16(weights=weights)
        features = list(vgg.features.children())

        self.pool3 = nn.Sequential(*features[:17])    # H/8,  256 ch
        self.pool4 = nn.Sequential(*features[17:24])  # H/16, 512 ch
        self.pool5 = nn.Sequential(*features[24:31])  # H/32, 512 ch

        self.conv6 = nn.Conv2d(512,  4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=dropout)   # dropout 0.5 (article)

        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=dropout)   # dropout 0.5 (article)

        self.score_fr    = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512,  num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256,  num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        # Skip connections : gaussienne tronquée à faible std (article II-B)
        for m in [self.score_fr, self.score_pool4, self.score_pool3]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        # Conv6/7 : Kaiming
        for m in [self.conv6, self.conv7]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        in_size = x.shape[2:]
        p3 = self.pool3(x)
        p4 = self.pool4(p3)
        p5 = self.pool5(p4)

        x     = self.drop6(self.relu6(self.conv6(p5)))
        x     = self.drop7(self.relu7(self.conv7(x)))
        score = self.score_fr(x)

        u2    = F.interpolate(score, size=p4.shape[2:], mode='bilinear', align_corners=False)
        fuse4 = u2 + self.score_pool4(p4)

        u4    = F.interpolate(fuse4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        fuse3 = u4 + self.score_pool3(p3)

        return F.interpolate(fuse3, size=in_size, mode='bilinear', align_corners=False)

class SegFormer(nn.Module):
    def __init__(self, n_classes=2, variant='b2', pretrained=False):
        super().__init__()
        self.register_buffer('img_mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('img_std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                f'nvidia/mit-{variant}', num_labels=n_classes, ignore_mismatched_sizes=True)
        else:
            cfg = SegformerForSemanticSegmentation.config_class.from_pretrained(
                f'nvidia/mit-{variant}', num_labels=n_classes, ignore_mismatched_sizes=True)
            self.model = SegformerForSemanticSegmentation(cfg)

    def forward(self, x):
        sz = x.shape[-2:]
        x  = (x - self.img_mean) / self.img_std
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, sz, mode='bilinear', align_corners=False)



# --------------------------------------------
# CHARGEMENT DES MODÈLES
# --------------------------------------------
def load_model(name, pth_path):
    """
    Charge différents formats de checkpoints :

    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model_state": ...}        # format VGG16 bestval
    - state_dict sauvegardé directement
    """
    ckpt = torch.load(
        pth_path,
        map_location=DEVICE,
        weights_only=False
    )

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Format de checkpoint non reconnu pour {pth_path.name}: "
            f"type={type(ckpt).__name__}"
        )

    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]

    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]

    elif "model_state" in ckpt:
        sd = ckpt["model_state"]

    else:
        # Cas où le fichier contient directement le state_dict
        sd = ckpt

    if not isinstance(sd, dict):
        raise TypeError(
            f"Le state_dict extrait de {pth_path.name} "
            f"n'est pas un dictionnaire."
        )

    fname = pth_path.name.lower()

    if 'segformer' in fname:
        variant = 'b1' if 'b1' in fname else 'b2' if 'b2' in fname else 'b0'
        model = SegFormer(n_classes=N_CLASSES, variant=variant, pretrained=True)
    elif 'unet' in fname:
        model = UNet(n_classes=N_CLASSES)
    else:
        model = FCN8VGG16(num_classes=N_CLASSES, pretrained=False).to(DEVICE)

    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()

    print(
        f"  OK: {name} | "
        f"checkpoint={pth_path.name} | "
        f"type={checkpoint_type(pth_path)}"
    )

    return model

# ============================================================
# CACHE DE MODÈLES
# ============================================================

print("=== Initialisation du cache modèles ===")

# Les modèles ne sont plus tous chargés au démarrage.
# Ils sont chargés uniquement lorsqu'ils sont réellement utilisés.
MODELS = {}

print(
    f"{len(MODELS_CONFIG)} checkpoint(s) disponible(s). "
    "Chargement à la demande."
)

def get_or_load_model(name, pth_path):
    """
    Charge un modèle uniquement lorsqu'il est demandé.

    Le modèle est ensuite conservé dans un cache afin d'éviter
    de recharger le même checkpoint plusieurs fois.
    """

    cache_key = str(Path(pth_path).resolve())

    if cache_key in MODELS:
        return MODELS[cache_key]

    print(
        f"[LOAD] Chargement à la demande : "
        f"{Path(pth_path).name}"
    )

    model = load_model(
        name,
        Path(pth_path)
    )

    MODELS[cache_key] = model

    return model

# --------------------------------------------
# UTILITAIRES IMAGE
# ---------------------------------------------
def img_to_tensor(pil_img, img_h=None, img_w=None):
    """
    Convertit une image PIL en tenseur, avec résolution configurable.
    Si aucune résolution n'est donnée, utilise IMG_H, IMG_W.
    """

    if img_h is None:
        img_h = IMG_H

    if img_w is None:
        img_w = IMG_W

    img = pil_img.convert('RGB').resize((img_w, img_h), Image.BILINEAR)
    return transforms.ToTensor()(img).unsqueeze(0)

def get_model_input_size(model_name, checkpoint_path=None):
    """
    Retourne la résolution attendue par le modèle.

    VGG16-FCN8 :
        360 x 480

    SegFormer :
        résolution lue depuis le metadata si disponible,
        sinon 352 x 480.
    """

    name_lower = str(model_name).lower()

    # --------------------------------------------------------
    # VGG16-FCN8
    # --------------------------------------------------------
    if (
        "vgg" in name_lower
        or "fcn" in name_lower
    ):
        return 360, 480

    # --------------------------------------------------------
    # Recherche dans les pipelines
    # --------------------------------------------------------
    if checkpoint_path is not None:

        pipelines = discover_pipelines(
            MODELS_DIR
        )

        checkpoint_resolved = Path(
            checkpoint_path
        ).resolve()

        for pipeline in pipelines.values():

            pipeline_checkpoint = Path(
                pipeline["checkpoint_path"]
            ).resolve()

            if pipeline_checkpoint == checkpoint_resolved:

                img_h = pipeline.get("img_h")
                img_w = pipeline.get("img_w")

                if img_h is not None and img_w is not None:
                    return int(img_h), int(img_w)

    # --------------------------------------------------------
    # Fallback SegFormer historique
    # --------------------------------------------------------
    return IMG_H, IMG_W

def predict_mask(model, tensor):
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

def make_overlay(img_rgb, pred, gt=None):
    overlay = img_rgb.copy().astype(np.float32)
    cmap    = np.zeros_like(img_rgb, dtype=np.uint8)
    amap    = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    if gt is not None:
        tp = (pred==1)&(gt==1); fp = (pred==1)&(gt==0); fn = (pred==0)&(gt==1)
        for mask, key in [(tp,'TP'),(fp,'FP'),(fn,'FN')]:
            cmap[mask] = COLORS[key]; amap[mask] = ALPHA
    else:
        cmap[pred==1] = COLORS['TP']; amap[pred==1] = ALPHA
    a3 = amap[:,:,None]
    return np.clip((1-a3)*overlay + a3*cmap, 0, 255).astype(np.uint8)

def compute_metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())

    total = tp + fp + fn + tn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    # FP % : proportion de faux positifs sur tous les pixels
    fp_percent = 100 * fp / (total + 1e-8)

    # FN % : proportion de faux négatifs sur tous les pixels
    fn_percent = 100 * fn / (total + 1e-8)

    # Fall-out / False Positive Rate :
    # parmi les vrais pixels fond, proportion prédite à tort comme posidonie
    fallout = fp / (fp + tn + 1e-8)

    return {
        'f1': round(f1, 3),
        'iou': round(iou, 3),
        'fallout': round(fallout, 4),
        'fallout_percent': f"{100 * fallout:.1f}%",
        'fp': f"{fp_percent:.1f}%",
        'fn': f"{fn_percent:.1f}%",
    }

def ensemble_predict(models_dict, tensor, gt_mask=None):
    """
    Combine les probabilités softmax de tous les modèles par moyenne
    puis décide la classe majoritaire pixel par pixel.
    """
    proba_sum = None
    n = 0

    for name, model in models_dict.items():
        proba = predict_proba(model, tensor)   # [H, W] float
        if proba_sum is None:
            proba_sum = proba.copy()
        else:
            proba_sum += proba
        n += 1

    proba_mean = proba_sum / n                          # moyenne des probas
    pred       = (proba_mean >= 0.5).astype(np.uint8)  # décision majoritaire
    overlay    = make_uncertainty_overlay(np.array(
                    Image.fromarray(
                        (tensor.squeeze(0).permute(1,2,0).numpy()*255)
                        .astype(np.uint8)
                    ).resize((IMG_W, IMG_H), Image.BILINEAR)
                 ), proba_mean, gt_mask)
    metrics    = compute_metrics(pred, gt_mask) if gt_mask is not None else None

    return {
        'name'   : 'Ensemble (vote majoritaire)',
        'overlay': arr_to_b64(overlay),
        'metrics': metrics,
        'n_models': n,
    }

def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def arr_to_b64(arr):
    return pil_to_b64(Image.fromarray(arr))

# --------------------------------------------
# FLASK
# ---------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

def load_gt_mask(gt_path, split, img_h=None, img_w=None):
    """
    Convertit le masque GT en binaire {0,1} selon le format du split.
    """

    if img_h is None:
        img_h = IMG_H

    if img_w is None:
        img_w = IMG_W

    mask_pil = Image.open(gt_path).resize((img_w, img_h), Image.NEAREST)

    if split == 'training':
        mask_np = np.array(mask_pil.convert('RGB'))
        return (
            (mask_np[:, :, 0] > 200) &
            (mask_np[:, :, 1] < 50) &
            (mask_np[:, :, 2] < 50)
        ).astype(np.uint8)

    else:
        return (np.array(mask_pil.convert('L')) > 127).astype(np.uint8)
    

def load_gt_mask_native_colors(gt_path, split, img_h=None, img_w=None):
    """
    Charge le masque GT en conservant ses couleurs natives.
    """

    if img_h is None:
        img_h = IMG_H

    if img_w is None:
        img_w = IMG_W

    mask_pil = Image.open(gt_path).resize((img_w, img_h), Image.NEAREST)

    if split == 'training':
        return np.array(mask_pil.convert('RGB'))

    else:
        gray = np.array(mask_pil.convert('L'))
        return np.stack([gray, gray, gray], axis=-1)
    


def predict_proba(model, tensor):
    """
    Retourne la carte de probabilité softmax pour la classe posidonie.
    Shape : (H, W), valeurs dans [0, 1]
    """
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))          # [1, 2, H, W]
        proba  = torch.softmax(logits, dim=1)      # [1, 2, H, W]
        proba_pos = proba[0, 1, :, :]              # [H, W] — classe posidonie
    return proba_pos.cpu().numpy()                 # np.ndarray float32



def make_uncertainty_overlay(img_rgb, proba_map, gt_mask=None, threshold=0.5):
    """
    4 zones visuelles :
      - Vert   : TP haute confiance  (p >= 0.7)
      - Jaune  : zone de doute       (0.3 < p < 0.7) - modèle incertain
      - Rouge  : FP haute confiance  (p >= 0.7, gt=fond)
      - Bleu   : FN haute confiance  (p <= 0.3, gt=posidonie)

    Un seuil que l'utilisateur peut ajuster permet de déplacer les frontières de ces zones.
    """
    overlay = img_rgb.astype(np.float32)
    result  = overlay.copy()
    pred    = (proba_map >= threshold).astype(np.uint8)  # ← threshold

    SEUIL_HAUT = min(threshold + 0.2, 0.95)  # zone de confiance haute
    SEUIL_BAS  = max(threshold - 0.2, 0.05)  # zone de confiance basse
    ALPHA      = 0.60

    if gt_mask is not None:
        doute = (proba_map > SEUIL_BAS) & (proba_map < SEUIL_HAUT)
        tp = (proba_map >= SEUIL_HAUT) & (gt_mask == 1)
        fp = (proba_map >= SEUIL_HAUT) & (gt_mask == 0)
        fn = (proba_map <= SEUIL_BAS)  & (gt_mask == 1)
        zones = [
            (tp,    np.array([0,   200,   0], dtype=np.float32)),
            (fp,    np.array([220,   0,   0], dtype=np.float32)),
            (fn,    np.array([0,    80, 220], dtype=np.float32)),
            (doute, np.array([255, 200,   0], dtype=np.float32)),
        ]
    else:
        doute     = (proba_map > SEUIL_BAS) & (proba_map < SEUIL_HAUT)
        posidonie = proba_map >= SEUIL_HAUT
        zones = [
            (posidonie, np.array([0,   200,   0], dtype=np.float32)),
            (doute,     np.array([255, 200,   0], dtype=np.float32)),
        ]

    for mask, color in zones:
        if not mask.any(): continue
        mask3  = mask[:, :, np.newaxis]
        result = np.where(mask3,
                          np.clip((1-ALPHA)*overlay + ALPHA*color, 0, 255),
                          result)
    return result.astype(np.uint8)

def make_uncertainty_only_overlay(pred, proba, seuil_bas=0.3, seuil_haut=0.7):
    """
    Affiche uniquement la prédiction + zone de doute, sans comparaison au GT.
    Utilisable pour des images hors dataset (pas de masque annoté disponible).
    - blanc : posidonie prédite avec confiance
    - noir  : fond prédit avec confiance
    - jaune : zone de doute (0.3 < proba < 0.7)
    """
    h, w = pred.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    result[pred == 0] = [0, 0, 0]
    result[pred == 1] = [255, 255, 255]

    doute = (proba > seuil_bas) & (proba < seuil_haut)
    result[doute] = [255, 200, 0]

    return result

# Nous allons ajouter le Direchlet pour l'incertitude d'ensemble, et une incertitude continue pour un modèle unique.

def compute_single_model_uncertainty(proba, seuil_bas=0.3, seuil_haut=0.7):
    """
    Calcule l'incertitude pour un modèle unique à partir de la probabilité
    de la classe Posidonie.

    proba : carte HxW contenant p(posidonie)
    """

    # Incertitude continue : maximale à p=0.5, faible vers 0 ou 1
    uncertainty_map = 1.0 - np.abs(2.0 * proba - 1.0)

    # Zone de doute déjà utilisée dans le projet
    uncertain_pixels = (proba > seuil_bas) & (proba < seuil_haut)

    stats = {
        "uncertainty_mean": float(np.mean(uncertainty_map)),
        "uncertainty_median": float(np.median(uncertainty_map)),
        "uncertainty_max": float(np.max(uncertainty_map)),
        "uncertain_pixel_ratio": float(np.mean(uncertain_pixels)),
        "uncertain_pixel_percent": float(np.mean(uncertain_pixels) * 100.0),
    }

    return uncertainty_map, uncertain_pixels.astype(np.uint8), stats

def compute_dirichlet_uncertainty_from_probas(proba_list):
    """
    Calcule une incertitude Dirichlet à partir d'une ou plusieurs cartes de probabilités.

    proba_list : liste de cartes HxW contenant p(posidonie)
                 - un seul élément pour un modèle unique
                 - plusieurs éléments pour un ensemble
    """

    probas = np.stack(proba_list, axis=0)  # [n_models, H, W]
    n_models = probas.shape[0]

    # Confiance : proche de 0 si p≈0.5, proche de 1 si p≈0 ou p≈1
    confidence = np.abs(2.0 * probas - 1.0)

    # Évidence apportée à chaque classe
    evidence_posidonie = np.sum(confidence * probas, axis=0)
    evidence_fond = np.sum(confidence * (1.0 - probas), axis=0)

    alpha_posidonie = 1.0 + evidence_posidonie
    alpha_fond = 1.0 + evidence_fond
    alpha_sum = alpha_posidonie + alpha_fond

    dirichlet_prob_posidonie = alpha_posidonie / alpha_sum

    # Incertitude brute
    dirichlet_uncertainty = 2.0 / alpha_sum

    # Normalisation pour l’affichage
    u_min = 2.0 / (2.0 + n_models)
    u_max = 1.0

    dirichlet_uncertainty_norm = (
        dirichlet_uncertainty - u_min
    ) / (u_max - u_min + 1e-8)

    dirichlet_uncertainty_norm = np.clip(
        dirichlet_uncertainty_norm,
        0.0,
        1.0
    )

    uncertain_pixels = dirichlet_uncertainty_norm > DIRICHLET_UNCERTAINTY_THRESHOLD

    stats = {
        "dirichlet_uncertainty_mean": float(np.mean(dirichlet_uncertainty)),
        "dirichlet_uncertainty_median": float(np.median(dirichlet_uncertainty)),
        "dirichlet_uncertainty_max": float(np.max(dirichlet_uncertainty)),

        "dirichlet_uncertainty_norm_mean": float(np.mean(dirichlet_uncertainty_norm)),
        "dirichlet_uncertainty_norm_median": float(np.median(dirichlet_uncertainty_norm)),
        "dirichlet_uncertainty_norm_max": float(np.max(dirichlet_uncertainty_norm)),

        "dirichlet_uncertain_pixel_ratio": float(np.mean(uncertain_pixels)),
        "dirichlet_uncertain_pixel_percent": float(np.mean(uncertain_pixels) * 100.0),
    }

    return {
        "dirichlet_prob_posidonie": dirichlet_prob_posidonie,
        "dirichlet_uncertainty": dirichlet_uncertainty,
        "dirichlet_uncertainty_norm": dirichlet_uncertainty_norm,
        "uncertain_pixels_dirichlet": uncertain_pixels.astype(np.uint8),
        "stats": stats,
    }


def compute_ensemble_uncertainty(pred_list, proba_list):
    """
    Calcule l'incertitude de l'ensemble avec deux informations :

    1. Désaccord entre modèles :
       basé sur les segmentations finales 0/1.

    2. Incertitude Dirichlet pondérée :
       basée sur les cartes de probabilité de chaque modèle.
    """

    # ======================================================
    # 1. Désaccord entre modèles à partir des votes 0/1
    # ======================================================
    votes = np.stack(pred_list, axis=0)  # [n_models, H, W]
    n_models = votes.shape[0]

    votes_posidonie = np.sum(votes == 1, axis=0)
    votes_fond = np.sum(votes == 0, axis=0)

    majority_votes = np.maximum(votes_posidonie, votes_fond)

    agreement_map = majority_votes / n_models
    disagreement_map = 1.0 - agreement_map

    uncertain_pixels_disagreement = disagreement_map > 0.0

    # ======================================================
    # 2. Incertitude Dirichlet pondérée par la confiance
    # ======================================================
    probas = np.stack(proba_list, axis=0)  # [n_models, H, W]

    # Confiance de chaque modèle :
    # proche de 0 si p ≈ 0.5
    # proche de 1 si p ≈ 0 ou p ≈ 1
    confidence = np.abs(2.0 * probas - 1.0)

    # Évidence apportée par chaque modèle
    evidence_posidonie = np.sum(confidence * probas, axis=0)
    evidence_fond = np.sum(confidence * (1.0 - probas), axis=0)

    # Paramètres Dirichlet
    alpha_posidonie = 1.0 + evidence_posidonie
    alpha_fond = 1.0 + evidence_fond
    alpha_sum = alpha_posidonie + alpha_fond

    # Probabilité moyenne selon la Dirichlet
    dirichlet_prob_posidonie = alpha_posidonie / alpha_sum

    # Incertitude Dirichlet :
    # forte si peu d'évidence, faible si les modèles sont confiants
    dirichlet_uncertainty = 2.0 / alpha_sum

    # Normalisation pour l'affichage
    # Avec n_models modèles :
    # U_min = 2 / (2 + n_models) lorsque tous les modèles sont très confiants
    # U_max = 1 lorsque tous les modèles sont totalement incertains autour de 0.5
    u_min = 2.0 / (2.0 + n_models)
    u_max = 1.0

    dirichlet_uncertainty_norm = (dirichlet_uncertainty - u_min) / (u_max - u_min + 1e-8)
    dirichlet_uncertainty_norm = np.clip(dirichlet_uncertainty_norm, 0.0, 1.0)

    # Pixels incertains selon Dirichlet
    # Le seuil 0.5 est indicatif : tu peux le changer après observation
    uncertain_pixels_dirichlet = dirichlet_uncertainty_norm > 0.5

    stats = {
        # Désaccord entre modèles
        "agreement_mean": float(np.mean(agreement_map)),
        "agreement_percent": float(np.mean(agreement_map) * 100.0),

        "disagreement_mean": float(np.mean(disagreement_map)),
        "disagreement_percent": float(np.mean(disagreement_map) * 100.0),

        "uncertain_pixel_ratio": float(np.mean(uncertain_pixels_disagreement)),
        "uncertain_pixel_percent": float(np.mean(uncertain_pixels_disagreement) * 100.0),

        # Dirichlet pondérée
        "dirichlet_uncertainty_mean": float(np.mean(dirichlet_uncertainty)),
        "dirichlet_uncertainty_median": float(np.median(dirichlet_uncertainty)),
        "dirichlet_uncertainty_max": float(np.max(dirichlet_uncertainty)),

        "dirichlet_uncertainty_norm_mean": float(np.mean(dirichlet_uncertainty_norm)),
        "dirichlet_uncertainty_norm_median": float(np.median(dirichlet_uncertainty_norm)),
        "dirichlet_uncertainty_norm_max": float(np.max(dirichlet_uncertainty_norm)),

        "dirichlet_uncertain_pixel_ratio": float(np.mean(uncertain_pixels_dirichlet)),
        "dirichlet_uncertain_pixel_percent": float(np.mean(uncertain_pixels_dirichlet) * 100.0),
    }

    return {
        "agreement_map": agreement_map,
        "disagreement_map": disagreement_map,

        "dirichlet_prob_posidonie": dirichlet_prob_posidonie,
        "dirichlet_uncertainty": dirichlet_uncertainty,
        "dirichlet_uncertainty_norm": dirichlet_uncertainty_norm,

        "uncertain_pixels_disagreement": uncertain_pixels_disagreement.astype(np.uint8),
        "uncertain_pixels_dirichlet": uncertain_pixels_dirichlet.astype(np.uint8),

        "stats": stats,
    }

def make_uncertainty_heatmap(uncertainty_map):
    """
    Transforme une carte d'incertitude HxW en image RGB.
    Plus l'incertitude est forte, plus l'image est claire.
    """

    uncertainty = np.clip(uncertainty_map, 0, 1)
    heat = (uncertainty * 255).astype(np.uint8)

    # Carte simple : noir = sûr, jaune = incertain
    rgb = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = heat
    rgb[:, :, 1] = heat
    rgb[:, :, 2] = 0

    return rgb

def make_probability_heatmap(proba_map, colormap="gray"):
    """
    Transforme une carte de probabilité HxW en image RGB affichable.

    proba_map : valeurs entre 0 et 1.
    - gray : noir = faible probabilité, blanc = forte probabilité.
    - jet  : carte couleur, plus visuelle mais moins proche de Martin-Abadal.
    """

    proba = np.clip(proba_map, 0.0, 1.0)
    proba_uint8 = (proba * 255).astype(np.uint8)

    if colormap == "jet":
        color = cv2.applyColorMap(proba_uint8, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color

    # Par défaut : niveaux de gris comme la figure de Martin-Abadal
    return np.stack([proba_uint8, proba_uint8, proba_uint8], axis=-1)

def ensemble_predict(models_dict, tensor, img_rgb,
                     gt_mask=None, threshold=0.5, method='soft'):
    """
    soft    : moyenne des probas softmax → seuil unique (méthode actuelle)
    hard    : chaque modèle vote avec son seuil global (threshold)
    """
    proba_list  = []
    pred_list   = []
    n           = len(models_dict)

    for name, model in models_dict.items():
        proba = predict_proba(model, tensor)
        proba_list.append(proba)

        pred_list.append((proba >= threshold).astype(np.uint8))

    if method == 'soft':
        proba_final = np.mean(proba_list, axis=0)
        pred        = (proba_final >= threshold).astype(np.uint8)
        label       = f'Ensemble -- Soft voting ({n} modèles)'

    elif method == 'hard':
        vote_sum    = np.stack(pred_list, axis=0).sum(axis=0)
        pred        = (vote_sum > n / 2).astype(np.uint8)
        proba_final = vote_sum / n
        label       = f'Ensemble -- Hard voting ({n} modèles)'


    overlay = make_uncertainty_overlay(img_rgb, proba_final, gt_mask, threshold)
    metrics = compute_metrics(pred, gt_mask) if gt_mask is not None else None

    return {
        'name'   : label,
        'overlay': arr_to_b64(overlay),
        'metrics': metrics,
    }

import matplotlib
matplotlib.use('Agg')  # backend sans interface graphique, requis pour Flask
import matplotlib.pyplot as plt
import seaborn as sns

def short_label_for_plot(name, pipeline_info=None):
    """
    Label propre pour les figures :
    - SegFormer-B2 (single split)
    - SegFormer-B2 (fold 3)
    - VGG16-FCN8 (fold 5)
    - Soft-voting ensemble
    """

    name_str = str(name)

    # Cas ensemble
    if "ensemble" in name_str.lower():
        if "soft" in name_str.lower():
            return "Soft-voting ensemble"
        if "hard" in name_str.lower():
            return "Hard-voting ensemble"
        return "Ensemble"

    # Cas où on a accès aux métadonnées du pipeline
    if pipeline_info is not None:
        architecture = str(pipeline_info.get("architecture", "")).lower()
        variant = str(pipeline_info.get("variant", "")).lower()
        fold = pipeline_info.get("fold", None)

        # Nom du modèle
        if "segformer" in architecture or variant in ["b0", "b1", "b2"]:
            if variant in ["b0", "b1", "b2"]:
                model_label = f"SegFormer-{variant.upper()}"
            else:
                # fallback si variant absent
                if "b0" in name_str.lower():
                    model_label = "SegFormer-B0"
                elif "b1" in name_str.lower():
                    model_label = "SegFormer-B1"
                elif "b2" in name_str.lower():
                    model_label = "SegFormer-B2"
                else:
                    model_label = "SegFormer"
        elif "vgg" in architecture or "fcn" in architecture or "vgg" in name_str.lower():
            model_label = "VGG16-FCN8"
        else:
            model_label = name_str

        # Fold ou single split
        if fold is None or str(fold).lower() in ["", "none", "null", "-", "nan"]:
            split_label = "single split"
        else:
            split_label = f"fold {fold}"

        return f"{model_label} ({split_label})"

    # Fallback si on n'a pas les métadonnées
    lower_name = name_str.lower()

    if "segformer-b0" in lower_name or "b0" in lower_name:
        model_label = "SegFormer-B0"
    elif "segformer-b1" in lower_name or "b1" in lower_name:
        model_label = "SegFormer-B1"
    elif "segformer-b2" in lower_name or "b2" in lower_name:
        model_label = "SegFormer-B2"
    elif "vgg" in lower_name or "fcn" in lower_name:
        model_label = "VGG16-FCN8"
    else:
        model_label = name_str[:35]

    import re
    match = re.search(r"fold[_\s-]*(\d+)", lower_name)
    if match:
        return f"{model_label} (fold {match.group(1)})"

    return f"{model_label} (single split)"

def generate_eval_figures(results_val, results_test, split_label, method, threshold,
                          model_thresholds=None, active_pipelines=None):    
    """
    Génère :
      - 1 figure ROC+PR par modèle (val vs test), empilées verticalement
      - 1 figure de matrices de confusion par modèle (val | test), empilées verticalement

    model_thresholds : dict {nom_modèle: seuil_utilisé} pour afficher le bon seuil par modèle
    """
    names = list(results_val.keys())
    n     = len(names)
    model_thresholds = model_thresholds or {}

    # === FIGURE 1 : ROC + PR - une ligne par modèle ===
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.5*n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, name in enumerate(names):
        mv, mt = results_val[name], results_test[name]

        pipeline_info = active_pipelines.get(name) if active_pipelines else None
        short_name = short_label_for_plot(name, pipeline_info)

        ax_roc = axes[i, 0]
        ax_roc.plot(
            mv['fpr'], mv['tpr'],
            label=f"{short_name} Val AUC={mv['auc_roc']:.4f}",
            lw=2
        )
        ax_roc.plot(
            mt['fpr'], mt['tpr'],
            label=f"{short_name} Test AUC={mt['auc_roc']:.4f}",
            lw=2
        )
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.set_title(f'{short_name} - ROC')
        ax_roc.legend(fontsize=7)
        ax_roc.grid(True)

        ax_pr = axes[i, 1]
        ax_pr.plot(
            mv['rec_curve'], mv['prec_curve'],
            label=f"{short_name} Val AUC-PR={mv['auc_pr']:.4f}",
            lw=2
        )
        ax_pr.plot(
            mt['rec_curve'], mt['prec_curve'],
            label=f"{short_name} Test AUC-PR={mt['auc_pr']:.4f}",
            lw=2
        )
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'{short_name} - PR')
        ax_pr.legend(fontsize=7)
        ax_pr.grid(True)

    plt.tight_layout()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    roc_pr_b64 = base64.b64encode(buf1.getvalue()).decode()

    # === FIGURE 2 : Matrices de confusion — une ligne par modèle, VAL | TEST ===
    fig2, axes2 = plt.subplots(n, 2, figsize=(13.5, 5.5*n))
    fig2.suptitle('Confusion Matrices - Validation and Test', fontsize=13)

    if n == 1:
        axes2 = axes2.reshape(1, 2)

    for i, name in enumerate(names):
        thr_display = model_thresholds.get(name, threshold)

        if thr_display is None:
            thr_display = DECISION_THRESHOLD # <- seuil propre au modèle

        for j, (m, split_name) in enumerate([
            (results_val[name], 'VAL'),
            (results_test[name], 'TEST')
        ]):
            ax = axes2[i, j]

            cm = np.array([
                [m['tn'], m['fp']],
                [m['fn'], m['tp']]
            ])

            cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

            sns.heatmap(
                cm_pct,
                annot=False,
                cmap='Blues',
                vmin=0,
                vmax=100,
                cbar=False,
                square=True,
                ax=ax,
                xticklabels=['Pred. background', 'Pred. posidonia'],
                yticklabels=['True background', 'True posidonia'],
                linewidths=1,
                linecolor='white'
            )

            for r in range(2):
                for c in range(2):
                    pct_value = cm_pct[r, c]
                    count = cm[r, c]

                    color = 'white' if pct_value > 50 else 'black'

                    ax.text(
                        c + 0.5,
                        r + 0.42,
                        f"{pct_value:.1f}%",
                        ha='center',
                        va='center',
                        color=color,
                        fontsize=12,
                        fontweight='bold'
                    )

                    ax.text(
                        c + 0.5,
                        r + 0.62,
                        f"{count:,} px".replace(",", " "),
                        ha='center',
                        va='center',
                        color=color,
                        fontsize=8
                    )

            pipeline_info = active_pipelines.get(name) if active_pipelines else None
            short_name = short_label_for_plot(name, pipeline_info)
            total_pixels = int(cm.sum())

            split_display = "Validation" if split_name == "VAL" else "Test"

            ax.set_title(
                (
                    f"{short_name} - {split_display} Set\n"
                    f"Threshold: {thr_display:.2f}   -   "
                    f"Accuracy: {m['accuracy'] * 100:.2f}%\n"
                    f"Dice/F1: {m['f1'] * 100:.2f}%   -   "
                    f"mIoU: {m['miou'] * 100:.2f}%   -  "
                    f"N: {total_pixels:,} px"
                ).replace(",", " "),
                fontsize=9,
                pad=12,
                linespacing=1.35
            )
    fig2.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.95],
        h_pad=3.0,
        w_pad=2.5
    )
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig2)
    cm_b64 = base64.b64encode(buf2.getvalue()).decode()

    return roc_pr_b64, cm_b64

def generate_eval_figures_single(results, split_label, method, threshold,
                                 model_thresholds=None, active_pipelines=None):
    """
    Version pour UN SEUL split (val ou test) : grille multi-modèles.

    model_thresholds : dict {nom_modèle: seuil_utilisé} pour afficher le bon seuil par modèle
    """
    names = list(results.keys())
    n     = len(names)
    model_thresholds = model_thresholds or {}

    # === FIGURE 1 : ROC + PR (tous les modèles superposés) ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{split_label} - {method} voting', fontsize=13)

    for name in names:
        m = results[name]
        pipeline_info = active_pipelines.get(name) if active_pipelines else None
        short_name = short_label_for_plot(name, pipeline_info)

        axes[0].plot(
            m['fpr'], m['tpr'],
            label=f"{short_name} AUC={m['auc_roc']*100:.2f}%",
            lw=2
        )

        axes[1].plot(
            m['rec_curve'], m['prec_curve'],
            label=f"{short_name} AUC-PR={m['auc_pr']*100:.2f}%",
            lw=2
        )
    axes[0].plot([0,1],[0,1],'k--', lw=1)
    axes[0].set_xlabel('Fall-out (FPR)'); axes[0].set_ylabel('Recall (TPR)')
    axes[0].set_title('ROC Curve'); axes[0].legend(fontsize=8); axes[0].grid(True)

    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve'); axes[1].legend(fontsize=8); axes[1].grid(True)

    plt.tight_layout()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    roc_pr_b64 = base64.b64encode(buf1.getvalue()).decode()

    # ==========================================================
    # FIGURE 2 : MATRICES DE CONFUSION — VERSION LISIBLE
    # ==========================================================

    names_cm = list(results.keys())
    n_cm = len(names_cm)

    n_cols = min(n_cm, 3)
    n_rows = (n_cm + n_cols - 1) // n_cols

    fig2, axes2 = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.4 * n_cols, 6.0 * n_rows)
    )

    fig2.suptitle(
        f"Matrices de confusion - {split_label}",
        fontsize=18,
        fontweight="bold",
        y=0.99
    )

    # Fonctionne aussi lorsqu'il n'y a qu'une seule matrice
    axes2_flat = np.array(axes2, ndmin=1).flatten()

    for i, name in enumerate(names_cm):

        m = results[name]
        ax = axes2_flat[i]

        # Seuil réellement utilisé pour ce modèle
        thr_display = model_thresholds.get(name, threshold)

        if thr_display is None:
            thr_display = DECISION_THRESHOLD

        # Matrice :
        # [[TN, FP],
        #  [FN, TP]]
        cm = np.array([
            [m["tn"], m["fp"]],
            [m["fn"], m["tp"]]
        ])

        # Pourcentages normalisés par classe réelle
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        ) * 100.0

        sns.heatmap(
            cm_pct,
            annot=False,
            cmap="Blues",
            vmin=0,
            vmax=100,
            cbar=False,
            square=True,
            ax=ax,
            xticklabels=["Fond", "Posidonie"],
            yticklabels=["Fond", "Posidonie"],
            linewidths=1.5,
            linecolor="white"
        )

        # Axes beaucoup plus lisibles
        ax.set_xlabel(
            "Classe prédite",
            fontsize=13,
            fontweight="bold",
            labelpad=10
        )

        ax.set_ylabel(
            "Classe réelle",
            fontsize=13,
            fontweight="bold",
            labelpad=10
        )

        ax.tick_params(
            axis="both",
            labelsize=12
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0
        )

        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0
        )

        # Valeurs dans les quatre cases
        for r in range(2):
            for c in range(2):

                pct_value = cm_pct[r, c]
                count = cm[r, c]

                text_color = (
                    "white"
                    if pct_value > 50
                    else "black"
                )

                # Pourcentage
                ax.text(
                    c + 0.5,
                    r + 0.40,
                    f"{pct_value:.1f} %",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=17,
                    fontweight="bold"
                )

                # Nombre de pixels
                ax.text(
                    c + 0.5,
                    r + 0.62,
                    f"{count:,} px".replace(",", " "),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11
                )

        # Nom court du modèle
        pipeline_info = (
            active_pipelines.get(name)
            if active_pipelines
            else None
        )

        short_name = short_label_for_plot(
            name,
            pipeline_info
        )

        # On garde seulement l'information indispensable
        # Les autres métriques sont déjà dans le tableau HTML
        ax.set_title(
            f"{short_name}\nSeuil = {thr_display:.2f}",
            fontsize=13,
            fontweight="bold",
            pad=14
        )

    # Masquer les axes inutilisés si la grille n'est pas complète
    for j in range(n_cm, len(axes2_flat)):
        axes2_flat[j].axis("off")

    fig2.tight_layout(
        rect=[0.02, 0.02, 0.98, 0.94],
        h_pad=3.0,
        w_pad=3.0
    )

    buf2 = io.BytesIO()

    fig2.savefig(
        buf2,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig2)

    cm_b64 = base64.b64encode(
        buf2.getvalue()
    ).decode()


    return roc_pr_b64, cm_b64

def make_native_class_overlay(pred, split_source):
    """
    Construit une image de segmentation avec les couleurs natives du dataset :
    - testing  : noir=fond, blanc=posidonie (comme les masques GT testing)
    - training : rouge=fond, magenta=posidonie (comme les masques GT training)
    Pas de blend avec l'image originale — couleurs pures pour rester
    cohérent visuellement avec le format des fichiers GT.
    """
    h, w = pred.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    if split_source == 'training':
        result[pred == 0] = [255, 0, 0]    # rouge = fond
        result[pred == 1] = [255, 0, 255]  # magenta = posidonie
    else:
        # testing (ou par défaut si image hors dataset) : noir/blanc
        result[pred == 0] = [0, 0, 0]      # noir = fond
        result[pred == 1] = [255, 255, 255]  # blanc = posidonie

    return result

def make_error_zones_overlay(pred, proba, gt_mask, threshold=0.5,
                               seuil_bas=0.3, seuil_haut=0.7):
    """
    Construit l'image des zones d'erreur :
    - blanc  : posidonie correctement détectée (TP)
    - noir   : fond correctement détecté (TN)
    - rouge  : fausse alarme (FP) — fond prédit posidonie
    - bleu   : posidonie manquée (FN) — posidonie prédite fond
    - jaune  : zone de doute (0.3 < proba < 0.7), prioritaire sur le reste
    """
    h, w = pred.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    tp = (pred == 1) & (gt_mask == 1)
    tn = (pred == 0) & (gt_mask == 0)
    fp = (pred == 1) & (gt_mask == 0)
    fn = (pred == 0) & (gt_mask == 1)
    doute = (proba > seuil_bas) & (proba < seuil_haut)

    result[tn] = [0, 0, 0]
    result[tp] = [255, 255, 255]
    result[fp] = [255, 0, 0]
    result[fn] = [0, 80, 220]
    result[doute] = [255, 200, 0]  # jaune, prioritaire — appliqué en dernier

    return result

def make_error_zones_overlay_without_uncertainty(pred, gt_mask):
    """
    Affiche uniquement les erreurs de segmentation, sans zone de doute 0.3-0.7.
    - noir   : TN fond correctement prédit
    - blanc  : TP posidonie correctement prédite
    - rouge  : FP fond prédit comme posidonie
    - bleu   : FN posidonie manquée
    """
    h, w = pred.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    tp = (pred == 1) & (gt_mask == 1)
    tn = (pred == 0) & (gt_mask == 0)
    fp = (pred == 1) & (gt_mask == 0)
    fn = (pred == 0) & (gt_mask == 1)

    result[tn] = [0, 0, 0]
    result[tp] = [255, 255, 255]
    result[fp] = [255, 0, 0]
    result[fn] = [0, 80, 220]

    return result

def load_model_from_pipeline(pipeline):
    """
    Charge à la demande le modèle correspondant au pipeline.
    """

    checkpoint_path = Path(
        pipeline["checkpoint_path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    label = (
        pipeline.get("label")
        or pipeline.get("experiment_name")
        or checkpoint_path.stem
    )

    return get_or_load_model(
        label,
        checkpoint_path
    )


# === CLAHE ===

def apply_clahe(img_rgb, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Égalisation locale du contraste (CLAHE) sur le canal L de LAB.

    Args:
        img_rgb : np.ndarray [H, W, 3] uint8 RGB
        clip_limit : seuil de limitation du contraste (default 3.0)
        tile_grid_size : taille de la grille de tuiles (default 8x8)

    Returns:
        np.ndarray [H, W, 3] uint8 RGB avec contraste local rehaussé

    Référence : Pizer et al. (1987), Zuiderveld (1994).
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def dynamic_expansion_rgb_eq(img_rgb, percentile_low=1, percentile_high=99):
    """
    Expansion dynamique par canal + égalisation des moyennes RGB.

    Étape 1 : étire chaque canal entre [percentile_low, percentile_high] vers [0, 255]
    Étape 2 : ajuste chaque canal pour avoir la même moyenne (supprime dominante couleur)

    Args:
        img_rgb : np.ndarray [H, W, 3] uint8 RGB
        percentile_low : percentile bas pour l'étirement (default 1)
        percentile_high : percentile haut pour l'étirement (default 99)

    Returns:
        np.ndarray [H, W, 3] uint8 RGB avec contraste maximisé et couleur neutralisée

    Référence : Bazeille et al. (2006).
    """
    img_out = img_rgb.astype(np.float32).copy()

    # === Étape 1 : expansion dynamique par canal ===
    for ch in range(3):
        low = np.percentile(img_out[:, :, ch], percentile_low)
        high = np.percentile(img_out[:, :, ch], percentile_high)
        if high > low:
            img_out[:, :, ch] = (img_out[:, :, ch] - low) / (high - low) * 255
            img_out[:, :, ch] = np.clip(img_out[:, :, ch], 0, 255)

    # === Étape 2 : égalisation des moyennes RGB ===
    target_mean = img_out.mean()  # moyenne globale = cible commune
    for ch in range(3):
        ch_mean = img_out[:, :, ch].mean()
        if ch_mean > 0:
            img_out[:, :, ch] *= (target_mean / ch_mean)

    img_out = np.clip(img_out, 0, 255)
    return img_out.astype(np.uint8)


def apply_pipeline_preprocess(pil_img, preprocess):
    """
    Applique le prétraitement associé au pipeline.
    Entrée  : image PIL
    Sortie  : image PIL
    """

    if preprocess is None or preprocess == "none":
        return pil_img.convert("RGB")

    img_rgb = np.array(pil_img.convert("RGB"))

    if preprocess == "clahe":
        img_out = apply_clahe(img_rgb)

    elif preprocess == "expansion_rgb":
        img_out = dynamic_expansion_rgb_eq(img_rgb)

    else:
        raise ValueError(f"Prétraitement inconnu : {preprocess}")

    return Image.fromarray(img_out.astype(np.uint8))

# ============================================================
# RECALIBRATION DES SEUILS VGG16-FCN8 BESTVAL
# ============================================================

def read_fold_validation_names(csv_path):
    """
    Lit les noms d'images contenus dans un CSV de validation.

    Colonnes reconnues :
      - image
      - filename
      - name
      - image_path
      - path

    Les extensions et les éventuels chemins sont supprimés.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV de validation introuvable : {csv_path}"
        )

    image_names = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                f"Le CSV {csv_path.name} ne contient pas d'en-tête."
            )

        normalized_fields = {
            str(field).strip().lower(): field
            for field in reader.fieldnames
            if field is not None
        }

        accepted_columns = (
            "image",
            "filename",
            "name",
            "image_path",
            "path",
        )

        selected_column = next(
            (
                normalized_fields[column]
                for column in accepted_columns
                if column in normalized_fields
            ),
            None
        )

        if selected_column is None:
            raise ValueError(
                f"Aucune colonne d'image reconnue dans {csv_path.name}. "
                f"Colonnes trouvées : {reader.fieldnames}"
            )

        for row in reader:
            raw_value = str(
                row.get(selected_column, "")
            ).strip()

            if not raw_value:
                continue

            # Compatible avec les chemins Windows et Unix.
            normalized_value = raw_value.replace("\\", "/")
            image_name = Path(normalized_value).stem

            if image_name:
                image_names.append(image_name)

    # Élimination des doublons sans modifier l'ordre.
    image_names = list(dict.fromkeys(image_names))

    if not image_names:
        raise ValueError(
            f"Aucun nom d'image trouvé dans {csv_path.name}."
        )

    return image_names


def find_existing_image(image_name, image_dir):
    """
    Retrouve une image indépendamment de son extension.
    """
    image_dir = Path(image_dir)

    for extension in (
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG",
    ):
        candidate = image_dir / f"{image_name}{extension}"

        if candidate.exists():
            return candidate

    return None


def compute_calibration_metrics(
    probabilities,
    labels,
    threshold
):
    """
    Calcule les métriques globales pour un seuil candidat.

    Le score de sélection est :
        (recall + specificity) / 2

    Cette expression est équivalente, pour la sélection du seuil,
    à maximiser recall - fall-out.
    """
    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    tp = int(
        ((predictions == 1) & (labels == 1)).sum()
    )
    fp = int(
        ((predictions == 1) & (labels == 0)).sum()
    )
    fn = int(
        ((predictions == 0) & (labels == 1)).sum()
    )
    tn = int(
        ((predictions == 0) & (labels == 0)).sum()
    )

    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / (total + 1e-8)
    )

    precision = (
        tp / (tp + fp + 1e-8)
    )

    recall = (
        tp / (tp + fn + 1e-8)
    )

    fallout = (
        fp / (fp + tn + 1e-8)
    )

    specificity = 1.0 - fallout

    f1 = (
        2.0 * precision * recall
        / (precision + recall + 1e-8)
    )

    iou_posidonia = (
        tp / (tp + fp + fn + 1e-8)
    )

    iou_background = (
        tn / (tn + fp + fn + 1e-8)
    )

    miou = (
        iou_posidonia + iou_background
    ) / 2.0

    calibration_score = (
        recall + specificity
    ) / 2.0

    return {
        "threshold": round(float(threshold), 4),
        "calibration_score": float(calibration_score),
        "recall_minus_fallout": float(
            recall - fallout
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fallout": float(fallout),
        "f1": float(f1),
        "dice": float(f1),
        "iou": float(iou_posidonia),
        "iou_background": float(iou_background),
        "miou": float(miou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def recalibrate_one_bestval_pipeline(
    pipeline,
    validation_csv,
    candidate_thresholds=None,
):
    """
    Recalibre le seuil d'un checkpoint VGG16-FCN8 bestval
    sur le sous-ensemble de validation de son fold.

    Le nouveau seuil est sauvegardé dans le metadata JSON
    correspondant au pipeline.
    """
    checkpoint_path = Path(
        pipeline["checkpoint_path"]
    )

    metadata_path = Path(
        pipeline["metadata_path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    if not checkpoint_path.stem.lower().endswith(
        "_bestval"
    ):
        raise ValueError(
            f"Le checkpoint n'est pas un bestval : "
            f"{checkpoint_path.name}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata introuvable : {metadata_path}"
        )

    if candidate_thresholds is None:
        candidate_thresholds = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]

    candidate_thresholds = [
        round(float(value), 4)
        for value in candidate_thresholds
    ]

    image_names = read_fold_validation_names(
        validation_csv
    )

    model = load_model_from_pipeline(pipeline)

    img_h = int(
        pipeline.get("img_h") or 360
    )
    img_w = int(
        pipeline.get("img_w") or 480
    )

    preprocess = pipeline.get("preprocess")

    all_probabilities = []
    all_labels = []
    skipped_images = []

    print(
        f"\n[RECALIBRATION] "
        f"{pipeline['experiment_name']}"
    )
    print(
        f"  Checkpoint : {checkpoint_path.name}"
    )
    print(
        f"  Validation : {Path(validation_csv).name}"
    )
    print(
        f"  Images attendues : {len(image_names)}"
    )

    for image_name in image_names:
        image_path = find_existing_image(
            image_name,
            TRAINING_IMGS
        )

        mask_path = (
            TRAINING_MSKS
            / f"{image_name}_gt.png"
        )

        if (
            image_path is None
            or not mask_path.exists()
        ):
            skipped_images.append(image_name)
            continue

        pil_image = Image.open(
            image_path
        ).convert("RGB")

        pil_for_model = apply_pipeline_preprocess(
            pil_image,
            preprocess,
        )

        tensor = img_to_tensor(
            pil_for_model,
            img_h=img_h,
            img_w=img_w,
        )

        probability = predict_proba(
            model,
            tensor,
        )

        ground_truth = load_gt_mask(
            mask_path,
            split="training",
            img_h=img_h,
            img_w=img_w,
        )

        if probability.shape != ground_truth.shape:
            probability = cv2.resize(
                probability,
                (
                    ground_truth.shape[1],
                    ground_truth.shape[0],
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        all_probabilities.append(
            probability.reshape(-1)
        )

        all_labels.append(
            ground_truth.reshape(-1)
        )

    if not all_probabilities:
        raise RuntimeError(
            f"Aucune image valide n'a été trouvée "
            f"pour {pipeline['experiment_name']}."
        )

    probabilities = np.concatenate(
        all_probabilities
    )

    labels = np.concatenate(
        all_labels
    ).astype(np.uint8)

    candidate_results = [
        compute_calibration_metrics(
            probabilities,
            labels,
            threshold,
        )
        for threshold in candidate_thresholds
    ]

    # Les seuils sont parcourus dans l'ordre croissant.
    # En cas d'égalité exacte, le premier est conservé.
    best_result = max(
        candidate_results,
        key=lambda result: result[
            "calibration_score"
        ],
    )

    best_threshold = float(
        best_result["threshold"]
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    previous_threshold = metadata.get(
        "val_threshold"
    )

    # Conservation de l'ancienne valeur pour la traçabilité.
    threshold_history = metadata.setdefault(
        "threshold_history",
        [],
    )

    threshold_history.append({
        "val_threshold": previous_threshold,
        "source": metadata.get(
            "val_threshold_source",
            "legacy_metadata",
        ),
        "checkpoint": metadata.get(
            "val_threshold_checkpoint"
        ),
    })

    metadata["val_threshold"] = (
        best_threshold
    )

    metadata["val_threshold_source"] = (
        "recomputed_on_fold_validation_"
        "with_bestval_checkpoint"
    )

    metadata["val_threshold_checkpoint"] = (
        checkpoint_path.name
    )

    metadata[
        "val_threshold_validation_csv"
    ] = Path(validation_csv).name

    metadata["val_threshold_criterion"] = (
        "mean_recall_specificity"
    )

    metadata[
        "val_threshold_candidate_values"
    ] = candidate_thresholds

    metadata[
        "val_threshold_metrics"
    ] = best_result

    metadata[
        "val_threshold_all_candidates"
    ] = candidate_results

    metadata["val_threshold_n_images"] = (
        len(all_probabilities)
    )

    metadata[
        "val_threshold_n_skipped"
    ] = len(skipped_images)

    metadata[
        "val_threshold_skipped_images"
    ] = skipped_images

    # Écriture dans un fichier temporaire,
    # puis remplacement du JSON original.
    temporary_path = metadata_path.with_name(
        metadata_path.stem
        + "_temporary.json"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(metadata_path)

    print(
        f"  Ancien seuil : {previous_threshold}"
    )
    print(
        f"  Nouveau seuil : {best_threshold}"
    )
    print(
        f"  Score : "
        f"{best_result['calibration_score']:.6f}"
    )
    print(
        f"  Images utilisées : "
        f"{len(all_probabilities)}"
    )
    print(
        f"  Images ignorées : "
        f"{len(skipped_images)}"
    )
    print(
        f"  JSON mis à jour : "
        f"{metadata_path.name}"
    )

    return {
        "experiment_name": (
            pipeline["experiment_name"]
        ),
        "checkpoint": checkpoint_path.name,
        "metadata": metadata_path.name,
        "validation_csv": (
            Path(validation_csv).name
        ),
        "previous_threshold": (
            previous_threshold
        ),
        "new_threshold": best_threshold,
        "n_images": len(all_probabilities),
        "n_skipped": len(skipped_images),
        "validation_metrics": best_result,
    }


@app.route(
    "/recalibrate_vgg_bestval",
    methods=["POST"],
)
def recalibrate_vgg_bestval():
    """
    Recalibre automatiquement les cinq folds VGG16-FCN8.

    Correspondance :
      pipeline fold 1 -> fold0_val.csv
      pipeline fold 2 -> fold1_val.csv
      ...
      pipeline fold 5 -> fold4_val.csv
    """
    data = request.get_json(
        silent=True
    ) or {}

    candidate_thresholds = data.get(
        "candidate_thresholds"
    )

    if candidate_thresholds is None:
        candidate_thresholds = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]

    if not VGG_FOLDS_DIR.exists():
        return jsonify({
            "error": (
                "Dossier des folds introuvable : "
                f"{VGG_FOLDS_DIR}"
            )
        }), 400

    pipelines = discover_pipelines(
        MODELS_DIR
    )

    vgg_pipelines = []

    for pipeline in pipelines.values():
        architecture = str(
            pipeline.get("architecture", "")
        ).lower()

        checkpoint_path = Path(
            pipeline["checkpoint_path"]
        )

        fold = pipeline.get("fold")

        is_vgg = (
            "vgg" in architecture
            or "fcn" in architecture
            or "fcn8" in checkpoint_path.name.lower()
        )

        is_bestval = (
            checkpoint_path.stem
            .lower()
            .endswith("_bestval")
        )

        if (
            is_vgg
            and is_bestval
            and fold is not None
        ):
            vgg_pipelines.append(pipeline)

    vgg_pipelines = sorted(
        vgg_pipelines,
        key=lambda pipeline: int(
            pipeline["fold"]
        ),
    )

    if not vgg_pipelines:
        return jsonify({
            "error": (
                "Aucun pipeline VGG16-FCN8 bestval "
                "n'a été détecté."
            )
        }), 400

    results = []
    errors = []

    for pipeline in vgg_pipelines:
        fold_label = int(
            pipeline["fold"]
        )

        # Fold affiché 1 -> fichier fold0_val.csv
        csv_index = fold_label - 1

        validation_csv = (
            VGG_FOLDS_DIR
            / f"fold{csv_index}_val.csv"
        )

        try:
            result = (
                recalibrate_one_bestval_pipeline(
                    pipeline=pipeline,
                    validation_csv=validation_csv,
                    candidate_thresholds=(
                        candidate_thresholds
                    ),
                )
            )

            results.append(result)

        except Exception as exc:
            errors.append({
                "experiment_name": (
                    pipeline.get(
                        "experiment_name"
                    )
                ),
                "fold": fold_label,
                "validation_csv": (
                    validation_csv.name
                ),
                "error": str(exc),
            })

    return jsonify({
        "message": (
            "Recalibration terminée. "
            "Redémarre l'application avant "
            "de relancer les évaluations."
        ),
        "folds_directory": str(
            VGG_FOLDS_DIR
        ),
        "candidate_thresholds": (
            candidate_thresholds
        ),
        "results": results,
        "errors": errors,
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': list(MODELS_CONFIG.keys())
    })


# On pourra afficher deux choix pour l'utilisateur :
# - seuil fixe 0.5 pour tous les modèles (fixed)
# - seuil calibré sur validation pour chaque modèle (calibrated)
# et aussi le mode d'incertitude : standard (Dirichlet classique) ou threshold (Dirichlet adapté au seuil de décision).
@app.route('/config', methods=['GET'])
def get_config():
    """
    Retourne les modèles disponibles, le seuil fixe
    et les seuils calibrés.

    MODELS_CONFIG contient les checkpoints disponibles.
    MODELS contient uniquement les modèles déjà chargés en mémoire.
    """

    available_models = list(MODELS_CONFIG.keys())

    fixed_thresholds = {
        name: DECISION_THRESHOLD
        for name in available_models
    }

    calibrated_thresholds = {
        name: MODEL_THRESHOLDS.get(
            name,
            DECISION_THRESHOLD
        )
        for name in available_models
    }

    return jsonify({
        'models': available_models,

        'decision_threshold': DECISION_THRESHOLD,

        'fixed_thresholds': fixed_thresholds,
        'calibrated_thresholds': calibrated_thresholds,

        'threshold_modes': [
            'fixed',
            'calibrated'
        ],

        'uncertainty_modes': [
            'standard',
            'threshold'
        ],

        'threshold_mode_default':
            THRESHOLD_MODE_DEFAULT,

        'uncertainty_mode_default':
            UNCERTAINTY_MODE_DEFAULT,
    })

def get_model_decision_threshold(model_name, threshold_mode="fixed"):
    """
    Retourne le seuil de décision utilisé pour produire le masque binaire.

    - fixed      : seuil fixe 0.5
    - calibrated : seuil calibré sur validation, chargé depuis MODEL_THRESHOLDS
    """
    if threshold_mode == "calibrated":
        return MODEL_THRESHOLDS.get(model_name, DECISION_THRESHOLD)

    return DECISION_THRESHOLD

def compute_threshold_adapted_dirichlet_uncertainty(proba_list, threshold=0.5):
    """
    Variante d'incertitude Dirichlet adaptée au seuil de décision.

    Ici, l'incertitude est maximale lorsque la probabilité est proche
    du seuil de décision choisi.
    """

    probas = np.stack(proba_list, axis=0)
    n_models = probas.shape[0]

    distance = np.abs(probas - threshold)

    max_distance = np.where(
        probas >= threshold,
        1.0 - threshold,
        threshold
    )

    confidence = distance / (max_distance + 1e-8)
    confidence = np.clip(confidence, 0.0, 1.0)

    evidence_posidonie = np.sum(confidence * probas, axis=0)
    evidence_fond = np.sum(confidence * (1.0 - probas), axis=0)

    alpha_posidonie = 1.0 + evidence_posidonie
    alpha_fond = 1.0 + evidence_fond
    alpha_sum = alpha_posidonie + alpha_fond

    dirichlet_prob_posidonie = alpha_posidonie / alpha_sum
    dirichlet_uncertainty = 2.0 / alpha_sum

    u_min = 2.0 / (2.0 + n_models)
    u_max = 1.0

    dirichlet_uncertainty_norm = (
        dirichlet_uncertainty - u_min
    ) / (u_max - u_min + 1e-8)

    dirichlet_uncertainty_norm = np.clip(dirichlet_uncertainty_norm, 0.0, 1.0)

    uncertain_pixels = dirichlet_uncertainty_norm > DIRICHLET_UNCERTAINTY_THRESHOLD

    stats = {
        "dirichlet_uncertainty_mean": float(np.mean(dirichlet_uncertainty)),
        "dirichlet_uncertainty_median": float(np.median(dirichlet_uncertainty)),
        "dirichlet_uncertainty_max": float(np.max(dirichlet_uncertainty)),

        "dirichlet_uncertainty_norm_mean": float(np.mean(dirichlet_uncertainty_norm)),
        "dirichlet_uncertainty_norm_median": float(np.median(dirichlet_uncertainty_norm)),
        "dirichlet_uncertainty_norm_max": float(np.max(dirichlet_uncertainty_norm)),

        "dirichlet_uncertain_pixel_ratio": float(np.mean(uncertain_pixels)),
        "dirichlet_uncertain_pixel_percent": float(np.mean(uncertain_pixels) * 100.0),
    }

    return {
        "dirichlet_prob_posidonie": dirichlet_prob_posidonie,
        "dirichlet_uncertainty": dirichlet_uncertainty,
        "dirichlet_uncertainty_norm": dirichlet_uncertainty_norm,
        "uncertain_pixels_dirichlet": uncertain_pixels.astype(np.uint8),
        "stats": stats,
    }

# Pour le choix du mode d'incertitude
def compute_uncertainty_by_mode(proba_list, uncertainty_mode="standard", threshold=0.5):
    """
    Choisit la méthode de calcul d'incertitude.

    - standard  : Dirichlet standard, centré sur 0.5
    - threshold : Dirichlet adapté au seuil de décision
    """
    if uncertainty_mode == "threshold":
        return compute_threshold_adapted_dirichlet_uncertainty(
            proba_list,
            threshold=threshold
        )

    return compute_dirichlet_uncertainty_from_probas(proba_list)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    # ==========================================================
    # 0. Paramètres reçus depuis le frontend
    # ==========================================================
    threshold_mode = request.form.get("threshold_mode", THRESHOLD_MODE_DEFAULT)
    uncertainty_mode = request.form.get("uncertainty_mode", UNCERTAINTY_MODE_DEFAULT)
    ensemble_method = request.form.get('ensemble_method', 'soft')
    inference_preprocess = request.form.get("inference_preprocess", "none")

    if inference_preprocess not in ("none", "clahe", "expansion_rgb"):
        inference_preprocess = "none"

    if threshold_mode not in ("fixed", "calibrated"):
        threshold_mode = THRESHOLD_MODE_DEFAULT

    if uncertainty_mode not in ("standard", "threshold"):
        uncertainty_mode = UNCERTAINTY_MODE_DEFAULT

    if ensemble_method not in ("soft", "hard"):
        ensemble_method = "soft"

    selected_models = request.form.getlist('selected_models')

    # Si le frontend envoie une seule chaîne séparée par des virgules
    if len(selected_models) == 1 and "," in selected_models[0]:
        selected_models = [
            m.strip()
            for m in selected_models[0].split(",")
            if m.strip()
        ]

    # ==========================================================
    # Chargement uniquement des modèles sélectionnés
    # ==========================================================

    if not selected_models:
        selected_models = list(
            MODELS_CONFIG.keys()
        )

    active_models = {}

    for name in selected_models:

        if name not in MODELS_CONFIG:
            continue

        checkpoint_path = MODELS_CONFIG[name]

        model = get_or_load_model(
            name,
            checkpoint_path
        )

        active_models[name] = model


    if not active_models:
        return jsonify({
            'error': 'Aucun modèle sélectionné ou disponible'
        }), 400

    # ==========================================================
    # 1. Lecture de l'image
    # ==========================================================
    file = request.files['image']
    pil_orig = Image.open(file.stream).convert("RGB")

    # Image éventuellement prétraitée pour l'inférence
    pil_for_inference = apply_pipeline_preprocess(
        pil_orig,
        None if inference_preprocess == "none" else inference_preprocess
    )

    img_rgb_original = np.array(
    pil_orig.convert("RGB")
    )

    # Image prétraitée affichable
    preprocessed_rgb = np.array(
        pil_for_inference.convert('RGB').resize((IMG_W, IMG_H), Image.BILINEAR)
    )

    filename = file.filename or ''
    base = Path(filename).stem

    # ==========================================================
    # 2. Recherche du masque GT si l'image appartient au dataset
    # ==========================================================
    gt_path_found = None
    gt_source = None

    for split_name, img_dir, mask_dir in [
        ('testing', TESTING_IMGS, TESTING_MSKS),
        ('training', TRAINING_IMGS, TRAINING_MSKS),
    ]:
        gt_path = mask_dir / (
            base + '_gt.png'
        )

        if gt_path.exists():
            gt_path_found = gt_path
            gt_source = split_name
            break

    orig_b64 = arr_to_b64(img_rgb_original)
    preprocessed_b64 = arr_to_b64(preprocessed_rgb)

    # Si l'image n'est pas dans le dataset, on utilise testing par défaut
    split_for_colors = gt_source if gt_source else 'testing'

    # ==========================================================
    # 3. Prédiction des modèles individuels
    # ==========================================================
    # Résolution commune utilisée uniquement pour la fusion d'ensemble
    ensemble_img_h = IMG_H   # 352
    ensemble_img_w = IMG_W   # 480

    results = []

    # On garde les cartes de probabilité et les segmentations binaires
    # pour construire ensuite le modèle d'ensemble.
    proba_list = []
    pred_list = []

    model_thresholds_used = {}

    # GT uniquement destiné à l'ensemble
    gt_mask_ensemble = None

    if gt_path_found is not None:
        gt_mask_ensemble = load_gt_mask(
            gt_path_found,
            gt_source,
            img_h=ensemble_img_h,
            img_w=ensemble_img_w
        )

    # ==========================================================
    # GT général conservé pour compatibilité avec le frontend
    # ==========================================================

    gt_general_b64 = None

    if gt_path_found is not None:

        gt_general_rgb = (
            load_gt_mask_native_colors(
                gt_path_found,
                gt_source,
                img_h=pil_orig.height,
                img_w=pil_orig.width
            )
        )

        gt_general_b64 = arr_to_b64(
            gt_general_rgb
        )

    for name, model in active_models.items():

        # ======================================================
        # 1. Checkpoint associé
        # ======================================================

        checkpoint_path = MODELS_CONFIG[
            name
        ]

        # ======================================================
        # 2. Résolution réellement utilisée par le pipeline
        # ======================================================

        model_img_h, model_img_w = (
            get_model_input_size(
                name,
                checkpoint_path
            )
        )

        print(
            f"[PREDICT] {name} | "
            f"entrée={model_img_h}x{model_img_w}"
        )

        # ======================================================
        # 3. Prétraitement demandé depuis l'interface
        # ======================================================

        pil_for_model = (
            apply_pipeline_preprocess(
                pil_orig,
                None
                if inference_preprocess == "none"
                else inference_preprocess
            )
        )

        # ======================================================
        # 4. Image réellement utilisée par le modèle
        # ======================================================

        model_input_pil = (
            pil_for_model
            .convert("RGB")
            .resize(
                (
                    model_img_w,
                    model_img_h
                ),
                Image.BILINEAR
            )
        )

        model_input_rgb = np.array(
            model_input_pil
        )

        tensor_model = (
            transforms
            .ToTensor()(
                model_input_pil
            )
            .unsqueeze(0)
        )

        # ======================================================
        # 5. Prédiction native
        # ======================================================

        proba = predict_proba(
            model,
            tensor_model
        )

        model_threshold = (
            get_model_decision_threshold(
                name,
                threshold_mode
            )
        )

        model_thresholds_used[
            name
        ] = model_threshold

        pred = (
            proba >= model_threshold
        ).astype(np.uint8)

        # ======================================================
        # 6. Ground truth à la résolution propre du modèle
        # ======================================================

        gt_mask_model = None
        gt_native_model_b64 = None

        if gt_path_found is not None:

            gt_mask_model = load_gt_mask(
                gt_path_found,
                gt_source,
                img_h=model_img_h,
                img_w=model_img_w
            )

            gt_native_model = (
                load_gt_mask_native_colors(
                    gt_path_found,
                    gt_source,
                    img_h=model_img_h,
                    img_w=model_img_w
                )
            )

            gt_native_model_b64 = (
                arr_to_b64(
                    gt_native_model
                )
            )

        # Les métriques sont calculées à la résolution réelle du modèle
        metrics = (
            compute_metrics(
                pred,
                gt_mask_model
            )
            if gt_mask_model is not None
            else None
        )

        # ======================================================
        # 7. Visualisations individuelles
        #    Elles restent à la résolution réelle du pipeline
        # ======================================================

        probability_heatmap = (
            make_probability_heatmap(
                proba,
                colormap="gray"
            )
        )

        class_overlay = (
            make_native_class_overlay(
                pred,
                split_for_colors
            )
        )

        if gt_mask_model is not None:

            error_overlay = (
                make_error_zones_overlay_without_uncertainty(
                    pred,
                    gt_mask_model
                )
            )

        else:
            error_overlay = None

        # ======================================================
        # 8. Incertitude individuelle
        # ======================================================

        single_uncertainty = (
            compute_uncertainty_by_mode(
                [proba],
                uncertainty_mode=uncertainty_mode,
                threshold=model_threshold
            )
        )

        uncertainty_heatmap = (
            make_uncertainty_heatmap(
                single_uncertainty[
                    "dirichlet_uncertainty_norm"
                ]
            )
        )

        uncertainty_stats = (
            single_uncertainty["stats"]
        )

        posidonia_cover = float(
            np.mean(pred) * 100.0
        )

        # ======================================================
        # 9. Résultat individuel envoyé au navigateur
        # ======================================================

        results.append({
            "name": name,

            "input_image": arr_to_b64(
                model_input_rgb
            ),

            "ground_truth": (
                gt_native_model_b64
            ),

            "class_overlay": arr_to_b64(
                class_overlay
            ),

            "probability_map": arr_to_b64(
                probability_heatmap
            ),

            "error_overlay": (
                arr_to_b64(error_overlay)
                if error_overlay is not None
                else None
            ),

            "uncertainty_overlay": arr_to_b64(
                uncertainty_heatmap
            ),

            # Dimensions réellement utilisées
            "input_height": model_img_h,
            "input_width": model_img_w,

            "resolution": (
                f"{model_img_h}x{model_img_w}"
            ),

            "resolution_type": "pipeline",

            "metrics": metrics,

            "threshold_used": (
                model_threshold
            ),

            "threshold_mode": (
                threshold_mode
            ),

            "posidonia_cover_percent": (
                posidonia_cover
            ),

            "uncertainty": {
                "type": (
                    "single_model_dirichlet"
                ),

                "mode": (
                    uncertainty_mode
                ),

                "description": (
                    "Incertitude Dirichlet standard centrée sur 0.5."
                    if uncertainty_mode == "standard"
                    else
                    "Incertitude Dirichlet adaptée au seuil de décision du modèle."
                ),

                **uncertainty_stats
            }
        })

        # ======================================================
        # 10. Copie UNIQUEMENT destinée à l'ensemble
        # ======================================================

        if proba.shape != (
            ensemble_img_h,
            ensemble_img_w
        ):

            proba_for_ensemble = cv2.resize(
                proba,
                (
                    ensemble_img_w,
                    ensemble_img_h
                ),
                interpolation=cv2.INTER_LINEAR
            )

        else:
            proba_for_ensemble = proba

        # Le seuil reste celui du modèle individuel
        pred_for_ensemble = (
            proba_for_ensemble
            >= model_threshold
        ).astype(np.uint8)

        proba_list.append(
            proba_for_ensemble
        )

        pred_list.append(
            pred_for_ensemble
        )

    # ==========================================================
    # 4. Prédiction du modèle ensemble
    # ==========================================================

    n_models = len(proba_list)

    threshold_list = [
        get_model_decision_threshold(
            name,
            threshold_mode
        )
        for name in active_models.keys()
    ]

    if threshold_mode == "calibrated":
        ensemble_threshold = float(
            np.mean(threshold_list)
        )
    else:
        ensemble_threshold = (
            DECISION_THRESHOLD
        )


    # ==========================================================
    # 4.1 Soft voting ou hard voting
    # ==========================================================

    if ensemble_method == "soft":

        # Toutes les cartes contenues dans proba_list
        # sont déjà à la résolution commune 352×480.
        proba_final = np.mean(
            proba_list,
            axis=0
        )

        pred_final = (
            proba_final
            >= ensemble_threshold
        ).astype(np.uint8)

        ensemble_label = (
            "Ensemble (soft voting)"
        )

        threshold_used_for_display = (
            ensemble_threshold
        )


    else:

        # Chaque prédiction contenue dans pred_list
        # est déjà à la résolution commune 352×480.
        vote_sum = (
            np.stack(
                pred_list,
                axis=0
            )
            .sum(axis=0)
        )

        # Majorité stricte.
        # En cas d'égalité, le fond est conservé.
        pred_final = (
            vote_sum > n_models / 2
        ).astype(np.uint8)

        # Pour le hard voting, cette carte correspond
        # à la proportion de modèles votant Posidonie.
        proba_final = (
            vote_sum / n_models
        )

        ensemble_label = (
            "Ensemble "
            "(hard voting / majority voting)"
        )

        if threshold_mode == "calibrated":
            threshold_used_for_display = (
                "individual_calibrated_thresholds"
            )
        else:
            threshold_used_for_display = (
                DECISION_THRESHOLD
            )
    # Carte de probabilité de l'ensemble : proba moyenne ou taux de vote posidonie
    ensemble_probability_heatmap = (
        make_probability_heatmap(
            proba_final,
            colormap="gray"
        )
    )

    # ==========================================================
    # 5. Désaccord entre modèles
    # ==========================================================

    votes = np.stack(
        pred_list,
        axis=0
    )

    votes_posidonie = np.sum(
        votes == 1,
        axis=0
    )

    votes_fond = np.sum(
        votes == 0,
        axis=0
    )

    majority_votes = np.maximum(
        votes_posidonie,
        votes_fond
    )

    agreement_map = (
        majority_votes / n_models
    )

    disagreement_map = (
        1.0 - agreement_map
    )

    disagreement_pixels = (
        disagreement_map > 0.0
    )

    agreement_percent = float(
        np.mean(agreement_map)
        * 100.0
    )

    disagreement_percent = float(
        np.mean(disagreement_map)
        * 100.0
    )

    disagreement_pixel_percent = float(
        np.mean(disagreement_pixels)
        * 100.0
    )

    # ==========================================================
    # 6. Incertitude Dirichlet de l'ensemble
    # ==========================================================

    ensemble_uncertainty_threshold = (
        ensemble_threshold
    )

    ensemble_dirichlet = (
        compute_uncertainty_by_mode(
            proba_list,
            uncertainty_mode=uncertainty_mode,
            threshold=
                ensemble_uncertainty_threshold
        )
    )

    ensemble_uncertainty_heatmap = (
        make_uncertainty_heatmap(
            ensemble_dirichlet[
                "dirichlet_uncertainty_norm"
            ]
        )
    )

    # ==========================================================
    # 7. Visualisations et métriques de l'ensemble
    # ==========================================================

    ens_class_overlay = (
        make_native_class_overlay(
            pred_final,
            split_for_colors
        )
    )


    if gt_mask_ensemble is not None:

        ens_error_overlay = (
            make_error_zones_overlay_without_uncertainty(
                pred_final,
                gt_mask_ensemble
            )
        )

    else:
        ens_error_overlay = None


    ens_metrics = (
        compute_metrics(
            pred_final,
            gt_mask_ensemble
        )
        if gt_mask_ensemble is not None
        else None
    )


    ens_posidonia_cover = float(
        np.mean(pred_final)
        * 100.0
    )

    gt_ensemble_b64 = None

    if gt_path_found is not None:

        gt_ensemble_rgb = (
            load_gt_mask_native_colors(
                gt_path_found,
                gt_source,
                img_h=ensemble_img_h,
                img_w=ensemble_img_w
            )
        )

        gt_ensemble_b64 = arr_to_b64(
            gt_ensemble_rgb
        )

        results.append({
        "name": ensemble_label,

        "ground_truth": (
            gt_ensemble_b64
        ),

        "class_overlay": arr_to_b64(
            ens_class_overlay
        ),

        "probability_map": arr_to_b64(
            ensemble_probability_heatmap
        ),

        "error_overlay": (
            arr_to_b64(
                ens_error_overlay
            )
            if ens_error_overlay is not None
            else None
        ),

        "uncertainty_overlay": arr_to_b64(
            ensemble_uncertainty_heatmap
        ),

        # Résolution de fusion
        "input_height": ensemble_img_h,
        "input_width": ensemble_img_w,

        "resolution": (
            f"{ensemble_img_h}x{ensemble_img_w}"
        ),

        "resolution_type": "fusion",

        "metrics": ens_metrics,

        "threshold_used":
            threshold_used_for_display,

        "threshold_mode":
            threshold_mode,

        "posidonia_cover_percent":
            ens_posidonia_cover,

        "uncertainty": {
            "type":
                "ensemble_dirichlet",

            "mode":
                uncertainty_mode,

            "description": (
                "Incertitude Dirichlet standard calculée "
                "à partir des probabilités des modèles."
                if uncertainty_mode == "standard"
                else
                "Incertitude Dirichlet adaptée au seuil "
                "de décision de l'ensemble."
            ),

            "agreement_percent":
                agreement_percent,

            "disagreement_percent":
                disagreement_percent,

            "uncertain_pixel_percent":
                disagreement_pixel_percent,

            **ensemble_dirichlet["stats"]
        }
    })
    # ==========================================================
    # 8. Réponse JSON
    # ==========================================================
    return jsonify({
        'original': orig_b64,
        'preprocessed': preprocessed_b64,
        'inference_preprocess': inference_preprocess,
        'inference_preprocess_label': (
            "CLAHE" if inference_preprocess == "clahe"
            else "Expansion RGB + égalisation" if inference_preprocess == "expansion_rgb"
            else "Aucun prétraitement"
        ),
        'gt_source': gt_source,

        'results': results,

        'ensemble_method': ensemble_method,

        'threshold_mode': threshold_mode,
        'uncertainty_mode': uncertainty_mode,
        'gt': gt_general_b64,
        'has_gt': gt_general_b64 is not None,
        'gt_source': gt_source,

        'decision_threshold': DECISION_THRESHOLD,
        'ensemble_threshold': ensemble_threshold,
        'model_thresholds_used': model_thresholds_used,
    })


# ── Route page évaluation ──
@app.route('/evaluate')
def evaluate_page():
    pipelines = discover_pipelines(MODELS_DIR)
    return render_template('evaluate.html', pipelines=pipelines)

@app.route('/run_evaluate', methods=['POST'])
def run_evaluate():

    try:
        return _run_evaluate_impl()

    except Exception as exc:
        import traceback

        traceback.print_exc()

        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__,
        }), 500


def _run_evaluate_impl():
    data = request.get_json() or {}

    split = data.get('split', 'both')       # 'val', 'test', ou 'both'
    method = data.get('method', 'soft')     # 'soft' ou 'hard'

    threshold_mode = data.get("threshold_mode", THRESHOLD_MODE_DEFAULT)
    uncertainty_mode = data.get("uncertainty_mode", UNCERTAINTY_MODE_DEFAULT)

    if threshold_mode not in ("fixed", "calibrated"):
        threshold_mode = THRESHOLD_MODE_DEFAULT

    if uncertainty_mode not in ("standard", "threshold"):
        uncertainty_mode = UNCERTAINTY_MODE_DEFAULT

    if method not in ('soft', 'hard'):
        method = 'soft'

    # ==========================================================
    # 0. Sélection des pipelines
    # ==========================================================
    pipeline_names = data.get("pipeline_names")

    # Compatibilité avec l'ancien cas d'un seul pipeline
    if pipeline_names is None:
        single_pipeline_name = data.get("pipeline_name")
        pipeline_names = [single_pipeline_name] if single_pipeline_name else []

    pipeline_names = [p for p in pipeline_names if p]

    pipeline_mode = len(pipeline_names) > 0
    ensemble_pipeline_mode = len(pipeline_names) >= 2

    selected_models = data.get('selected_models', [])

    selected_pipelines = []
    active_pipelines = {}

    if pipeline_mode:
        pipelines = discover_pipelines(MODELS_DIR)

        for pipeline_name in pipeline_names:
            if pipeline_name not in pipelines:
                return jsonify({
                    "error": f"Pipeline inconnu : {pipeline_name}"
                }), 400

            selected_pipelines.append(pipelines[pipeline_name])

        active_models = {}
        active_pipelines = {}

        for pipeline in selected_pipelines:
            model = load_model_from_pipeline(pipeline)

            label = pipeline["label"]

            # Évite d'écraser deux modèles qui auraient le même label
            if label in active_models:
                label = f"{label} ({pipeline['experiment_name']})"

            active_models[label] = model
            active_pipelines[label] = pipeline

    else:
        if not selected_models:
            selected_models = list(
                MODELS_CONFIG.keys()
            )

        active_models = {}

        for name in selected_models:

            if name not in MODELS_CONFIG:
                continue

            checkpoint_path = MODELS_CONFIG[name]

            model = get_or_load_model(
                name,
                checkpoint_path
            )

            active_models[name] = model

        if not active_models:
            return jsonify({
                "error": "Aucun modèle sélectionné ou disponible"
            }), 400

    # ----------------------------------------------------------
    # Liste des images selon le split
    # ----------------------------------------------------------
    def get_image_list(split_name):
        if split_name == 'test':
            img_dir, mask_dir = TESTING_IMGS, TESTING_MSKS

            image_names = sorted([
                f.stem for f in img_dir.iterdir()
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
            ])

            mask_format = 'testing'

        else:
            img_dir, mask_dir = TRAINING_IMGS, TRAINING_MSKS
            val_csv = BASE_DIR / "Dataset" / "val_split_stratified.csv"

            image_names = []

            if not val_csv.exists():
                print(f"[WARN] Fichier validation introuvable : {val_csv}")
                return img_dir, mask_dir, image_names, 'training'

            with open(val_csv, newline='') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    name = Path(row.get('image', row.get('filename', ''))).stem
                    if name:
                        image_names.append(name)

            mask_format = 'training'

        return img_dir, mask_dir, image_names, mask_format

    # ----------------------------------------------------------
    # Métriques globales
    # ----------------------------------------------------------
    def compute_full_metrics(probas_list, labels, thr, preds_list=None):
        probas_np = np.concatenate(probas_list)

        if preds_list is not None:
            preds = np.concatenate(preds_list).astype(np.uint8)
        else:
            preds = (probas_np >= thr).astype(np.uint8)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())

        total = tp + fp + fn + tn

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        iou = tp / (tp + fp + fn + 1e-8)
        iou_bg = tn / (tn + fn + fp + 1e-8)
        miou = (iou + iou_bg) / 2

        accuracy = (tp + tn) / (total + 1e-8)
        fallout = fp / (fp + tn + 1e-8)

        from sklearn.metrics import roc_curve, auc, precision_recall_curve

        fpr, tpr, _ = roc_curve(labels, probas_np)
        auc_roc = float(auc(fpr, tpr))

        prec_c, rec_c, _ = precision_recall_curve(labels, probas_np)
        auc_pr = float(auc(rec_c, prec_c))

        step_roc = max(1, len(fpr) // 500)
        step_pr = max(1, len(prec_c) // 500)

        return {
            'accuracy': round(float(accuracy), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'iou': round(float(iou), 4),
            'miou': round(float(miou), 4),
            'fallout': round(float(fallout), 4),
            'auc_roc': round(float(auc_roc), 4),
            'auc_pr': round(float(auc_pr), 4),

            'fpr': fpr[::step_roc].tolist(),
            'tpr': tpr[::step_roc].tolist(),
            'prec_curve': prec_c[::step_pr].tolist(),
            'rec_curve': rec_c[::step_pr].tolist(),

            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def get_threshold_for_active_model(model_name):
        """
        Retourne le seuil réellement demandé par l'utilisateur.

        - fixed      : seuil fixe 0.5
        - calibrated : val_threshold enregistré dans le pipeline
        """

        if threshold_mode == "fixed":
            return DECISION_THRESHOLD

        if threshold_mode == "calibrated":
            if pipeline_mode:
                pipeline = active_pipelines[model_name]
                val_threshold = pipeline.get("val_threshold")

                if val_threshold is None:
                    print(
                        f"[WARN] Aucun val_threshold pour {model_name}. "
                        f"Utilisation du seuil 0.5."
                    )
                    return DECISION_THRESHOLD

                return float(val_threshold)

            return get_model_decision_threshold(
                model_name,
                threshold_mode="calibrated"
            )

        return DECISION_THRESHOLD

    
    def build_kfold_summary(results_dict):
        """
        Calcule moyenne ± écart-type uniquement si l'utilisateur a sélectionné
        plusieurs folds du même protocole expérimental.
        """

        if not pipeline_mode:
            return None

        # On récupère les pipelines actifs qui sont des folds
        kfold_items = [
            (label, pipeline)
            for label, pipeline in active_pipelines.items()
            if pipeline.get("fold") is not None
        ]

        if len(kfold_items) < 2:
            return None

        # Vérification : tous les folds doivent appartenir à la même configuration
        group_keys = set()

        for _, pipeline in kfold_items:
            group_key = (
                pipeline.get("architecture"),
                pipeline.get("variant"),
                pipeline.get("loss"),
                pipeline.get("img_h"),
                pipeline.get("img_w"),
                pipeline.get("preprocess"),
            )
            group_keys.add(group_key)

        if len(group_keys) != 1:
            return {
                "available": False,
                "reason": (
                    "Les pipelines sélectionnés ne correspondent pas aux folds "
                    "d'un même protocole expérimental."
                )
            }

        metrics_to_summarize = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "iou",
            "miou",
            "fallout",
            "auc_roc",
            "auc_pr",
        ]

        summary = {}

        for metric in metrics_to_summarize:
            values = []

            for label, _ in kfold_items:
                if label in results_dict and metric in results_dict[label]:
                    values.append(float(results_dict[label][metric]))

            if len(values) >= 2:
                arr = np.array(values, dtype=float)

                summary[metric] = {
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std(ddof=1)), 4),
                    "n": int(len(values)),
                }

        first_pipeline = kfold_items[0][1]

        folds = [
            int(pipeline.get("fold"))
            for _, pipeline in kfold_items
        ]

        folds = sorted(folds)

        return {
            "available": True,
            "model": first_pipeline.get("architecture"),
            "variant": first_pipeline.get("variant"),
            "loss": first_pipeline.get("loss"),
            "img_h": first_pipeline.get("img_h"),
            "img_w": first_pipeline.get("img_w"),
            "preprocess": first_pipeline.get("preprocess"),
            "folds": folds,
            "n_folds": len(folds),
            "metrics": summary,
        }

    def synchronize_if_cuda():
        """
        Synchronise le GPU pour mesurer correctement le temps d'inférence.
        Sans cela, les opérations CUDA sont asynchrones et le temps mesuré peut être faux.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ----------------------------------------------------------
    # Évaluation d'un split
    # ----------------------------------------------------------
    def run_split_eval(split_name):
        img_dir, mask_dir, image_names, mask_format = get_image_list(split_name)

        if not image_names:
            return None

        ensemble_is_active = (not pipeline_mode) or ensemble_pipeline_mode
        # Résolution commune utilisée uniquement pour l'ensemble
        ensemble_img_h = IMG_H   # 352
        ensemble_img_w = IMG_W   # 480

        all_probas = {
            name: []
            for name in active_models
        }

        all_preds = {
            name: []
            for name in active_models
        }

        # Chaque modèle garde ses propres GT à sa résolution native
        all_labels_by_model = {
            name: []
            for name in active_models
        }

        # GT distinct pour l'ensemble à la résolution de fusion
        all_labels_ensemble = []

# pour calculer le temps d'inférence 
        all_inference_times = {name: [] for name in active_models}
        all_ensemble_times = []

        if ensemble_is_active:
            all_probas['__ensemble__'] = []
            all_preds['__ensemble__'] = []


        all_single_dirichlet_uncertainties = {name: [] for name in active_models}
        all_single_dirichlet_uncertain_pixels = {name: [] for name in active_models}

        all_ensemble_dirichlet_uncertainties = []
        all_ensemble_dirichlet_uncertain_pixels = []
        all_ensemble_disagreement = []
        all_ensemble_disagreement_pixels = []

        n_images = 0
        n_skipped = 0

        model_thresholds_for_split = {
            name: get_threshold_for_active_model(name)
            for name in active_models.keys()
        }

        if threshold_mode == "calibrated":
            ensemble_threshold = float(
                np.mean(list(model_thresholds_for_split.values()))
            )
        else:
            ensemble_threshold = DECISION_THRESHOLD

        for img_name in image_names:
            img_path = img_dir / (img_name + '.png')

            if not img_path.exists():
                img_path = img_dir / (img_name + '.jpg')

            if not img_path.exists():
                img_path = img_dir / (img_name + '.jpeg')

            if not img_path.exists():
                n_skipped += 1
                continue

            mask_path = mask_dir / (img_name + '_gt.png')

            if not mask_path.exists():
                n_skipped += 1
                continue

            pil_img = Image.open(img_path).convert("RGB")


            proba_list = []
            pred_list = []

            # Masque destiné uniquement à l'ensemble
            gt_mask_ensemble = None

            if ensemble_is_active:
                gt_mask_ensemble = load_gt_mask(
                    mask_path,
                    mask_format,
                    img_h=ensemble_img_h,
                    img_w=ensemble_img_w
                )
            for name, model in active_models.items():
                if pipeline_mode:
                    pipeline = active_pipelines[name]

                    preprocess = pipeline.get("preprocess")
                    img_h = int(pipeline.get("img_h") or IMG_H)
                    img_w = int(pipeline.get("img_w") or IMG_W)

                    pil_for_model = apply_pipeline_preprocess(
                        pil_img,
                        preprocess
                    )

                    tensor_model = img_to_tensor(
                        pil_for_model,
                        img_h=img_h,
                        img_w=img_w
                    )

                else:
                    img_h = IMG_H
                    img_w = IMG_W

                    tensor_model = img_to_tensor(
                        pil_img,
                        img_h=img_h,
                        img_w=img_w
                    )

                # GT propre au modèle, à sa résolution native
                gt_mask_model = load_gt_mask(
                    mask_path,
                    mask_format,
                    img_h=img_h,
                    img_w=img_w
                )

                synchronize_if_cuda()
                start_time = time.perf_counter()

# Les modèles sont exécutés les uns après les autres
# Ainsi le temps pour le modèle d'ensemble sera la somme du temps de tous les modèles
                proba = predict_proba(model, tensor_model)

                synchronize_if_cuda()
                end_time = time.perf_counter()

                inference_time = end_time - start_time
                all_inference_times[name].append(inference_time)

                if proba.shape != gt_mask_model.shape:
                    proba = cv2.resize(
                        proba,
                        (
                            gt_mask_model.shape[1],
                            gt_mask_model.shape[0]
                        ),
                        interpolation=cv2.INTER_LINEAR
                    )

                model_threshold = model_thresholds_for_split[name]
                pred = (proba >= model_threshold).astype(np.uint8)

                all_probas[name].append(proba.flatten())
                all_preds[name].append(pred.flatten())

                all_labels_by_model[name].append(
                    gt_mask_model.flatten()
                )

                # Pour l'ensemble, toutes les cartes sont ramenées
                # explicitement à 352 × 480
                if ensemble_is_active:
                    if proba.shape != (
                        ensemble_img_h,
                        ensemble_img_w
                    ):
                        proba_for_ensemble = cv2.resize(
                            proba,
                            (
                                ensemble_img_w,
                                ensemble_img_h
                            ),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        proba_for_ensemble = proba

                    pred_for_ensemble = (
                        proba_for_ensemble >= model_threshold
                    ).astype(np.uint8)

                    proba_list.append(proba_for_ensemble)
                    pred_list.append(pred_for_ensemble)

                single_dirichlet = compute_uncertainty_by_mode(
                    [proba],
                    uncertainty_mode=uncertainty_mode,
                    threshold=model_threshold
                )

                all_single_dirichlet_uncertainties[name].append(
                    single_dirichlet["dirichlet_uncertainty_norm"].flatten()
                )

                all_single_dirichlet_uncertain_pixels[name].append(
                    single_dirichlet["uncertain_pixels_dirichlet"].flatten()
                )

            if len(proba_list) == 0:
                n_skipped += 1
                continue

            if ensemble_is_active:
                n_models = len(proba_list)
                ensemble_start = time.perf_counter()
                if method == 'soft':
                    proba_ens = np.mean(proba_list, axis=0)
                    pred_ens = (proba_ens >= ensemble_threshold).astype(np.uint8)

                else:
                    vote_sum = np.stack(pred_list, axis=0).sum(axis=0)
                    pred_ens = (vote_sum > n_models / 2).astype(np.uint8)
                    proba_ens = vote_sum / n_models

                ensemble_end = time.perf_counter()
                fusion_time = ensemble_end - ensemble_start

                image_models_time = sum(
                    all_inference_times[name][-1]
                    for name in active_models.keys()
                )

                all_ensemble_times.append(image_models_time + fusion_time)

                all_probas['__ensemble__'].append(proba_ens.flatten())
                all_preds['__ensemble__'].append(pred_ens.flatten())

                votes = np.stack(pred_list, axis=0)

                votes_posidonie = np.sum(votes == 1, axis=0)
                votes_fond = np.sum(votes == 0, axis=0)

                majority_votes = np.maximum(votes_posidonie, votes_fond)

                agreement_map = majority_votes / n_models
                disagreement_map = 1.0 - agreement_map
                disagreement_pixels = disagreement_map > 0.0

                all_ensemble_disagreement.append(disagreement_map.flatten())
                all_ensemble_disagreement_pixels.append(
                    disagreement_pixels.astype(np.uint8).flatten()
                )

                ensemble_dirichlet = compute_uncertainty_by_mode(
                    proba_list,
                    uncertainty_mode=uncertainty_mode,
                    threshold=ensemble_threshold
                )

                all_ensemble_dirichlet_uncertainties.append(
                    ensemble_dirichlet["dirichlet_uncertainty_norm"].flatten()
                )

                all_ensemble_dirichlet_uncertain_pixels.append(
                    ensemble_dirichlet["uncertain_pixels_dirichlet"].flatten()
                )

            if ensemble_is_active:
                all_labels_ensemble.append(
                    gt_mask_ensemble.flatten()
                )

            n_images += 1

        if n_images == 0:
            return None

        labels_by_model_np = {
            name: np.concatenate(
                all_labels_by_model[name]
            )
            for name in active_models
        }

        if ensemble_is_active:
            labels_ensemble_np = np.concatenate(
                all_labels_ensemble
            )
        else:
            labels_ensemble_np = None

        def summarize_single_uncertainty(model_name):
            dirichlet_u = np.concatenate(
                all_single_dirichlet_uncertainties[model_name]
            )

            dirichlet_pixels = np.concatenate(
                all_single_dirichlet_uncertain_pixels[model_name]
            )

            return {
                'type': 'single_model_dirichlet',
                'mode': uncertainty_mode,
                'dirichlet_uncertainty_norm_mean': round(float(np.mean(dirichlet_u)), 4),
                'dirichlet_uncertainty_norm_median': round(float(np.median(dirichlet_u)), 4),
                'dirichlet_uncertainty_norm_max': round(float(np.max(dirichlet_u)), 4),
                'dirichlet_uncertain_pixel_ratio': round(float(np.mean(dirichlet_pixels)), 4),
                'dirichlet_uncertain_pixel_percent': round(float(np.mean(dirichlet_pixels) * 100.0), 2),
            }

        def summarize_ensemble_uncertainty():
            dirichlet_u = np.concatenate(all_ensemble_dirichlet_uncertainties)
            dirichlet_pixels = np.concatenate(all_ensemble_dirichlet_uncertain_pixels)

            disagreement = np.concatenate(all_ensemble_disagreement)
            disagreement_pixels = np.concatenate(all_ensemble_disagreement_pixels)

            return {
                'type': 'ensemble_dirichlet',
                'mode': uncertainty_mode,
                'agreement_percent': round(float((1.0 - np.mean(disagreement)) * 100.0), 2),
                'disagreement_percent': round(float(np.mean(disagreement) * 100.0), 2),
                'uncertain_pixel_ratio': round(float(np.mean(disagreement_pixels)), 4),
                'uncertain_pixel_percent': round(float(np.mean(disagreement_pixels) * 100.0), 2),
                'dirichlet_uncertainty_norm_mean': round(float(np.mean(dirichlet_u)), 4),
                'dirichlet_uncertainty_norm_median': round(float(np.median(dirichlet_u)), 4),
                'dirichlet_uncertainty_norm_max': round(float(np.max(dirichlet_u)), 4),
                'dirichlet_uncertain_pixel_ratio': round(float(np.mean(dirichlet_pixels)), 4),
                'dirichlet_uncertain_pixel_percent': round(float(np.mean(dirichlet_pixels) * 100.0), 2),
            }

        results = {}

        ensemble_name = f'Ensemble ({method} voting)'

        model_names_to_report = list(active_models.keys())

        if ensemble_is_active:
            model_names_to_report.append('__ensemble__')

        for name in model_names_to_report:
            if name == '__ensemble__':
                label = ensemble_name

                results[label] = compute_full_metrics(
                    all_probas[name],
                    labels_ensemble_np,
                    ensemble_threshold,
                    preds_list=all_preds[name]
                )

                results[label]['uncertainty'] = summarize_ensemble_uncertainty()
                results[label]['threshold_used'] = (
                    ensemble_threshold
                    if method == 'soft'
                    else (
                        'individual_calibrated_thresholds'
                        if threshold_mode == 'calibrated'
                        else DECISION_THRESHOLD
                    )
                )
        
                ens_times = np.array(all_ensemble_times, dtype=float)

                results[label]['inference_time_mean_s'] = round(float(ens_times.mean()), 6)
                results[label]['inference_time_std_s'] = round(float(ens_times.std(ddof=1)), 6) if len(ens_times) > 1 else 0.0
                results[label]['inference_time_total_s'] = round(float(ens_times.sum()), 4)

            else:
                label = name
                model_threshold = model_thresholds_for_split[name]

                results[label] = compute_full_metrics(
                    all_probas[name],
                    labels_by_model_np[name],
                    model_threshold,
                    preds_list=all_preds[name]
                )

                results[label]['uncertainty'] = summarize_single_uncertainty(name)
                results[label]['threshold_used'] = model_threshold
                times = np.array(all_inference_times[name], dtype=float)

                results[label]['inference_time_mean_s'] = round(float(times.mean()), 6)
                results[label]['inference_time_std_s'] = round(float(times.std(ddof=1)), 6) if len(times) > 1 else 0.0
                results[label]['inference_time_total_s'] = round(float(times.sum()), 4)

        return {
            'results': results,
            'n_images': n_images,
            'n_skipped': n_skipped,
            'model_thresholds_used': model_thresholds_for_split,
            'ensemble_threshold_used': ensemble_threshold if ensemble_is_active else None,
        }

    # ----------------------------------------------------------
    # Seuils utilisés pour affichage
    # ----------------------------------------------------------
    model_thresholds_used = {
        name: get_threshold_for_active_model(name)
        for name in active_models.keys()
    }

    ensemble_is_active_global = (not pipeline_mode) or ensemble_pipeline_mode

    if ensemble_is_active_global:
        if threshold_mode == "calibrated":
            ensemble_threshold_used = float(
                np.mean(list(model_thresholds_used.values()))
            )
        else:
            ensemble_threshold_used = DECISION_THRESHOLD

        ensemble_name = f'Ensemble ({method} voting)'
        model_thresholds_used[ensemble_name] = ensemble_threshold_used
    else:
        ensemble_threshold_used = None

    # ----------------------------------------------------------
    # Exécution
    # ----------------------------------------------------------
    results_val = run_split_eval('val') if split in ('val', 'both') else None
    results_test = run_split_eval('test') if split in ('test', 'both') else None

    kfold_summary_val = (
        build_kfold_summary(results_val["results"])
        if results_val is not None
        else None
    )

    kfold_summary_test = (
        build_kfold_summary(results_test["results"])
        if results_test is not None
        else None
    )

    if results_val is None and results_test is None:
        return jsonify({'error': 'Aucune image valide trouvée'}), 400

    # ----------------------------------------------------------
    # Figures
    # ----------------------------------------------------------
    if split == 'both':
        roc_pr_b64, cm_b64 = generate_eval_figures(
            results_val['results'],
            results_test['results'],
            'Val + Test',
            method,
            ensemble_threshold_used,
            model_thresholds=model_thresholds_used,
            active_pipelines=active_pipelines if pipeline_mode else None
        )

        n_images_info = (
            f"Val: {results_val['n_images']} "
            f"/ Test: {results_test['n_images']}"
        )

    elif split == 'val':
        roc_pr_b64, cm_b64 = generate_eval_figures_single(
            results_val['results'],
            'Validation',
            method,
            ensemble_threshold_used,
            model_thresholds=model_thresholds_used,
            active_pipelines=active_pipelines if pipeline_mode else None
        )

        n_images_info = f"Val: {results_val['n_images']}"

    else:
        roc_pr_b64, cm_b64 = generate_eval_figures_single(
            results_test['results'],
            'Test',
            method,
            ensemble_threshold_used,
            model_thresholds=model_thresholds_used,
            active_pipelines=active_pipelines if pipeline_mode else None
        )

        n_images_info = f"Test: {results_test['n_images']}"

    # ----------------------------------------------------------
    # Réponse JSON
    # ----------------------------------------------------------
    return jsonify({
        'split': split,
        'n_images_info': n_images_info,

        'method': method,
        'threshold_mode': threshold_mode,
        'uncertainty_mode': uncertainty_mode,

        'decision_threshold': DECISION_THRESHOLD,
        'ensemble_threshold_used': ensemble_threshold_used,
        'model_thresholds_used': model_thresholds_used,

        'kfold_summary_val': kfold_summary_val,
        'kfold_summary_test': kfold_summary_test,

        'results_val': results_val['results'] if results_val else None,
        'results_test': results_test['results'] if results_test else None,

        'roc_pr_image': roc_pr_b64,
        'confusion_image': cm_b64,

        'pipeline_mode': pipeline_mode,
        'ensemble_pipeline_mode': ensemble_pipeline_mode,

        'selected_pipelines': [
            {
                'experiment_name': p['experiment_name'],
                'label': p['label'],
                'checkpoint_path': p['checkpoint_path'],
                'metadata_path': p['metadata_path'],
                'architecture': p['architecture'],
                'loss': p['loss'],
                'img_h': p['img_h'],
                'img_w': p['img_w'],
                'preprocess': p.get('preprocess'),
                'val_threshold': p['val_threshold'],
                'test_threshold_saved': p['test_threshold'],
            }
            for p in selected_pipelines
        ] if pipeline_mode else [],
    })



if __name__ == '__main__':
    app.run(debug=False, port=5000)
