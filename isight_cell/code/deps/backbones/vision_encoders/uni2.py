"""UNI2-h vision backbone (timm ViT-giant).

Output: (B, 1536) global feature. Override the checkpoint location with
the required env var UNI2_CKPT_PATH.
"""
import os

import timm
import torch
import torch.nn as nn
from torchvision import transforms

UNI2_PATH = os.environ.get("UNI2_CKPT_PATH", "")   # UNI2-h weights (MahmoodLab/UNI2-h)
EMBED_DIM = 1536
INPUT_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    # NOTE: ColorJitter intentionally REMOVED.
    # Two of our 5 tasks (`intensity` and `quantity`) directly measure DAB
    # staining strength/coverage; any brightness/contrast perturbation
    # corrupts these labels (borderline weak ↔ moderate cases get flipped).
    # HPA is single-center standardized scanning so cross-scanner color
    # robustness is not needed. Regularization is already strong
    # (drop_path=0.2, drop_rate=0.2, attn_drop=0.05, head_dropout=0.25) +
    # 10M training samples → no extra need for color aug.
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def build_uni2(drop_path_rate: float = 0.1,
                drop_rate: float = 0.0,
                attn_drop_rate: float = 0.0):
    """UNI2-h (timm vit_giant_patch14_224 + SwiGLU + reg).

    Args:
        drop_path_rate: stochastic depth (residual drop). Default 0.1 — main
                         regularizer for ViT fine-tuning. Try 0.2 if overfitting.
        drop_rate:      projection/MLP dropout (timm `drop_rate`).
                         0.0 keeps timm default. Try 0.1–0.2 for fine-tune.
        attn_drop_rate: attention-map dropout (timm `attn_drop_rate`).
                         Usually 0.0; try 0.05 if needed.
    """
    timm_kwargs = {
        "img_size": INPUT_SIZE, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": EMBED_DIM, "mlp_ratio": 2.66667 * 2,
        "num_classes": 0, "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
        "drop_path_rate": drop_path_rate,
        "drop_rate": drop_rate,
        "attn_drop_rate": attn_drop_rate,
    }
    model = timm.create_model("vit_giant_patch14_224", pretrained=False,
                                **timm_kwargs)
    if not UNI2_PATH:
        raise RuntimeError("set UNI2_CKPT_PATH to the UNI2-h pytorch_model.bin "
                           "(https://huggingface.co/MahmoodLab/UNI2-h)")
    state = torch.load(UNI2_PATH, weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model
