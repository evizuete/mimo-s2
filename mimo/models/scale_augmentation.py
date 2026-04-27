# scale_augmentation.py

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import numpy as np


@dataclass
class ScaleAugmentConfig:
    enabled: bool = True
    prob: float = 0.60
    mult_low: float = 0.97
    mult_high: float = 1.03
    noise_std: float = 0.01
    context_noise_std: float = 0.005
    seed: int = 42


def _apply_aug_3d(
    x: np.ndarray,
    rng: np.random.Generator,
    mult_low: float,
    mult_high: float,
    noise_std: float,
    skip_idx: Optional[Iterable[int]] = None,
) -> np.ndarray:
    x = x.copy().astype(np.float32)
    if x.size == 0:
        return x

    n_feat = x.shape[-1]
    mask = np.ones(n_feat, dtype=bool)
    if skip_idx is not None:
        mask[list(skip_idx)] = False

    mult = rng.uniform(mult_low, mult_high, size=(x.shape[0], 1, n_feat)).astype(np.float32)
    noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    mult[:, :, ~mask] = 1.0
    noise[:, :, ~mask] = 0.0

    return x * mult + noise


def _apply_aug_2d(
    x: np.ndarray,
    rng: np.random.Generator,
    mult_low: float,
    mult_high: float,
    noise_std: float,
    skip_idx: Optional[Iterable[int]] = None,
) -> np.ndarray:
    x = x.copy().astype(np.float32)
    if x.size == 0:
        return x

    n_feat = x.shape[-1]
    mask = np.ones(n_feat, dtype=bool)
    if skip_idx is not None:
        mask[list(skip_idx)] = False

    mult = rng.uniform(mult_low, mult_high, size=(x.shape[0], n_feat)).astype(np.float32)
    noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    mult[:, ~mask] = 1.0
    noise[:, ~mask] = 0.0

    return x * mult + noise


def augment_train_batch(
    X: Dict[str, np.ndarray],
    cfg: ScaleAugmentConfig,
    skip_map: Optional[Dict[str, Iterable[int]]] = None,
) -> Dict[str, np.ndarray]:
    if not cfg.enabled:
        return X

    rng = np.random.default_rng(cfg.seed)
    out = {k: v.copy() for k, v in X.items()}

    n = len(out["seq_short"])
    apply_mask = rng.random(n) < cfg.prob

    if apply_mask.sum() == 0:
        return out

    idx = np.where(apply_mask)[0]

    skip_map = skip_map or {}

    out["seq_short"][idx] = _apply_aug_3d(
        out["seq_short"][idx], rng,
        cfg.mult_low, cfg.mult_high, cfg.noise_std,
        skip_idx=skip_map.get("seq_short"),
    )

    out["seq_long"][idx] = _apply_aug_3d(
        out["seq_long"][idx], rng,
        cfg.mult_low, cfg.mult_high, cfg.noise_std,
        skip_idx=skip_map.get("seq_long"),
    )

    out["context"][idx] = _apply_aug_2d(
        out["context"][idx], rng,
        cfg.mult_low, cfg.mult_high, cfg.context_noise_std,
        skip_idx=skip_map.get("context"),
    )

    # time no se toca
    return out

def _build_aug_skip_map(pipeline_fold, side: str) -> dict:
    fe = pipeline_fold.feature_engineer
    fe.set_side(side)
    fe._assign_features_to_inputs()
    cols = fe.feature_columns

    no_scale = pipeline_fold.no_scale_by_block

    def idxs(block_name: str, key_name: str):
        block_cols = cols[key_name]
        blocked = set(no_scale.get(block_name, set()))
        return [i for i, c in enumerate(block_cols) if c in blocked]

    return {
        "seq_short": idxs("seq_short", "sequence_short"),
        "seq_long": idxs("seq_long", "sequence_long"),
        "context": idxs("context", "context"),
    }