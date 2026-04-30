from typing import Dict

import torch
import torch.nn as nn


def bce_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    per_elem = criterion(logits, targets)  # [B, T, 88]
    masked = per_elem * mask.unsqueeze(-1)
    denom = mask.sum() * logits.shape[-1]
    return masked.sum() / max(denom, 1.0)


def compute_micro_prf(preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    tp = float(((preds == 1.0) & (targets == 1.0)).float().mul(valid_mask).sum().item())
    fp = float(((preds == 1.0) & (targets == 0.0)).float().mul(valid_mask).sum().item())
    fn = float(((preds == 0.0) & (targets == 1.0)).float().mul(valid_mask).sum().item())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }
