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


def focal_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    alpha_pos: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)
    alpha_pos = alpha_pos.view(1, 1, -1).to(device=logits.device, dtype=logits.dtype)
    alpha_t = torch.where(targets == 1.0, alpha_pos, 1.0 - alpha_pos)
    pt = torch.where(targets == 1.0, probs, 1.0 - probs)
    per_elem = -alpha_t * torch.pow(1.0 - pt, gamma) * torch.log(pt)
    masked = per_elem * mask.unsqueeze(-1)
    denom = mask.sum() * logits.shape[-1]
    return masked.sum() / max(denom, 1.0)


def onsets_frames_loss(
    outputs: dict[str, torch.Tensor],
    frame_targets: torch.Tensor,
    onset_targets: torch.Tensor,
    mask: torch.Tensor,
    frame_criterion: nn.Module,
    onset_criterion: nn.Module,
    onset_loss_weight: float,
    onset_loss_type: str = "bce",
    onset_focal_gamma: float = 2.0,
    onset_focal_alpha_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_loss = bce_masked_loss(
        logits=outputs["frame_logits"],
        targets=frame_targets,
        mask=mask,
        criterion=frame_criterion,
    )
    if onset_loss_type == "bce":
        onset_loss = bce_masked_loss(
            logits=outputs["onset_logits"],
            targets=onset_targets,
            mask=mask,
            criterion=onset_criterion,
        )
    elif onset_loss_type == "focal":
        if onset_focal_alpha_pos is None:
            raise ValueError("onset_focal_alpha_pos is required when onset_loss_type='focal'")
        onset_loss = focal_masked_loss(
            logits=outputs["onset_logits"],
            targets=onset_targets,
            mask=mask,
            alpha_pos=onset_focal_alpha_pos,
            gamma=onset_focal_gamma,
        )
    else:
        raise ValueError(f"Unknown onset_loss_type: {onset_loss_type}")
    loss = frame_loss + onset_loss_weight * onset_loss
    return loss, {
        "loss": float(loss.item()),
        "frame_loss": float(frame_loss.item()),
        "onset_loss": float(onset_loss.item()),
    }


def decode_onsets_frames(
    onset_probs: torch.Tensor,
    frame_probs: torch.Tensor,
    onset_threshold: float,
    frame_threshold: float,
) -> torch.Tensor:
    """Decode note activity with onset-gated starts and frame-gated continuations.

    Args:
        onset_probs: [B, T, 88]
        frame_probs: [B, T, 88]

    Returns:
        Binary decoded frame roll [B, T, 88].
    """
    if onset_probs.shape != frame_probs.shape:
        raise ValueError("onset_probs and frame_probs must have the same shape")

    batch, timesteps, keys = frame_probs.shape
    active = torch.zeros((batch, keys), dtype=torch.bool, device=frame_probs.device)
    decoded = torch.zeros_like(frame_probs, dtype=torch.float32)

    for t in range(timesteps):
        frame_on = frame_probs[:, t, :] > frame_threshold
        onset_on = onset_probs[:, t, :] > onset_threshold
        starts = (~active) & onset_on & frame_on
        continues = active & frame_on
        active = starts | continues
        decoded[:, t, :] = active.to(dtype=torch.float32)

    return decoded


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


def micro_counts(preds: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> tuple[float, float, float]:
    tp = float(((preds == 1.0) & (targets == 1.0)).float().mul(valid_mask).sum().item())
    fp = float(((preds == 1.0) & (targets == 0.0)).float().mul(valid_mask).sum().item())
    fn = float(((preds == 0.0) & (targets == 1.0)).float().mul(valid_mask).sum().item())
    return tp, fp, fn


def prf_from_counts(tp: float, fp: float, fn: float, prefix: str = "") -> Dict[str, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        f"{prefix}precision_micro": precision,
        f"{prefix}recall_micro": recall,
        f"{prefix}f1_micro": f1,
    }
