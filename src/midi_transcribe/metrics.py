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


def state4_masked_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    # logits: [B, T, 88, 4], targets: [B, T, 88]
    per_elem = criterion(logits.permute(0, 3, 1, 2), targets)  # [B, T, 88]
    masked = per_elem * mask.unsqueeze(-1)
    denom = mask.sum() * targets.shape[-1]
    return masked.sum() / max(denom, 1.0)


def decode_state4_predictions(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    off_state: int = 0,
    sustain_state: int = 1,
    offset_state: int = 2,
    onset_state: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode per-frame state predictions into active-frame and onset rolls.

    `off` closes on the previous frame, so the current off frame is inactive.
    `sustain` can start activity only on the first frame of a segment.
    """

    batch, timesteps, keys = states.shape
    active = torch.zeros((batch, keys), dtype=torch.bool, device=states.device)
    decoded_active = torch.zeros((batch, timesteps, keys), dtype=torch.float32, device=states.device)
    decoded_onset = torch.zeros_like(decoded_active)

    if mask is None:
        valid_steps = torch.ones((batch, timesteps), dtype=torch.bool, device=states.device)
    else:
        valid_steps = mask.to(device=states.device) > 0

    for t in range(timesteps):
        valid = valid_steps[:, t].unsqueeze(-1)
        state_t = states[:, t, :]

        onset = (state_t == onset_state) & valid
        sustain = (state_t == sustain_state) & valid
        offset = (state_t == offset_state) & valid
        off = (state_t == off_state) & valid

        decoded_onset[:, t, :] = onset.to(dtype=torch.float32)

        # Onset opens a note on the current frame even if another note was already active.
        active = torch.where(onset, torch.ones_like(active), active)
        decoded_active[:, t, :] = torch.where(
            onset,
            torch.ones_like(decoded_active[:, t, :]),
            decoded_active[:, t, :],
        )

        if t == 0:
            active = torch.where(sustain & ~active, torch.ones_like(active), active)
        decoded_active[:, t, :] = torch.where(
            sustain & active,
            torch.ones_like(decoded_active[:, t, :]),
            decoded_active[:, t, :],
        )

        decoded_active[:, t, :] = torch.where(
            offset & active,
            torch.ones_like(decoded_active[:, t, :]),
            decoded_active[:, t, :],
        )
        active = torch.where(offset & active, torch.zeros_like(active), active)

        # Current off frame is inactive; it only terminates a note that was active before.
        active = torch.where(off & active, torch.zeros_like(active), active)

        active = torch.where(valid, active, torch.zeros_like(active))

    if mask is not None:
        decoded_active = decoded_active * mask.unsqueeze(-1)
        decoded_onset = decoded_onset * mask.unsqueeze(-1)
    return decoded_active, decoded_onset


def state4_macro_f1(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    num_states: int = 4,
) -> Dict[str, float]:
    valid = valid_mask.to(dtype=torch.bool).unsqueeze(-1)
    f1_values = []
    result: Dict[str, float] = {}
    for state in range(num_states):
        pred_state = preds == state
        target_state = targets == state
        tp = float((pred_state & target_state & valid).sum().item())
        fp = float((pred_state & ~target_state & valid).sum().item())
        fn = float((~pred_state & target_state & valid).sum().item())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        result[f"state_{state}_f1"] = f1
        f1_values.append(f1)
    result["state_macro_f1"] = sum(f1_values) / max(len(f1_values), 1)
    return result


def state4_metric_counts(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, tuple[float, float, float]]:
    valid = mask.unsqueeze(-1)
    pred_frame = (pred_states != 0).to(dtype=torch.float32)
    target_frame = (target_states != 0).to(dtype=torch.float32)
    pred_onset = (pred_states == 3).to(dtype=torch.float32)
    target_onset = (target_states == 3).to(dtype=torch.float32)

    decoded_frame, decoded_onset = decode_state4_predictions(pred_states, mask=mask)
    target_decoded_frame, target_decoded_onset = decode_state4_predictions(target_states, mask=mask)

    return {
        "frame": micro_counts(pred_frame, target_frame, valid),
        "onset": micro_counts(pred_onset, target_onset, valid),
        "decoded_frame": micro_counts(decoded_frame, target_decoded_frame, valid),
        "decoded_onset": micro_counts(decoded_onset, target_decoded_onset, valid),
    }
