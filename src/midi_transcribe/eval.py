from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from .data import iterate_batches
from .data import iterate_onsets_frames_batches
from .metrics import (
    bce_masked_loss,
    decode_onsets_frames,
    micro_counts,
    onsets_frames_loss,
    prf_from_counts,
)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    files: List[Path],
    device: torch.device,
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    tp = 0.0
    fp = 0.0
    fn = 0.0

    for xb, yb, mb in iterate_batches(
        files=files,
        batch_size=batch_size,
        segment_frames=segment_frames,
        segment_stride=segment_stride,
        shuffle_files=False,
    ):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(xb)
            loss = bce_masked_loss(logits, yb, mb, criterion)

        total_loss += float(loss.item())
        total_batches += 1

        preds = (torch.sigmoid(logits) >= 0.5).float()
        valid = mb.unsqueeze(-1)

        tp += float(((preds == 1.0) & (yb == 1.0)).float().mul(valid).sum().item())
        fp += float(((preds == 1.0) & (yb == 0.0)).float().mul(valid).sum().item())
        fn += float(((preds == 0.0) & (yb == 1.0)).float().mul(valid).sum().item())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    return {
        "loss": total_loss / max(total_batches, 1),
        "precision_micro": precision,
        "recall_micro": recall,
        "f1_micro": f1,
    }


@torch.no_grad()
def evaluate_onsets_frames_split(
    model: nn.Module,
    files: List[Path],
    device: torch.device,
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
    frame_criterion: nn.Module,
    onset_criterion: nn.Module,
    onset_loss_weight: float,
    frame_threshold: float,
    onset_threshold: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_frame_loss = 0.0
    total_onset_loss = 0.0
    total_batches = 0

    frame_tp = 0.0
    frame_fp = 0.0
    frame_fn = 0.0
    onset_tp = 0.0
    onset_fp = 0.0
    onset_fn = 0.0
    decoded_frame_tp = 0.0
    decoded_frame_fp = 0.0
    decoded_frame_fn = 0.0

    for xb, frame_yb, onset_yb, mb in iterate_onsets_frames_batches(
        files=files,
        batch_size=batch_size,
        segment_frames=segment_frames,
        segment_stride=segment_stride,
        shuffle_files=False,
    ):
        xb = xb.to(device, non_blocking=True)
        frame_yb = frame_yb.to(device, non_blocking=True)
        onset_yb = onset_yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(xb)
            loss, loss_parts = onsets_frames_loss(
                outputs=outputs,
                frame_targets=frame_yb,
                onset_targets=onset_yb,
                mask=mb,
                frame_criterion=frame_criterion,
                onset_criterion=onset_criterion,
                onset_loss_weight=onset_loss_weight,
            )

        total_loss += float(loss.item())
        total_frame_loss += loss_parts["frame_loss"]
        total_onset_loss += loss_parts["onset_loss"]
        total_batches += 1

        valid = mb.unsqueeze(-1)
        frame_probs = torch.sigmoid(outputs["frame_logits"])
        onset_probs = torch.sigmoid(outputs["onset_logits"])
        frame_preds = (frame_probs >= frame_threshold).float()
        onset_preds = (onset_probs >= onset_threshold).float()
        decoded_frame_preds = decode_onsets_frames(
            onset_probs=onset_probs,
            frame_probs=frame_probs,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )

        tp, fp, fn = micro_counts(frame_preds, frame_yb, valid)
        frame_tp += tp
        frame_fp += fp
        frame_fn += fn

        tp, fp, fn = micro_counts(onset_preds, onset_yb, valid)
        onset_tp += tp
        onset_fp += fp
        onset_fn += fn

        tp, fp, fn = micro_counts(decoded_frame_preds, frame_yb, valid)
        decoded_frame_tp += tp
        decoded_frame_fp += fp
        decoded_frame_fn += fn

    return {
        "loss": total_loss / max(total_batches, 1),
        "frame_loss": total_frame_loss / max(total_batches, 1),
        "onset_loss": total_onset_loss / max(total_batches, 1),
        **prf_from_counts(frame_tp, frame_fp, frame_fn, prefix="frame_"),
        **prf_from_counts(onset_tp, onset_fp, onset_fn, prefix="onset_"),
        **prf_from_counts(
            decoded_frame_tp,
            decoded_frame_fp,
            decoded_frame_fn,
            prefix="decoded_frame_",
        ),
    }
