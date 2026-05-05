from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .data import iterate_batches
from .data import iterate_onsets_frames_batches
from .data import iterate_state4_batches
from .metrics import (
    bce_masked_loss,
    decode_onsets_frames,
    micro_counts,
    onsets_frames_loss,
    prf_from_counts,
    state4_macro_f1,
    state4_masked_cross_entropy_loss,
    state4_metric_counts,
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
    onset_loss_type: str = "bce",
    onset_focal_gamma: float = 2.0,
    onset_focal_alpha_pos: torch.Tensor | None = None,
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
                onset_loss_type=onset_loss_type,
                onset_focal_gamma=onset_focal_gamma,
                onset_focal_alpha_pos=onset_focal_alpha_pos,
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


@torch.no_grad()
def evaluate_onsets_frames_threshold_grid(
    model: nn.Module,
    files: List[Path],
    device: torch.device,
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
    threshold_pairs: Iterable[tuple[float, float]],
) -> List[Dict[str, float]]:
    model.eval()
    pairs = list(threshold_pairs)
    counts = {
        pair: {
            "frame": [0.0, 0.0, 0.0],
            "onset": [0.0, 0.0, 0.0],
            "decoded_frame": [0.0, 0.0, 0.0],
        }
        for pair in pairs
    }

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
        valid = mb.unsqueeze(-1)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(xb)

        frame_probs = torch.sigmoid(outputs["frame_logits"])
        onset_probs = torch.sigmoid(outputs["onset_logits"])

        for frame_threshold, onset_threshold in pairs:
            frame_preds = (frame_probs >= frame_threshold).float()
            onset_preds = (onset_probs >= onset_threshold).float()
            decoded_frame_preds = decode_onsets_frames(
                onset_probs=onset_probs,
                frame_probs=frame_probs,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
            )

            for name, preds, targets in (
                ("frame", frame_preds, frame_yb),
                ("onset", onset_preds, onset_yb),
                ("decoded_frame", decoded_frame_preds, frame_yb),
            ):
                tp, fp, fn = micro_counts(preds, targets, valid)
                counts[(frame_threshold, onset_threshold)][name][0] += tp
                counts[(frame_threshold, onset_threshold)][name][1] += fp
                counts[(frame_threshold, onset_threshold)][name][2] += fn

    results = []
    for frame_threshold, onset_threshold in pairs:
        rec: Dict[str, float] = {
            "frame_threshold": frame_threshold,
            "onset_threshold": onset_threshold,
        }
        for prefix in ("frame", "onset", "decoded_frame"):
            tp, fp, fn = counts[(frame_threshold, onset_threshold)][prefix]
            rec.update(prf_from_counts(tp, fp, fn, prefix=f"{prefix}_"))
        results.append(rec)

    return sorted(results, key=lambda x: x["decoded_frame_f1_micro"], reverse=True)


@torch.no_grad()
def evaluate_state4_split(
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
    counts = {
        "frame": [0.0, 0.0, 0.0],
        "onset": [0.0, 0.0, 0.0],
        "decoded_frame": [0.0, 0.0, 0.0],
        "decoded_onset": [0.0, 0.0, 0.0],
    }
    macro_f1_totals = {f"state_{state}_f1": 0.0 for state in range(4)}
    macro_f1_totals["state_macro_f1"] = 0.0

    for xb, yb, mb in iterate_state4_batches(
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
            outputs = model(xb)
            loss = state4_masked_cross_entropy_loss(
                logits=outputs["state_logits"],
                targets=yb,
                mask=mb,
                criterion=criterion,
            )

        total_loss += float(loss.item())
        total_batches += 1

        preds = outputs["state_logits"].argmax(dim=-1)
        batch_counts = state4_metric_counts(preds, yb, mb)
        for name, (tp, fp, fn) in batch_counts.items():
            counts[name][0] += tp
            counts[name][1] += fp
            counts[name][2] += fn

        macro_f1 = state4_macro_f1(preds, yb, mb)
        for key, value in macro_f1.items():
            macro_f1_totals[key] += value

    metrics: Dict[str, float] = {"loss": total_loss / max(total_batches, 1)}
    for prefix, (tp, fp, fn) in counts.items():
        metrics.update(prf_from_counts(tp, fp, fn, prefix=f"{prefix}_"))
    for key, value in macro_f1_totals.items():
        metrics[key] = value / max(total_batches, 1)
    return metrics
