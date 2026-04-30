from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from .data import iterate_batches
from .metrics import bce_masked_loss


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
