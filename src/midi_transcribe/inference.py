from pathlib import Path
from typing import Dict

import torch

from .data import load_piece
from .model import create_model


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get("config", {})
    model_name = str(model_cfg.get("model_name", "mlp_baseline"))
    hidden_dim = int(model_cfg.get("hidden_dim", 512))

    model = create_model(
        model_name=model_name,
        input_dim=128,
        output_dim=88,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_segment_from_pt(
    model: torch.nn.Module,
    pt_path: Path,
    start_frame: int = 0,
    segment_frames: int = 512,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device

    features, targets = load_piece(pt_path)  # [T,128], [T,88]
    total_frames = features.shape[0]
    start = max(0, min(start_frame, max(total_frames - 1, 0)))
    end = min(start + segment_frames, total_frames)

    seg_x = features[start:end]
    seg_y = targets[start:end]
    valid_len = end - start

    if valid_len < segment_frames:
        pad = segment_frames - valid_len
        seg_x = torch.cat([seg_x, torch.zeros((pad, seg_x.shape[1]), dtype=seg_x.dtype)], dim=0)
        seg_y = torch.cat([seg_y, torch.zeros((pad, seg_y.shape[1]), dtype=seg_y.dtype)], dim=0)

    x = seg_x.unsqueeze(0).to(device, non_blocking=True)  # [1,T,128]
    logits = model(x).squeeze(0).cpu()  # [T,88]
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()

    valid_mask = torch.zeros(segment_frames, dtype=torch.float32)
    valid_mask[:valid_len] = 1.0

    return {
        "features": seg_x[:valid_len].cpu(),  # [T_valid,128]
        "targets": seg_y[:valid_len].cpu(),   # [T_valid,88]
        "logits": logits[:valid_len],         # [T_valid,88]
        "probs": probs[:valid_len],           # [T_valid,88]
        "pred": pred[:valid_len],             # [T_valid,88]
        "valid_mask": valid_mask[:valid_len], # [T_valid]
        "start_frame": torch.tensor(start),
        "end_frame": torch.tensor(end),
    }
