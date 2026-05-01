from pathlib import Path
from typing import Dict

import torch

from .data import load_piece, load_onsets_frames_piece
from .metrics import decode_onsets_frames
from .model import create_model


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get("config", {})
    model_name = str(model_cfg.get("model_name", "mlp_baseline"))
    hidden_dim = int(model_cfg.get("hidden_dim", model_cfg.get("head_hidden", 512)))
    use_temporal_convs = bool(model_cfg.get("use_temporal_convs", False))
    rnn_type = str(model_cfg.get("rnn_type", "lstm"))
    rnn_hidden_size = int(model_cfg.get("rnn_hidden_size", 128))
    rnn_num_layers = int(model_cfg.get("rnn_num_layers", 1))
    rnn_bidirectional = bool(model_cfg.get("rnn_bidirectional", True))

    model = create_model(
        model_name=model_name,
        input_dim=128,
        output_dim=88,
        hidden_dim=hidden_dim,
        use_temporal_convs=use_temporal_convs,
        rnn_type=rnn_type,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        rnn_bidirectional=rnn_bidirectional,
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
    frame_threshold: float | None = None,
    onset_threshold: float | None = None,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device

    try:
        features, targets, onset_targets = load_onsets_frames_piece(pt_path)
    except KeyError:
        features, targets = load_piece(pt_path)  # [T,128], [T,88]
        onset_targets = None

    frame_threshold = threshold if frame_threshold is None else frame_threshold
    onset_threshold = threshold if onset_threshold is None else onset_threshold

    total_frames = features.shape[0]
    start = max(0, min(start_frame, max(total_frames - 1, 0)))
    end = min(start + segment_frames, total_frames)

    seg_x = features[start:end]
    seg_y = targets[start:end]
    seg_onset_y = onset_targets[start:end] if onset_targets is not None else None
    valid_len = end - start

    if valid_len < segment_frames:
        pad = segment_frames - valid_len
        seg_x = torch.cat([seg_x, torch.zeros((pad, seg_x.shape[1]), dtype=seg_x.dtype)], dim=0)
        seg_y = torch.cat([seg_y, torch.zeros((pad, seg_y.shape[1]), dtype=seg_y.dtype)], dim=0)
        if seg_onset_y is not None:
            seg_onset_y = torch.cat(
                [
                    seg_onset_y,
                    torch.zeros((pad, seg_onset_y.shape[1]), dtype=seg_onset_y.dtype),
                ],
                dim=0,
            )

    x = seg_x.unsqueeze(0).to(device, non_blocking=True)  # [1,T,128]
    outputs = model(x)
    if isinstance(outputs, dict):
        frame_logits = outputs["frame_logits"].squeeze(0).cpu()  # [T,88]
        onset_logits = outputs["onset_logits"].squeeze(0).cpu()  # [T,88]
        frame_probs = torch.sigmoid(frame_logits)
        onset_probs = torch.sigmoid(onset_logits)
        raw_frame_pred = (frame_probs >= frame_threshold).float()
        onset_pred = (onset_probs >= onset_threshold).float()
        pred = decode_onsets_frames(
            onset_probs=onset_probs.unsqueeze(0),
            frame_probs=frame_probs.unsqueeze(0),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        ).squeeze(0)
        logits = frame_logits
        probs = frame_probs
    else:
        logits = outputs.squeeze(0).cpu()  # [T,88]
        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).float()
        raw_frame_pred = pred
        onset_logits = None
        onset_probs = None
        onset_pred = None

    valid_mask = torch.zeros(segment_frames, dtype=torch.float32)
    valid_mask[:valid_len] = 1.0

    result = {
        "features": seg_x[:valid_len].cpu(),  # [T_valid,128]
        "targets": seg_y[:valid_len].cpu(),   # [T_valid,88]
        "logits": logits[:valid_len],         # [T_valid,88]
        "probs": probs[:valid_len],           # [T_valid,88]
        "pred": pred[:valid_len],             # decoded for onsets+frames, raw for frame-only
        "raw_frame_pred": raw_frame_pred[:valid_len],
        "valid_mask": valid_mask[:valid_len], # [T_valid]
        "start_frame": torch.tensor(start),
        "end_frame": torch.tensor(end),
    }
    if onset_targets is not None and seg_onset_y is not None:
        result["onset_targets"] = seg_onset_y[:valid_len].cpu()
    if onset_logits is not None and onset_probs is not None and onset_pred is not None:
        result["onset_logits"] = onset_logits[:valid_len]
        result["onset_probs"] = onset_probs[:valid_len]
        result["onset_pred"] = onset_pred[:valid_len]
    return result
