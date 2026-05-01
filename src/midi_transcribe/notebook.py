from pathlib import Path

import torch

from .inference import build_model_from_checkpoint, predict_segment_from_pt
from .visualization import save_prediction_figure


def predict_and_plot_segment(
    checkpoint_path: str | Path,
    pt_path: str | Path,
    output_path: str | Path,
    start_frame: int = 0,
    segment_frames: int = 512,
    threshold: float = 0.5,
    frame_threshold: float | None = None,
    onset_threshold: float | None = None,
    device: str | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    pt_path = Path(pt_path)
    output_path = Path(output_path)

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    model = build_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device_obj)
    pred_pack = predict_segment_from_pt(
        model=model,
        pt_path=pt_path,
        start_frame=start_frame,
        segment_frames=segment_frames,
        threshold=threshold,
        frame_threshold=frame_threshold,
        onset_threshold=onset_threshold,
        device=device_obj,
    )
    save_prediction_figure(
        features=pred_pack["features"],
        pred_roll=pred_pack["pred"],
        target_roll=pred_pack["targets"],
        output_path=output_path,
        title=f"{pt_path.name} [{int(pred_pack['start_frame'])}:{int(pred_pack['end_frame'])}]",
    )
    return output_path


def predict_and_plot_decoded_onsets_frames_segment(
    checkpoint_path: str | Path,
    pt_path: str | Path,
    output_path: str | Path,
    start_frame: int = 0,
    segment_frames: int = 512,
    frame_threshold: float = 0.5,
    onset_threshold: float = 0.5,
    device: str | None = None,
) -> Path:
    return predict_and_plot_segment(
        checkpoint_path=checkpoint_path,
        pt_path=pt_path,
        output_path=output_path,
        start_frame=start_frame,
        segment_frames=segment_frames,
        threshold=frame_threshold,
        frame_threshold=frame_threshold,
        onset_threshold=onset_threshold,
        device=device,
    )
