#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn

from midi_transcribe.data import list_pt_files
from midi_transcribe.eval import evaluate_onsets_frames_split, evaluate_onsets_frames_threshold_grid
from midi_transcribe.inference import build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an onsets+frames checkpoint")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-frames", type=int, default=512)
    parser.add_argument("--segment-stride", type=int, default=512)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-grid", type=str, default=None)
    parser.add_argument("--optimize-thresholds-on", choices=("validation", "test"), default=None)
    parser.add_argument("--frame-pos-weight", type=float, default=1.0)
    parser.add_argument("--onset-pos-weight", type=float, default=1.0)
    parser.add_argument("--onset-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def parse_threshold_values(raw: str) -> list[float]:
    values = [float(x.strip()) for x in re.split(r"[,:;\s]+", raw) if x.strip()]
    if not values:
        raise ValueError("--threshold-grid must contain at least one value")
    return values


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(Path(args.checkpoint_path), device=device)
    files = list_pt_files(Path(args.data_root), args.split)

    frame_weight = torch.full((88,), args.frame_pos_weight, device=device)
    onset_weight = torch.full((88,), args.onset_pos_weight, device=device)
    frame_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=frame_weight)
    onset_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=onset_weight)

    best_thresholds = None
    frame_threshold = args.frame_threshold
    onset_threshold = args.onset_threshold

    if args.optimize_thresholds_on is not None:
        threshold_values = parse_threshold_values(args.threshold_grid or "0.4,0.5,0.6")
        threshold_pairs = [(f, o) for f in threshold_values for o in threshold_values]
        threshold_files = list_pt_files(Path(args.data_root), args.optimize_thresholds_on)
        threshold_search = evaluate_onsets_frames_threshold_grid(
            model=model,
            files=threshold_files,
            device=device,
            batch_size=args.batch_size,
            segment_frames=args.segment_frames,
            segment_stride=args.segment_stride,
            threshold_pairs=threshold_pairs,
        )
        best_thresholds = threshold_search[0]
        frame_threshold = best_thresholds["frame_threshold"]
        onset_threshold = best_thresholds["onset_threshold"]

    metrics = evaluate_onsets_frames_split(
        model=model,
        files=files,
        device=device,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
        segment_stride=args.segment_stride,
        frame_criterion=frame_criterion,
        onset_criterion=onset_criterion,
        onset_loss_weight=args.onset_loss_weight,
        frame_threshold=frame_threshold,
        onset_threshold=onset_threshold,
    )

    output = {
        "checkpoint_path": args.checkpoint_path,
        "data_root": args.data_root,
        "split": args.split,
        "batch_size": args.batch_size,
        "segment_frames": args.segment_frames,
        "segment_stride": args.segment_stride,
        "frame_threshold": frame_threshold,
        "onset_threshold": onset_threshold,
        "optimized_on": args.optimize_thresholds_on,
        "best_thresholds": best_thresholds,
        "metrics": metrics,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
