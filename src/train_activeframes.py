#!/usr/bin/env python3
import argparse
from pathlib import Path

from midi_transcribe.config import TrainConfig
from midi_transcribe.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train active-frames baseline model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mlp_baseline",
        choices=("mlp_baseline", "cnn_context5"),
        help="Model architecture to train",
    )
    parser.add_argument("--data-root", required=True, help="Precomputed dataset root")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--segment-frames", type=int, default=512)
    parser.add_argument("--segment-stride", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        model_name=args.model_name,
        data_root=Path(args.data_root),
        run_dir=Path(args.run_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
        segment_stride=args.segment_stride,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        prediction_threshold=args.prediction_threshold,
        seed=args.seed,
    )
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
