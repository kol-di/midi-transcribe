#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from midi_transcribe.data import (
    iterate_onsets_frames_batches,
    list_pt_files,
    load_onsets_frames_piece,
    segment_starts,
)
from midi_transcribe.eval import evaluate_onsets_frames_split, evaluate_onsets_frames_threshold_grid
from midi_transcribe.metrics import decode_onsets_frames, onsets_frames_loss
from midi_transcribe.model import create_model
from midi_transcribe.visualization import save_prediction_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train onset+frame CNN/CRNN baseline")
    parser.add_argument("--data-root", required=True, help="Precomputed onsets+frames dataset root")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument(
        "--model-name",
        choices=["cnn_onsets_frames", "crnn_onsets_frames"],
        default="cnn_onsets_frames",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--segment-frames", type=int, default=512)
    parser.add_argument("--segment-stride", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--pooled-freq-bands", type=int, default=8)
    parser.add_argument("--use-temporal-convs", action="store_true")
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--rnn-input-dim", type=int, default=256)
    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--rnn-num-layers", type=int, default=1)
    parser.add_argument("--rnn-bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use-lr-scheduler", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--onset-loss-weight", type=float, default=1.0)
    parser.add_argument("--frame-threshold", type=float, default=0.5)
    parser.add_argument("--onset-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-grid", type=str, default="0.3,0.4,0.5,0.6")
    parser.add_argument("--frame-pos-weight", type=float, default=None)
    parser.add_argument("--onset-pos-weight", type=float, default=None)
    parser.add_argument("--max-batches-for-pos-weight", type=int, default=64)
    parser.add_argument("--pos-weight-min", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_weight(weight: torch.Tensor) -> Dict[str, Any]:
    return {
        "mean": float(weight.mean().item()),
        "min": float(weight.min().item()),
        "max": float(weight.max().item()),
        "values": [float(x) for x in weight.cpu().tolist()],
    }


def parse_threshold_values(raw: str) -> List[float]:
    values = [float(x.strip()) for x in re.split(r"[,:;\s]+", raw) if x.strip()]
    if not values:
        raise ValueError("--threshold-grid must contain at least one value")
    return values


def count_onsets_frames_batches(
    files: List[Path],
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
) -> tuple[int, int]:
    total_segments = 0
    for path in files:
        features, _, _ = load_onsets_frames_piece(path)
        total_segments += len(segment_starts(features.shape[0], segment_frames, segment_stride))
    total_batches = math.ceil(total_segments / batch_size)
    return total_segments, total_batches


def estimate_pos_weights(
    train_files: List[Path],
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
    max_batches: int,
    clamp_min: float,
    clamp_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_pos = torch.zeros(88, dtype=torch.float64)
    onset_pos = torch.zeros(88, dtype=torch.float64)
    valid_frames = 0.0

    for batch_idx, (_, frame_yb, onset_yb, mb) in enumerate(
        iterate_onsets_frames_batches(
            files=train_files,
            batch_size=batch_size,
            segment_frames=segment_frames,
            segment_stride=segment_stride,
            shuffle_files=True,
        )
    ):
        valid = mb.unsqueeze(-1).to(dtype=torch.float64)
        frame_pos += (frame_yb.to(dtype=torch.float64) * valid).sum(dim=(0, 1))
        onset_pos += (onset_yb.to(dtype=torch.float64) * valid).sum(dim=(0, 1))
        valid_frames += float(mb.sum().item())

        if batch_idx + 1 >= max_batches:
            break

    total_per_key = torch.full((88,), valid_frames, dtype=torch.float64)
    frame_neg = total_per_key - frame_pos
    onset_neg = total_per_key - onset_pos

    frame_weight = frame_neg / frame_pos.clamp_min(1.0)
    onset_weight = onset_neg / onset_pos.clamp_min(1.0)
    return (
        frame_weight.clamp(clamp_min, clamp_max).to(dtype=torch.float32),
        onset_weight.clamp(clamp_min, clamp_max).to(dtype=torch.float32),
    )


def save_prediction_preview(
    model: nn.Module,
    files: List[Path],
    device: torch.device,
    output_dir: Path,
    segment_frames: int,
    frame_threshold: float,
    onset_threshold: float,
) -> None:
    features, frame_targets, _ = next(
        iterate_onsets_frames_batches(
            files=[files[0]],
            batch_size=1,
            segment_frames=segment_frames,
            segment_stride=segment_frames,
            shuffle_files=False,
        )
    )[:3]

    with torch.no_grad():
        outputs = model(features.to(device))
        frame_probs = torch.sigmoid(outputs["frame_logits"]).cpu()
        onset_probs = torch.sigmoid(outputs["onset_logits"]).cpu()
        frame_pred = decode_onsets_frames(
            onset_probs=onset_probs,
            frame_probs=frame_probs,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )

    save_prediction_figure(
        features=features[0],
        pred_roll=frame_pred[0],
        target_roll=frame_targets[0],
        output_path=output_dir / "prediction_preview.png",
        title=f"Onsets+Frames frame preview: {files[0].name}",
    )
    (output_dir / "prediction_preview_meta.json").write_text(
        json.dumps({"pt_file": str(files[0]), "start_frame": 0, "segment_frames": segment_frames}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    device = torch.device("cuda")
    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_pt_files(data_root, "train")
    val_files = list_pt_files(data_root, "validation")
    test_files = list_pt_files(data_root, "test")
    if not train_files or not val_files or not test_files:
        raise RuntimeError("One or more splits are empty. Check data-root path and split folders.")

    if args.frame_pos_weight is None or args.onset_pos_weight is None:
        auto_frame_weight, auto_onset_weight = estimate_pos_weights(
            train_files=train_files,
            batch_size=args.batch_size,
            segment_frames=args.segment_frames,
            segment_stride=args.segment_stride,
            max_batches=args.max_batches_for_pos_weight,
            clamp_min=args.pos_weight_min,
            clamp_max=args.pos_weight_max,
        )
    else:
        auto_frame_weight = torch.full((88,), args.frame_pos_weight, dtype=torch.float32)
        auto_onset_weight = torch.full((88,), args.onset_pos_weight, dtype=torch.float32)

    frame_weight = (
        torch.full((88,), args.frame_pos_weight, dtype=torch.float32)
        if args.frame_pos_weight is not None
        else auto_frame_weight
    ).to(device)
    onset_weight = (
        torch.full((88,), args.onset_pos_weight, dtype=torch.float32)
        if args.onset_pos_weight is not None
        else auto_onset_weight
    ).to(device)

    model = create_model(
        model_name=args.model_name,
        input_dim=128,
        output_dim=88,
        hidden_dim=args.head_hidden,
        use_temporal_convs=args.use_temporal_convs,
        pooled_freq_bands=args.pooled_freq_bands,
        rnn_type=args.rnn_type,
        rnn_input_dim=args.rnn_input_dim,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        rnn_bidirectional=args.rnn_bidirectional,
    ).to(device)

    frame_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=frame_weight)
    onset_criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=onset_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    train_segments, train_batches_per_epoch = count_onsets_frames_batches(
        files=train_files,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
        segment_stride=args.segment_stride,
    )
    total_steps = args.epochs * train_batches_per_epoch
    warmup_steps = min(args.warmup_epochs * train_batches_per_epoch, max(total_steps - 1, 1))
    scheduler = None
    if args.use_lr_scheduler:
        if total_steps <= 0:
            raise RuntimeError("Cannot create LR scheduler with zero training steps.")
        if warmup_steps > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    run_config = {
        "data_root": str(data_root),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "segment_frames": args.segment_frames,
        "segment_stride": args.segment_stride,
        "head_hidden": args.head_hidden,
        "pooled_freq_bands": args.pooled_freq_bands,
        "use_temporal_convs": args.use_temporal_convs,
        "rnn_type": args.rnn_type,
        "rnn_input_dim": args.rnn_input_dim,
        "rnn_hidden_size": args.rnn_hidden_size,
        "rnn_num_layers": args.rnn_num_layers,
        "rnn_bidirectional": args.rnn_bidirectional,
        "lr": args.lr,
        "lr_scheduler": {
            "enabled": args.use_lr_scheduler,
            "kind": "LinearLR+CosineAnnealingLR via SequentialLR" if args.use_lr_scheduler else None,
            "warmup_epochs": args.warmup_epochs,
            "warmup_steps": warmup_steps if args.use_lr_scheduler else 0,
            "total_steps": total_steps,
            "train_segments": train_segments,
            "train_batches_per_epoch": train_batches_per_epoch,
            "start_factor": 0.1 if args.use_lr_scheduler else None,
        },
        "weight_decay": args.weight_decay,
        "onset_loss_weight": args.onset_loss_weight,
        "frame_threshold": args.frame_threshold,
        "onset_threshold": args.onset_threshold,
        "threshold_grid": parse_threshold_values(args.threshold_grid),
        "max_batches_for_pos_weight": args.max_batches_for_pos_weight,
        "frame_pos_weight": summarize_weight(frame_weight.detach().cpu()),
        "onset_pos_weight": summarize_weight(onset_weight.detach().cpu()),
        "seed": args.seed,
        "device": str(device),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (artifacts_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    val_log_path = artifacts_dir / "metrics_val_history.jsonl"
    best_val_frame_f1 = -math.inf
    start_time = time.time()

    with val_log_path.open("w", encoding="utf-8") as val_log_f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            lr_epoch_start = optimizer.param_groups[0]["lr"]
            epoch_loss = 0.0
            epoch_frame_loss = 0.0
            epoch_onset_loss = 0.0
            epoch_batches = 0
            epoch_start = time.time()

            for xb, frame_yb, onset_yb, mb in iterate_onsets_frames_batches(
                files=train_files,
                batch_size=args.batch_size,
                segment_frames=args.segment_frames,
                segment_stride=args.segment_stride,
                shuffle_files=True,
            ):
                xb = xb.to(device, non_blocking=True)
                frame_yb = frame_yb.to(device, non_blocking=True)
                onset_yb = onset_yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(xb)
                    loss, loss_parts = onsets_frames_loss(
                        outputs=outputs,
                        frame_targets=frame_yb,
                        onset_targets=onset_yb,
                        mask=mb,
                        frame_criterion=frame_criterion,
                        onset_criterion=onset_criterion,
                        onset_loss_weight=args.onset_loss_weight,
                    )

                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                if scheduler is not None and new_scale >= old_scale:
                    scheduler.step()

                epoch_loss += loss_parts["loss"]
                epoch_frame_loss += loss_parts["frame_loss"]
                epoch_onset_loss += loss_parts["onset_loss"]
                epoch_batches += 1

            val_metrics = evaluate_onsets_frames_split(
                model=model,
                files=val_files,
                device=device,
                batch_size=args.batch_size,
                segment_frames=args.segment_frames,
                segment_stride=args.segment_stride,
                frame_criterion=frame_criterion,
                onset_criterion=onset_criterion,
                onset_loss_weight=args.onset_loss_weight,
                frame_threshold=args.frame_threshold,
                onset_threshold=args.onset_threshold,
            )
            val_rec = {
                "epoch": epoch,
                "lr_epoch_start": lr_epoch_start,
                "lr_epoch_end": optimizer.param_groups[0]["lr"],
                "train_loss_mean": epoch_loss / max(epoch_batches, 1),
                "train_frame_loss_mean": epoch_frame_loss / max(epoch_batches, 1),
                "train_onset_loss_mean": epoch_onset_loss / max(epoch_batches, 1),
                **val_metrics,
            }
            val_log_f.write(json.dumps(val_rec) + "\n")
            val_log_f.flush()

            print(
                "epoch={epoch} lr_start={lr_start:.6g} lr_end={lr_end:.6g} "
                "train_loss={train_loss:.6f} val_frame_f1={frame_f1:.4f} "
                "val_onset_f1={onset_f1:.4f} val_loss={val_loss:.6f} epoch_sec={sec:.1f}".format(
                    epoch=epoch,
                    lr_start=val_rec["lr_epoch_start"],
                    lr_end=val_rec["lr_epoch_end"],
                    train_loss=val_rec["train_loss_mean"],
                    frame_f1=val_metrics["frame_f1_micro"],
                    onset_f1=val_metrics["onset_f1_micro"],
                    val_loss=val_metrics["loss"],
                    sec=time.time() - epoch_start,
                ),
                flush=True,
            )

            if val_metrics["frame_f1_micro"] > best_val_frame_f1:
                best_val_frame_f1 = val_metrics["frame_f1_micro"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_frame_f1": best_val_frame_f1,
                        "config": run_config,
                    },
                    artifacts_dir / "best_model.pt",
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": run_config,
            "best_val_frame_f1": best_val_frame_f1,
            "epochs_trained": args.epochs,
        },
        artifacts_dir / "final_model.pt",
    )

    best_checkpoint = torch.load(artifacts_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    threshold_values = parse_threshold_values(args.threshold_grid)
    threshold_pairs = [(f, o) for f in threshold_values for o in threshold_values]
    threshold_search_results = evaluate_onsets_frames_threshold_grid(
        model=model,
        files=val_files,
        device=device,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
        segment_stride=args.segment_stride,
        threshold_pairs=threshold_pairs,
    )
    best_thresholds = threshold_search_results[0]
    # (artifacts_dir / "threshold_search_val.json").write_text(
    #     json.dumps(threshold_search_results, indent=2),
    #     encoding="utf-8",
    # )
    (artifacts_dir / "best_thresholds.json").write_text(
        json.dumps(best_thresholds, indent=2),
        encoding="utf-8",
    )

    test_metrics = evaluate_onsets_frames_split(
        model=model,
        files=test_files,
        device=device,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
        segment_stride=args.segment_stride,
        frame_criterion=frame_criterion,
        onset_criterion=onset_criterion,
        onset_loss_weight=args.onset_loss_weight,
        frame_threshold=best_thresholds["frame_threshold"],
        onset_threshold=best_thresholds["onset_threshold"],
    )
    metrics_test = {
        "test": test_metrics,
        "best_val_frame_f1": best_val_frame_f1,
        "best_thresholds": best_thresholds,
        "total_train_time_sec": time.time() - start_time,
    }
    (artifacts_dir / "metrics_test.json").write_text(
        json.dumps(metrics_test, indent=2),
        encoding="utf-8",
    )

    try:
        save_prediction_preview(
            model=model,
            files=test_files,
            device=device,
            output_dir=artifacts_dir,
            segment_frames=args.segment_frames,
            frame_threshold=best_thresholds["frame_threshold"],
            onset_threshold=best_thresholds["onset_threshold"],
        )
    except Exception as exc:
        print(f"[warn] failed to generate prediction preview: {exc}", flush=True)

    env_lines = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_device_name={torch.cuda.get_device_name(0)}",
        f"cuda_available={torch.cuda.is_available()}",
        f"train_files={len(train_files)}",
        f"validation_files={len(val_files)}",
        f"test_files={len(test_files)}",
    ]
    (artifacts_dir / "env.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print(
        "test_frame_f1={frame_f1:.4f} test_onset_f1={onset_f1:.4f} test_loss={loss:.6f}".format(
            frame_f1=test_metrics["frame_f1_micro"],
            onset_f1=test_metrics["onset_f1_micro"],
            loss=test_metrics["loss"],
        ),
        flush=True,
    )
    print(f"Artifacts written to: {artifacts_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
