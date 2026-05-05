#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from midi_transcribe.data import (
    iterate_state4_batches,
    list_pt_files,
    load_state4_piece,
    segment_starts,
)
from midi_transcribe.eval import evaluate_state4_split
from midi_transcribe.metrics import decode_state4_predictions, state4_masked_cross_entropy_loss
from midi_transcribe.model import create_model
from midi_transcribe.visualization import save_prediction_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN state4 baseline")
    parser.add_argument("--data-root", required=True, help="Precomputed state4 dataset root")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--model-name", choices=["crnn_state4"], default="crnn_state4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--segment-frames", type=int, default=512)
    parser.add_argument("--segment-stride", type=int, default=512)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--pooled-freq-bands", type=int, default=16)
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="gru")
    parser.add_argument("--rnn-input-dim", type=int, default=256)
    parser.add_argument("--rnn-hidden-size", type=int, default=128)
    parser.add_argument("--rnn-num-layers", type=int, default=2)
    parser.add_argument("--rnn-bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-lr-scheduler", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weight-min", type=float, default=0.05)
    parser.add_argument("--class-weight-max", type=float, default=20.0)
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
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


class State4FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 4, T, 88], targets: [B, T, 88]
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        gather_idx = targets.unsqueeze(1)
        target_log_probs = log_probs.gather(dim=1, index=gather_idx).squeeze(1)
        target_probs = probs.gather(dim=1, index=gather_idx).squeeze(1)
        focal_factor = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_factor * target_log_probs
        if self.weight is not None:
            target_weight = self.weight.to(device=logits.device, dtype=logits.dtype)[targets]
            loss = loss * target_weight
        return loss


def count_state4_batches(
    files: List[Path],
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
) -> tuple[int, int]:
    total_segments = 0
    for path in files:
        features, _ = load_state4_piece(path)
        total_segments += len(segment_starts(features.shape[0], segment_frames, segment_stride))
    total_batches = math.ceil(total_segments / batch_size)
    return total_segments, total_batches


def estimate_state4_class_weights(
    files: List[Path],
    num_states: int,
    clamp_min: float,
    clamp_max: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    counts = torch.zeros(num_states, dtype=torch.float64)
    for path in files:
        _, targets = load_state4_piece(path)
        counts += torch.bincount(targets.reshape(-1), minlength=num_states).to(dtype=torch.float64)

    total = counts.sum().clamp_min(1.0)
    frequencies = counts / total
    weights = torch.rsqrt(frequencies.clamp_min(1.0 / total))
    weights = weights / weights.mean().clamp_min(1e-8)
    weights = weights.clamp(clamp_min, clamp_max).to(dtype=torch.float32)
    summary = {
        "method": "inverse_sqrt_frequency_mean_normalized",
        "counts": [float(x) for x in counts.tolist()],
        "frequencies": [float(x) for x in frequencies.tolist()],
        "weights": summarize_weight(weights),
    }
    return weights, summary


def save_state4_prediction_preview(
    model: nn.Module,
    files: List[Path],
    device: torch.device,
    output_dir: Path,
    segment_frames: int,
) -> None:
    features, targets, mask = next(
        iterate_state4_batches(
            files=[files[0]],
            batch_size=1,
            segment_frames=segment_frames,
            segment_stride=segment_frames,
            shuffle_files=False,
        )
    )

    with torch.no_grad():
        outputs = model(features.to(device))
        pred_states = outputs["state_logits"].argmax(dim=-1).cpu()

    pred_roll, _ = decode_state4_predictions(pred_states, mask=mask)
    target_roll, _ = decode_state4_predictions(targets, mask=mask)

    save_prediction_figure(
        features=features[0],
        pred_roll=pred_roll[0],
        target_roll=target_roll[0],
        output_path=output_dir / "prediction_preview.png",
        title=f"State4 decoded frame preview: {files[0].name}",
    )
    (output_dir / "prediction_preview_meta.json").write_text(
        json.dumps(
            {
                "pt_file": str(files[0]),
                "start_frame": 0,
                "segment_frames": segment_frames,
                "prediction": "decoded active frames from argmax state4 predictions",
                "target": "decoded active frames from target state4 labels",
            },
            indent=2,
        ),
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

    if args.use_class_weights:
        class_weights, class_weight_summary = estimate_state4_class_weights(
            files=train_files,
            num_states=4,
            clamp_min=args.class_weight_min,
            clamp_max=args.class_weight_max,
        )
    else:
        class_weights = torch.ones(4, dtype=torch.float32)
        class_weight_summary = {
            "counts": None,
            "frequencies": None,
            "weights": summarize_weight(class_weights),
        }
    class_weights = class_weights.to(device)

    model = create_model(
        model_name=args.model_name,
        input_dim=128,
        output_dim=88,
        hidden_dim=args.head_hidden,
        pooled_freq_bands=args.pooled_freq_bands,
        rnn_type=args.rnn_type,
        rnn_input_dim=args.rnn_input_dim,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_num_layers,
        rnn_bidirectional=args.rnn_bidirectional,
    ).to(device)

    if args.loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(reduction="none", weight=class_weights)
    else:
        criterion = State4FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    train_segments, train_batches_per_epoch = count_state4_batches(
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

    run_config = {
        "data_root": str(data_root),
        "model_name": args.model_name,
        "target_mode": "state4",
        "target_state_mapping": {"off": 0, "sustain": 1, "offset": 2, "onset": 3},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "segment_frames": args.segment_frames,
        "segment_stride": args.segment_stride,
        "head_hidden": args.head_hidden,
        "pooled_freq_bands": args.pooled_freq_bands,
        "rnn_type": args.rnn_type,
        "rnn_input_dim": args.rnn_input_dim,
        "rnn_hidden_size": args.rnn_hidden_size,
        "rnn_num_layers": args.rnn_num_layers,
        "rnn_bidirectional": args.rnn_bidirectional,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss": {
            "type": args.loss_type,
            "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
            "class_weighting": "enabled" if args.use_class_weights else "disabled",
        },
        "class_weights": class_weight_summary,
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
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "monitor": "val_loss",
            "patience": args.early_stopping_patience,
        },
        "seed": args.seed,
        "device": str(device),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (artifacts_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    val_log_path = artifacts_dir / "metrics_val_history.jsonl"
    best_val_loss = math.inf
    epochs_without_val_loss_improvement = 0
    stopped_early = False
    last_epoch = 0
    start_time = time.time()

    with val_log_path.open("w", encoding="utf-8") as val_log_f:
        for epoch in range(1, args.epochs + 1):
            last_epoch = epoch
            model.train()
            lr_epoch_start = optimizer.param_groups[0]["lr"]
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_start = time.time()

            for xb, yb, mb in iterate_state4_batches(
                files=train_files,
                batch_size=args.batch_size,
                segment_frames=args.segment_frames,
                segment_stride=args.segment_stride,
                shuffle_files=True,
            ):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(xb)
                    loss = state4_masked_cross_entropy_loss(
                        logits=outputs["state_logits"],
                        targets=yb,
                        mask=mb,
                        criterion=criterion,
                    )

                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                if scheduler is not None and new_scale >= old_scale:
                    scheduler.step()

                epoch_loss += float(loss.item())
                epoch_batches += 1

            val_metrics = evaluate_state4_split(
                model=model,
                files=val_files,
                device=device,
                batch_size=args.batch_size,
                segment_frames=args.segment_frames,
                segment_stride=args.segment_stride,
                criterion=criterion,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_val_loss_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_loss": best_val_loss,
                        "config": run_config,
                    },
                    artifacts_dir / "best_model.pt",
                )
            else:
                epochs_without_val_loss_improvement += 1

            val_rec = {
                "epoch": epoch,
                "lr_epoch_start": lr_epoch_start,
                "lr_epoch_end": optimizer.param_groups[0]["lr"],
                "train_loss_mean": epoch_loss / max(epoch_batches, 1),
                "best_val_loss": best_val_loss,
                "epochs_without_val_loss_improvement": epochs_without_val_loss_improvement,
                **val_metrics,
            }
            val_log_f.write(json.dumps(val_rec) + "\n")
            val_log_f.flush()

            print(
                "epoch={epoch} lr_start={lr_start:.6g} lr_end={lr_end:.6g} "
                "train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                "val_frame_f1={frame_f1:.4f} val_onset_f1={onset_f1:.4f} "
                "val_decoded_frame_f1={decoded_frame_f1:.4f} val_state_macro_f1={macro_f1:.4f} "
                "epoch_sec={sec:.1f}".format(
                    epoch=epoch,
                    lr_start=val_rec["lr_epoch_start"],
                    lr_end=val_rec["lr_epoch_end"],
                    train_loss=val_rec["train_loss_mean"],
                    val_loss=val_metrics["loss"],
                    frame_f1=val_metrics["frame_f1_micro"],
                    onset_f1=val_metrics["onset_f1_micro"],
                    decoded_frame_f1=val_metrics["decoded_frame_f1_micro"],
                    macro_f1=val_metrics["state_macro_f1"],
                    sec=time.time() - epoch_start,
                ),
                flush=True,
            )

            if (
                args.early_stopping_patience > 0
                and epochs_without_val_loss_improvement >= args.early_stopping_patience
            ):
                stopped_early = True
                print(
                    "early_stopping triggered at epoch={epoch} best_val_loss={best_val_loss:.6f} "
                    "patience={patience}".format(
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        patience=args.early_stopping_patience,
                    ),
                    flush=True,
                )
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": run_config,
            "best_val_loss": best_val_loss,
            "epochs_trained": last_epoch,
            "stopped_early": stopped_early,
        },
        artifacts_dir / "final_model.pt",
    )

    best_checkpoint = torch.load(artifacts_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    test_metrics = evaluate_state4_split(
        model=model,
        files=test_files,
        device=device,
        batch_size=args.batch_size,
        segment_frames=args.segment_frames,
        segment_stride=args.segment_stride,
        criterion=criterion,
    )
    metrics_test = {
        "test": test_metrics,
        "best_val_loss": best_val_loss,
        "epochs_trained": last_epoch,
        "stopped_early": stopped_early,
        "total_train_time_sec": time.time() - start_time,
    }
    (artifacts_dir / "metrics_test.json").write_text(
        json.dumps(metrics_test, indent=2),
        encoding="utf-8",
    )

    try:
        save_state4_prediction_preview(
            model=model,
            files=test_files,
            device=device,
            output_dir=artifacts_dir,
            segment_frames=args.segment_frames,
        )
    except Exception as exc:
        print(f"[warn] failed to generate state4 prediction preview: {exc}", flush=True)

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
        "test_frame_f1={frame_f1:.4f} test_onset_f1={onset_f1:.4f} "
        "test_decoded_frame_f1={decoded_frame_f1:.4f} test_loss={loss:.6f}".format(
            frame_f1=test_metrics["frame_f1_micro"],
            onset_f1=test_metrics["onset_f1_micro"],
            decoded_frame_f1=test_metrics["decoded_frame_f1_micro"],
            loss=test_metrics["loss"],
        ),
        flush=True,
    )
    print(f"Artifacts written to: {artifacts_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
