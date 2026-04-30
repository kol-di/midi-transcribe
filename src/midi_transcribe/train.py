import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from .config import TrainConfig
from .data import iterate_batches, list_pt_files
from .eval import evaluate_split
from .inference import predict_segment_from_pt
from .metrics import bce_masked_loss
from .model import create_model
from .visualization import save_prediction_figure


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(config: TrainConfig) -> Dict[str, float]:
    set_seed(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    artifacts_dir = config.run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_pt_files(config.data_root, "train")
    val_files = list_pt_files(config.data_root, "validation")
    test_files = list_pt_files(config.data_root, "test")

    if not train_files or not val_files or not test_files:
        raise RuntimeError("One or more splits are empty. Check data-root path and split folders.")

    device = torch.device("cuda")
    model = create_model(
        model_name=config.model_name,
        input_dim=128,
        output_dim=88,
        hidden_dim=config.hidden_dim,
    ).to(device)

    pos_weight = torch.full((88,), config.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    run_config = {
        "data_root": str(config.data_root),
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "segment_frames": config.segment_frames,
        "segment_stride": config.segment_stride,
        "hidden_dim": config.hidden_dim,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "pos_weight": config.pos_weight,
        "prediction_threshold": config.prediction_threshold,
        "seed": config.seed,
        "device": str(device),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (artifacts_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    val_log_path = artifacts_dir / "metrics_val_history.jsonl"

    best_val_f1 = -math.inf

    start_time = time.time()
    with val_log_path.open("w", encoding="utf-8") as val_log_f:
        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_losses: List[float] = []
            epoch_start = time.time()

            for xb, yb, mb in iterate_batches(
                files=train_files,
                batch_size=config.batch_size,
                segment_frames=config.segment_frames,
                segment_stride=config.segment_stride,
                shuffle_files=True,
            ):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = bce_masked_loss(logits, yb, mb, criterion)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_v = float(loss.item())
                epoch_losses.append(loss_v)

            train_loss_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
            val_metrics = evaluate_split(
                model=model,
                files=val_files,
                device=device,
                batch_size=config.batch_size,
                segment_frames=config.segment_frames,
                segment_stride=config.segment_stride,
                criterion=criterion,
            )
            val_rec = {"epoch": epoch, "train_loss_mean": train_loss_mean, **val_metrics}
            val_log_f.write(json.dumps(val_rec) + "\n")
            val_log_f.flush()

            print(
                "epoch={epoch} train_loss_mean={train_loss:.6f} val_loss={val_loss:.6f} "
                "val_p={p:.4f} val_r={r:.4f} val_f1={f1:.4f} epoch_sec={sec:.1f}".format(
                    epoch=epoch,
                    train_loss=train_loss_mean,
                    val_loss=val_metrics["loss"],
                    p=val_metrics["precision_micro"],
                    r=val_metrics["recall_micro"],
                    f1=val_metrics["f1_micro"],
                    sec=time.time() - epoch_start,
                ),
                flush=True,
            )

            if val_metrics["f1_micro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_micro"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_f1": best_val_f1,
                        "config": run_config,
                    },
                    artifacts_dir / "best_model.pt",
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": run_config,
            "best_val_f1": best_val_f1,
            "epochs_trained": config.epochs,
        },
        artifacts_dir / "final_model.pt",
    )

    test_metrics = evaluate_split(
        model=model,
        files=test_files,
        device=device,
        batch_size=config.batch_size,
        segment_frames=config.segment_frames,
        segment_stride=config.segment_stride,
        criterion=criterion,
    )
    metrics_test = {
        "test": test_metrics,
        "best_val_f1": best_val_f1,
        "total_train_time_sec": time.time() - start_time,
    }
    (artifacts_dir / "metrics_test.json").write_text(
        json.dumps(metrics_test, indent=2),
        encoding="utf-8",
    )

    # Save one qualitative prediction figure for quick artifact inspection.
    try:
        example_pt = test_files[0]
        pred_pack = predict_segment_from_pt(
            model=model,
            pt_path=example_pt,
            start_frame=0,
            segment_frames=config.segment_frames,
            threshold=config.prediction_threshold,
            device=device,
        )
        save_prediction_figure(
            features=pred_pack["features"],
            pred_roll=pred_pack["pred"],
            target_roll=pred_pack["targets"],
            output_path=artifacts_dir / "prediction_preview.png",
            title=f"Prediction Preview: {example_pt.name}",
        )
        (artifacts_dir / "prediction_preview_meta.json").write_text(
            json.dumps(
                {
                    "pt_file": str(example_pt),
                    "start_frame": int(pred_pack["start_frame"]),
                    "end_frame": int(pred_pack["end_frame"]),
                    "threshold": config.prediction_threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
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
        "test_loss={loss:.6f} test_p={p:.4f} test_r={r:.4f} test_f1={f1:.4f}".format(
            loss=test_metrics["loss"],
            p=test_metrics["precision_micro"],
            r=test_metrics["recall_micro"],
            f1=test_metrics["f1_micro"],
        ),
        flush=True,
    )
    print(f"Artifacts written to: {artifacts_dir}", flush=True)
    return metrics_test
