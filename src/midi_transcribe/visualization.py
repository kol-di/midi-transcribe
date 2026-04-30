from pathlib import Path

import torch


def save_prediction_figure(
    features: torch.Tensor,
    pred_roll: torch.Tensor,
    target_roll: torch.Tensor,
    output_path: Path,
    title: str = "Prediction vs Ground Truth",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for visualization") from exc

    feat_np = features.detach().cpu().numpy().T  # [128,T]
    pred_np = pred_roll.detach().cpu().numpy().T  # [88,T]
    targ_np = target_roll.detach().cpu().numpy().T  # [88,T]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    fig.suptitle(title)

    im0 = axes[0].imshow(feat_np, origin="lower", aspect="auto", interpolation="nearest")
    axes[0].set_ylabel("Mel Bin")
    axes[0].set_title("Input Mel Spectrogram")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)

    axes[1].imshow(pred_np, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
    axes[1].set_ylabel("Piano Key")
    axes[1].set_title("Predicted Active Notes")

    axes[2].imshow(targ_np, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
    axes[2].set_ylabel("Piano Key")
    axes[2].set_xlabel("Frame")
    axes[2].set_title("Ground Truth Active Notes")

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
