from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    data_root: Path
    run_dir: Path
    epochs: int
    batch_size: int
    segment_frames: int
    segment_stride: int
    hidden_dim: int
    lr: float
    weight_decay: float
    pos_weight: float
    prediction_threshold: float
    seed: int
