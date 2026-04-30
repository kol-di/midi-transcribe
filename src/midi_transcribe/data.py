import random
from pathlib import Path
from typing import Iterable, List, Tuple

import torch


def list_pt_files(data_root: Path, split: str) -> List[Path]:
    split_dir = data_root / split
    return sorted(split_dir.rglob("*.pt"))


def segment_starts(n_frames: int, segment_frames: int, segment_stride: int) -> List[int]:
    if n_frames <= 0:
        return []
    starts = list(range(0, max(n_frames - segment_frames + 1, 1), segment_stride))
    last_start = max(n_frames - segment_frames, 0)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def load_piece(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    features = obj["features"].to(dtype=torch.float32).transpose(0, 1).contiguous()  # [T, 128]
    targets = obj["targets"].to(dtype=torch.float32).contiguous()  # [T, 88]
    return features, targets


def iterate_batches(
    files: List[Path],
    batch_size: int,
    segment_frames: int,
    segment_stride: int,
    shuffle_files: bool,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    file_order = files[:]
    if shuffle_files:
        random.shuffle(file_order)

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []

    for path in file_order:
        features, targets = load_piece(path)
        total_frames = features.shape[0]
        starts = segment_starts(total_frames, segment_frames, segment_stride)

        for start in starts:
            end = min(start + segment_frames, total_frames)
            x = features[start:end]
            y = targets[start:end]

            valid_len = end - start
            if valid_len < segment_frames:
                pad = segment_frames - valid_len
                x = torch.cat([x, torch.zeros((pad, x.shape[1]), dtype=x.dtype)], dim=0)
                y = torch.cat([y, torch.zeros((pad, y.shape[1]), dtype=y.dtype)], dim=0)

            mask = torch.zeros(segment_frames, dtype=torch.float32)
            mask[:valid_len] = 1.0

            xs.append(x)
            ys.append(y)
            ms.append(mask)

            if len(xs) == batch_size:
                yield torch.stack(xs, dim=0), torch.stack(ys, dim=0), torch.stack(ms, dim=0)
                xs.clear()
                ys.clear()
                ms.clear()

    if xs:
        yield torch.stack(xs, dim=0), torch.stack(ys, dim=0), torch.stack(ms, dim=0)
