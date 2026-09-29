"""Input preparation for SkateFormer, ported from the feeders of the original repository.

A SkateFormer checkpoint does **not** consume raw skeletons: the joints are reordered into
skeletal partitions (right arm / left arm / right leg / left leg / torso), the clip is
resampled to a fixed window, and the sampled frame timestamps are handed to the model as
``index_t``. This module reproduces the *evaluation-time* path of
``feeders/feeder_ntu.py``, ``feeders/feeder_ntu_inter.py`` and ``feeders/feeder_ucla.py``
(no random augmentation).

Reference: `feeders/tools.py::valid_crop_resize`, `valid_crop_uniform`, `joint2bone`.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Joint partitions (1-indexed in the original code, converted to 0-indexed here)
# ---------------------------------------------------------------------------

# NTU RGB+D 25-joint skeleton -> 24 partitioned joints (joint 21 / "spine" is dropped).
NTU_PARTITION_INDEX = np.concatenate([
    np.array([7, 8, 22, 23]) - 1,    # right arm
    np.array([11, 12, 24, 25]) - 1,  # left arm
    np.array([13, 14, 15, 16]) - 1,  # right leg
    np.array([17, 18, 19, 20]) - 1,  # left leg
    np.array([5, 9, 6, 10]) - 1,     # height-wise torso
    np.array([2, 3, 1, 4]) - 1,      # width-wise torso
])

# NW-UCLA 20-joint skeleton -> 20 partitioned joints.
NW_UCLA_PARTITION_INDEX = np.concatenate([
    np.array([5, 6, 7, 8]) - 1,      # right arm
    np.array([9, 10, 11, 12]) - 1,   # left arm
    np.array([13, 14, 15, 16]) - 1,  # right leg
    np.array([17, 18, 19, 20]) - 1,  # left leg
    np.array([2, 3, 1, 4]) - 1,      # torso
])

PARTITION_INDEX = {"ntu": NTU_PARTITION_INDEX, "nw_ucla": NW_UCLA_PARTITION_INDEX}

# ---------------------------------------------------------------------------
# Bone edges, as (child, parent) 0-indexed joint pairs
# ---------------------------------------------------------------------------

NTU_BONE_PAIRS = [
    (0, 1), (1, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
    (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
    (19, 18), (20, 1), (21, 7), (22, 7), (23, 11), (24, 11),
]

NW_UCLA_BONE_PAIRS = [
    (v1 - 1, v2 - 1) for v1, v2 in
    [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10),
     (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
]

BONE_PAIRS = {"ntu": NTU_BONE_PAIRS, "nw_ucla": NW_UCLA_BONE_PAIRS}


def joint_to_bone(data: np.ndarray, layout: str = "ntu") -> np.ndarray:
    """Convert joint coordinates to bone vectors. ``data`` is ``[C, T, V, M]``."""
    bone = np.zeros_like(data)
    for v1, v2 in BONE_PAIRS[layout]:
        bone[:, :, v1, :] = data[:, :, v1, :] - data[:, :, v2, :]
    return bone


def to_motion(data: np.ndarray) -> np.ndarray:
    """Temporal difference (motion) modality. ``data`` is ``[C, T, V, M]``."""
    motion = np.zeros_like(data)
    motion[:, :-1] = data[:, 1:] - data[:, :-1]
    return motion


def partition_joints(data: np.ndarray, layout: str = "ntu") -> np.ndarray:
    """Reorder joints into skeletal partitions. ``data`` is ``[C, T, V, M]``."""
    return data[:, :, PARTITION_INDEX[layout]]


def crop_resize(
    data: np.ndarray,
    valid_frame_num: int,
    window: int = 64,
    p: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic centre-crop + linear resize to ``window`` frames.

    Mirrors ``tools.valid_crop_resize`` with a single-element ``p_interval`` (the setting
    every `config/test/*.yaml` uses).

    Args:
        data: ``[C, T, V, M]`` skeleton sequence.
        valid_frame_num: number of non-padded frames at the start of ``data``.
        window: target clip length (64 for all released checkpoints).
        p: fraction of the valid sequence to keep.

    Returns:
        ``([C, window, V, M], [window])`` — the resampled clip and its ``index_t``
        timestamps normalised to ``[-1, 1]``.
    """
    C, T, V, M = data.shape
    valid_size = valid_frame_num
    bias = int((1 - p) * valid_size / 2)
    c_b, c_e = bias, valid_size - bias
    cropped = data[:, c_b:c_e]

    x = torch.tensor(cropped, dtype=torch.float)
    x = x.permute(2, 3, 0, 1).contiguous().view(V * M, C, cropped.shape[1])
    x = F.interpolate(x, size=window, mode="linear", align_corners=False)
    x = x.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()

    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float)
    index_t = F.interpolate(index_t[None, None, :], size=window, mode="linear", align_corners=False).squeeze()
    index_t = 2 * index_t / valid_size - 1
    return x, index_t.numpy()


def prepare_input(
    data: np.ndarray,
    valid_frame_num: int = None,
    layout: str = "ntu",
    modality: str = "j",
    window: int = 64,
    p: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end evaluation preprocessing for a single sequence.

    Args:
        data: raw skeleton of shape ``[C, T, V, M]`` — 3 channels, ``T`` frames,
            25 joints (``layout="ntu"``) or 20 joints (``layout="nw_ucla"``),
            2 people (NTU) or 1 (NW-UCLA).
        valid_frame_num: non-padded frame count; defaults to ``T``.
        layout: ``"ntu"`` or ``"nw_ucla"``.
        modality: ``"j"`` (joint), ``"b"`` (bone), ``"jm"`` or ``"bm"`` (motion variants).
            Must match the checkpoint you loaded.
        window: clip length, 64 for all released checkpoints.
        p: fraction of the valid sequence to keep.

    Returns:
        ``(input, index_t)`` batched with a leading dimension of 1, ready for
        ``model(input, index_t)``.
    """
    if valid_frame_num is None:
        valid_frame_num = data.shape[1]

    clip, index_t = crop_resize(data, valid_frame_num, window=window, p=p)

    if modality in ("b", "bm"):
        clip = joint_to_bone(clip, layout)
    if modality in ("jm", "bm"):
        clip = to_motion(clip)

    clip = partition_joints(clip, layout)

    return (
        torch.from_numpy(np.ascontiguousarray(clip)).float().unsqueeze(0),
        torch.from_numpy(np.ascontiguousarray(index_t)).float().unsqueeze(0),
    )
