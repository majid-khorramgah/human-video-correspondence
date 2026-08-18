"""
05_temporal_correspondence.py

PhySense-Human / Human Video Correspondence
Stage:
    02_temporal_redundancy / 05_temporal_correspondence

Purpose
-------
Analyze temporal correspondence between frames of the same video using
dense optical flow.

The analysis evaluates whether visual information can be spatially
corresponded from one frame to another as temporal distance increases.

The script:

1. Loads the master dataset index.
2. Supports all dataset shards automatically.
3. Uses train / val / test frames from master_index.csv.
4. Builds temporal frame pairs for configurable temporal distances.
5. Computes dense optical flow.
6. Computes forward-backward consistency.
7. Computes image warping error.
8. Computes human-region correspondence when a full-human mask exists.
9. Produces aggregate CSV / JSON results.
10. Produces video-level statistics.
11. Produces figures suitable for GitHub documentation.

Important
---------
This stage measures correspondence using optical-flow-based proxies.
It does NOT claim ground-truth correspondence accuracy.

Ground-truth motion vectors are not available in the dataset, so
forward-backward consistency and photometric warping error are used as
self-consistency indicators.

Default temporal distances:
    1, 2, 5, 10

Default sampling:
    per_video

Default maximum pairs:
    50 pairs per video per temporal distance

Example
-------
python 05_temporal_correspondence.py

More pairs:
-------
python 05_temporal_correspondence.py --max-pairs-video 100

Fewer pairs for a quick test:
-------
python 05_temporal_correspondence.py --max-pairs-video 10

Specific lags:
-------
python 05_temporal_correspondence.py --lags 1 2 5 10

Save detailed pair-level CSV:
-------
python 05_temporal_correspondence.py --save-pairs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_NAME = "05_temporal_correspondence.py"
STAGE_NAME = "02_temporal_redundancy / 05_temporal_correspondence"

DEFAULT_LAGS = [1, 2, 5, 10]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MAX_PAIRS_VIDEO = 50
DEFAULT_SAMPLING_MODE = "per_video"

DEFAULT_FB_THRESHOLD = 1.5

FLOW_PYRAMID_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_FLAGS = 0


# =============================================================================
# PATH HELPERS
# =============================================================================

def find_project_root() -> Path:
    """
    Locate project root based on this script location.

    Expected:
        project_root/
            02_temporal_redundancy/
                05_temporal_correspondence.py
    """
    script_path = Path(__file__).resolve()

    if script_path.parent.name == "02_temporal_redundancy":
        return script_path.parent.parent

    current = script_path.parent

    for candidate in [current] + list(current.parents):
        if (
            (candidate / "01_dataset_audit").exists()
            and (candidate / "02_temporal_redundancy").exists()
        ):
            return candidate

    return current


def build_paths(project_root: Path) -> Dict[str, Path]:
    result_root = (
        project_root
        / "02_temporal_redundancy"
        / "results"
        / "05_temporal_correspondence"
    )

    correspondence_root = result_root / "correspondence"
    figures_root = result_root / "figures"

    return {
        "project_root": project_root,
        "master_index": (
            project_root
            / "01_dataset_audit"
            / "results"
            / "integrity"
            / "master_index.csv"
        ),
        "result_root": result_root,
        "correspondence_root": correspondence_root,
        "figures_root": figures_root,
    }


def ensure_directories(paths: Dict[str, Path]) -> None:
    paths["correspondence_root"].mkdir(parents=True, exist_ok=True)
    paths["figures_root"].mkdir(parents=True, exist_ok=True)


# =============================================================================
# PRINTING
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# =============================================================================
# JSON UTILITIES
# =============================================================================

def json_safe(value):
    """
    Convert numpy / pandas scalar values into JSON-compatible Python values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal correspondence analysis using dense optical flow."
    )

    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=DEFAULT_LAGS,
        help="Temporal distances to analyze.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Analysis image size. Images are resized to square dimensions.",
    )

    parser.add_argument(
        "--max-pairs-video",
        type=int,
        default=DEFAULT_MAX_PAIRS_VIDEO,
        help="Maximum successful candidate pairs per video and lag.",
    )

    parser.add_argument(
        "--sampling-mode",
        choices=["per_video", "sequential"],
        default=DEFAULT_SAMPLING_MODE,
        help="Pair sampling strategy.",
    )

    parser.add_argument(
        "--fb-threshold",
        type=float,
        default=DEFAULT_FB_THRESHOLD,
        help="Forward-backward consistency threshold in analysis pixels.",
    )

    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Dataset splits to analyze.",
    )

    parser.add_argument(
        "--save-pairs",
        action="store_true",
        help="Save detailed pair-level correspondence CSV.",
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


# =============================================================================
# MASTER INDEX
# =============================================================================

def load_master_index(
    master_index_path: Path,
    selected_splits: List[str],
) -> pd.DataFrame:

    if not master_index_path.exists():
        raise FileNotFoundError(
            f"Master index not found:\n{master_index_path}"
        )

    print(f"Master index:\n  {master_index_path}")

    required_columns = [
        "shard",
        "split",
        "video_id",
        "frame_id",
        "hr_filename",
        "hr_extension",
        "has_Mask_Full_HR",
        "Mask_Full_HR_filename",
        "Mask_Full_HR_extension",
    ]

    header = pd.read_csv(
        master_index_path,
        nrows=0,
        encoding="utf-8",
    )

    available = set(header.columns)

    use_columns = [
        c for c in required_columns
        if c in available
    ]

    missing_required = {
        "shard",
        "split",
        "video_id",
        "frame_id",
        "hr_filename",
    } - set(use_columns)

    if missing_required:
        raise RuntimeError(
            "Master index is missing required columns:\n"
            + ", ".join(sorted(missing_required))
        )

    print("Reading required columns only...")

    df = pd.read_csv(
        master_index_path,
        usecols=use_columns,
        encoding="utf-8",
    )

    df = df[df["split"].isin(selected_splits)].copy()

    df["frame_id"] = pd.to_numeric(
        df["frame_id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["shard", "split", "video_id", "frame_id", "hr_filename"]
    ).copy()

    df["frame_id"] = df["frame_id"].astype(int)

    df["video_id"] = df["video_id"].astype(str)
    df["shard"] = df["shard"].astype(str)
    df["split"] = df["split"].astype(str)
    df["hr_filename"] = df["hr_filename"].astype(str)

    if "has_Mask_Full_HR" not in df.columns:
        df["has_Mask_Full_HR"] = False

    if "Mask_Full_HR_filename" not in df.columns:
        df["Mask_Full_HR_filename"] = ""

    df["has_Mask_Full_HR"] = (
        df["has_Mask_Full_HR"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df["Mask_Full_HR_filename"] = (
        df["Mask_Full_HR_filename"]
        .fillna("")
        .astype(str)
    )

    print(f"Rows loaded after split filtering: {len(df):,}")

    return df


def validate_master_index(df: pd.DataFrame) -> None:

    print_section("Validating master index")

    duplicate_count = int(
        df.duplicated(
            subset=["video_id", "frame_id"]
        ).sum()
    )

    print(
        f"Duplicate (video_id, frame_id) records: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate temporal identities detected in master index."
        )

    print(f"Videos: {df['video_id'].nunique():,}")
    print(f"Shards: {df['shard'].nunique():,}")

    print("Splits:")

    for split, count in (
        df.groupby("split")
        .size()
        .sort_index()
        .items()
    ):
        print(f"  {split}: {count:,} frames")

    print("Shards:")

    for shard, count in (
        df.groupby("shard")
        .size()
        .sort_index()
        .items()
    ):
        print(f"  {shard}: {count:,} frames")


# =============================================================================
# DATASET PATH DISCOVERY
# =============================================================================

def find_case_insensitive_child(
    directory: Path,
    target_name: str,
) -> Optional[Path]:

    if not directory.exists():
        return None

    target_lower = target_name.lower()

    try:
        for child in directory.iterdir():
            if child.name.lower() == target_lower:
                return child
    except OSError:
        return None

    return None


def find_modality_directory(
    shard_dir: Path,
    split: str,
    modality: str,
) -> Optional[Path]:
    """
    Find a modality directory under:

        shard/split/

    Uses case-insensitive matching and a few common naming variants.
    """

    split_dir = find_case_insensitive_child(
        shard_dir,
        split,
    )

    if split_dir is None:
        return None

    candidates = [
        modality,
        modality.replace("_", ""),
        modality.lower(),
    ]

    for candidate in candidates:
        result = find_case_insensitive_child(
            split_dir,
            candidate,
        )

        if result is not None and result.is_dir():
            return result

    # Exact normalized fallback
    normalized_target = (
        modality.lower()
        .replace("_", "")
        .replace("-", "")
    )

    try:
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue

            normalized = (
                child.name.lower()
                .replace("_", "")
                .replace("-", "")
            )

            if normalized == normalized_target:
                return child
    except OSError:
        pass

    return None


def resolve_frame_path(
    project_root: Path,
    row: pd.Series,
) -> Optional[Path]:

    shard_dir = project_root / "Dataset" / str(row["shard"])

    if not shard_dir.exists():
        return None

    split = str(row["split"])
    filename = str(row["hr_filename"])

    hr_dir = find_modality_directory(
        shard_dir,
        split,
        "Img_HR",
    )

    if hr_dir is not None:
        candidate = hr_dir / filename

        if candidate.exists():
            return candidate

    # Fallback: recursively search only under split directory.
    split_dir = find_case_insensitive_child(
        shard_dir,
        split,
    )

    if split_dir is None:
        return None

    try:
        matches = list(split_dir.rglob(filename))

        if matches:
            return matches[0]
    except OSError:
        pass

    return None


def resolve_mask_path(
    project_root: Path,
    row: pd.Series,
) -> Optional[Path]:

    if not bool(row.get("has_Mask_Full_HR", False)):
        return None

    mask_filename = str(
        row.get("Mask_Full_HR_filename", "")
    )

    if not mask_filename or mask_filename.lower() == "nan":
        return None

    shard_dir = project_root / "Dataset" / str(row["shard"])

    if not shard_dir.exists():
        return None

    mask_dir = find_modality_directory(
        shard_dir,
        str(row["split"]),
        "Mask_Full_HR",
    )

    if mask_dir is not None:
        candidate = mask_dir / mask_filename

        if candidate.exists():
            return candidate

    split_dir = find_case_insensitive_child(
        shard_dir,
        str(row["split"]),
    )

    if split_dir is None:
        return None

    try:
        matches = list(split_dir.rglob(mask_filename))

        if matches:
            return matches[0]
    except OSError:
        pass

    return None


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(
    path: Path,
    image_size: int,
) -> Optional[np.ndarray]:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    return image


def load_mask(
    path: Optional[Path],
    image_size: int,
) -> Optional[np.ndarray]:

    if path is None:
        return None

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        return None

    mask = cv2.resize(
        mask,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask > 0


# =============================================================================
# OPTICAL FLOW
# =============================================================================

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )


def compute_dense_flow(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:

    source_gray = rgb_to_gray(source)
    target_gray = rgb_to_gray(target)

    flow = cv2.calcOpticalFlowFarneback(
        source_gray,
        target_gray,
        None,
        FLOW_PYRAMID_SCALE,
        FLOW_LEVELS,
        FLOW_WINSIZE,
        FLOW_ITERATIONS,
        FLOW_POLY_N,
        FLOW_POLY_SIGMA,
        FLOW_FLAGS,
    )

    return flow.astype(np.float32)


# =============================================================================
# FLOW SAMPLING
# =============================================================================

def build_pixel_grid(
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    return x, y


def remap_flow(
    flow: np.ndarray,
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
) -> np.ndarray:

    fx = cv2.remap(
        flow[..., 0],
        coordinates_x,
        coordinates_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    fy = cv2.remap(
        flow[..., 1],
        coordinates_x,
        coordinates_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return np.stack([fx, fy], axis=-1)


def compute_forward_backward_error(
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    height, width = forward_flow.shape[:2]

    grid_x, grid_y = build_pixel_grid(
        height,
        width,
    )

    destination_x = grid_x + forward_flow[..., 0]
    destination_y = grid_y + forward_flow[..., 1]

    valid_destination = (
        (destination_x >= 0)
        & (destination_x <= width - 1)
        & (destination_y >= 0)
        & (destination_y <= height - 1)
    )

    sampled_backward = remap_flow(
        backward_flow,
        destination_x,
        destination_y,
    )

    consistency_vector = (
        forward_flow + sampled_backward
    )

    fb_error = np.linalg.norm(
        consistency_vector,
        axis=-1,
    )

    fb_error[~valid_destination] = np.nan

    return fb_error, valid_destination


# =============================================================================
# WARPING
# =============================================================================

def warp_source_with_flow(
    source: np.ndarray,
    flow: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    height, width = source.shape[:2]

    grid_x, grid_y = build_pixel_grid(
        height,
        width,
    )

    # Backward sampling approximation:
    # destination pixel samples from source using negative flow.
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]

    valid = (
        (map_x >= 0)
        & (map_x <= width - 1)
        & (map_y >= 0)
        & (map_y <= height - 1)
    )

    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return warped, valid


# =============================================================================
# METRICS
# =============================================================================

def safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.median(values))


def safe_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:

    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.percentile(values, percentile))


def calculate_pair_metrics(
    source: np.ndarray,
    target: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    human_mask: Optional[np.ndarray],
    fb_threshold: float,
) -> Dict[str, float]:

    fb_error, valid_destination = (
        compute_forward_backward_error(
            forward_flow,
            backward_flow,
        )
    )

    flow_magnitude = np.linalg.norm(
        forward_flow,
        axis=-1,
    )

    valid_fb = (
        np.isfinite(fb_error)
        & valid_destination
    )

    consistent = (
        valid_fb
        & (fb_error <= fb_threshold)
    )

    warped_source, warp_valid = (
        warp_source_with_flow(
            source,
            forward_flow,
        )
    )

    target_float = target.astype(np.float32)
    warped_float = warped_source.astype(np.float32)

    pixel_error = np.mean(
        np.abs(
            warped_float - target_float
        ),
        axis=-1,
    )

    valid_warp = warp_valid

    metrics = {
        "mean_flow_magnitude": safe_mean(
            flow_magnitude
        ),
        "median_flow_magnitude": safe_median(
            flow_magnitude
        ),
        "p90_flow_magnitude": safe_percentile(
            flow_magnitude,
            90,
        ),
        "mean_fb_error": safe_mean(
            fb_error
        ),
        "median_fb_error": safe_median(
            fb_error
        ),
        "p90_fb_error": safe_percentile(
            fb_error,
            90,
        ),
        "fb_consistency_ratio": (
            float(np.mean(consistent[valid_fb]))
            if np.any(valid_fb)
            else float("nan")
        ),
        "valid_flow_ratio": (
            float(np.mean(valid_fb))
            if valid_fb.size
            else float("nan")
        ),
        "mean_warp_error": safe_mean(
            pixel_error[valid_warp]
        ),
        "median_warp_error": safe_median(
            pixel_error[valid_warp]
        ),
        "warp_valid_ratio": (
            float(np.mean(valid_warp))
            if valid_warp.size
            else float("nan")
        ),
    }

    # -------------------------------------------------------------------------
    # Human-region metrics
    # -------------------------------------------------------------------------

    if human_mask is not None:
        mask = human_mask.astype(bool)

        human_valid = (
            mask
            & valid_fb
        )

        human_consistent = (
            human_valid
            & (fb_error <= fb_threshold)
        )

        human_flow = flow_magnitude[mask]
        human_fb = fb_error[human_valid]
        human_warp = pixel_error[
            mask & valid_warp
        ]

        metrics.update(
            {
                "human_mask_coverage": float(
                    np.mean(mask)
                ),
                "human_mean_flow_magnitude": safe_mean(
                    human_flow
                ),
                "human_median_flow_magnitude": safe_median(
                    human_flow
                ),
                "human_p90_flow_magnitude": safe_percentile(
                    human_flow,
                    90,
                ),
                "human_mean_fb_error": safe_mean(
                    human_fb
                ),
                "human_fb_consistency_ratio": (
                    float(
                        np.mean(
                            human_consistent[
                                human_valid
                            ]
                        )
                    )
                    if np.any(human_valid)
                    else float("nan")
                ),
                "human_mean_warp_error": safe_mean(
                    human_warp
                ),
            }
        )

    else:
        metrics.update(
            {
                "human_mask_coverage": float("nan"),
                "human_mean_flow_magnitude": float("nan"),
                "human_median_flow_magnitude": float("nan"),
                "human_p90_flow_magnitude": float("nan"),
                "human_mean_fb_error": float("nan"),
                "human_fb_consistency_ratio": float("nan"),
                "human_mean_warp_error": float("nan"),
            }
        )

    return metrics


# =============================================================================
# TEMPORAL PAIR SAMPLING
# =============================================================================

def build_video_groups(
    df: pd.DataFrame,
) -> Dict[Tuple[str, str, str], pd.DataFrame]:

    groups = {}

    for key, group in df.groupby(
        ["shard", "split", "video_id"],
        sort=False,
    ):
        group = group.sort_values(
            "frame_id"
        ).reset_index(drop=True)

        groups[key] = group

    return groups


def sample_pairs_for_video(
    group: pd.DataFrame,
    lag: int,
    max_pairs: int,
    mode: str,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:

    frame_ids = group["frame_id"].to_numpy(
        dtype=np.int64
    )

    frame_to_position = {
        int(frame_id): index
        for index, frame_id in enumerate(frame_ids)
    }

    candidates = []

    for index, frame_id in enumerate(frame_ids):
        target_frame_id = int(frame_id) + lag

        if target_frame_id in frame_to_position:
            target_index = frame_to_position[
                target_frame_id
            ]

            candidates.append(
                (index, target_index)
            )

    if not candidates:
        return []

    if mode == "sequential":
        return candidates[:max_pairs]

    if len(candidates) <= max_pairs:
        return candidates

    selected_indices = rng.choice(
        len(candidates),
        size=max_pairs,
        replace=False,
    )

    selected_indices = sorted(
        selected_indices.tolist()
    )

    return [
        candidates[i]
        for i in selected_indices
    ]


# =============================================================================
# FIGURES
# =============================================================================

def save_empty_figure(
    path: Path,
    title: str,
    message: str,
) -> None:

    plt.figure(
        figsize=(9, 5)
    )

    plt.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def plot_metric_by_lag(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:

    if summary.empty:
        save_empty_figure(
            output_path,
            title,
            "No valid data available.",
        )
        return

    plt.figure(
        figsize=(9, 6)
    )

    for split in sorted(
        summary["split"].dropna().unique()
    ):

        subset = summary[
            summary["split"] == split
        ].sort_values(
            "temporal_distance"
        )

        if subset.empty:
            continue

        plt.plot(
            subset["temporal_distance"],
            subset[metric],
            marker="o",
            label=split,
        )

    plt.xlabel("Temporal distance (frames)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_distribution(
    pair_df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:

    if pair_df.empty or metric not in pair_df.columns:
        save_empty_figure(
            output_path,
            title,
            "No valid pair-level data available.",
        )
        return

    plt.figure(
        figsize=(9, 6)
    )

    for lag in sorted(
        pair_df["temporal_distance"]
        .dropna()
        .unique()
    ):

        values = pair_df.loc[
            pair_df["temporal_distance"] == lag,
            metric,
        ].dropna()

        if values.empty:
            continue

        plt.hist(
            values,
            bins=40,
            alpha=0.45,
            label=f"lag={int(lag)}",
        )

    plt.xlabel(xlabel)
    plt.ylabel("Pair count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_video_distribution(
    video_df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:

    if video_df.empty or metric not in video_df.columns:
        save_empty_figure(
            output_path,
            title,
            "No video-level data available.",
        )
        return

    values = video_df[metric].dropna()

    if values.empty:
        save_empty_figure(
            output_path,
            title,
            "No valid video-level data available.",
        )
        return

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        values,
        bins=35,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Video count")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def generate_figures(
    summary_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    video_df: pd.DataFrame,
    figures_root: Path,
) -> None:

    print_section("Generating figures")

    plot_metric_by_lag(
        summary_df,
        "mean_flow_magnitude",
        "Mean flow magnitude (pixels)",
        "Mean Motion Magnitude by Temporal Distance",
        figures_root / "correspondence_motion_by_temporal_distance.png",
    )

    plot_metric_by_lag(
        summary_df,
        "fb_consistency_ratio",
        "Forward-backward consistency ratio",
        "Temporal Correspondence Consistency",
        figures_root / "forward_backward_consistency.png",
    )

    plot_metric_by_lag(
        summary_df,
        "mean_warp_error",
        "Mean warping error",
        "Warping Error by Temporal Distance",
        figures_root / "warping_error_by_temporal_distance.png",
    )

    plot_metric_by_lag(
        summary_df,
        "human_fb_consistency_ratio",
        "Human-region consistency ratio",
        "Human-region Correspondence Consistency",
        figures_root / "human_correspondence_consistency.png",
    )

    plot_distribution(
        pair_df,
        "fb_consistency_ratio",
        "Pair-level Correspondence Consistency Distribution",
        "Forward-backward consistency ratio",
        figures_root / "correspondence_consistency_distribution.png",
    )

    plot_distribution(
        pair_df,
        "mean_flow_magnitude",
        "Pair-level Flow Magnitude Distribution",
        "Mean flow magnitude (pixels)",
        figures_root / "flow_magnitude_distribution.png",
    )

    plot_video_distribution(
        video_df,
        "mean_flow_magnitude",
        "Video-level Motion Magnitude Distribution",
        "Mean flow magnitude (pixels)",
        figures_root / "video_motion_distribution.png",
    )

    print(f"Figures saved to:\n  {figures_root}")


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_summary(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:

    if pair_df.empty:
        return pd.DataFrame()

    metric_columns = [
        "mean_flow_magnitude",
        "median_flow_magnitude",
        "p90_flow_magnitude",
        "mean_fb_error",
        "median_fb_error",
        "p90_fb_error",
        "fb_consistency_ratio",
        "valid_flow_ratio",
        "mean_warp_error",
        "median_warp_error",
        "warp_valid_ratio",
        "human_mask_coverage",
        "human_mean_flow_magnitude",
        "human_median_flow_magnitude",
        "human_p90_flow_magnitude",
        "human_mean_fb_error",
        "human_fb_consistency_ratio",
        "human_mean_warp_error",
    ]

    available_metrics = [
        c
        for c in metric_columns
        if c in pair_df.columns
    ]

    grouped = pair_df.groupby(
        ["temporal_distance", "split"],
        dropna=False,
    )

    summary = grouped[
        available_metrics
    ].mean().reset_index()

    counts = grouped.size().reset_index(
        name="pairs"
    )

    summary = summary.merge(
        counts,
        on=["temporal_distance", "split"],
        how="left",
    )

    videos = grouped["video_id"].nunique().reset_index(
        name="videos"
    )

    summary = summary.merge(
        videos,
        on=["temporal_distance", "split"],
        how="left",
    )

    summary = summary.sort_values(
        ["temporal_distance", "split"]
    )

    return summary


def aggregate_video_statistics(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:

    if pair_df.empty:
        return pd.DataFrame()

    metrics = [
        "mean_flow_magnitude",
        "median_flow_magnitude",
        "p90_flow_magnitude",
        "mean_fb_error",
        "fb_consistency_ratio",
        "mean_warp_error",
        "human_mask_coverage",
        "human_mean_flow_magnitude",
        "human_mean_fb_error",
        "human_fb_consistency_ratio",
        "human_mean_warp_error",
    ]

    available = [
        c for c in metrics
        if c in pair_df.columns
    ]

    grouped = pair_df.groupby(
        ["shard", "split", "video_id"],
        dropna=False,
    )

    result = grouped[
        available
    ].mean().reset_index()

    pair_counts = grouped.size().reset_index(
        name="pairs"
    )

    result = result.merge(
        pair_counts,
        on=["shard", "split", "video_id"],
        how="left",
    )

    return result.sort_values(
        ["split", "video_id"]
    )


# =============================================================================
# PAIR ANALYSIS
# =============================================================================

def analyze_pair(
    project_root: Path,
    row_a: pd.Series,
    row_b: pd.Series,
    lag: int,
    image_size: int,
    fb_threshold: float,
) -> Tuple[Optional[Dict], Optional[str]]:

    source_path = resolve_frame_path(
        project_root,
        row_a,
    )

    target_path = resolve_frame_path(
        project_root,
        row_b,
    )

    if source_path is None:
        return None, (
            f"Source HR image not found: "
            f"{row_a['hr_filename']}"
        )

    if target_path is None:
        return None, (
            f"Target HR image not found: "
            f"{row_b['hr_filename']}"
        )

    source = load_image(
        source_path,
        image_size,
    )

    target = load_image(
        target_path,
        image_size,
    )

    if source is None:
        return None, (
            f"Unable to read source image: "
            f"{source_path}"
        )

    if target is None:
        return None, (
            f"Unable to read target image: "
            f"{target_path}"
        )

    mask_path = resolve_mask_path(
        project_root,
        row_a,
    )

    human_mask = load_mask(
        mask_path,
        image_size,
    )

    try:
        forward_flow = compute_dense_flow(
            source,
            target,
        )

        backward_flow = compute_dense_flow(
            target,
            source,
        )

        metrics = calculate_pair_metrics(
            source,
            target,
            forward_flow,
            backward_flow,
            human_mask,
            fb_threshold,
        )

    except Exception as exc:
        return None, (
            f"Flow calculation failed: {repr(exc)}"
        )

    result = {
        "shard": str(row_a["shard"]),
        "split": str(row_a["split"]),
        "video_id": str(row_a["video_id"]),
        "frame_id_source": int(row_a["frame_id"]),
        "frame_id_target": int(row_b["frame_id"]),
        "temporal_distance": int(lag),
        "source_filename": str(row_a["hr_filename"]),
        "target_filename": str(row_b["hr_filename"]),
        "human_mask_available": bool(
            human_mask is not None
        ),
    }

    result.update(metrics)

    return result, None


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(
    project_root: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], Dict]:

    rng = np.random.default_rng(
        args.seed
    )

    groups = build_video_groups(
        df
    )

    group_items = list(groups.items())

    if args.limit_videos is not None:
        group_items = group_items[
            :args.limit_videos
        ]

    print(f"Videos available: {len(group_items):,}")

    all_results: List[Dict] = []
    errors: List[Dict] = []

    candidate_pairs = 0
    successful_pairs = 0

    videos_processed = 0

    start_time = time.time()

    for key, group in group_items:

        shard, split, video_id = key

        video_success = 0

        for lag in args.lags:

            sampled_pairs = sample_pairs_for_video(
                group,
                lag,
                args.max_pairs_video,
                args.sampling_mode,
                rng,
            )

            candidate_pairs += len(
                sampled_pairs
            )

            for source_pos, target_pos in sampled_pairs:

                row_a = group.iloc[source_pos]
                row_b = group.iloc[target_pos]

                result, error = analyze_pair(
                    project_root,
                    row_a,
                    row_b,
                    lag,
                    args.image_size,
                    args.fb_threshold,
                )

                if result is not None:
                    all_results.append(
                        result
                    )

                    successful_pairs += 1
                    video_success += 1

                else:
                    errors.append(
                        {
                            "shard": shard,
                            "split": split,
                            "video_id": video_id,
                            "frame_id_source": int(
                                row_a["frame_id"]
                            ),
                            "frame_id_target": int(
                                row_b["frame_id"]
                            ),
                            "temporal_distance": lag,
                            "error": error,
                        }
                    )

        videos_processed += 1

        if (
            videos_processed == 1
            or videos_processed % 10 == 0
            or videos_processed == len(group_items)
        ):
            elapsed = time.time() - start_time

            rate = (
                videos_processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Videos processed: "
                f"{videos_processed:,} / "
                f"{len(group_items):,} | "
                f"Pairs successful: "
                f"{successful_pairs:,} | "
                f"Rate: "
                f"{rate:.2f} videos/s"
            )

    pair_df = pd.DataFrame(
        all_results
    )

    video_df = aggregate_video_statistics(
        pair_df
    )

    summary_df = aggregate_summary(
        pair_df
    )

    elapsed = time.time() - start_time

    run_metadata = {
        "script": SCRIPT_NAME,
        "stage": STAGE_NAME,
        "lags": args.lags,
        "image_size": args.image_size,
        "sampling_mode": args.sampling_mode,
        "max_pairs_per_video": args.max_pairs_video,
        "fb_threshold": args.fb_threshold,
        "selected_splits": args.split,
        "seed": args.seed,
        "videos_available": len(group_items),
        "videos_processed": videos_processed,
        "candidate_pairs": candidate_pairs,
        "successful_pairs": successful_pairs,
        "failed_pairs": len(errors),
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60.0,
    }

    return (
        pair_df,
        summary_df,
        errors,
        run_metadata,
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    paths: Dict[str, Path],
    pair_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    video_df: pd.DataFrame,
    errors: List[Dict],
    metadata: Dict,
    save_pairs: bool,
) -> None:

    print_section("Saving correspondence results")

    output_root = paths["correspondence_root"]

    summary_csv = (
        output_root
        / "correspondence_summary.csv"
    )

    video_csv = (
        output_root
        / "video_correspondence_statistics.csv"
    )

    error_csv = (
        output_root
        / "correspondence_errors.csv"
    )

    summary_json = (
        output_root
        / "correspondence_summary.json"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8",
    )

    video_df.to_csv(
        video_csv,
        index=False,
        encoding="utf-8",
    )

    pd.DataFrame(errors).to_csv(
        error_csv,
        index=False,
        encoding="utf-8",
    )

    if save_pairs:
        pair_csv = (
            output_root
            / "frame_pair_correspondence_statistics.csv"
        )

        pair_df.to_csv(
            pair_csv,
            index=False,
            encoding="utf-8",
        )

        print(f"Saved detailed pairs:\n  {pair_csv}")

    else:
        print(
            "Detailed pair CSV not saved "
            "(use --save-pairs if needed)."
        )

    metadata = json_safe(metadata)

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved:\n  {summary_csv}")
    print(f"Saved:\n  {video_csv}")
    print(f"Saved:\n  {error_csv}")
    print(f"Saved:\n  {summary_json}")


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(
    df: pd.DataFrame,
    pair_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    errors: List[Dict],
    metadata: Dict,
    paths: Dict[str, Path],
) -> None:

    print()
    print("=" * 78)
    print("FINAL TEMPORAL CORRESPONDENCE REPORT")
    print("=" * 78)

    print(
        f"Frames available: {len(df):,}"
    )

    print(
        f"Videos: {df['video_id'].nunique():,}"
    )

    print(
        f"Shards: {df['shard'].nunique():,}"
    )

    print(
        f"Temporal distances: "
        f"{metadata['lags']}"
    )

    print(
        f"Sampling mode: "
        f"{metadata['sampling_mode']}"
    )

    print(
        f"Candidate pairs: "
        f"{metadata['candidate_pairs']:,}"
    )

    print(
        f"Successful comparisons: "
        f"{metadata['successful_pairs']:,}"
    )

    print(
        f"Failed comparisons: "
        f"{metadata['failed_pairs']:,}"
    )

    if not summary_df.empty:

        print()
        print("Correspondence summary:")

        display_columns = [
            "temporal_distance",
            "split",
            "pairs",
            "videos",
            "mean_flow_magnitude",
            "mean_fb_error",
            "fb_consistency_ratio",
            "mean_warp_error",
            "human_fb_consistency_ratio",
        ]

        display_columns = [
            c
            for c in display_columns
            if c in summary_df.columns
        ]

        display_df = summary_df[
            display_columns
        ].copy()

        print(
            display_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print()

    if metadata["successful_pairs"] > 0:
        print(
            "CORE TEMPORAL CORRESPONDENCE ANALYSIS: COMPLETE"
        )
    else:
        print(
            "CORE TEMPORAL CORRESPONDENCE ANALYSIS: FAILED"
        )

    print(
        f"\nRuntime: "
        f"{metadata['runtime_minutes']:.2f} minutes"
    )

    print()
    print("Results:")
    print(
        f"  {paths['correspondence_root']}"
    )
    print(
        f"  {paths['figures_root']}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    total_start = time.time()

    args = parse_args()

    print_header(
        "TEMPORAL CORRESPONDENCE ANALYSIS"
    )

    print(
        "PhySense-Human Research Pipeline"
    )

    print(
        f"Stage: {STAGE_NAME}"
    )

    print()

    project_root = find_project_root()

    print(
        f"Project root:\n  {project_root}"
    )

    paths = build_paths(
        project_root
    )

    ensure_directories(
        paths
    )

    print_section(
        "Configuration"
    )

    print(
        f"Temporal distances: {args.lags}"
    )

    print(
        f"Image size: "
        f"{args.image_size} x {args.image_size}"
    )

    print(
        f"Sampling mode: "
        f"{args.sampling_mode}"
    )

    print(
        f"Maximum pairs/video/lag: "
        f"{args.max_pairs_video}"
    )

    print(
        f"Forward-backward threshold: "
        f"{args.fb_threshold} pixels"
    )

    print_section(
        "Loading master dataset index"
    )

    df = load_master_index(
        paths["master_index"],
        args.split,
    )

    validate_master_index(
        df
    )

    print_section(
        "Computing temporal correspondence"
    )

    print(
        "Correspondence method:"
    )

    print(
        "  Dense optical flow"
    )

    print(
        "  Forward-backward consistency"
    )

    print(
        "  Photometric warping error"
    )

    print(
        "  Human-region correspondence"
    )

    try:

        (
            pair_df,
            summary_df,
            errors,
            metadata,
        ) = run_analysis(
            project_root,
            df,
            args,
        )

    except KeyboardInterrupt:

        print()
        print(
            "Analysis interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print(
            "FATAL ERROR:"
        )

        print(
            repr(exc)
        )

        traceback.print_exc()

        sys.exit(1)

    print_section(
        "Aggregating correspondence statistics"
    )

    video_df = aggregate_video_statistics(
        pair_df
    )

    print_section(
        "Saving CSV / JSON results"
    )

    save_results(
        paths,
        pair_df,
        summary_df,
        video_df,
        errors,
        metadata,
        args.save_pairs,
    )

    generate_figures(
        summary_df,
        pair_df,
        video_df,
        paths["figures_root"],
    )

    total_runtime = (
        time.time() - total_start
    )

    metadata["total_runtime_seconds"] = (
        total_runtime
    )

    metadata["total_runtime_minutes"] = (
        total_runtime / 60.0
    )

    # Update JSON with total runtime.
    summary_json = (
        paths["correspondence_root"]
        / "correspondence_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(metadata),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print_final_report(
        df,
        pair_df,
        summary_df,
        errors,
        metadata,
        paths,
    )


if __name__ == "__main__":
    main()