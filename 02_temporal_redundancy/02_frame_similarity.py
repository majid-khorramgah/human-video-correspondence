"""
02_frame_similarity.py

PhySense-Human Research Pipeline
Stage:
    02_temporal_redundancy / 02_frame_similarity

Purpose
-------
Measure visual similarity between temporally related HR frames.

The script uses the master dataset index created during dataset integrity
analysis and works across all dataset shards and all splits.

Main measurements
-----------------
For selected temporal distances (lags):

    t -> t+1
    t -> t+2
    t -> t+5
    t -> t+10

the script computes:

    - MSE
    - MAE
    - PSNR
    - SSIM

The default configuration produces research-friendly aggregate outputs
without generating a huge pair-level CSV.

Outputs
-------
02_temporal_redundancy/
└── results/
    ├── similarity/
    │   ├── similarity_summary.csv
    │   ├── video_similarity_statistics.csv
    │   ├── similarity_summary.json
    │   ├── similarity_errors.csv
    │   └── pair_similarity_statistics.csv   [optional]
    │
    └── figures/
        ├── similarity_distribution_ssim.png
        ├── similarity_by_temporal_distance.png
        ├── similarity_by_split.png
        └── video_mean_ssim_distribution.png

Examples
--------
Default:
    python 02_frame_similarity.py

Use only adjacent frames:
    python 02_frame_similarity.py --lags 1

Use several temporal distances:
    python 02_frame_similarity.py --lags 1 2 5 10

Analyze every eligible pair:
    python 02_frame_similarity.py --sample-mode all

Analyze 20% of eligible pairs:
    python 02_frame_similarity.py --sample-mode fraction --sample-fraction 0.2

Maximum 50 pairs per video for each lag:
    python 02_frame_similarity.py --sample-mode per_video --max-pairs-per-video 50

Save detailed pair-level CSV:
    python 02_frame_similarity.py --save-pairs

Resize images before comparison:
    python 02_frame_similarity.py --image-size 224

Notes
-----
This stage measures raw image similarity.

It does NOT perform:

    - optical flow
    - geometric alignment
    - feature correspondence
    - pose alignment
    - masking
    - reconstruction

Those belong to later stages of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from skimage.metrics import structural_similarity


# ---------------------------------------------------------------------------
# PIL configuration
# ---------------------------------------------------------------------------

ImageFile.LOAD_TRUNCATED_IMAGES = False

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="skimage",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_NAME = "02_temporal_redundancy / 02_frame_similarity"

DEFAULT_LAGS = [1, 2, 5, 10]

DEFAULT_IMAGE_SIZE = 224

DEFAULT_SAMPLE_MODE = "per_video"

DEFAULT_MAX_PAIRS_PER_VIDEO = 50

DEFAULT_SAMPLE_FRACTION = 0.10

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Console utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

def discover_project_root() -> Path:
    """
    Assume this script is located at:

        <project_root>/02_temporal_redundancy/02_frame_similarity.py

    and therefore project root is two parents above this file.
    """
    script_path = Path(__file__).resolve()

    # <project_root>/02_temporal_redundancy/02_frame_similarity.py
    project_root = script_path.parent.parent

    return project_root


def build_paths(project_root: Path) -> Dict[str, Path]:

    master_index = (
        project_root
        / "01_dataset_audit"
        / "results"
        / "integrity"
        / "master_index.csv"
    )

    similarity_dir = (
        project_root
        / "02_temporal_redundancy"
        / "results"
        / "similarity"
    )

    figures_dir = (
        project_root
        / "02_temporal_redundancy"
        / "results"
        / "figures"
    )

    return {
        "project_root": project_root,
        "master_index": master_index,
        "similarity_dir": similarity_dir,
        "figures_dir": figures_dir,
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Measure visual similarity between temporally related "
            "HR frames in the PhySense-Human dataset."
        )
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Project root. Default: automatically inferred from script location."
        ),
    )

    parser.add_argument(
        "--master-index",
        type=str,
        default=None,
        help=(
            "Path to master_index.csv. "
            "Default: project_root/01_dataset_audit/results/integrity/master_index.csv"
        ),
    )

    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=DEFAULT_LAGS,
        help=(
            "Temporal distances to compare. "
            "Example: --lags 1 2 5 10"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=(
            "Resize images to this square size before comparison. "
            "Default: 224"
        ),
    )

    parser.add_argument(
        "--sample-mode",
        type=str,
        choices=[
            "all",
            "fraction",
            "per_video",
        ],
        default=DEFAULT_SAMPLE_MODE,
        help=(
            "Sampling strategy. "
            "all = every eligible pair; "
            "fraction = random fraction; "
            "per_video = maximum number of pairs per video and lag."
        ),
    )

    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help=(
            "Fraction of eligible pairs to sample when "
            "--sample-mode fraction is used. Default: 0.10"
        ),
    )

    parser.add_argument(
        "--max-pairs-per-video",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_VIDEO,
        help=(
            "Maximum number of pairs per video for each temporal lag "
            "when --sample-mode per_video is used. Default: 50"
        ),
    )

    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Splits to analyze.",
    )

    parser.add_argument(
        "--save-pairs",
        action="store_true",
        help=(
            "Save detailed pair-level results. "
            "Can create a large CSV."
        ),
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help=(
            "Unused currently; kept for reproducibility/config compatibility."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. "
            "Example: --max-videos 10"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Master index
# ---------------------------------------------------------------------------

def load_master_index(
    master_index_path: Path,
    splits: Sequence[str],
) -> pd.DataFrame:

    if not master_index_path.exists():
        raise FileNotFoundError(
            f"Master index not found:\n{master_index_path}"
        )

    print(f"Master index:")
    print(f"  {master_index_path}")

    required_candidates = [
        "shard",
        "split",
        "video_id",
        "frame_id",
        "hr_filename",
        "hr_extension",
    ]

    header = pd.read_csv(
        master_index_path,
        nrows=0,
    )

    available_columns = set(header.columns)

    missing = [
        c
        for c in required_candidates
        if c not in available_columns
    ]

    if missing:
        raise RuntimeError(
            "Required columns are missing from master_index.csv:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    print("Reading required columns only...")

    df = pd.read_csv(
        master_index_path,
        usecols=required_candidates,
    )

    df = df[df["split"].isin(splits)].copy()

    df["video_id"] = df["video_id"].astype(str)

    df["split"] = df["split"].astype(str)

    df["shard"] = df["shard"].astype(str)

    df["frame_id"] = pd.to_numeric(
        df["frame_id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "video_id",
            "frame_id",
            "hr_filename",
        ]
    )

    df["frame_id"] = df["frame_id"].astype(np.int64)

    print(f"Rows loaded after split filtering: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_master_index(df: pd.DataFrame) -> None:

    print_section("Validating master index")

    duplicates = df.duplicated(
        subset=[
            "video_id",
            "frame_id",
        ],
        keep=False,
    ).sum()

    print(
        f"Duplicate (video_id, frame_id) records: {duplicates:,}"
    )

    if duplicates > 0:
        raise RuntimeError(
            "Duplicate temporal identities detected."
        )

    videos = df["video_id"].nunique()

    shards = df["shard"].nunique()

    print(f"Videos: {videos:,}")
    print(f"Shards: {shards:,}")

    print("Splits:")
    for split, count in df["split"].value_counts().sort_index().items():
        print(f"  {split}: {count:,} frames")

    print("Shards:")
    for shard, count in df["shard"].value_counts().sort_index().items():
        print(f"  {shard}: {count:,} frames")


# ---------------------------------------------------------------------------
# File path resolution
# ---------------------------------------------------------------------------

def resolve_image_path(
    project_root: Path,
    row: pd.Series,
) -> Path:

    shard = str(row["shard"])
    split = str(row["split"])
    filename = str(row["hr_filename"])

    direct_path = (
        project_root
        / shard
        / split
        / "Img_HR"
        / filename
    )

    if direct_path.exists():
        return direct_path

    # -----------------------------------------------------------------------
    # Extension-independent fallback
    # -----------------------------------------------------------------------

    stem = Path(filename).stem

    modality_dir = (
        project_root
        / shard
        / split
        / "Img_HR"
    )

    if modality_dir.exists():

        candidates = list(
            modality_dir.glob(
                stem + ".*"
            )
        )

        image_candidates = [
            p
            for p in candidates
            if p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }
        ]

        if len(image_candidates) == 1:
            return image_candidates[0]

    return direct_path


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(
    path: Path,
    image_size: int,
) -> np.ndarray:

    with Image.open(path) as image:

        image = image.convert("RGB")

        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

    array /= 255.0

    return array


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def calculate_metrics(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> Dict[str, float]:

    difference = image_a - image_b

    mae = float(
        np.mean(
            np.abs(difference)
        )
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    if mse <= 1e-12:
        psnr = float("inf")
    else:
        psnr = float(
            10.0 * np.log10(
                1.0 / mse
            )
        )

    # SSIM on RGB image.
    #
    # data_range is explicitly specified because the images are
    # floating point arrays normalized to [0, 1].
    ssim = float(
        structural_similarity(
            image_a,
            image_b,
            data_range=1.0,
            channel_axis=-1,
        )
    )

    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def select_pair_positions(
    frame_ids: np.ndarray,
    lag: int,
    sample_mode: str,
    sample_fraction: float,
    max_pairs_per_video: int,
    rng: np.random.Generator,
) -> List[int]:

    if len(frame_ids) <= lag:
        return []

    # -----------------------------------------------------------------------
    # Candidate positions.
    #
    # We use positional indexing after sorting by frame_id.
    # Because frame IDs were already validated for continuity in the
    # previous stage, position i and i+lag correspond to temporal distance.
    # -----------------------------------------------------------------------

    candidates = np.arange(
        0,
        len(frame_ids) - lag,
        dtype=np.int64,
    )

    if len(candidates) == 0:
        return []

    # -----------------------------------------------------------------------
    # ALL
    # -----------------------------------------------------------------------

    if sample_mode == "all":
        return candidates.tolist()

    # -----------------------------------------------------------------------
    # FRACTION
    # -----------------------------------------------------------------------

    if sample_mode == "fraction":

        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError(
                "--sample-fraction must be in (0, 1]."
            )

        target = max(
            1,
            int(
                math.ceil(
                    len(candidates)
                    * sample_fraction
                )
            ),
        )

        target = min(
            target,
            len(candidates),
        )

        selected = rng.choice(
            candidates,
            size=target,
            replace=False,
        )

        return sorted(
            selected.tolist()
        )

    # -----------------------------------------------------------------------
    # PER VIDEO
    # -----------------------------------------------------------------------

    if sample_mode == "per_video":

        if max_pairs_per_video <= 0:
            raise ValueError(
                "--max-pairs-per-video must be > 0."
            )

        target = min(
            max_pairs_per_video,
            len(candidates),
        )

        if target == len(candidates):
            return candidates.tolist()

        selected = rng.choice(
            candidates,
            size=target,
            replace=False,
        )

        return sorted(
            selected.tolist()
        )

    raise ValueError(
        f"Unknown sample mode: {sample_mode}"
    )


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_video_pairs(
    video_df: pd.DataFrame,
    lag: int,
    sample_mode: str,
    sample_fraction: float,
    max_pairs_per_video: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:

    frame_ids = (
        video_df["frame_id"]
        .to_numpy(
            dtype=np.int64
        )
    )

    positions = select_pair_positions(
        frame_ids=frame_ids,
        lag=lag,
        sample_mode=sample_mode,
        sample_fraction=sample_fraction,
        max_pairs_per_video=max_pairs_per_video,
        rng=rng,
    )

    pairs = []

    for position in positions:

        source_position = position

        target_position = position + lag

        source_frame_id = int(
            frame_ids[source_position]
        )

        target_frame_id = int(
            frame_ids[target_position]
        )

        pairs.append(
            (
                source_frame_id,
                target_frame_id,
            )
        )

    return pairs


# ---------------------------------------------------------------------------
# Single pair comparison
# ---------------------------------------------------------------------------

def compare_pair(
    project_root: Path,
    source_row: pd.Series,
    target_row: pd.Series,
    image_size: int,
) -> Dict[str, object]:

    source_path = resolve_image_path(
        project_root,
        source_row,
    )

    target_path = resolve_image_path(
        project_root,
        target_row,
    )

    source_image = load_image(
        source_path,
        image_size,
    )

    target_image = load_image(
        target_path,
        image_size,
    )

    metrics = calculate_metrics(
        source_image,
        target_image,
    )

    return {
        "shard": str(source_row["shard"]),
        "split": str(source_row["split"]),
        "video_id": str(source_row["video_id"]),
        "frame_id_a": int(source_row["frame_id"]),
        "frame_id_b": int(target_row["frame_id"]),
        "temporal_distance": int(
            int(target_row["frame_id"])
            - int(source_row["frame_id"])
        ),
        "filename_a": str(
            source_row["hr_filename"]
        ),
        "filename_b": str(
            target_row["hr_filename"]
        ),
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
    }


# ---------------------------------------------------------------------------
# Main similarity computation
# ---------------------------------------------------------------------------

def compute_similarity(
    df: pd.DataFrame,
    project_root: Path,
    lags: Sequence[int],
    sample_mode: str,
    sample_fraction: float,
    max_pairs_per_video: int,
    image_size: int,
    seed: int,
    max_videos: Optional[int],
    save_pairs: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    print_section("Preparing temporal sequences")

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # Group by video.
    #
    # IMPORTANT:
    # We group using video_id and sort by frame_id.
    # -----------------------------------------------------------------------

    grouped = df.groupby(
        [
            "video_id",
        ],
        sort=False,
    )

    total_videos = len(grouped)

    print(f"Videos available: {total_videos:,}")

    if max_videos is not None:
        print(
            f"DEBUG MODE: limiting to {max_videos:,} videos."
        )

    pair_records = []

    error_records = []

    processed_videos = 0

    total_pairs = 0

    successful_pairs = 0

    start_time = time.time()

    for video_id, video_df in grouped:

        if (
            max_videos is not None
            and processed_videos >= max_videos
        ):
            break

        video_df = (
            video_df
            .sort_values(
                "frame_id"
            )
            .reset_index(
                drop=True
            )
        )

        # ---------------------------------------------------------------
        # Sanity check.
        # ---------------------------------------------------------------

        split_values = video_df["split"].unique()

        if len(split_values) != 1:

            error_records.append(
                {
                    "video_id": str(video_id),
                    "error_type": "multiple_splits",
                    "error_message": (
                        f"Video appears in multiple splits: "
                        f"{list(split_values)}"
                    ),
                }
            )

            continue

        split = str(
            split_values[0]
        )

        # ---------------------------------------------------------------
        # Analyze every requested temporal distance.
        # ---------------------------------------------------------------

        for lag in lags:

            if lag <= 0:
                continue

            pairs = generate_video_pairs(
                video_df=video_df,
                lag=lag,
                sample_mode=sample_mode,
                sample_fraction=sample_fraction,
                max_pairs_per_video=max_pairs_per_video,
                rng=rng,
            )

            total_pairs += len(pairs)

            # -----------------------------------------------------------
            # Compare each selected pair.
            # -----------------------------------------------------------

            frame_lookup = {
                int(row["frame_id"]): row
                for _, row in video_df.iterrows()
            }

            for frame_a, frame_b in pairs:

                source_row = frame_lookup.get(
                    frame_a
                )

                target_row = frame_lookup.get(
                    frame_b
                )

                if source_row is None or target_row is None:

                    error_records.append(
                        {
                            "video_id": str(video_id),
                            "split": split,
                            "lag": lag,
                            "frame_id_a": frame_a,
                            "frame_id_b": frame_b,
                            "error_type": "frame_not_found",
                            "error_message": (
                                "Frame ID could not be resolved."
                            ),
                        }
                    )

                    continue

                try:

                    result = compare_pair(
                        project_root=project_root,
                        source_row=source_row,
                        target_row=target_row,
                        image_size=image_size,
                    )

                    result["requested_lag"] = lag

                    if save_pairs:
                        pair_records.append(
                            result
                        )

                    else:
                        # We still need statistics.
                        # Store the metrics in memory temporarily.
                        pair_records.append(
                            result
                        )

                    successful_pairs += 1

                except Exception as exc:

                    error_records.append(
                        {
                            "video_id": str(video_id),
                            "split": split,
                            "lag": lag,
                            "frame_id_a": frame_a,
                            "frame_id_b": frame_b,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

        processed_videos += 1

        if (
            processed_videos % 10 == 0
            or processed_videos == 1
        ):

            elapsed = time.time() - start_time

            rate = (
                processed_videos / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Videos processed: "
                f"{processed_videos:,} / {total_videos:,} | "
                f"Pairs successful: {successful_pairs:,} | "
                f"Rate: {rate:.2f} videos/s"
            )

    print()
    print(
        f"Videos processed: {processed_videos:,}"
    )

    print(
        f"Candidate pairs: {total_pairs:,}"
    )

    print(
        f"Successful comparisons: {successful_pairs:,}"
    )

    print(
        f"Failed comparisons: {len(error_records):,}"
    )

    pair_df = pd.DataFrame(
        pair_records
    )

    error_df = pd.DataFrame(
        error_records
    )

    if pair_df.empty:

        raise RuntimeError(
            "No successful frame comparisons were produced."
        )

    # -----------------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------------

    print_section("Aggregating similarity statistics")

    group_columns = [
        "requested_lag",
        "split",
    ]

    summary_records = []

    grouped_metrics = pair_df.groupby(
        group_columns,
        dropna=False,
    )

    for keys, group in grouped_metrics:

        lag, split = keys

        summary_records.append(
            {
                "temporal_distance": int(lag),
                "split": str(split),
                "pairs": int(len(group)),
                "videos": int(
                    group["video_id"].nunique()
                ),
                "mean_ssim": float(
                    group["ssim"].mean()
                ),
                "median_ssim": float(
                    group["ssim"].median()
                ),
                "std_ssim": float(
                    group["ssim"].std(
                        ddof=0
                    )
                ),
                "p10_ssim": float(
                    group["ssim"].quantile(0.10)
                ),
                "p25_ssim": float(
                    group["ssim"].quantile(0.25)
                ),
                "p75_ssim": float(
                    group["ssim"].quantile(0.75)
                ),
                "p90_ssim": float(
                    group["ssim"].quantile(0.90)
                ),
                "mean_mae": float(
                    group["mae"].mean()
                ),
                "median_mae": float(
                    group["mae"].median()
                ),
                "mean_mse": float(
                    group["mse"].mean()
                ),
                "mean_psnr": float(
                    group["psnr"]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .mean()
                ),
                "median_psnr": float(
                    group["psnr"]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .median()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_records
    ).sort_values(
        [
            "temporal_distance",
            "split",
        ]
    )

    # -----------------------------------------------------------------------
    # Per-video statistics
    # -----------------------------------------------------------------------

    video_group_columns = [
        "video_id",
        "split",
        "shard",
        "requested_lag",
    ]

    video_records = []

    grouped_videos = pair_df.groupby(
        video_group_columns,
        dropna=False,
    )

    for keys, group in grouped_videos:

        video_id, split, shard, lag = keys

        video_records.append(
            {
                "video_id": str(video_id),
                "split": str(split),
                "shard": str(shard),
                "temporal_distance": int(lag),
                "pairs": int(len(group)),
                "mean_ssim": float(
                    group["ssim"].mean()
                ),
                "median_ssim": float(
                    group["ssim"].median()
                ),
                "std_ssim": float(
                    group["ssim"].std(
                        ddof=0
                    )
                ),
                "mean_mae": float(
                    group["mae"].mean()
                ),
                "mean_mse": float(
                    group["mse"].mean()
                ),
                "mean_psnr": float(
                    group["psnr"]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .mean()
                ),
            }
        )

    video_df = pd.DataFrame(
        video_records
    ).sort_values(
        [
            "temporal_distance",
            "split",
            "video_id",
        ]
    )

    return (
        summary_df,
        video_df,
        pair_df,
        error_df,
    )


# ---------------------------------------------------------------------------
# Figure utilities
# ---------------------------------------------------------------------------

def save_similarity_distribution(
    pair_df: pd.DataFrame,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(9, 6)
    )

    for lag in sorted(
        pair_df["requested_lag"].unique()
    ):

        values = pair_df.loc[
            pair_df["requested_lag"] == lag,
            "ssim",
        ].dropna()

        if len(values) == 0:
            continue

        plt.hist(
            values,
            bins=50,
            alpha=0.45,
            label=f"t → t+{lag}",
        )

    plt.xlabel(
        "SSIM"
    )

    plt.ylabel(
        "Number of frame pairs"
    )

    plt.title(
        "Distribution of Temporal Frame Similarity"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def save_similarity_by_distance(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(9, 6)
    )

    for split in sorted(
        summary_df["split"].unique()
    ):

        subset = summary_df[
            summary_df["split"] == split
        ].sort_values(
            "temporal_distance"
        )

        plt.plot(
            subset["temporal_distance"],
            subset["mean_ssim"],
            marker="o",
            label=split,
        )

    plt.xlabel(
        "Temporal Distance (frames)"
    )

    plt.ylabel(
        "Mean SSIM"
    )

    plt.title(
        "Mean Visual Similarity vs Temporal Distance"
    )

    plt.xticks(
        sorted(
            summary_df[
                "temporal_distance"
            ].unique()
        )
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def save_similarity_by_split(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(9, 6)
    )

    distances = sorted(
        summary_df[
            "temporal_distance"
        ].unique()
    )

    splits = sorted(
        summary_df[
            "split"
        ].unique()
    )

    x = np.arange(
        len(distances)
    )

    width = (
        0.8 / max(
            len(splits),
            1,
        )
    )

    for i, split in enumerate(splits):

        subset = (
            summary_df[
                summary_df["split"] == split
            ]
            .set_index(
                "temporal_distance"
            )
            .reindex(
                distances
            )
        )

        positions = (
            x
            - 0.4
            + width / 2
            + i * width
        )

        plt.bar(
            positions,
            subset["mean_ssim"],
            width=width,
            label=split,
        )

    plt.xlabel(
        "Temporal Distance (frames)"
    )

    plt.ylabel(
        "Mean SSIM"
    )

    plt.title(
        "Temporal Similarity by Dataset Split"
    )

    plt.xticks(
        x,
        [str(d) for d in distances],
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def save_video_mean_distribution(
    video_df: pd.DataFrame,
    output_path: Path,
) -> None:

    plt.figure(
        figsize=(9, 6)
    )

    for lag in sorted(
        video_df[
            "temporal_distance"
        ].unique()
    ):

        values = video_df.loc[
            video_df["temporal_distance"] == lag,
            "mean_ssim",
        ].dropna()

        if len(values) == 0:
            continue

        plt.hist(
            values,
            bins=40,
            alpha=0.45,
            label=f"t → t+{lag}",
        )

    plt.xlabel(
        "Per-video Mean SSIM"
    )

    plt.ylabel(
        "Number of videos"
    )

    plt.title(
        "Distribution of Mean Temporal Similarity Across Videos"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ---------------------------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------------------------

def sanitize_for_json(value):

    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    return value


def dataframe_records_to_json_safe(
    df: pd.DataFrame,
) -> List[Dict]:

    records = df.to_dict(
        orient="records"
    )

    safe_records = []

    for record in records:

        safe = {}

        for key, value in record.items():

            if pd.isna(value):
                safe[key] = None
            else:
                safe[key] = sanitize_for_json(
                    value
                )

        safe_records.append(
            safe
        )

    return safe_records


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary_json(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    video_df: pd.DataFrame,
    error_df: pd.DataFrame,
    args: argparse.Namespace,
    project_root: Path,
) -> Dict:

    all_ssim = summary_df[
        "mean_ssim"
    ].to_numpy(
        dtype=float
    )

    summary = {

        "stage": STAGE_NAME,

        "project": "PhySense-Human",

        "analysis": {
            "purpose": (
                "Measure raw visual similarity between temporally "
                "related HR frames."
            ),
            "metric": [
                "MAE",
                "MSE",
                "PSNR",
                "SSIM",
            ],
        },

        "dataset": {
            "frames_analyzed": int(
                len(df)
            ),
            "videos_analyzed": int(
                df["video_id"].nunique()
            ),
            "shards": sorted(
                df["shard"].unique().tolist()
            ),
            "splits": sorted(
                df["split"].unique().tolist()
            ),
        },

        "configuration": {
            "lags": [
                int(x)
                for x in args.lags
            ],
            "image_size": int(
                args.image_size
            ),
            "sample_mode": args.sample_mode,
            "sample_fraction": (
                float(args.sample_fraction)
                if args.sample_mode == "fraction"
                else None
            ),
            "max_pairs_per_video": (
                int(args.max_pairs_per_video)
                if args.sample_mode == "per_video"
                else None
            ),
            "random_seed": int(
                args.seed
            ),
            "save_pairs": bool(
                args.save_pairs
            ),
        },

        "results": {
            "successful_pair_comparisons": int(
                len(
                    summary_df
                )
                and (
                    video_df["pairs"].sum()
                )
            ),
            "failed_comparisons": int(
                len(error_df)
            ),
        },

        "interpretation_note": (
            "SSIM here measures raw image similarity between resized "
            "RGB frames. It does not perform motion compensation, "
            "geometric alignment, optical flow, pose alignment, or "
            "feature correspondence."
        ),

        "summary_by_distance_and_split":
            dataframe_records_to_json_safe(
                summary_df
            ),

        "runtime": {
            "project_root": str(
                project_root
            ),
        },
    }

    return summary


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    similarity_dir: Path,
    figures_dir: Path,
    summary_df: pd.DataFrame,
    video_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    error_df: pd.DataFrame,
    summary_json: Dict,
    save_pairs: bool,
) -> None:

    similarity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section("Saving CSV results")

    summary_csv = (
        similarity_dir
        / "similarity_summary.csv"
    )

    video_csv = (
        similarity_dir
        / "video_similarity_statistics.csv"
    )

    errors_csv = (
        similarity_dir
        / "similarity_errors.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    video_df.to_csv(
        video_csv,
        index=False,
    )

    error_df.to_csv(
        errors_csv,
        index=False,
    )

    print(
        f"Saved: {summary_csv}"
    )

    print(
        f"Saved: {video_csv}"
    )

    print(
        f"Saved: {errors_csv}"
    )

    if save_pairs:

        pair_csv = (
            similarity_dir
            / "pair_similarity_statistics.csv"
        )

        pair_df.to_csv(
            pair_csv,
            index=False,
        )

        print(
            f"Saved detailed pair CSV: {pair_csv}"
        )

    else:

        print(
            "Detailed pair CSV not saved "
            "(use --save-pairs if needed)."
        )

    # -----------------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------------

    json_path = (
        similarity_dir
        / "similarity_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved: {json_path}"
    )

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------

    print_section("Generating figures")

    save_similarity_distribution(
        pair_df,
        figures_dir
        / "similarity_distribution_ssim.png",
    )

    save_similarity_by_distance(
        summary_df,
        figures_dir
        / "similarity_by_temporal_distance.png",
    )

    save_similarity_by_split(
        summary_df,
        figures_dir
        / "similarity_by_split.png",
    )

    save_video_mean_distribution(
        video_df,
        figures_dir
        / "video_mean_ssim_distribution.png",
    )

    print(
        f"Figures saved to: {figures_dir}"
    )


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    video_df: pd.DataFrame,
    error_df: pd.DataFrame,
    args: argparse.Namespace,
    start_time: float,
) -> None:

    elapsed = (
        time.time()
        - start_time
    )

    print_header(
        "FINAL FRAME SIMILARITY REPORT"
    )

    print(
        f"Frames available: "
        f"{len(df):,}"
    )

    print(
        f"Videos: "
        f"{df['video_id'].nunique():,}"
    )

    print(
        f"Shards: "
        f"{df['shard'].nunique():,}"
    )

    print(
        f"Lags analyzed: "
        f"{args.lags}"
    )

    print(
        f"Sampling mode: "
        f"{args.sample_mode}"
    )

    print(
        f"Successful pair comparisons: "
        f"{int(video_df['pairs'].sum()):,}"
    )

    print(
        f"Failed comparisons: "
        f"{len(error_df):,}"
    )

    print()

    print(
        "Similarity summary:"
    )

    display_columns = [
        "temporal_distance",
        "split",
        "pairs",
        "videos",
        "mean_ssim",
        "median_ssim",
        "p10_ssim",
        "p90_ssim",
        "mean_mae",
        "mean_psnr",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Runtime: "
        f"{elapsed / 60.0:.2f} minutes"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This stage measures raw visual similarity."
    )

    print(
        "It does not measure motion-compensated "
        "correspondence or reconstruction usefulness."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    start_time = time.time()

    args = parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    print_header(
        "FRAME SIMILARITY ANALYSIS"
    )

    print(
        "PhySense-Human Research Pipeline"
    )

    print(
        f"Stage: {STAGE_NAME}"
    )

    # -----------------------------------------------------------------------
    # Project root
    # -----------------------------------------------------------------------

    if args.project_root is not None:

        project_root = Path(
            args.project_root
        ).resolve()

    else:

        project_root = discover_project_root()

    paths = build_paths(
        project_root
    )

    if args.master_index is not None:

        master_index_path = Path(
            args.master_index
        ).resolve()

    else:

        master_index_path = paths[
            "master_index"
        ]

    print()
    print(
        "Project root:"
    )
    print(
        f"  {project_root}"
    )

    print()
    print(
        "Master index:"
    )
    print(
        f"  {master_index_path}"
    )

    # -----------------------------------------------------------------------
    # Validate configuration
    # -----------------------------------------------------------------------

    lags = sorted(
        set(
            int(x)
            for x in args.lags
        )
    )

    if any(
        lag <= 0
        for lag in lags
    ):
        raise ValueError(
            "All temporal lags must be positive integers."
        )

    if args.image_size < 32:
        raise ValueError(
            "--image-size should be >= 32."
        )

    # -----------------------------------------------------------------------
    # Load master index
    # -----------------------------------------------------------------------

    print_section(
        "Loading master dataset index"
    )

    df = load_master_index(
        master_index_path,
        args.splits,
    )

    validate_master_index(
        df
    )

    # -----------------------------------------------------------------------
    # Compute similarity
    # -----------------------------------------------------------------------

    print_section(
        "Computing temporal visual similarity"
    )

    print(
        f"Temporal distances: {lags}"
    )

    print(
        f"Image size: "
        f"{args.image_size} x {args.image_size}"
    )

    print(
        f"Sampling mode: "
        f"{args.sample_mode}"
    )

    if args.sample_mode == "fraction":

        print(
            f"Sampling fraction: "
            f"{args.sample_fraction}"
        )

    if args.sample_mode == "per_video":

        print(
            f"Maximum pairs/video/lag: "
            f"{args.max_pairs_per_video}"
        )

    (
        summary_df,
        video_df,
        pair_df,
        error_df,
    ) = compute_similarity(
        df=df,
        project_root=project_root,
        lags=lags,
        sample_mode=args.sample_mode,
        sample_fraction=args.sample_fraction,
        max_pairs_per_video=args.max_pairs_per_video,
        image_size=args.image_size,
        seed=args.seed,
        max_videos=args.max_videos,
        save_pairs=args.save_pairs,
    )

    # -----------------------------------------------------------------------
    # Build JSON
    # -----------------------------------------------------------------------

    summary_json = build_summary_json(
        df=df,
        summary_df=summary_df,
        video_df=video_df,
        error_df=error_df,
        args=args,
        project_root=project_root,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    save_outputs(
        similarity_dir=paths[
            "similarity_dir"
        ],
        figures_dir=paths[
            "figures_dir"
        ],
        summary_df=summary_df,
        video_df=video_df,
        pair_df=pair_df,
        error_df=error_df,
        summary_json=summary_json,
        save_pairs=args.save_pairs,
    )

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    print_final_report(
        df=df,
        summary_df=summary_df,
        video_df=video_df,
        error_df=error_df,
        args=args,
        start_time=start_time,
    )

    print()
    print(
        "CORE FRAME SIMILARITY ANALYSIS: COMPLETE"
    )

    print()
    print(
        "Results:"
    )

    print(
        f"  {paths['similarity_dir']}"
    )

    print(
        f"  {paths['figures_dir']}"
    )


if __name__ == "__main__":
    main()