"""
===============================================================================
04_motion_analysis.py

PhySense-Human / Human Video Correspondence Research Pipeline

Stage:
    02_temporal_redundancy / 04_motion_analysis

Purpose:
    Analyze temporal motion between video frames using dense optical flow.

Main objectives:
    1. Measure frame-to-frame motion magnitude.
    2. Measure motion at multiple temporal distances.
    3. Compare full-frame motion with human-region motion.
    4. Estimate background motion.
    5. Analyze motion direction and spatial coverage.
    6. Produce reproducible CSV / JSON / PNG research outputs.

Input:
    01_dataset_audit/results/integrity/master_index.csv

Image source:
    Img_HR

Dataset:
    All available shards and train / val / test splits.

Method:
    Dense Optical Flow using OpenCV Farneback algorithm.

Important:
    This analysis measures image-space motion.
    It does NOT establish semantic correspondence by itself.

===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_INDEX = (
    PROJECT_ROOT
    / "01_dataset_audit"
    / "results"
    / "integrity"
    / "master_index.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "02_temporal_redundancy"
    / "results"
    / "04_motion_analysis"
)

STATISTICS_DIR = RESULTS_ROOT / "motion_statistics"
FIGURES_DIR = RESULTS_ROOT / "figures"


DEFAULT_LAGS = [1, 2, 5, 10]

DEFAULT_IMAGE_SIZE = 224

DEFAULT_MAX_PAIRS_PER_VIDEO = 50

DEFAULT_SAMPLING_MODE = "per_video"

DEFAULT_SPLITS = ["train", "val", "test"]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subsection(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def ensure_directories() -> None:
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def json_safe(value):
    """
    Convert NumPy / pandas objects to JSON-safe Python objects.
    """

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    return value


def safe_mean(values):
    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return np.nan

    return float(np.nanmean(values))


def safe_median(values):
    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return np.nan

    return float(np.nanmedian(values))


def safe_percentile(values, percentile):
    if len(values) == 0:
        return np.nan

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return np.nan

    return float(np.nanpercentile(values, percentile))


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze temporal motion using dense optical flow "
            "across all dataset shards."
        )
    )

    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=DEFAULT_LAGS,
        help=(
            "Temporal distances to analyze. "
            "Example: --lags 1 2 5 10"
        ),
    )

    parser.add_argument(
        "--max-pairs-per-video",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_VIDEO,
        help=(
            "Maximum number of frame pairs analyzed per "
            "video and temporal distance."
        ),
    )

    parser.add_argument(
        "--sampling",
        choices=["per_video", "all"],
        default=DEFAULT_SAMPLING_MODE,
        help=(
            "Sampling strategy. "
            "per_video samples approximately equal numbers of pairs "
            "from each video."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Resize frames to this square size before optical flow.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=["train", "val", "test"],
        help="Dataset splits to analyze.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    parser.add_argument(
        "--save-pairs",
        action="store_true",
        help=(
            "Save detailed pair-level motion measurements. "
            "This can create a large CSV."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N videos.",
    )

    return parser.parse_args()


# =============================================================================
# MASTER INDEX
# =============================================================================

def load_master_index(
    master_index_path: Path,
    splits: List[str],
) -> pd.DataFrame:

    print_subsection("Loading master dataset index")

    print(f"Master index:")
    print(f"  {master_index_path}")

    if not master_index_path.exists():
        raise FileNotFoundError(
            f"Master index not found:\n{master_index_path}"
        )

    required_columns = [
        "shard",
        "split",
        "video_id",
        "frame_id",
        "Img_HR_filename",
        "has_Img_HR",
    ]

    print("Reading required columns only...")

    df = pd.read_csv(
        master_index_path,
        usecols=required_columns,
    )

    df = df[df["split"].isin(splits)].copy()

    df["frame_id"] = pd.to_numeric(
        df["frame_id"],
        errors="coerce",
    )

    df = df.dropna(subset=["frame_id"])

    df["frame_id"] = df["frame_id"].astype(int)

    df["video_id"] = df["video_id"].astype(str)

    df["shard"] = df["shard"].astype(str)

    df["split"] = df["split"].astype(str)

    df["Img_HR_filename"] = df["Img_HR_filename"].astype(str)

    print(f"Rows loaded after split filtering: {len(df):,}")

    return df


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_master_index(df: pd.DataFrame) -> None:

    print_subsection("Validating master index")

    duplicate_count = int(
        df.duplicated(
            subset=["video_id", "frame_id"]
        ).sum()
    )

    print(
        "Duplicate (video_id, frame_id) records: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate temporal identities detected."
        )

    print(f"Videos: {df['video_id'].nunique():,}")
    print(f"Shards: {df['shard'].nunique():,}")

    print("Splits:")

    for split in sorted(df["split"].unique()):
        count = int((df["split"] == split).sum())
        print(f"  {split}: {count:,} frames")

    print("Shards:")

    for shard in sorted(df["shard"].unique()):
        count = int((df["shard"] == shard).sum())
        print(f"  {shard}: {count:,} frames")


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def resolve_image_path(
    row: pd.Series,
) -> Path:

    shard = row["shard"]
    split = row["split"]
    filename = row["Img_HR_filename"]

    path = (
        PROJECT_ROOT
        / "Dataset"
        / shard
        / split
        / "Img_HR"
        / filename
    )

    if path.exists():
        return path

    # Fallback:
    # Some local dataset layouts may use the dataset shards
    # directly under the project root.
    fallback = (
        PROJECT_ROOT
        / shard
        / split
        / "Img_HR"
        / filename
    )

    return fallback


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_gray_image(
    path: Path,
    image_size: int,
) -> Optional[np.ndarray]:

    if not path.exists():
        return None

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    if image_size is not None:
        image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

    return image


# =============================================================================
# OPTICAL FLOW
# =============================================================================

def compute_optical_flow(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[np.ndarray]:

    if frame_a is None or frame_b is None:
        return None

    try:

        flow = cv2.calcOpticalFlowFarneback(
            frame_a,
            frame_b,
            None,

            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        return flow

    except Exception:
        return None


# =============================================================================
# FLOW STATISTICS
# =============================================================================

def compute_flow_statistics(
    flow: np.ndarray,
) -> Dict[str, float]:

    dx = flow[..., 0]
    dy = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        dx,
        dy,
        angleInDegrees=True,
    )

    magnitude = magnitude.astype(np.float32)

    valid = np.isfinite(magnitude)

    if not np.any(valid):
        return {
            "mean_magnitude": np.nan,
            "median_magnitude": np.nan,
            "p10_magnitude": np.nan,
            "p90_magnitude": np.nan,
            "max_magnitude": np.nan,
            "motion_coverage": np.nan,
            "mean_dx": np.nan,
            "mean_dy": np.nan,
            "mean_direction": np.nan,
        }

    mag = magnitude[valid]

    dx_valid = dx[valid]
    dy_valid = dy[valid]

    threshold = 0.5

    motion_pixels = mag > threshold

    motion_coverage = (
        float(np.mean(motion_pixels))
    )

    mean_dx = float(np.mean(dx_valid))
    mean_dy = float(np.mean(dy_valid))

    mean_direction = float(
        math.degrees(
            math.atan2(
                mean_dy,
                mean_dx,
            )
        )
    )

    return {
        "mean_magnitude": float(np.mean(mag)),
        "median_magnitude": float(np.median(mag)),
        "p10_magnitude": float(np.percentile(mag, 10)),
        "p90_magnitude": float(np.percentile(mag, 90)),
        "max_magnitude": float(np.max(mag)),
        "motion_coverage": motion_coverage,
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "mean_direction": mean_direction,
    }


# =============================================================================
# REGION STATISTICS
# =============================================================================

def region_flow_statistics(
    flow: np.ndarray,
    mask: Optional[np.ndarray],
) -> Dict[str, float]:

    if mask is None:
        return {
            "mean_magnitude": np.nan,
            "median_magnitude": np.nan,
            "motion_coverage": np.nan,
        }

    if mask.shape != flow.shape[:2]:

        mask = cv2.resize(
            mask,
            (
                flow.shape[1],
                flow.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = mask > 127

    if not np.any(mask):
        return {
            "mean_magnitude": np.nan,
            "median_magnitude": np.nan,
            "motion_coverage": np.nan,
        }

    magnitude = np.sqrt(
        flow[..., 0] ** 2
        + flow[..., 1] ** 2
    )

    values = magnitude[mask]

    motion_coverage = float(
        np.mean(values > 0.5)
    )

    return {
        "mean_magnitude": float(np.mean(values)),
        "median_magnitude": float(np.median(values)),
        "motion_coverage": motion_coverage,
    }


# =============================================================================
# SAMPLING
# =============================================================================

def sample_pairs_for_video(
    video_df: pd.DataFrame,
    lag: int,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:

    frame_ids = sorted(
        video_df["frame_id"].astype(int).tolist()
    )

    frame_set = set(frame_ids)

    candidates = []

    for frame_id in frame_ids:

        target = frame_id + lag

        if target in frame_set:
            candidates.append(
                (frame_id, target)
            )

    if max_pairs is not None:
        if len(candidates) > max_pairs:

            indices = rng.choice(
                len(candidates),
                size=max_pairs,
                replace=False,
            )

            candidates = [
                candidates[int(i)]
                for i in indices
            ]

    return candidates


# =============================================================================
# FRAME LOOKUP
# =============================================================================

def build_frame_lookup(
    video_df: pd.DataFrame,
) -> Dict[int, pd.Series]:

    lookup = {}

    for _, row in video_df.iterrows():

        frame_id = int(row["frame_id"])

        lookup[frame_id] = row

    return lookup


# =============================================================================
# SINGLE PAIR ANALYSIS
# =============================================================================

def analyze_pair(
    row_a: pd.Series,
    row_b: pd.Series,
    image_size: int,
) -> Optional[Dict]:

    path_a = resolve_image_path(row_a)
    path_b = resolve_image_path(row_b)

    frame_a = load_gray_image(
        path_a,
        image_size,
    )

    frame_b = load_gray_image(
        path_b,
        image_size,
    )

    if frame_a is None or frame_b is None:
        return None

    flow = compute_optical_flow(
        frame_a,
        frame_b,
    )

    if flow is None:
        return None

    stats = compute_flow_statistics(flow)

    result = {
        "shard": row_a["shard"],
        "split": row_a["split"],
        "video_id": row_a["video_id"],
        "frame_id_a": int(row_a["frame_id"]),
        "frame_id_b": int(row_b["frame_id"]),
        "temporal_distance": int(
            row_b["frame_id"] - row_a["frame_id"]
        ),
        **stats,
    }

    return result


# =============================================================================
# MAIN MOTION ANALYSIS
# =============================================================================

def run_motion_analysis(
    df: pd.DataFrame,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print_subsection(
        "Computing temporal motion using dense optical flow"
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
        f"{args.sampling}"
    )

    print(
        f"Maximum pairs/video/lag: "
        f"{args.max_pairs_per_video}"
    )

    rng = np.random.default_rng(
        args.seed
    )

    grouped = list(
        df.groupby(
            ["video_id"],
            sort=True,
        )
    )

    total_videos = len(grouped)

    print(
        f"Videos available: "
        f"{total_videos:,}"
    )

    results = []
    errors = []

    candidate_pairs = 0

    processed_videos = 0

    start_time = time.time()

    for video_id, video_df in grouped:

        processed_videos += 1

        video_df = video_df.sort_values(
            "frame_id"
        )

        frame_lookup = build_frame_lookup(
            video_df
        )

        for lag in args.lags:

            pairs = sample_pairs_for_video(
                video_df=video_df,
                lag=lag,
                max_pairs=(
                    None
                    if args.sampling == "all"
                    else args.max_pairs_per_video
                ),
                rng=rng,
            )

            candidate_pairs += len(pairs)

            for frame_a, frame_b in pairs:

                row_a = frame_lookup.get(
                    frame_a
                )

                row_b = frame_lookup.get(
                    frame_b
                )

                if row_a is None or row_b is None:
                    errors.append(
                        {
                            "video_id": video_id,
                            "frame_id_a": frame_a,
                            "frame_id_b": frame_b,
                            "temporal_distance": lag,
                            "error": "frame_lookup_failed",
                        }
                    )

                    continue

                result = analyze_pair(
                    row_a,
                    row_b,
                    args.image_size,
                )

                if result is None:

                    errors.append(
                        {
                            "video_id": video_id,
                            "frame_id_a": frame_a,
                            "frame_id_b": frame_b,
                            "temporal_distance": lag,
                            "error": "image_or_flow_failed",
                        }
                    )

                    continue

                results.append(result)

        if (
            processed_videos == 1
            or processed_videos % args.progress_every == 0
            or processed_videos == total_videos
        ):

            elapsed = time.time() - start_time

            rate = (
                processed_videos / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Videos processed: "
                f"{processed_videos:,} / "
                f"{total_videos:,} | "
                f"Successful pairs: "
                f"{len(results):,} | "
                f"Rate: "
                f"{rate:.2f} videos/s"
            )

    results_df = pd.DataFrame(results)

    errors_df = pd.DataFrame(errors)

    print()
    print(
        f"Candidate pairs: "
        f"{candidate_pairs:,}"
    )

    print(
        f"Successful comparisons: "
        f"{len(results_df):,}"
    )

    print(
        f"Failed comparisons: "
        f"{len(errors_df):,}"
    )

    return results_df, errors_df


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_motion_statistics(
    results_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print_subsection(
        "Aggregating motion statistics"
    )

    if results_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    summary = (
        results_df
        .groupby(
            [
                "temporal_distance",
                "split",
            ]
        )
        .agg(
            pairs=("video_id", "count"),
            videos=("video_id", "nunique"),

            mean_motion_magnitude=(
                "mean_magnitude",
                "mean",
            ),

            median_motion_magnitude=(
                "mean_magnitude",
                "median",
            ),

            p10_motion_magnitude=(
                "mean_magnitude",
                lambda x: np.percentile(x, 10),
            ),

            p90_motion_magnitude=(
                "mean_magnitude",
                lambda x: np.percentile(x, 90),
            ),

            mean_motion_coverage=(
                "motion_coverage",
                "mean",
            ),

            mean_dx=(
                "mean_dx",
                "mean",
            ),

            mean_dy=(
                "mean_dy",
                "mean",
            ),

            mean_direction=(
                "mean_direction",
                "mean",
            ),
        )
        .reset_index()
    )

    video_summary = (
        results_df
        .groupby(
            [
                "video_id",
                "split",
                "shard",
                "temporal_distance",
            ]
        )
        .agg(
            pairs=("video_id", "count"),

            mean_motion_magnitude=(
                "mean_magnitude",
                "mean",
            ),

            median_motion_magnitude=(
                "mean_magnitude",
                "median",
            ),

            p90_motion_magnitude=(
                "mean_magnitude",
                lambda x: np.percentile(x, 90),
            ),

            mean_motion_coverage=(
                "motion_coverage",
                "mean",
            ),

            mean_dx=(
                "mean_dx",
                "mean",
            ),

            mean_dy=(
                "mean_dy",
                "mean",
            ),

            mean_direction=(
                "mean_direction",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary, video_summary


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def save_figure(
    filename: str,
) -> None:

    path = FIGURES_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# FIGURE 1
# =============================================================================

def plot_motion_by_temporal_distance(
    summary: pd.DataFrame,
) -> None:

    if summary.empty:
        return

    plt.figure(figsize=(9, 6))

    for split in sorted(
        summary["split"].unique()
    ):

        subset = summary[
            summary["split"] == split
        ].sort_values(
            "temporal_distance"
        )

        plt.plot(
            subset["temporal_distance"],
            subset["mean_motion_magnitude"],
            marker="o",
            label=split,
        )

    plt.xlabel(
        "Temporal distance (frames)"
    )

    plt.ylabel(
        "Mean optical-flow magnitude"
    )

    plt.title(
        "Motion Magnitude vs Temporal Distance"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    save_figure(
        "motion_by_temporal_distance.png"
    )


# =============================================================================
# FIGURE 2
# =============================================================================

def plot_motion_by_split(
    summary: pd.DataFrame,
) -> None:

    if summary.empty:
        return

    distances = sorted(
        summary[
            "temporal_distance"
        ].unique()
    )

    for lag in distances:

        subset = summary[
            summary["temporal_distance"] == lag
        ].sort_values("split")

        plt.figure(figsize=(8, 5))

        plt.bar(
            subset["split"],
            subset["mean_motion_magnitude"],
        )

        plt.xlabel("Split")

        plt.ylabel(
            "Mean optical-flow magnitude"
        )

        plt.title(
            f"Motion Magnitude by Split "
            f"(temporal distance = {lag})"
        )

        save_figure(
            f"motion_by_split_lag_{lag}.png"
        )


# =============================================================================
# FIGURE 3
# =============================================================================

def plot_motion_coverage(
    summary: pd.DataFrame,
) -> None:

    if summary.empty:
        return

    plt.figure(figsize=(9, 6))

    for split in sorted(
        summary["split"].unique()
    ):

        subset = summary[
            summary["split"] == split
        ].sort_values(
            "temporal_distance"
        )

        plt.plot(
            subset["temporal_distance"],
            subset["mean_motion_coverage"],
            marker="o",
            label=split,
        )

    plt.xlabel(
        "Temporal distance (frames)"
    )

    plt.ylabel(
        "Mean motion coverage"
    )

    plt.title(
        "Spatial Motion Coverage vs Temporal Distance"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    save_figure(
        "motion_coverage_by_temporal_distance.png"
    )


# =============================================================================
# FIGURE 4
# =============================================================================

def plot_motion_distribution(
    results_df: pd.DataFrame,
) -> None:

    if results_df.empty:
        return

    distances = sorted(
        results_df[
            "temporal_distance"
        ].unique()
    )

    for lag in distances:

        subset = results_df[
            results_df["temporal_distance"] == lag
        ]

        if subset.empty:
            continue

        plt.figure(figsize=(9, 6))

        plt.hist(
            subset["mean_magnitude"],
            bins=50,
        )

        plt.xlabel(
            "Mean optical-flow magnitude"
        )

        plt.ylabel(
            "Number of frame pairs"
        )

        plt.title(
            f"Motion Magnitude Distribution "
            f"(temporal distance = {lag})"
        )

        save_figure(
            f"motion_distribution_lag_{lag}.png"
        )


# =============================================================================
# FIGURE 5
# =============================================================================

def plot_video_motion_distribution(
    video_summary: pd.DataFrame,
) -> None:

    if video_summary.empty:
        return

    subset = video_summary[
        video_summary["temporal_distance"]
        == video_summary["temporal_distance"].min()
    ]

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))

    plt.hist(
        subset["mean_motion_magnitude"],
        bins=50,
    )

    plt.xlabel(
        "Mean video-level motion magnitude"
    )

    plt.ylabel(
        "Number of videos"
    )

    plt.title(
        "Distribution of Video-Level Motion"
    )

    save_figure(
        "video_motion_distribution.png"
    )


# =============================================================================
# GENERATE FIGURES
# =============================================================================

def generate_figures(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    video_summary_df: pd.DataFrame,
) -> None:

    print_subsection(
        "Generating motion figures"
    )

    plot_motion_by_temporal_distance(
        summary_df
    )

    plot_motion_by_split(
        summary_df
    )

    plot_motion_coverage(
        summary_df
    )

    plot_motion_distribution(
        results_df
    )

    plot_video_motion_distribution(
        video_summary_df
    )

    print(
        f"Figures saved to:\n"
        f"  {FIGURES_DIR}"
    )


# =============================================================================
# SUMMARY JSON
# =============================================================================

def create_summary_json(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    args,
    runtime_seconds: float,
) -> Dict:

    summary = {
        "analysis": {
            "name": "Temporal Motion Analysis",
            "stage": (
                "02_temporal_redundancy/"
                "04_motion_analysis"
            ),
            "method": (
                "Dense Optical Flow "
                "(OpenCV Farneback)"
            ),
        },

        "input": {
            "master_index": str(
                MASTER_INDEX
            ),
            "image_modality": "Img_HR",
            "splits": args.splits,
        },

        "dataset": {
            "frames_available": int(
                len(df)
            ),
            "videos": int(
                df["video_id"].nunique()
            ),
            "shards": int(
                df["shard"].nunique()
            ),
        },

        "configuration": {
            "lags": args.lags,
            "sampling_mode": args.sampling,
            "max_pairs_per_video": (
                args.max_pairs_per_video
            ),
            "image_size": args.image_size,
            "seed": args.seed,
        },

        "results": {
            "successful_comparisons": int(
                len(results_df)
            ),
            "failed_comparisons": int(
                len(errors_df)
            ),
        },

        "interpretation": {
            "meaning": (
                "Optical-flow magnitude measures "
                "image-space displacement between "
                "two frames."
            ),
            "limitation": (
                "High motion magnitude does not "
                "necessarily imply semantic motion. "
                "Camera motion, illumination changes, "
                "occlusion, and background movement "
                "can contribute."
            ),
            "research_relevance": (
                "The analysis quantifies how much "
                "temporal information changes as "
                "frame distance increases."
            ),
        },

        "runtime_seconds": float(
            runtime_seconds
        ),
    }

    return json_safe(summary)


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    video_summary_df: pd.DataFrame,
    df: pd.DataFrame,
    args,
    runtime_seconds: float,
) -> None:

    print_subsection(
        "Saving motion analysis results"
    )

    summary_path = (
        STATISTICS_DIR
        / "motion_summary.csv"
    )

    video_path = (
        STATISTICS_DIR
        / "video_motion_statistics.csv"
    )

    errors_path = (
        STATISTICS_DIR
        / "motion_errors.csv"
    )

    json_path = (
        STATISTICS_DIR
        / "motion_summary.json"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    video_summary_df.to_csv(
        video_path,
        index=False,
    )

    errors_df.to_csv(
        errors_path,
        index=False,
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {video_path}"
    )

    print(
        f"Saved: {errors_path}"
    )

    if args.save_pairs:

        pair_path = (
            STATISTICS_DIR
            / "frame_pair_motion_statistics.csv"
        )

        results_df.to_csv(
            pair_path,
            index=False,
        )

        print(
            f"Saved: {pair_path}"
        )

    summary_json = create_summary_json(
        df=df,
        results_df=results_df,
        errors_df=errors_df,
        summary_df=summary_df,
        args=args,
        runtime_seconds=runtime_seconds,
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


# =============================================================================
# FINAL REPORT
# =============================================================================

def print_final_report(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    args,
    runtime_seconds: float,
) -> None:

    print()
    print("=" * 78)
    print("FINAL MOTION ANALYSIS REPORT")
    print("=" * 78)

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
        f"{args.sampling}"
    )

    print(
        f"Successful comparisons: "
        f"{len(results_df):,}"
    )

    print(
        f"Failed comparisons: "
        f"{len(errors_df):,}"
    )

    if not results_df.empty:

        print()
        print(
            "Overall motion statistics:"
        )

        print(
            f"  Mean motion magnitude: "
            f"{results_df['mean_magnitude'].mean():.6f}"
        )

        print(
            f"  Median motion magnitude: "
            f"{results_df['mean_magnitude'].median():.6f}"
        )

        print(
            f"  Mean motion coverage: "
            f"{results_df['motion_coverage'].mean():.6f}"
        )

        print()
        print(
            "Motion summary by temporal distance:"
        )

        display_cols = [
            "temporal_distance",
            "split",
            "pairs",
            "videos",
            "mean_motion_magnitude",
            "median_motion_magnitude",
            "mean_motion_coverage",
        ]

        # Recompute compact summary for console
        console_summary = (
            results_df
            .groupby(
                [
                    "temporal_distance",
                    "split",
                ]
            )
            .agg(
                pairs=("video_id", "count"),
                videos=("video_id", "nunique"),
                mean_motion_magnitude=(
                    "mean_magnitude",
                    "mean",
                ),
                median_motion_magnitude=(
                    "mean_magnitude",
                    "median",
                ),
                mean_motion_coverage=(
                    "motion_coverage",
                    "mean",
                ),
            )
            .reset_index()
        )

        print(
            console_summary[
                display_cols
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "CORE MOTION ANALYSIS: COMPLETE"
    )

    print()
    print(
        f"Runtime: "
        f"{runtime_seconds / 60:.2f} minutes"
    )

    print()
    print(
        "Results:"
    )

    print(
        f"  {STATISTICS_DIR}"
    )

    print(
        f"  {FIGURES_DIR}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_time = time.time()

    args = parse_args()

    print("=" * 78)
    print("MOTION ANALYSIS")
    print("=" * 78)

    print(
        "PhySense-Human Research Pipeline"
    )

    print(
        "Stage: "
        "02_temporal_redundancy / "
        "04_motion_analysis"
    )

    print()
    print(
        "Project root:"
    )

    print(
        f"  {PROJECT_ROOT}"
    )

    print()
    print(
        "Master index:"
    )

    print(
        f"  {MASTER_INDEX}"
    )

    ensure_directories()

    df = load_master_index(
        MASTER_INDEX,
        args.splits,
    )

    validate_master_index(
        df
    )

    results_df, errors_df = run_motion_analysis(
        df,
        args,
    )

    summary_df, video_summary_df = (
        aggregate_motion_statistics(
            results_df
        )
    )

    save_results(
        results_df=results_df,
        errors_df=errors_df,
        summary_df=summary_df,
        video_summary_df=video_summary_df,
        df=df,
        args=args,
        runtime_seconds=(
            time.time() - start_time
        ),
    )

    generate_figures(
        results_df=results_df,
        summary_df=summary_df,
        video_summary_df=video_summary_df,
    )

    runtime_seconds = (
        time.time() - start_time
    )

    print_final_report(
        df=df,
        results_df=results_df,
        errors_df=errors_df,
        args=args,
        runtime_seconds=runtime_seconds,
    )


if __name__ == "__main__":
    main()