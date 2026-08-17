from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
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

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "02_temporal_redundancy"
    / "results"
    / "03_temporal_difference"
)

STATS_DIR = OUTPUT_ROOT / "temporal_difference"
FIGURES_DIR = OUTPUT_ROOT / "figures"


EXPECTED_MASKS = {
    "full": "Mask_Full_HR",
    "cloth": "Mask_Cloth_HR",
    "skin": "Mask_Skin_HR",
}


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
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal difference analysis for PhySense-Human."
    )

    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 2, 5, 10],
        help="Temporal distances to analyze.",
    )

    parser.add_argument(
        "--max-pairs-per-video",
        type=int,
        default=50,
        help="Maximum sampled pairs per video and lag.",
    )

    parser.add_argument(
        "--sampling",
        choices=["per_video", "all"],
        default="per_video",
        help="Pair sampling strategy.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Images are resized to this square size.",
    )

    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to analyze.",
    )

    parser.add_argument(
        "--save-pairs",
        action="store_true",
        help="Save detailed pair-level CSV. Can be large.",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


# =============================================================================
# UTILITY
# =============================================================================

def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    return value


def safe_mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return float(np.mean(values))


def safe_median(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return float(np.median(values))


# =============================================================================
# MASTER INDEX
# =============================================================================

def load_master_index(
    master_index: Path,
    splits: List[str],
) -> pd.DataFrame:

    if not master_index.exists():
        raise FileNotFoundError(
            f"Master index not found:\n{master_index}"
        )

    print(f"Master index:")
    print(f"  {master_index}")

    required = [
        "shard",
        "split",
        "video_id",
        "frame_id",
        "Img_HR_filename",
        "Mask_Full_HR_filename",
        "Mask_Cloth_HR_filename",
        "Mask_Skin_HR_filename",
    ]

    print("Reading required columns only...")

    df = pd.read_csv(
        master_index,
        usecols=lambda c: c in required,
    )

    df["split"] = df["split"].astype(str)

    df = df[df["split"].isin(splits)].copy()

    df["video_id"] = df["video_id"].astype(str)

    df["frame_id"] = pd.to_numeric(
        df["frame_id"],
        errors="coerce",
    )

    df = df.dropna(subset=["video_id", "frame_id"])

    df["frame_id"] = df["frame_id"].astype(int)

    print(f"Rows loaded after split filtering: {len(df):,}")

    return df


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def resolve_image_path(
    row: pd.Series,
    filename_column: str,
) -> Optional[Path]:
    """
    Resolve an image path from one master-index row.

    Expected physical dataset structure:

    Dataset/
      Dataset_Shard_x/
        train/
          Img_HR/
        val/
          Img_HR/
        test/
          Img_HR/

    The master index stores only filenames, so the shard/split
    information is used to reconstruct the path.
    """

    filename = row.get(filename_column)

    if pd.isna(filename):
        return None

    filename = str(filename)

    if not filename:
        return None

    shard = str(row["shard"])
    split = str(row["split"])

    candidates = [
        PROJECT_ROOT / "Dataset" / shard / split,
        PROJECT_ROOT / "Dataset" / shard,
        PROJECT_ROOT,
    ]

    modality = filename_column.replace("_filename", "")

    for base in candidates:

        direct = base / modality / filename

        if direct.exists():
            return direct

    # Fallback:
    # Search only inside the corresponding shard.
    shard_root = PROJECT_ROOT / "Dataset" / shard

    if shard_root.exists():

        for path in shard_root.rglob(filename):
            return path

    return None


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_rgb_image(
    path: Path,
    image_size: int,
) -> np.ndarray:

    image = Image.open(path).convert("RGB")

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


def load_mask(
    path: Optional[Path],
    image_size: int,
) -> Optional[np.ndarray]:

    if path is None or not path.exists():
        return None

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST,
    )

    array = np.asarray(
        mask,
        dtype=np.float32,
    )

    array /= 255.0

    return array > 0.5


# =============================================================================
# DIFFERENCE METRICS
# =============================================================================

def compute_difference(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:

    difference = np.abs(image_a - image_b)

    # Average over RGB channels.
    pixel_difference = np.mean(
        difference,
        axis=2,
    )

    squared_difference = np.square(
        image_a - image_b
    )

    pixel_mse = np.mean(
        squared_difference,
        axis=2,
    )

    if mask is not None:

        valid = mask.astype(bool)

        if np.any(valid):

            mae = float(
                np.mean(pixel_difference[valid])
            )

            mse = float(
                np.mean(pixel_mse[valid])
            )

            changed_fraction = float(
                np.mean(
                    pixel_difference[valid] > 0.05
                )
            )

            return {
                "mae": mae,
                "mse": mse,
                "rmse": float(math.sqrt(mse)),
                "changed_fraction": changed_fraction,
                "valid_pixels": int(np.sum(valid)),
            }

    mae = float(np.mean(pixel_difference))

    mse = float(np.mean(pixel_mse))

    changed_fraction = float(
        np.mean(pixel_difference > 0.05)
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "changed_fraction": changed_fraction,
        "valid_pixels": int(pixel_difference.size),
    }


# =============================================================================
# PAIR SAMPLING
# =============================================================================

def sample_pairs_for_video(
    frame_ids: np.ndarray,
    lag: int,
    max_pairs: int,
    sampling: str,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:

    frame_ids = np.sort(
        np.asarray(frame_ids, dtype=int)
    )

    if len(frame_ids) <= lag:
        return []

    frame_set = set(frame_ids.tolist())

    candidates = []

    for frame_id in frame_ids:

        target = int(frame_id + lag)

        if target in frame_set:
            candidates.append(
                (int(frame_id), target)
            )

    if sampling == "all":
        return candidates

    if len(candidates) <= max_pairs:
        return candidates

    indices = rng.choice(
        len(candidates),
        size=max_pairs,
        replace=False,
    )

    indices = np.sort(indices)

    return [
        candidates[int(i)]
        for i in indices
    ]


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def process_video(
    video_df: pd.DataFrame,
    lag: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[Dict], int]:

    video_df = video_df.sort_values(
        "frame_id"
    )

    frame_lookup = {
        int(row.frame_id): row
        for row in video_df.itertuples(index=False)
    }

    frame_ids = np.array(
        sorted(frame_lookup.keys()),
        dtype=int,
    )

    pairs = sample_pairs_for_video(
        frame_ids=frame_ids,
        lag=lag,
        max_pairs=args.max_pairs_per_video,
        sampling=args.sampling,
        rng=rng,
    )

    results = []
    failed = 0

    for frame_a, frame_b in pairs:

        row_a = frame_lookup[frame_a]
        row_b = frame_lookup[frame_b]

        row_a = pd.Series(row_a._asdict())
        row_b = pd.Series(row_b._asdict())

        path_a = resolve_image_path(
            row_a,
            "Img_HR_filename",
        )

        path_b = resolve_image_path(
            row_b,
            "Img_HR_filename",
        )

        if path_a is None or path_b is None:
            failed += 1
            continue

        try:

            image_a = load_rgb_image(
                path_a,
                args.image_size,
            )

            image_b = load_rgb_image(
                path_b,
                args.image_size,
            )

        except Exception:
            failed += 1
            continue

        result = {
            "shard": str(row_a["shard"]),
            "split": str(row_a["split"]),
            "video_id": str(row_a["video_id"]),
            "frame_id_a": frame_a,
            "frame_id_b": frame_b,
            "temporal_distance": lag,
        }

        # ---------------------------------------------------------------------
        # FULL IMAGE
        # ---------------------------------------------------------------------

        full = compute_difference(
            image_a,
            image_b,
            mask=None,
        )

        result.update({
            "full_mae": full["mae"],
            "full_mse": full["mse"],
            "full_rmse": full["rmse"],
            "full_changed_fraction": full["changed_fraction"],
        })

        # ---------------------------------------------------------------------
        # HUMAN MASK
        # ---------------------------------------------------------------------

        mask_a_path = resolve_image_path(
            row_a,
            "Mask_Full_HR_filename",
        )

        mask_b_path = resolve_image_path(
            row_b,
            "Mask_Full_HR_filename",
        )

        mask_a = load_mask(
            mask_a_path,
            args.image_size,
        )

        mask_b = load_mask(
            mask_b_path,
            args.image_size,
        )

        full_mask = None

        if mask_a is not None and mask_b is not None:
            full_mask = mask_a | mask_b

        human = compute_difference(
            image_a,
            image_b,
            mask=full_mask,
        )

        result.update({
            "human_mae": human["mae"],
            "human_mse": human["mse"],
            "human_rmse": human["rmse"],
            "human_changed_fraction": human["changed_fraction"],
            "human_valid_pixels": human["valid_pixels"],
        })

        # ---------------------------------------------------------------------
        # CLOTH MASK
        # ---------------------------------------------------------------------

        cloth_a_path = resolve_image_path(
            row_a,
            "Mask_Cloth_HR_filename",
        )

        cloth_b_path = resolve_image_path(
            row_b,
            "Mask_Cloth_HR_filename",
        )

        cloth_a = load_mask(
            cloth_a_path,
            args.image_size,
        )

        cloth_b = load_mask(
            cloth_b_path,
            args.image_size,
        )

        cloth_mask = None

        if cloth_a is not None and cloth_b is not None:
            cloth_mask = cloth_a | cloth_b

        cloth = compute_difference(
            image_a,
            image_b,
            mask=cloth_mask,
        )

        result.update({
            "cloth_mae": cloth["mae"],
            "cloth_mse": cloth["mse"],
            "cloth_rmse": cloth["rmse"],
            "cloth_changed_fraction": cloth["changed_fraction"],
            "cloth_valid_pixels": cloth["valid_pixels"],
        })

        # ---------------------------------------------------------------------
        # SKIN MASK
        # ---------------------------------------------------------------------

        skin_a_path = resolve_image_path(
            row_a,
            "Mask_Skin_HR_filename",
        )

        skin_b_path = resolve_image_path(
            row_b,
            "Mask_Skin_HR_filename",
        )

        skin_a = load_mask(
            skin_a_path,
            args.image_size,
        )

        skin_b = load_mask(
            skin_b_path,
            args.image_size,
        )

        skin_mask = None

        if skin_a is not None and skin_b is not None:
            skin_mask = skin_a | skin_b

        skin = compute_difference(
            image_a,
            image_b,
            mask=skin_mask,
        )

        result.update({
            "skin_mae": skin["mae"],
            "skin_mse": skin["mse"],
            "skin_rmse": skin["rmse"],
            "skin_changed_fraction": skin["changed_fraction"],
            "skin_valid_pixels": skin["valid_pixels"],
        })

        results.append(result)

    return results, failed


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_statistics(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:

    if pair_df.empty:
        return pd.DataFrame()

    grouped = (
        pair_df
        .groupby(
            [
                "temporal_distance",
                "split",
            ],
            dropna=False,
        )
        .agg(
            pairs=("video_id", "size"),
            videos=("video_id", "nunique"),

            mean_full_mae=("full_mae", "mean"),
            median_full_mae=("full_mae", "median"),
            p10_full_mae=("full_mae", lambda x: x.quantile(0.10)),
            p90_full_mae=("full_mae", lambda x: x.quantile(0.90)),

            mean_human_mae=("human_mae", "mean"),
            median_human_mae=("human_mae", "median"),

            mean_cloth_mae=("cloth_mae", "mean"),
            median_cloth_mae=("cloth_mae", "median"),

            mean_skin_mae=("skin_mae", "mean"),
            median_skin_mae=("skin_mae", "median"),

            mean_full_changed_fraction=(
                "full_changed_fraction",
                "mean",
            ),

            mean_human_changed_fraction=(
                "human_changed_fraction",
                "mean",
            ),

            mean_cloth_changed_fraction=(
                "cloth_changed_fraction",
                "mean",
            ),

            mean_skin_changed_fraction=(
                "skin_changed_fraction",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "temporal_distance",
                "split",
            ]
        )
    )

    return grouped


def aggregate_video_statistics(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:

    if pair_df.empty:
        return pd.DataFrame()

    result = (
        pair_df
        .groupby(
            [
                "shard",
                "split",
                "video_id",
                "temporal_distance",
            ],
            dropna=False,
        )
        .agg(
            pairs=("frame_id_a", "size"),

            mean_full_mae=("full_mae", "mean"),
            median_full_mae=("full_mae", "median"),

            mean_human_mae=("human_mae", "mean"),
            mean_cloth_mae=("cloth_mae", "mean"),
            mean_skin_mae=("skin_mae", "mean"),

            mean_full_changed_fraction=(
                "full_changed_fraction",
                "mean",
            ),

            mean_human_changed_fraction=(
                "human_changed_fraction",
                "mean",
            ),

            mean_cloth_changed_fraction=(
                "cloth_changed_fraction",
                "mean",
            ),

            mean_skin_changed_fraction=(
                "skin_changed_fraction",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "video_id",
                "temporal_distance",
            ]
        )
    )

    return result


# =============================================================================
# FIGURES
# =============================================================================

def save_figures(
    summary_df: pd.DataFrame,
    video_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if summary_df.empty:
        return

    # -------------------------------------------------------------------------
    # 1. Difference vs temporal distance
    # -------------------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    for split in sorted(
        summary_df["split"].unique()
    ):

        subset = summary_df[
            summary_df["split"] == split
        ]

        plt.plot(
            subset["temporal_distance"],
            subset["mean_full_mae"],
            marker="o",
            label=split,
        )

    plt.xlabel("Temporal distance (frames)")
    plt.ylabel("Mean absolute difference")
    plt.title("Temporal Difference vs Temporal Distance")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "difference_by_temporal_distance.png",
        dpi=180,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # 2. Difference by split
    # -------------------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    distances = sorted(
        summary_df["temporal_distance"].unique()
    )

    for distance in distances:

        subset = summary_df[
            summary_df["temporal_distance"] == distance
        ]

        plt.plot(
            subset["split"],
            subset["mean_full_mae"],
            marker="o",
            label=f"lag={distance}",
        )

    plt.xlabel("Dataset split")
    plt.ylabel("Mean absolute difference")
    plt.title("Temporal Difference by Dataset Split")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_dir / "difference_by_split.png",
        dpi=180,
    )

    plt.close()

    # -------------------------------------------------------------------------
    # 3. Difference distribution
    # -------------------------------------------------------------------------

    if not video_df.empty:

        plt.figure(figsize=(9, 6))

        for distance in sorted(
            video_df["temporal_distance"].unique()
        ):

            subset = video_df[
                video_df["temporal_distance"] == distance
            ]

            values = subset[
                "mean_full_mae"
            ].dropna()

            if len(values) == 0:
                continue

            plt.hist(
                values,
                bins=30,
                alpha=0.45,
                label=f"lag={distance}",
            )

        plt.xlabel("Video mean MAE")
        plt.ylabel("Number of videos")
        plt.title("Distribution of Video-level Temporal Difference")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_dir / "difference_distribution.png",
            dpi=180,
        )

        plt.close()

    # -------------------------------------------------------------------------
    # 4. Human vs full image
    # -------------------------------------------------------------------------

    human_available = (
        "mean_human_mae" in summary_df.columns
    )

    if human_available:

        plt.figure(figsize=(9, 6))

        for split in sorted(
            summary_df["split"].unique()
        ):

            subset = summary_df[
                summary_df["split"] == split
            ]

            plt.plot(
                subset["temporal_distance"],
                subset["mean_full_mae"],
                marker="o",
                label=f"{split} - full",
            )

            plt.plot(
                subset["temporal_distance"],
                subset["mean_human_mae"],
                marker="s",
                linestyle="--",
                label=f"{split} - human",
            )

        plt.xlabel("Temporal distance (frames)")
        plt.ylabel("Mean absolute difference")
        plt.title("Full Image vs Human-region Temporal Difference")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_dir / "full_vs_human_difference.png",
            dpi=180,
        )

        plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main() -> None:

    start_time = time.time()

    args = parse_args()

    np.random.seed(args.seed)

    rng = np.random.default_rng(
        args.seed
    )

    print_header(
        "TEMPORAL DIFFERENCE ANALYSIS"
    )

    print("PhySense-Human Research Pipeline")
    print(
        "Stage: 02_temporal_redundancy / "
        "03_temporal_difference"
    )

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Master index:")
    print(f"  {MASTER_INDEX}")

    print()
    print("Output root:")
    print(f"  {OUTPUT_ROOT}")

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
        f"Sampling mode: {args.sampling}"
    )

    print(
        f"Maximum pairs/video/lag: "
        f"{args.max_pairs_per_video}"
    )

    print(
        f"Splits: {args.split}"
    )

    print_section(
        "Loading master dataset index"
    )

    df = load_master_index(
        MASTER_INDEX,
        args.split,
    )

    if df.empty:
        raise RuntimeError(
            "No rows available after split filtering."
        )

    print_section(
        "Validating dataset"
    )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "video_id",
                "frame_id",
            ]
        ).sum()
    )

    print(
        "Duplicate (video_id, frame_id) records: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:
        raise RuntimeError(
            "Duplicate temporal identities detected."
        )

    print(
        f"Frames: {len(df):,}"
    )

    print(
        f"Videos: {df['video_id'].nunique():,}"
    )

    print(
        f"Shards: {df['shard'].nunique():,}"
    )

    print()
    print("Splits:")

    split_counts = (
        df["split"]
        .value_counts()
        .sort_index()
    )

    for split, count in split_counts.items():
        print(
            f"  {split}: {count:,} frames"
        )

    print()
    print("Shards:")

    shard_counts = (
        df["shard"]
        .value_counts()
        .sort_index()
    )

    for shard, count in shard_counts.items():
        print(
            f"  {shard}: {count:,} frames"
        )

    # -------------------------------------------------------------------------
    # Video groups
    # -------------------------------------------------------------------------

    grouped_videos = list(
        df.groupby(
            [
                "shard",
                "split",
                "video_id",
            ],
            sort=True,
        )
    )

    if args.max_videos is not None:

        grouped_videos = grouped_videos[
            :args.max_videos
        ]

    print_section(
        "Computing temporal image differences"
    )

    print(
        f"Videos available: "
        f"{len(grouped_videos):,}"
    )

    all_results = []

    total_failed = 0

    total_candidates = 0

    processed = 0

    last_report = time.time()

    for (
        (
            shard,
            split,
            video_id
        ),
        video_df,
    ) in grouped_videos:

        for lag in args.lags:

            candidates = sample_pairs_for_video(
                frame_ids=video_df["frame_id"].values,
                lag=lag,
                max_pairs=args.max_pairs_per_video,
                sampling=args.sampling,
                rng=rng,
            )

            total_candidates += len(
                candidates
            )

            if not candidates:
                continue

            results, failed = process_video(
                video_df=video_df,
                lag=lag,
                args=args,
                rng=rng,
            )

            total_failed += failed

            all_results.extend(
                results
            )

        processed += 1

        now = time.time()

        if (
            processed == 1
            or processed % 10 == 0
            or now - last_report > 60
        ):

            elapsed = (
                now - start_time
            )

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"Videos processed: "
                f"{processed:,} / "
                f"{len(grouped_videos):,} | "
                f"Successful comparisons: "
                f"{len(all_results):,} | "
                f"Rate: {rate:.2f} videos/s"
            )

            last_report = now

    print()
    print(
        f"Videos processed: "
        f"{processed:,}"
    )

    print(
        f"Candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Successful comparisons: "
        f"{len(all_results):,}"
    )

    print(
        f"Failed comparisons: "
        f"{total_failed:,}"
    )

    if not all_results:
        raise RuntimeError(
            "No successful temporal-difference comparisons."
        )

    pair_df = pd.DataFrame(
        all_results
    )

    print_section(
        "Aggregating difference statistics"
    )

    summary_df = aggregate_statistics(
        pair_df
    )

    video_statistics_df = (
        aggregate_video_statistics(
            pair_df
        )
    )

    # -------------------------------------------------------------------------
    # Output directories
    # -------------------------------------------------------------------------

    STATS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section(
        "Saving CSV results"
    )

    summary_path = (
        STATS_DIR
        / "difference_summary.csv"
    )

    video_path = (
        STATS_DIR
        / "video_difference_statistics.csv"
    )

    errors_path = (
        STATS_DIR
        / "difference_errors.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    video_statistics_df.to_csv(
        video_path,
        index=False,
    )

    pd.DataFrame(
        columns=[
            "error_type",
            "count",
        ]
    ).to_csv(
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

    # -------------------------------------------------------------------------
    # Optional pair-level CSV
    # -------------------------------------------------------------------------

    pair_path = None

    if args.save_pairs:

        pair_path = (
            STATS_DIR
            / "pair_difference_statistics.csv"
        )

        pair_df.to_csv(
            pair_path,
            index=False,
        )

        print(
            f"Saved detailed pair CSV: "
            f"{pair_path}"
        )

    else:

        print(
            "Detailed pair CSV not saved "
            "(use --save-pairs if needed)."
        )

    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------

    json_summary = {
        "analysis": "temporal_difference",
        "project": "PhySense-Human",
        "stage": "02_temporal_redundancy/03_temporal_difference",

        "project_root": str(
            PROJECT_ROOT
        ),

        "master_index": str(
            MASTER_INDEX
        ),

        "frames_available": int(
            len(df)
        ),

        "videos_available": int(
            df["video_id"].nunique()
        ),

        "shards": sorted(
            df["shard"].unique().tolist()
        ),

        "splits": sorted(
            df["split"].unique().tolist()
        ),

        "temporal_distances": args.lags,

        "sampling_mode": args.sampling,

        "max_pairs_per_video": (
            args.max_pairs_per_video
        ),

        "image_size": args.image_size,

        "candidate_pairs": (
            total_candidates
        ),

        "successful_comparisons": (
            len(all_results)
        ),

        "failed_comparisons": (
            total_failed
        ),

        "mask_regions": [
            "full",
            "human",
            "cloth",
            "skin",
        ],

        "summary": json_safe(
            summary_df.to_dict(
                orient="records"
            )
        ),
    }

    json_path = (
        STATS_DIR
        / "difference_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved: {json_path}"
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print_section(
        "Generating figures"
    )

    save_figures(
        summary_df,
        video_statistics_df,
        FIGURES_DIR,
    )

    print(
        f"Figures saved to: "
        f"{FIGURES_DIR}"
    )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    elapsed = (
        time.time() - start_time
    )

    print_header(
        "FINAL TEMPORAL DIFFERENCE REPORT"
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
        f"{args.sampling}"
    )

    print(
        f"Successful comparisons: "
        f"{len(all_results):,}"
    )

    print(
        f"Failed comparisons: "
        f"{total_failed:,}"
    )

    print()
    print(
        "Difference summary:"
    )

    display_columns = [
        "temporal_distance",
        "split",
        "pairs",
        "videos",
        "mean_full_mae",
        "median_full_mae",
        "mean_human_mae",
        "mean_cloth_mae",
        "mean_skin_mae",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in summary_df.columns
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
        "CORE TEMPORAL DIFFERENCE ANALYSIS: COMPLETE"
    )

    print()
    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        "Results:"
    )

    print(
        f"  {STATS_DIR}"
    )

    print(
        f"  {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()