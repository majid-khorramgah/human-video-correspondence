from pathlib import Path
from collections import defaultdict
import re
import json
import time

import pandas as pd
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# Parent directory containing all Dataset_Shard_* folders.
#
# Example:
#
# F:\PhySense-Human\
# ├── Dataset_Shard_0000000_0099999
# ├── Dataset_Shard_0100000_0199999
# ├── Dataset_Shard_0200000_0299999
# ├── Dataset_Shard_0300000_0399999
# ├── Dataset_Shard_0400000_0499999
# └── Dataset_Shard_0500000_0599999

DATA_ROOT = Path(r"F:\PhySense-Human")


# Results will automatically be placed beside this script:
#
# F:\PhySense-Human\01_dataset_audit\
# ├── dataset_integrity.py
# └── results\
#     └── integrity\

OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "results"
    / "integrity"
)


SPLITS = [
    "train",
    "val",
    "test",
]


MODALITIES = [
    "Img_HR",
    "Img_LR",
    "Mask_Cloth_HR",
    "Mask_Cloth_LR",
    "Mask_Full_HR",
    "Mask_Full_LR",
    "Mask_Skin_HR",
    "Mask_Skin_LR",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# FILENAME PARSER
# ============================================================

# Expected:
#
# frame_0000033_-Hs-zuBOlQE.jpg
#
# frame ID:
#     33
#
# video ID:
#     -Hs-zuBOlQE
#
# IMPORTANT:
# Extension is deliberately NOT part of the identity.

FILENAME_PATTERN = re.compile(
    r"^frame_(\d+)_([^./]+)\.[^.]+$",
    re.IGNORECASE,
)


def parse_filename(filename):
    """
    Parse frame filename.

    Returns:
        frame_id
        video_id
        parse_ok
    """

    match = FILENAME_PATTERN.match(filename)

    if match is None:
        return {
            "frame_id": None,
            "video_id": None,
            "parse_ok": False,
        }

    return {
        "frame_id": int(match.group(1)),
        "video_id": match.group(2),
        "parse_ok": True,
    }


def make_frame_key(filename):
    """
    Create extension-independent frame identity.

    Identity:
        (video_id, frame_id)

    Therefore:

        frame_0000033_VIDEO.jpg
        frame_0000033_VIDEO.png

    are considered the same frame.
    """

    parsed = parse_filename(filename)

    if not parsed["parse_ok"]:
        return None

    return (
        parsed["video_id"],
        parsed["frame_id"],
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def list_image_files(directory):
    """
    Return all image files inside directory.
    """

    if not directory.exists():
        return []

    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )


def build_frame_index(directory):
    """
    Build:

        (video_id, frame_id) -> file path

    Extension is ignored.

    This is the central fix compared with the previous version.
    """

    index = {}

    files = list_image_files(directory)

    for path in files:

        key = make_frame_key(path.name)

        if key is None:
            continue

        # If two files inside the same modality have
        # exactly the same video/frame identity,
        # we keep both for duplicate detection later.

        if key not in index:
            index[key] = []

        index[key].append(path)

    return index


# ============================================================
# IMAGE VALIDATION
# ============================================================

def inspect_image(path):
    """
    Try to open an image and obtain basic properties.

    This does NOT load the entire image into memory.
    """

    try:

        with Image.open(path) as img:

            width, height = img.size
            mode = img.mode
            image_format = img.format

            # Verify image integrity.
            img.verify()

        return {
            "readable": True,
            "width": width,
            "height": height,
            "mode": mode,
            "format": image_format,
            "error": None,
        }

    except Exception as exc:

        return {
            "readable": False,
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "error": str(exc),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("DATASET INTEGRITY AUDIT - V2")
    print("=" * 70)

    print()
    print("Dataset root:")
    print(DATA_ROOT.resolve())

    print()
    print("Output root:")
    print(OUTPUT_ROOT.resolve())

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # VALIDATE DATA ROOT
    # ========================================================

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"DATA_ROOT does not exist:\n{DATA_ROOT}"
        )

    # ========================================================
    # DISCOVER SHARDS
    # ========================================================

    print()
    print("Discovering dataset shards...")

    shard_paths = sorted(
        [
            p
            for p in DATA_ROOT.iterdir()
            if p.is_dir()
            and p.name.startswith("Dataset_Shard_")
        ]
    )

    print(
        f"Shards discovered: {len(shard_paths)}"
    )

    for shard in shard_paths:

        print(
            f"  - {shard.name}"
        )

    if len(shard_paths) == 0:

        raise RuntimeError(
            "No Dataset_Shard_* directories found."
        )

    # ========================================================
    # SPLIT DIRECTORY CHECK
    # ========================================================

    print()
    print("Checking train / val / test directories...")

    missing_split_dirs = []

    for shard in shard_paths:

        for split in SPLITS:

            split_path = shard / split

            if not split_path.exists():

                missing_split_dirs.append(
                    {
                        "shard": shard.name,
                        "split": split,
                        "path": str(split_path),
                    }
                )

    if missing_split_dirs:

        print()
        print(
            "WARNING: Missing split directories:"
        )

        for item in missing_split_dirs:

            print(
                f"  {item['shard']} / "
                f"{item['split']}"
            )

    else:

        print(
            "All shards contain train / val / test."
        )

    # ========================================================
    # MODALITY DIRECTORY CHECK
    # ========================================================

    print()
    print("Checking modality directories...")

    missing_modalities = []

    for shard in shard_paths:

        for split in SPLITS:

            for modality in MODALITIES:

                path = (
                    shard
                    / split
                    / modality
                )

                if not path.exists():

                    missing_modalities.append(
                        {
                            "shard": shard.name,
                            "split": split,
                            "modality": modality,
                            "path": str(path),
                        }
                    )

    if missing_modalities:

        print()
        print(
            "WARNING: Missing modality directories:"
        )

        for item in missing_modalities:

            print(
                f"  {item['shard']} / "
                f"{item['split']} / "
                f"{item['modality']}"
            )

    else:

        print(
            "All expected modality directories exist."
        )

    # ========================================================
    # BUILD MASTER INDEX
    # ========================================================

    print()
    print(
        "Building extension-independent master index..."
    )

    master_records = []

    # Duplicate identities inside individual modalities
    duplicate_file_records = []

    # All files that cannot be parsed
    parse_error_records = []

    for shard in shard_paths:

        print()
        print(
            f"Processing: {shard.name}"
        )

        for split in SPLITS:

            split_path = (
                shard / split
            )

            modality_indexes = {}

            for modality in MODALITIES:

                modality_path = (
                    split_path / modality
                )

                modality_indexes[
                    modality
                ] = build_frame_index(
                    modality_path
                )

            # ------------------------------------------------
            # Use Img_HR as primary frame universe
            # ------------------------------------------------

            primary_index = (
                modality_indexes[
                    "Img_HR"
                ]
            )

            print(
                f"  {split}: "
                f"{sum(len(v) for v in primary_index.values()):,} "
                f"HR files"
            )

            # ------------------------------------------------
            # Detect duplicate identities in every modality
            # ------------------------------------------------

            for modality in MODALITIES:

                index = modality_indexes[
                    modality
                ]

                for key, paths in index.items():

                    if len(paths) > 1:

                        video_id, frame_id = key

                        for path in paths:

                            duplicate_file_records.append(
                                {
                                    "shard":
                                        shard.name,

                                    "split":
                                        split,

                                    "modality":
                                        modality,

                                    "video_id":
                                        video_id,

                                    "frame_id":
                                        frame_id,

                                    "filename":
                                        path.name,
                                }
                            )

            # ------------------------------------------------
            # Parse all files
            # ------------------------------------------------

            for modality in MODALITIES:

                modality_path = (
                    split_path / modality
                )

                for path in list_image_files(
                    modality_path
                ):

                    parsed = parse_filename(
                        path.name
                    )

                    if not parsed[
                        "parse_ok"
                    ]:

                        parse_error_records.append(
                            {
                                "shard":
                                    shard.name,

                                "split":
                                    split,

                                "modality":
                                    modality,

                                "filename":
                                    path.name,
                            }
                        )

            # ------------------------------------------------
            # Create records from Img_HR
            # ------------------------------------------------

            for key, hr_paths in (
                primary_index.items()
            ):

                video_id, frame_id = key

                # Usually exactly one HR file.
                # If multiple exist, record the first
                # for the master index and report duplicates
                # separately.

                hr_path = hr_paths[0]

                record = {

                    "shard":
                        shard.name,

                    "split":
                        split,

                    "video_id":
                        video_id,

                    "frame_id":
                        frame_id,

                    "hr_filename":
                        hr_path.name,

                    "hr_extension":
                        hr_path.suffix.lower(),
                }

                # ------------------------------------------------
                # Check every modality by frame identity
                # rather than extension.
                # ------------------------------------------------

                for modality in MODALITIES:

                    index = modality_indexes[
                        modality
                    ]

                    paths = index.get(
                        key,
                        []
                    )

                    record[
                        f"has_{modality}"
                    ] = (
                        len(paths) > 0
                    )

                    record[
                        f"{modality}_count"
                    ] = len(paths)

                    if paths:

                        record[
                            f"{modality}_filename"
                        ] = paths[0].name

                        record[
                            f"{modality}_extension"
                        ] = (
                            paths[0]
                            .suffix
                            .lower()
                        )

                    else:

                        record[
                            f"{modality}_filename"
                        ] = None

                        record[
                            f"{modality}_extension"
                        ] = None

                master_records.append(
                    record
                )

    master_df = pd.DataFrame(
        master_records
    )

    print()
    print(
        f"Master index created: "
        f"{len(master_df):,} rows"
    )

    # ========================================================
    # PARSE ERRORS
    # ========================================================

    parse_errors_df = pd.DataFrame(
        parse_error_records
    )

    print()
    print(
        "Filename parsing errors: "
        f"{len(parse_errors_df):,}"
    )

    # ========================================================
    # MISSING CORRESPONDING FILES
    # ========================================================

    print()
    print(
        "Checking cross-modality correspondence..."
    )

    missing_records = []

    for _, row in master_df.iterrows():

        missing = []

        for modality in MODALITIES:

            if not row[
                f"has_{modality}"
            ]:

                missing.append(
                    modality
                )

        if missing:

            missing_records.append(
                {
                    "shard":
                        row["shard"],

                    "split":
                        row["split"],

                    "video_id":
                        row["video_id"],

                    "frame_id":
                        row["frame_id"],

                    "hr_filename":
                        row["hr_filename"],

                    "missing_modalities":
                        ";".join(
                            missing
                        ),
                }
            )

    missing_df = pd.DataFrame(
        missing_records
    )

    print(
        f"Frames with missing corresponding files: "
        f"{len(missing_df):,}"
    )

    # ========================================================
    # FILES NOT PRESENT IN IMG_HR
    # ========================================================

    print()
    print(
        "Checking modality files that have no matching HR frame..."
    )

    unexpected_records = []

    for shard in shard_paths:

        for split in SPLITS:

            split_path = (
                shard / split
            )

            hr_index = build_frame_index(
                split_path / "Img_HR"
            )

            hr_keys = set(
                hr_index.keys()
            )

            for modality in MODALITIES:

                if modality == "Img_HR":
                    continue

                modality_index = (
                    build_frame_index(
                        split_path
                        / modality
                    )
                )

                for key, paths in (
                    modality_index.items()
                ):

                    if key not in hr_keys:

                        video_id, frame_id = key

                        for path in paths:

                            unexpected_records.append(
                                {
                                    "shard":
                                        shard.name,

                                    "split":
                                        split,

                                    "modality":
                                        modality,

                                    "video_id":
                                        video_id,

                                    "frame_id":
                                        frame_id,

                                    "filename":
                                        path.name,

                                    "reason":
                                        "not_present_in_Img_HR",
                                }
                            )

    unexpected_df = pd.DataFrame(
        unexpected_records
    )

    print(
        f"Files without matching HR frame: "
        f"{len(unexpected_df):,}"
    )

    # ========================================================
    # DUPLICATES
    # ========================================================

    duplicate_df = pd.DataFrame(
        duplicate_file_records
    )

    print()
    print(
        "Duplicate frame identities:"
        f" {len(duplicate_df):,}"
    )

    # ========================================================
    # FRAME-LEVEL SPLIT OVERLAP
    # ========================================================

    print()
    print(
        "Checking frame-level split leakage..."
    )

    split_frame_sets = {}

    for split in SPLITS:

        subset = master_df[
            master_df["split"]
            == split
        ]

        split_frame_sets[
            split
        ] = set(
            zip(
                subset["video_id"],
                subset["frame_id"],
            )
        )

    frame_overlap_records = []

    for i in range(len(SPLITS)):

        for j in range(
            i + 1,
            len(SPLITS)
        ):

            split_a = SPLITS[i]
            split_b = SPLITS[j]

            overlap = (
                split_frame_sets[
                    split_a
                ]
                &
                split_frame_sets[
                    split_b
                ]
            )

            frame_overlap_records.append(
                {
                    "split_a":
                        split_a,

                    "split_b":
                        split_b,

                    "overlap_count":
                        len(overlap),
                }
            )

    frame_overlap_df = pd.DataFrame(
        frame_overlap_records
    )

    print(
        frame_overlap_df.to_string(
            index=False
        )
    )

    # ========================================================
    # VIDEO-LEVEL SPLIT OVERLAP
    # ========================================================

    print()
    print(
        "Checking VIDEO-LEVEL split leakage..."
    )

    split_video_sets = {}

    for split in SPLITS:

        subset = master_df[
            master_df["split"]
            == split
        ]

        split_video_sets[
            split
        ] = set(
            subset[
                "video_id"
            ].dropna()
        )

    video_overlap_records = []

    for i in range(len(SPLITS)):

        for j in range(
            i + 1,
            len(SPLITS)
        ):

            split_a = SPLITS[i]
            split_b = SPLITS[j]

            overlap = (
                split_video_sets[
                    split_a
                ]
                &
                split_video_sets[
                    split_b
                ]
            )

            for video_id in sorted(
                overlap
            ):

                video_overlap_records.append(
                    {
                        "split_a":
                            split_a,

                        "split_b":
                            split_b,

                        "video_id":
                            video_id,
                    }
                )

    video_overlap_df = pd.DataFrame(
        video_overlap_records
    )

    print(
        f"Video-level leakage: "
        f"{len(video_overlap_df):,}"
    )

    if len(
        video_overlap_df
    ) == 0:

        print(
            "PASS: No video-level overlap detected."
        )

    # ========================================================
    # CROSS-SHARD VIDEO CONTINUITY
    # ========================================================

    print()
    print(
        "Checking videos spanning multiple shards..."
    )

    video_to_shards = defaultdict(set)

    for _, row in master_df.iterrows():

        video_to_shards[
            row["video_id"]
        ].add(
            row["shard"]
        )

    cross_shard_records = []

    for video_id, shards in (
        video_to_shards.items()
    ):

        if len(shards) > 1:

            cross_shard_records.append(
                {
                    "video_id":
                        video_id,

                    "num_shards":
                        len(shards),

                    "shards":
                        ";".join(
                            sorted(shards)
                        ),
                }
            )

    cross_shard_df = pd.DataFrame(
        cross_shard_records
    )

    print(
        f"Videos spanning multiple shards: "
        f"{len(cross_shard_df):,}"
    )

    # ========================================================
    # DATASET STRUCTURE
    # ========================================================

    print()
    print(
        "Computing dataset structure..."
    )

    structure_records = []

    for shard in shard_paths:

        for split in SPLITS:

            subset = master_df[
                (master_df["shard"]
                 == shard.name)
                &
                (master_df["split"]
                 == split)
            ]

            structure_records.append(
                {
                    "shard":
                        shard.name,

                    "split":
                        split,

                    "frames":
                        len(subset),

                    "videos":
                        subset[
                            "video_id"
                        ].nunique(),
                }
            )

    structure_df = pd.DataFrame(
        structure_records
    )

    print()
    print(
        structure_df.to_string(
            index=False
        )
    )

    # ========================================================
    # VIDEO SEQUENCE CHECK
    # ========================================================

    print()
    print(
        "Checking temporal frame sequences..."
    )

    sequence_records = []

    for (
        shard,
        split,
        video_id,
    ), group in master_df.groupby(
        [
            "shard",
            "split",
            "video_id",
        ]
    ):

        frame_ids = sorted(
            group[
                "frame_id"
            ]
            .astype(int)
            .tolist()
        )

        if not frame_ids:
            continue

        diffs = np.diff(
            frame_ids
        )

        sequence_records.append(
            {
                "shard":
                    shard,

                "split":
                    split,

                "video_id":
                    video_id,

                "num_frames":
                    len(frame_ids),

                "min_frame_id":
                    min(frame_ids),

                "max_frame_id":
                    max(frame_ids),

                "num_gaps":
                    int(
                        np.sum(
                            diffs > 1
                        )
                    ),

                "max_gap":
                    int(
                        max(diffs)
                    )
                    if len(diffs)
                    else 0,

                "monotonic":
                    bool(
                        np.all(
                            diffs > 0
                        )
                    )
                    if len(diffs)
                    else True,
            }
        )

    sequence_df = pd.DataFrame(
        sequence_records
    )

    videos_with_gaps = int(
        (
            sequence_df["num_gaps"]
            > 0
        ).sum()
    )

    print(
        f"Videos with frame-ID gaps: "
        f"{videos_with_gaps:,}"
    )

    # ========================================================
    # IMAGE READABILITY AUDIT
    # ========================================================

    print()
    print(
        "Checking image readability..."
    )

    # We check every unique HR frame.
    # The full 8-modality pixel audit can be added later,
    # but HR is the canonical image universe for this stage.

    unreadable_records = []

    total_hr_checked = 0

    for shard in shard_paths:

        for split in SPLITS:

            hr_path = (
                shard
                / split
                / "Img_HR"
            )

            files = list_image_files(
                hr_path
            )

            for path in files:

                total_hr_checked += 1

                info = inspect_image(
                    path
                )

                if not info[
                    "readable"
                ]:

                    parsed = (
                        parse_filename(
                            path.name
                        )
                    )

                    unreadable_records.append(
                        {
                            "shard":
                                shard.name,

                            "split":
                                split,

                            "filename":
                                path.name,

                            "video_id":
                                parsed[
                                    "video_id"
                                ],

                            "frame_id":
                                parsed[
                                    "frame_id"
                                ],

                            "error":
                                info[
                                    "error"
                                ],
                        }
                    )

    unreadable_df = pd.DataFrame(
        unreadable_records
    )

    print(
        f"HR images checked: "
        f"{total_hr_checked:,}"
    )

    print(
        f"Unreadable HR images: "
        f"{len(unreadable_df):,}"
    )

    # ========================================================
    # MODALITY COUNTS
    # ========================================================

    print()
    print(
        "Counting files per modality..."
    )

    count_records = []

    for shard in shard_paths:

        for split in SPLITS:

            for modality in MODALITIES:

                path = (
                    shard
                    / split
                    / modality
                )

                count = len(
                    list_image_files(
                        path
                    )
                )

                count_records.append(
                    {
                        "shard":
                            shard.name,

                        "split":
                            split,

                        "modality":
                            modality,

                        "file_count":
                            count,
                    }
                )

    counts_df = pd.DataFrame(
        count_records
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()
    print(
        "Saving results..."
    )

    master_df.to_csv(
        OUTPUT_ROOT
        / "master_index.csv",
        index=False,
    )

    missing_df.to_csv(
        OUTPUT_ROOT
        / "missing_files.csv",
        index=False,
    )

    unexpected_df.to_csv(
        OUTPUT_ROOT
        / "mismatched_files.csv",
        index=False,
    )

    duplicate_df.to_csv(
        OUTPUT_ROOT
        / "duplicate_frames.csv",
        index=False,
    )

    parse_errors_df.to_csv(
        OUTPUT_ROOT
        / "filename_parse_errors.csv",
        index=False,
    )

    frame_overlap_df.to_csv(
        OUTPUT_ROOT
        / "frame_split_overlap.csv",
        index=False,
    )

    video_overlap_df.to_csv(
        OUTPUT_ROOT
        / "split_overlap.csv",
        index=False,
    )

    cross_shard_df.to_csv(
        OUTPUT_ROOT
        / "cross_shard_video_overlap.csv",
        index=False,
    )

    sequence_df.to_csv(
        OUTPUT_ROOT
        / "video_sequence_statistics.csv",
        index=False,
    )

    structure_df.to_csv(
        OUTPUT_ROOT
        / "dataset_structure.csv",
        index=False,
    )

    counts_df.to_csv(
        OUTPUT_ROOT
        / "modality_counts.csv",
        index=False,
    )

    unreadable_df.to_csv(
        OUTPUT_ROOT
        / "unreadable_hr_images.csv",
        index=False,
    )

    # ========================================================
    # CHECKS
    # ========================================================

    frame_overlap_zero = (
        len(frame_overlap_df) == 0
        or
        int(
            frame_overlap_df[
                "overlap_count"
            ].sum()
        ) == 0
    )

    checks = {

        "shards_found":
            bool(
                len(shard_paths) > 0
            ),

        "all_split_directories_present":
            bool(
                len(
                    missing_split_dirs
                ) == 0
            ),

        "all_modality_directories_present":
            bool(
                len(
                    missing_modalities
                ) == 0
            ),

        "all_filenames_parse_correctly":
            bool(
                len(
                    parse_errors_df
                ) == 0
            ),

        "no_missing_corresponding_files":
            bool(
                len(
                    missing_df
                ) == 0
            ),

        "no_files_without_hr_counterpart":
            bool(
                len(
                    unexpected_df
                ) == 0
            ),

        "no_duplicate_frame_identities":
            bool(
                len(
                    duplicate_df
                ) == 0
            ),

        "no_frame_split_overlap":
            bool(
                frame_overlap_zero
            ),

        "no_video_split_overlap":
            bool(
                len(
                    video_overlap_df
                ) == 0
            ),

        "all_hr_images_readable":
            bool(
                len(
                    unreadable_df
                ) == 0
            ),
    }

    core_integrity_pass = bool(
        all(
            checks.values()
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "dataset_root":
            str(
                DATA_ROOT.resolve()
            ),

        "num_shards":
            int(
                len(shard_paths)
            ),

        "num_hr_frames":
            int(
                len(master_df)
            ),

        "num_videos":
            int(
                master_df[
                    "video_id"
                ].nunique()
            ),

        "num_missing_corresponding_files":
            int(
                len(missing_df)
            ),

        "num_files_without_hr_counterpart":
            int(
                len(unexpected_df)
            ),

        "num_duplicate_frame_records":
            int(
                len(duplicate_df)
            ),

        "num_filename_parse_errors":
            int(
                len(parse_errors_df)
            ),

        "num_video_split_leakage":
            int(
                len(video_overlap_df)
            ),

        "num_frame_split_overlap":
            int(
                frame_overlap_df[
                    "overlap_count"
                ].sum()
            ),

        "num_videos_spanning_multiple_shards":
            int(
                len(cross_shard_df)
            ),

        "num_videos_with_frame_gaps":
            int(
                videos_with_gaps
            ),

        "num_unreadable_hr_images":
            int(
                len(unreadable_df)
            ),

        "checks":
            checks,

        "core_integrity_pass":
            core_integrity_pass,

        "notes": {

            "cross_shard_videos":
                "Videos appearing in multiple shards are not considered an error. "
                "Shards may split long videos across storage boundaries.",

            "incomplete_final_shard":
                "The final shard may contain fewer frames because extraction "
                "ended there. This is not considered an integrity failure.",

            "extension_matching":
                "Frame correspondence is determined by video_id and frame_id, "
                "not by filename extension."
        }
    }

    # ========================================================
    # SAVE JSON SAFELY
    # ========================================================

    summary_path = (
        OUTPUT_ROOT
        / "integrity_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print(
        "FINAL DATASET INTEGRITY REPORT"
    )
    print("=" * 70)

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{status:>5}  {name}"
        )

    print()
    print(
        f"HR frames: "
        f"{len(master_df):,}"
    )

    print(
        f"Videos: "
        f"{master_df['video_id'].nunique():,}"
    )

    print(
        f"Videos spanning multiple shards: "
        f"{len(cross_shard_df):,}"
    )

    print(
        f"Videos with frame-ID gaps: "
        f"{videos_with_gaps:,}"
    )

    print()
    print("=" * 70)

    if core_integrity_pass:

        print(
            "CORE INTEGRITY: PASS"
        )

    else:

        print(
            "CORE INTEGRITY: FAIL"
        )

    print("=" * 70)

    print()
    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_ROOT.resolve()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
