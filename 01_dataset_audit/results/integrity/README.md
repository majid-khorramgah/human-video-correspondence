# Dataset Integrity Audit Results

This directory contains the results of the first integrity audit of the PhySense-Human dataset.

The purpose of this stage is not to evaluate model performance or establish a research result. Its purpose is to verify that the dataset is structurally consistent, that all modalities correspond to the same video/frame identities, and that there is no train/validation/test leakage.

## Dataset Audited

The audit was performed on:

- 500,507 HR frames
- 547 unique YouTube videos
- 6 storage shards
- 3 splits: train / val / test
- 8 image modalities per frame

The eight modalities are:

1. Img_HR
2. Img_LR
3. Mask_Cloth_HR
4. Mask_Cloth_LR
5. Mask_Full_HR
6. Mask_Full_LR
7. Mask_Skin_HR
8. Mask_Skin_LR

The final shard contains the remaining frames produced by the automatic extraction process. Its smaller size is expected and is not considered an integrity error.

---

## What Was Verified?

The integrity audit verifies the following properties:

### 1. Dataset shard structure

All expected dataset shards were discovered:

- Dataset_Shard_0000000_0099999
- Dataset_Shard_0100000_0199999
- Dataset_Shard_0200000_0299999
- Dataset_Shard_0300000_0399999
- Dataset_Shard_0400000_0499999
- Dataset_Shard_0500000_0599999

Each shard contains the expected train, validation, and test directories.

### 2. Modality completeness

Every expected modality directory was checked.

The audit verifies that each HR frame has corresponding files in all other modalities.

Result:

- Missing corresponding files: 0
- Files without an HR counterpart: 0

### 3. Filename consistency

Frame identities are derived from:

    video_id + frame_id

For example:

    frame_0000003_-Hs-zuBOlQE.jpg

is interpreted as:

    video_id = -Hs-zuBOlQE
    frame_id = 3

File extensions are deliberately ignored when determining frame identity.

This is important because RGB images use JPEG while segmentation masks use PNG. Therefore:

    frame_0000003_-Hs-zuBOlQE.jpg

and

    frame_0000003_-Hs-zuBOlQE.png

represent the same frame identity when they belong to the corresponding modalities.

Result:

- Filename parsing errors: 0

### 4. Duplicate frame identities

The audit checks for multiple files representing the same:

    (video_id, frame_id)

identity within the same modality.

Result:

- Duplicate frame identities: 0

### 5. Frame-level split leakage

Frame identities were compared between:

- train vs validation
- train vs test
- validation vs test

Result:

- Train/validation overlap: 0
- Train/test overlap: 0
- Validation/test overlap: 0

### 6. Video-level split leakage

Because this dataset contains temporal frames extracted from videos, checking only individual frames is insufficient.

The audit therefore verifies that a complete video does not appear in more than one split.

For example, the following would be invalid:

    video_A -> train
    video_A -> test

Result:

- Video-level leakage: 0

This is an important property for future temporal and video-based experiments because frames from the same source video must not appear across training and evaluation splits.

### 7. Cross-shard video continuity

Five videos appear across more than one storage shard.

This is not considered an error.

The shards are storage partitions and may split a long video across shard boundaries. For example:

    Shard 1:
        video_A -> frames 0 ... 9999

    Shard 2:
        video_A -> frames 10000 ... 19999

This represents one continuous video stored across multiple shards, not duplicate data.

The audit records these cases in:

    cross_shard_video_overlap.csv

Importantly, no such video crosses train/validation/test boundaries.

### 8. Temporal frame sequence consistency

The frame IDs belonging to each video were checked for gaps.

Result:

- Videos with frame-ID gaps: 0

This provides an initial structural verification of the extracted temporal sequences.

### 9. HR image readability

All 500,507 HR images were opened and verified using an image integrity check.

Result:

- HR images checked: 500,507
- Unreadable HR images: 0

---

# Files in This Directory

## integrity_summary.json

This is the main machine-readable summary of the integrity audit.

It contains:

- dataset root
- number of shards
- number of frames
- number of videos
- number of missing files
- number of duplicate frames
- split leakage statistics
- cross-shard video statistics
- temporal gap statistics
- image readability statistics
- final integrity checks

This is the primary file to inspect if someone wants a quick overview of the audit.

---

## dataset_structure.csv

Contains the number of frames and videos for every:

    shard × split

combination.

For example:

    shard                         split    frames    videos
    Dataset_Shard_...             train    ...
    Dataset_Shard_...             val      ...
    Dataset_Shard_...             test     ...

This file makes the distribution of the dataset across storage shards and splits explicit.

---

## modality_counts.csv

Contains the number of files found for every modality in every shard and split.

It is used to verify that the expected modalities contain the expected number of files.

This provides a compact view of dataset completeness without storing the complete frame-level index.

---

## video_sequence_statistics.csv

Contains per-video temporal statistics, including:

- number of frames
- minimum frame ID
- maximum frame ID
- number of frame-ID gaps
- maximum frame gap
- monotonicity of frame IDs

This file is particularly relevant for future temporal reconstruction and video-based experiments.

---

## cross_shard_video_overlap.csv

Lists videos that appear in more than one storage shard.

These are not considered errors.

They indicate that a video may have been split across storage boundaries during automatic dataset sharding.

This file is provided for transparency and reproducibility.

---

## frame_split_overlap.csv

Contains frame-level overlap counts between:

- train / val
- train / test
- val / test

Expected result for a clean dataset:

    overlap_count = 0

---

## split_overlap.csv

Contains video-level split overlap information.

This is more important than frame-level overlap for temporal learning because all frames belonging to a source video should remain within a single split.

Expected result for a clean dataset:

    no video appears in multiple splits

---

## missing_files.csv

Contains frames for which one or more expected modalities are missing.

For the current dataset this file is empty because:

    missing corresponding files = 0

The empty file is intentionally retained as an audit artifact. It demonstrates that the check was performed and that no missing files were found.

---

## mismatched_files.csv

Contains modality files whose frame identity does not have a corresponding HR frame.

Frame identity is determined using:

    video_id + frame_id

For the current dataset this file is empty.

---

## duplicate_frames.csv

Contains duplicate frame identities detected within the same modality.

For the current dataset this file is empty.

The empty file is retained to document the result of the duplicate detection procedure.

---

## filename_parse_errors.csv

Contains files whose filenames could not be interpreted using the expected frame naming convention.

Expected naming pattern:

    frame_<frame_id>_<video_id>.<extension>

For the current dataset:

    filename parsing errors = 0

Therefore this file is empty.

---

## unreadable_hr_images.csv

Contains HR images that could not be opened or verified successfully.

For the current dataset:

    unreadable HR images = 0

Therefore this file is empty.

---

# Large File: master_index.csv

The complete frame-level master index was also generated during the audit.

It contains one row per HR frame and records the correspondence between the eight modalities.

For each frame it stores information such as:

- shard
- split
- video ID
- frame ID
- HR filename
- HR extension
- LR filename
- LR extension
- cloth mask filename
- full-body mask filename
- skin mask filename
- modality existence flags
- modality file counts

The current master index contains approximately 500,507 rows and is approximately 213 MB.

It is intentionally NOT committed to the normal Git repository.

The reason is practical rather than scientific: a 213 MB generated CSV is unnecessarily large for source-code version control and makes cloning and repository maintenance inefficient.

The complete file is retained locally and can be regenerated from the dataset using the integrity audit script.

Therefore:

    master_index.csv

is a generated intermediate artifact rather than a required source file for reproducing the audit logic.

The smaller summary files in this directory are committed to GitHub so that the audit results remain transparent and inspectable without requiring the full 213 MB index.

---

# Reproducibility

The integrity audit was generated using the corresponding dataset integrity script:

    01_dataset_audit/scripts/dataset_integrity.py

The script discovers all Dataset_Shard_* directories automatically and performs the integrity checks across all available shards.

The audit does not modify the original dataset.

It only reads the dataset and writes audit results.

---

# Final Integrity Status

The current dataset passed all core integrity checks.

Summary:

    HR frames:                         500,507
    Unique videos:                         547
    Missing modality files:                 0
    Files without HR counterpart:           0
    Duplicate frame identities:             0
    Filename parsing errors:                0
    Frame-level split overlap:              0
    Video-level split leakage:              0
    Videos with frame-ID gaps:              0
    Unreadable HR images:                   0

Final status:

    CORE INTEGRITY: PASS

This audit establishes the structural integrity of the dataset before proceeding to dataset statistics, visual analysis, baseline construction, and research experiments.

---

# Next Stage

The next stage is dataset characterization rather than model training.

The following properties will be investigated:

- frame and video distributions
- video length distributions
- train/validation/test composition
- image resolutions
- HR/LR scale relationships
- temporal sampling characteristics
- human appearance and pose variation
- mask coverage
- temporal redundancy
- inter-frame visual correspondence
- potential properties that may motivate a research question

The goal is to understand what is scientifically distinctive about the dataset before defining the final reconstruction or super-resolution research problem.
