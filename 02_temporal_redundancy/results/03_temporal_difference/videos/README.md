# Execution Video — Temporal Difference Analysis

This directory contains video documentation of the real execution of the temporal difference analysis stage.

The video provides an execution record showing that `03_temporal_difference.py` was actually run as part of the research pipeline.

---

## Experiment

**Script**

`02_temporal_redundancy/03_temporal_difference.py`

**Stage**

`02_temporal_redundancy / 03_temporal_difference`

**Purpose**

The purpose of this analysis is to quantify visual differences between temporally separated video frames and to characterize how temporal variation changes across different frame distances.

---

## Dataset

The analysis was performed on the complete available dataset.

| Property | Value |
|---|---:|
| Frames | 500,507 |
| Videos | 547 |
| Shards | 6 |
| Splits | Train / Validation / Test |
| Temporal distances | 1, 2, 5, 10 |
| Sampling mode | Per-video |
| Successful comparisons | 110,291 |
| Failed comparisons | 0 |

---

## Regions Analyzed

The analysis evaluates temporal differences in several image regions:

- Full image
- Human region
- Clothing region
- Skin region

This allows temporal variation to be examined not only at the full-image level but also within human-related regions.

---

## Main Result

The analysis shows a consistent increase in visual difference as temporal distance increases.

In general:

Temporal distance ↑

→ Visual difference ↑

Human-related regions also exhibit distinct temporal behavior compared with the full image.

These observations provide evidence that the dataset contains meaningful temporal variation and motivate subsequent investigation of motion and temporal correspondence.

---

## Important Interpretation Boundary

This experiment measures raw visual temporal differences.

It does not directly measure:

- Motion estimation accuracy
- Optical flow accuracy
- Frame correspondence quality
- Reconstruction quality
- Super-resolution performance
- Reconstruction improvement

Therefore, the results should be interpreted as dataset-level temporal observations rather than final conclusions about reconstruction.

---

## Execution Video

The complete execution of this stage is documented on YouTube:

[Watch the execution on YouTube](https://youtu.be/j-QVnoIde3I)

The video is provided as an execution record accompanying the numerical results and figures in this directory.

---

## Generated Results

Numerical results are available in:

`../temporal_difference/`

Visual results are available in:

`../figures/`

The corresponding figures include:

- `difference_by_split.png`
- `difference_by_temporal_distance.png`
- `difference_distribution.png`
- `full_vs_human_difference.png`

---

## Reproducibility

The experiment was executed using the project's temporal difference analysis script:

`02_temporal_redundancy/03_temporal_difference.py`

The execution completed successfully with:

- 110,291 successful comparisons
- 0 failed comparisons

Runtime:

Approximately 244.88 minutes.

The execution video is provided to make the computational experiment independently inspectable and to document the actual execution of the analysis pipeline.

---

## Relation to the Research Pipeline

This analysis is part of the temporal redundancy stage:

`01_temporal_structure.py`

→ establishes temporal continuity

`02_frame_similarity.py`

→ measures visual similarity between frames

`03_temporal_difference.py`

→ measures temporal visual differences

`04_motion_analysis.py`

→ investigates motion characteristics

`05_temporal_correspondence.py`

→ investigates temporal correspondence

The observations obtained from these analyses will later contribute to the formulation of the research hypothesis and research question.

---

## Status

**Temporal Difference Analysis: COMPLETE**

**Execution: SUCCESSFUL**

**Dataset coverage: FULL**

**Video documentation: AVAILABLE**