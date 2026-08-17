# Motion Analysis — Execution Video

> ▶️ **Watch the complete execution video:**  
> https://youtu.be/VWimTFAPOSQ

This video provides a direct execution record of the **Motion Analysis** stage of the Human Video Correspondence research pipeline.

It documents the actual execution of:

`02_temporal_redundancy/04_motion_analysis.py`

and demonstrates the generation of the corresponding motion statistics and visualizations.

---

## Experiment

### Stage

`02_temporal_redundancy / 04_motion_analysis`

### Script

`04_motion_analysis.py`

### Purpose

The purpose of this experiment is to quantify the amount and spatial extent of visual motion between frames of the same video sequence.

The analysis investigates how motion changes as the temporal distance between two frames increases.

The evaluated temporal distances are:

- 1 frame
- 2 frames
- 5 frames
- 10 frames

---

## Dataset Coverage

The experiment uses the project's master dataset index:

`01_dataset_audit/results/integrity/master_index.csv`

The analysis covers the complete available dataset:

| Property | Value |
|---|---:|
| Dataset shards | 6 |
| Videos | 547 |
| Frames | 500,507 |
| Splits | train / val / test |
| Temporal distances | 1, 2, 5, 10 |

The experiment therefore evaluates temporal motion across the complete dataset rather than relying on a single example video.

---

## What Is Being Measured?

The analysis focuses on the relationship between temporal distance and visual motion.

Conceptually:

`Frame(t) → Frame(t + 1)`

represents a short temporal displacement.

While:

`Frame(t) → Frame(t + 10)`

represents a substantially larger temporal displacement.

The experiment measures how the visual difference and motion characteristics evolve across these temporal distances.

---

## Why This Experiment Matters

The broader research question concerns whether information from neighboring video frames can be exploited for human-video reconstruction.

Temporal information may contain useful complementary information, but its usefulness depends on how much the scene changes between frames.

If two frames are extremely similar, the second frame may contain little additional information.

If two frames are substantially different, the second frame may contain useful new information, but correspondence and alignment become more difficult.

Motion analysis helps quantify this trade-off.

---

## Position in the Research Pipeline

This experiment is the fourth component of the temporal redundancy analysis:

`01_temporal_structure.py`

↓

`02_frame_similarity.py`

↓

`03_temporal_difference.py`

↓

`04_motion_analysis.py`

↓

`05_temporal_correspondence.py`

Each stage provides a different piece of evidence.

### 01 — Temporal Structure

Verifies temporal continuity and frame ordering.

### 02 — Frame Similarity

Measures visual similarity between frames at different temporal distances.

### 03 — Temporal Difference

Measures pixel-level visual changes between frames.

### 04 — Motion Analysis

Quantifies motion magnitude and spatial coverage.

### 05 — Temporal Correspondence

Will investigate whether information across frames can be reliably matched and aligned.

---

## Execution Evidence

The accompanying video serves as an execution record for this stage.

It demonstrates that the analysis is not only described theoretically but is implemented and executed on the actual dataset.

The video shows the computational execution associated with the generated research outputs.

This provides an additional layer of reproducibility and transparency for the project.

---

## Generated Results

The execution produces numerical statistics and visualizations stored under:

`02_temporal_redundancy/results/04_motion_analysis/`

The main result categories are:

### Motion Statistics

Contains:

- aggregate motion statistics,
- video-level motion statistics,
- motion errors,
- detailed frame-pair statistics.

### Figures

Contains visualizations describing:

- motion by temporal distance,
- motion by dataset split,
- motion coverage,
- motion distributions,
- video-level motion variation.

---

## Large Generated File

The detailed file:

`frame_pair_motion_statistics.csv`

contains frame-pair-level measurements and is intentionally excluded from Git version control because of its size.

The compact statistical summaries remain version-controlled so that the main research results are available directly from the repository.

---

## Research Interpretation

The motion analysis is not itself a reconstruction model.

Its role is to provide empirical evidence needed before designing the correspondence and reconstruction stages.

The results help answer questions such as:

- How rapidly does visual motion increase with temporal distance?
- How different are temporal dynamics across videos?
- Does the dataset contain mostly low-motion or high-motion sequences?
- How much additional information can potentially be obtained from neighboring frames?
- At what temporal distance might correspondence become more challenging?
- Is temporal information structured enough to motivate a correspondence-based reconstruction approach?

These observations will inform the next stages of the research pipeline.

---

## Reproducibility

The experiment can be reproduced using:

`02_temporal_redundancy/04_motion_analysis.py`

with the project's master dataset index.

The source dataset itself is not included in the GitHub repository.

The repository instead contains the analysis code, compact results, figures, documentation, and execution records.

---

## Execution Video

The complete execution is available on YouTube:

**Motion Analysis of Human Video Sequences | PhySense-Human | Temporal Redundancy**

https://youtu.be/VWimTFAPOSQ

---

## Status

**Motion Analysis: COMPLETE**

The analysis was successfully executed across:

- 6 dataset shards
- 547 videos
- 500,507 frames
- train / validation / test splits
- temporal distances of 1, 2, 5, and 10 frames

The generated numerical results and figures are available in this result directory.

The execution video provides a visual record of the computational experiment.

---

## Project

**Human Video Correspondence**

Research pipeline:

`Dataset Audit`
→ `Temporal Redundancy`
→ `Research Question`
→ `Baselines`
→ `Reconstruction`
→ `Experiments`
→ `Results`

This experiment contributes to the **Temporal Redundancy** stage by quantitatively characterizing motion across human video sequences.