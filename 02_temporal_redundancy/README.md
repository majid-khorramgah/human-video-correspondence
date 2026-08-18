# Human Video Temporal Redundancy

## Research Question

**How much temporal redundancy exists in human videos, and how does the reliability of visual correspondence change as temporal distance increases?**

The central hypothesis is that **nearby frames contain substantial redundant information, but this redundancy progressively decreases as the temporal distance between frames increases**. This project tests that hypothesis quantitatively rather than assuming it.

---

## What is this project about?

A video contains many frames, but consecutive frames are not independent observations. For human videos in particular, neighboring frames often depict highly similar visual content.

This raises a fundamental question:

> **How many frames are actually needed to preserve the temporal information contained in a human video?**

To investigate this, the project builds a five-stage analysis pipeline that moves from basic temporal structure toward motion-aware frame correspondence:

**Temporal Structure → Frame Similarity → Temporal Difference → Motion → Temporal Correspondence**

The goal is to determine **when temporal redundancy breaks down and frame-to-frame correspondence becomes unreliable.**

---

## Five-Stage Analysis

### 01 — Temporal Structure

Characterize how frames are distributed across videos and establish the temporal sampling structure of the dataset.

### 02 — Frame Similarity

Measure how visual similarity changes as the temporal distance between frames increases.

### 03 — Temporal Difference

Quantify how much visual information changes between frames at different temporal distances.

### 04 — Motion Analysis

Measure temporal motion using optical-flow-based statistics to distinguish visual similarity from actual spatial movement.

### 05 — Temporal Correspondence

Test whether visual content can still be reliably mapped between frames using:

* Optical flow magnitude
* Forward-backward consistency
* Warping error
* Valid-flow ratio
* Human-region correspondence
* Human-region motion
* Human-region consistency

The final correspondence analysis evaluates temporal distances of **1, 2, 5, and 10 frames**.

---

## Evidence So Far

The final stage processed:

* **552 videos**
* **110,291 successful frame pairs**
* **0 failed pairs**
* Train / validation / test splits
* Temporal distances: **1, 2, 5, 10 frames**
* Image resolution: **224 × 224**
* Maximum **50 sampled pairs per video**
* Fixed seed: **42**

The results show a consistent degradation of correspondence as temporal distance increases.

### Forward-Backward Correspondence Consistency

| Temporal distance | Train | Validation |  Test |
| ----------------: | ----: | ---------: | ----: |
|                 1 | 0.815 |      0.805 | 0.825 |
|                 2 | 0.761 |      0.750 | 0.777 |
|                 5 | 0.700 |      0.674 | 0.719 |
|                10 | 0.656 |      0.639 | 0.681 |

**Interpretation:** correspondence consistency decreases monotonically as temporal distance increases.

### Human-Region Correspondence Consistency

| Temporal distance | Train | Validation |  Test |
| ----------------: | ----: | ---------: | ----: |
|                 1 | 0.713 |      0.692 | 0.714 |
|                 2 | 0.638 |      0.615 | 0.642 |
|                 5 | 0.565 |      0.527 | 0.565 |
|                10 | 0.521 |      0.499 | 0.519 |

The same pattern appears when correspondence is restricted to the **human region**, suggesting that the observed degradation is not merely caused by background motion.

### Motion and Warping Error

Mean flow magnitude increases with temporal distance, while warping error also increases.

For example, in the training split:

* Flow magnitude: **3.16 → 5.17**
* Mean warping error: **12.94 → 27.40**
* Forward-backward consistency: **0.815 → 0.656**
* Human-region consistency: **0.713 → 0.521**

from temporal distance **1 → 10 frames**.

Together, these measurements provide converging evidence that **visual correspondence becomes progressively less reliable as temporal distance increases.**

---

## Key Finding

The current evidence supports the following empirical conclusion:

> **Temporal redundancy is strongest at short temporal distances, while motion, correspondence error, and correspondence uncertainty increase as the temporal gap grows.**

Importantly, this is not based on a single similarity metric. The same temporal trend appears simultaneously in **motion magnitude, forward-backward consistency, warping error, and human-region correspondence**.

The next research question is therefore not simply whether redundancy exists, but:

> **Can we identify a principled temporal sampling interval beyond which additional frames provide substantially new information rather than redundant observations?**

---

## Visual Evidence

### Correspondence Consistency Distribution

![Correspondence consistency distribution](figures/correspondence_consistency_distribution.png)

### Motion Across Temporal Distance

![Correspondence motion by temporal distance](figures/correspondence_motion_by_temporal_distance.png)

### Flow Magnitude Distribution

![Flow magnitude distribution](figures/flow_magnitude_distribution.png)

### Forward-Backward Consistency

![Forward-backward consistency](figures/forward_backward_consistency.png)

### Human-Region Correspondence

![Human correspondence consistency](figures/human_correspondence_consistency.png)

### Video Motion Distribution

![Video motion distribution](figures/video_motion_distribution.png)

### Warping Error Across Temporal Distance

![Warping error by temporal distance](figures/warping_error_by_temporal_distance.png)

---

## Reproducibility

The final stage produces frame-level, video-level, and aggregate statistics:

* `correspondence_summary.csv`
* `correspondence_summary.json`
* `video_correspondence_statistics.csv`
* `frame_pair_correspondence_statistics.csv`
* `correspondence_errors.csv`

The large frame-pair CSV is kept outside version control because of its size.

---

## Research Direction

The long-term objective is to move from **measuring temporal redundancy** to **using temporal redundancy to design more efficient video representations and sampling strategies**.

The central research direction is:

**Temporal redundancy → measurable correspondence decay → principled temporal sampling → more efficient video understanding**

---

## Project Status

**Five-stage analysis pipeline completed.**

The current results provide quantitative evidence that temporal correspondence systematically degrades with increasing temporal distance in human videos.

The next step is to determine whether these measurements can be converted into a **principled criterion for temporal sampling and redundancy-aware video representation.**
