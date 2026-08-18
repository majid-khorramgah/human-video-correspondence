# Human Video Temporal Redundancy

## Research Question

> Can temporal redundancy in human videos be quantified through correspondence decay, and can this signal be converted into a principled, content-adaptive temporal sampling strategy?

## Motivation

Video models process sequences containing many highly correlated frames, yet the amount of new temporal information contributed by an additional frame is not uniform.

A fixed sampling rate treats all parts of a video similarly, despite differences in motion, pose change, and scene dynamics.

This project asks whether temporal redundancy can instead be measured empirically and used to determine when an additional frame is sufficiently informative to retain.

The central hypothesis is:

**Temporal distance → correspondence decay → increasing visual change → adaptive sampling opportunity**

## Approach

I built a five-stage empirical pipeline to progressively characterize temporal redundancy:

**Temporal Structure → Frame Similarity → Temporal Difference → Motion → Temporal Correspondence**

The first four stages characterize temporal variation at increasingly specific levels. The final stage tests whether measurable visual correspondence systematically degrades as the temporal gap between frames increases.

The correspondence analysis evaluates temporal gaps of **1, 2, 5, and 10 frames** using:

- Optical-flow magnitude
- Forward-backward consistency
- Warping error
- Valid-flow ratio
- Human-region motion
- Human-region correspondence consistency

## Preliminary Evidence

The final correspondence experiment covered:

- **552 videos**
- **110,291 successful frame pairs**
- **0 failed pairs**
- Train / validation / test splits
- **224 × 224** frames
- Temporal gaps of **1, 2, 5, and 10 frames**
- Fixed seed: **42**

The observed trend is consistent across all three splits.

### Train Split

| Temporal Gap | Flow Magnitude | FB Consistency | Warping Error | Human-Region Consistency |
|-------------:|---------------:|---------------:|--------------:|-------------------------:|
| 1            | 3.16           | 0.815          | 12.94         | 0.713                    |
| 2            | 3.84           | 0.761          | 17.31         | 0.638                    |
| 5            | 4.61           | 0.700          | 23.06         | 0.565                    |
| 10           | 5.17           | 0.656          | 27.40         | 0.521                    |

As temporal distance increases:

**motion increases → warping error increases → correspondence consistency decreases.**

The same qualitative behavior is observed when correspondence is evaluated specifically within the human region, indicating that the trend is not attributable solely to background motion.

## What the Results Suggest

The current results provide evidence of a systematic **correspondence decay with increasing temporal distance**.

Importantly, the effect is observed across multiple complementary measurements rather than a single similarity metric:

**Flow magnitude ↑**  
**Warping error ↑**  
**Forward-backward consistency ↓**  
**Human-region consistency ↓**

This makes correspondence decay a potentially useful **measurable signal of temporal redundancy**.

However, the current experiment does not yet establish an optimal sampling rule.

That leads to the central next question:

> When does an additional frame provide enough new information that it should be retained rather than skipped?

## Core Research Hypothesis

I hypothesize that the appropriate temporal sampling interval is **content-dependent rather than fixed**.

A useful sampling criterion may depend on observable video dynamics such as:

**Motion → Pose Change → Scene Dynamics → Correspondence Reliability**

Rather than prescribing a fixed interval, the goal is to determine whether measurable correspondence decay can identify when the current frame has become sufficiently different from the previously retained frame.

## Next Research Step

The next experiment is to connect **correspondence decay** to **marginal information gain**.

Specifically:

> Can correspondence-based measurements predict whether retaining an additional frame contributes sufficiently new information to justify its computational cost?

If so, correspondence decay could provide the basis for a **content-adaptive temporal sampling criterion**, potentially reducing redundant frames while preserving meaningful temporal dynamics.

## Current Status

**Five-stage empirical pipeline completed.**

The current study establishes a reproducible correspondence-decay pattern across temporal distances in a dataset of 552 human videos.

The open research problem is now to move from:

**Measuring redundancy**

to

**Using redundancy to decide what to sample.**

That transition—from descriptive temporal redundancy to a **principled, content-adaptive sampling rule**—is the main direction I would like to investigate.