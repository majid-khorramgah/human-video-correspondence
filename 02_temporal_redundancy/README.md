# Temporal Information in Human Video

## Research Question

> **Under what conditions can visual information from one frame remain useful in another frame of a human video?**

## Project Goal

This project investigates whether temporal information in human videos can be reliably exploited across frames, particularly under **motion and increasing temporal distance**.

The study uses **PhySense-Human**, a diverse human try-on video dataset covering a wide range of clothing and seasonal styles, with **500,507 frames across 547 videos**. Frames are sampled at approximately **1 frame per second** from the original videos (e.g., frame 0 → frame 30 → frame 60 in a 30 FPS video), while preserving their temporal order.

---

## What We Found

We progressively analyzed:

**Temporal Structure → Similarity → Difference → Motion → Correspondence**

- **Temporal structure:** **500,507 frames → 547 videos → 499,960 temporal transitions → 100% adjacent continuity → 0 frame-ID gaps.**
- **Temporal similarity:** Across **109,389 frame pairs**, visual similarity consistently decreases from **lag 1 → 2 → 5 → 10**, confirming measurable but diminishing temporal redundancy.
- **Temporal difference:** In the training split, full-frame MAE increases from **0.0670 → 0.0845 → 0.1076 → 0.1241**, while human-region MAE increases from **0.1463 → 0.1716 → 0.1988 → 0.2146**.
- **Motion:** Estimated optical-flow magnitude increases from **3.159 → 3.837 → 4.609 → 5.167 px** as temporal distance increases from lag 1 to 10.
- **Correspondence:** Forward-backward consistency decreases from **0.815 → 0.761 → 0.700 → 0.656**, while warping error increases from **12.938 → 17.309 → 23.059 → 27.398**.

---

## Key Finding

> **Temporal information is present and potentially useful, but its reliability depends strongly on temporal distance, motion, and correspondence quality.**

As temporal distance increases, **visual similarity decreases, temporal difference and motion increase, and cross-frame correspondence becomes less reliable.**

The central challenge is therefore to determine **what visual information can be reliably transferred across time, under what conditions, and what this enables.**