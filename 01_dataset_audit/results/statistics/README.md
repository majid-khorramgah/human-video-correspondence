# Dataset Statistics

This directory contains the statistical analysis of the **PhySense-Human**
dataset.

## Dataset overview

- Total frames: 500,507
- Total videos: 547
- Total shards: 6

## Dataset generation

The dataset was generated from video sources.

The generation pipeline samples frames approximately once per second using:

`frame_idx % fps == 0`

The source frame is resized to:

- HR: 1920 × 1080
- LR: 640 × 360

The HR image is saved as JPEG with quality 95.

Segmentation masks are stored as PNG.

The dataset contains:

- RGB/appearance HR images
- RGB/appearance LR images
- Full-body masks
- Cloth masks
- Skin masks

## Important LR definition

The LR images are **synthetically generated** rather than independently
captured low-resolution observations.

The generation pipeline first creates the 1920 × 1080 representation and
then creates the 640 × 360 representation by resizing.

Therefore, experiments using this dataset should describe the LR input as
synthetically degraded/generated LR data.

## Temporal information

The source material is video and the dataset preserves:

- video identity
- frame identity
- ordering through frame IDs

However, the original video FPS, exact timestamps, duration, codec, and
original video resolution were not persisted in the dataset metadata.

Therefore, this statistics report does not claim to reconstruct exact
original timestamps.

## Split policy

The generation code assigns each video to a split using an MD5 hash of the
original filename.

Target split proportions:

- train: 80%
- val: 10%
- test: 10%

The integrity audit separately verifies frame-level and video-level leakage.

## Sharding

Frames are stored in shards of nominal capacity 100,000 frames.

The final shard may be partially filled.

Videos can span multiple shards because sharding is performed using the
global frame counter at the time each frame is saved.

## Files

### dataset_summary.json

Machine-readable high-level summary of the dataset.

### video_statistics.csv

One row per video containing:

- video ID
- number of frames
- split
- shard membership
- first frame ID
- last frame ID
- frame-ID span
- frame gaps
- sequence continuity

### shard_statistics.csv

Frame and video counts for every shard.

### split_statistics.csv

Frame and video counts for train/validation/test.

### frame_statistics.csv

Global frame-level statistics.

### resolution_statistics.csv

Resolution, image format, readability and file-size information for sampled
images.

### modality_statistics.csv

Coverage of every expected modality.

### file_size_statistics.csv

Exact file-size statistics when the exact filesystem scan is enabled.

### video_sequence_statistics.csv

Temporal/frame-ID continuity statistics per video.

### youtube_metadata_statistics.csv

Optional information obtained from the original `youtube_links.txt` file if
available.

## Interpretation

The most important structural property for temporal research is that the
dataset retains video identity and frame ordering. This allows later
experiments to compare single-frame and temporally informed approaches.

The statistics should therefore be considered not only as image statistics,
but also as video-derived dataset statistics.
