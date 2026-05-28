---
audience: user
stability: stable
P26-05-28
---

# Camera-to-camera recording protocol

A field guide for collecting paired board observations that the `camera2camera`
bundle adjustment can actually solve. Complements
[docs/camera2camera_quickstart.md](camera2camera_quickstart.md) — read that
first for the install + run command.

## Why this doc exists

The single biggest source of `camera2camera` failures is the recording
protocol, not the math. The pipeline's hidden assumption is that **the
chessboard is at the same physical pose when the parent and child cameras
capture their respective frames of a "pair"**. Two things break that:

1. **Sync slack between RTSP streams.** Each camera's frame at a given
   wall-clock instant arrives ~50–200 ms apart depending on handshake
   jitter, decoder buffering, and network path.
2. **A moving board.** If the operator is walking, swaying, or
   continuously re-tilting the board, those 50–200 ms of slack become real
   centimeters of pose offset between what each camera observed.

The bundle adjustment is set up to detect this and reject pairs that
disagree about the parent→child transform — see "Failure signature" below.
But the right fix is upstream of the math: hold the board still per pose.

## Failure signature

When the protocol is wrong, you'll see one of two errors at calibration
time:

```
Only N stereo pairs survived extraction; need at least M.
```

or a successful run with most pairs flagged
`skip_reason: inconsistent_relative_transform` in
`outputs/<pair>/diagnostics/extraction_entries.csv`.

The pipeline computes a per-pair relative transform `T_parent→child_i`
from each pair's two PnP poses, builds a consensus across all pairs, and
rejects any pair whose `T_i` is more than 0.25 m or 3° from consensus
(`camera2camera/reference_pipeline.py` and `models.py`). When the board
was moving, per-pair estimates scatter and most fall outside that
tolerance.

You can confirm this is the cause by checking
`extraction_entries.csv`:

- If `relative_transform_delta_to_consensus.translation_norm_m > 0.1`
  and `rotation_deg > 5` on most rejected pairs → board motion.
- If the deltas grow monotonically across `pair_NNNN` order → operator
  started moving partway through the recording.

## Static-pose recording protocol

Treat each recording as a series of **discrete still poses**, not
continuous motion.

```
1. Start recorders on BOTH cameras simultaneously, ideally via one
   command. Let RTSP settle for ~5 seconds before walking in.

2. Operator walks into the cameras' FOV overlap region, holding the
   board.

3. Hold board still in pose 1 for ~3 seconds. Visibly relaxed grip,
   no swaying.

4. Slowly transition to pose 2 (1-2 second walk). Do not try to
   film during transit.

5. Hold pose 2 for ~3 seconds.

6. Repeat for 15-20 distinct poses, deliberately varying:
   - depth      : 3-4 close, 3-4 mid, 3-4 far
   - tilt       : roll 15-30 deg left, right, forward, back
   - position   : left third / center / right third of overlap

7. Stop recorders. Give a 5-second buffer to avoid clipping the
   last pose.
```

Hold time of ~3 s matters because the offline extractor samples roughly
one pair per second (default) and uses a short lookback window to verify
the board didn't move; a 3-second hold guarantees the sampler catches at
least one stationary instant per pose.

## Where to stand for a wide-angle outward-pointing rig

If cameras point outward (typical for surround view, ADAS, security
rings), their FOV overlap is a wedge to one side of each pair. For
adjacent cameras with ~40 % overlap each, the operator stands in the
wedge between them, not directly in front of either.

Quick check before recording in earnest: open both RTSP streams in two
viewers side-by-side and walk the board around until you find a region
where the **entire board with margin** is visible in both views. Mark
that region with tape on the floor and confine the recording session to
it.

## How long the recording needs to be

20 poses × 3 s/pose ≈ 60 s per pair. Plus ~5 s at the start (RTSP
warm-up) and ~5 s at the end (clean stop). Budget **~75 s of recording
per pair**.

If you're using segmented recording (e.g.
`splitmuxsink max-size-time=60s`), expect 1–2 segments per pair.

## Verifying the recording before extraction

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 parent.mkv
ffprobe -v error -show_entries format=duration -of csv=p=0 child.mkv
```

The two durations should differ by less than ~1 s. If they differ by
several seconds, the recorders started or stopped at noticeably different
moments, sync slack will be larger, and you may need to hold longer per
pose (4–5 s) to give the extractor more stationary moments.

Spot-check the recording itself: a 60–75 s clip with 15–20 distinct
stationary poses should be visually obvious in any video player.

## The extractor's stability filter

`tools/camera2camera/extract_pairs.py` accepts a stability check
(default on): at each sample time `t`, it also reads frames at
`t - stability_window_ms` from both cameras and rejects the sample if any
chessboard corner moved more than `stability_max_px` in that window in
either camera.

```bash
python tools/camera2camera/extract_pairs.py <pair_name> <parent.mkv> <child.mkv>
# stricter — recommended once the operator protocol is reliable
python tools/camera2camera/extract_pairs.py <pair_name> <parent.mkv> <child.mkv> \
    --stability-max-px 2.0 --stability-window-ms 500
# off — only if you can't get static-pose recording and accept c2c-side rejection
python tools/camera2camera/extract_pairs.py <pair_name> <parent.mkv> <child.mkv> --no-stability
```

The manifest at `pairs/<pair_name>/extraction_manifest.yaml` records
`parent_motion_px` and `child_motion_px` per saved pair, so you can audit
how still the operator was at each accepted moment.

Running the extractor multiple times with the same `<pair_name>` but
different video files appends pairs across segments and records each
pair's source video in the manifest. Pair numbering continues across
runs (resumes from the last `pair_NNNN.png`).

## Pre-calibration sanity check

```bash
# matching stems in parent/ and child/
diff <(ls pairs/<pair_name>/parent) <(ls pairs/<pair_name>/child)   # should print nothing
# enough pairs for min_pairs gate
ls pairs/<pair_name>/parent | wc -l
```

You want this count comfortably above `optimization.min_pairs` in the
c2c config (default 8). 15+ pairs gives the bundle adjustment enough
constraint to clear all the diversity/coverage gates that the doc-level
quickstart describes.

## Reading the extraction diagnostics after a c2c run

`outputs/<pair_name>/diagnostics/extraction_entries.csv` lists every
paired frame with its `skip_reason` and `relative_transform_delta_to_consensus`
values. If you see:

| skip_reason | typical cause |
|---|---|
| `parent_checkerboard_not_found` / `child_checkerboard_not_found` | board not detected in that camera; check focus, occlusion, lighting |
| `board_too_small` | operator too far from camera; move closer |
| `board_too_close_to_edge` | board cropped; back off slightly or center in overlap |
| `parent_pnp_failed` / `child_pnp_failed` | pose estimation diverged; usually motion blur or partial board |
| `inconsistent_relative_transform` | **the protocol failure this doc is about** |

A run where the dominant skip reason is `inconsistent_relative_transform`
needs a re-record, not a threshold tweak. A run where the dominant skip
reason is `board_too_small` or `parent_checkerboard_not_found` needs an
operator standing position change, not a protocol change.

## Acceptance gates for a good extrinsic

After `camera2camera-calibrate` succeeds, the pass bar is in
[docs/camera2camera_quickstart.md](camera2camera_quickstart.md) but for
quick reference:

- `final_rms_px < 1.0`
- `pair_rms_p95 < 1.5`
- `holdout_rms_p95 < 1.5`
- `epipolar_p95 < 1.0`
- both `*_image_coverage_heatmap.png` light up ≥ 4 grid cells
- `pose_diversity` shows depth span ≥ 0.3 m and tilt span ≥ 30°
- `data_quality.yaml.release_ready: true`

Use `tools/camera2camera/analyze.py <run_dir>` for a compact one-row
summary with verdict.
