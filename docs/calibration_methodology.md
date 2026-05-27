---
audience: dev
stability: stable
P26-05-25
---

# Calibration methodology, design choices, and practical SOTA context

This document explains how the repo is designed, what method each calibration
module uses, what the practical industry baseline looks like today, and which
references are most relevant when evolving the stack.

## 1. Repo-level design philosophy

The repo intentionally separates calibration into three layers:

1. **Extraction / data layer**  
   Normalize raw data, count usable samples, and expose data-quality facts early.
2. **Algorithm layer**  
   Run the solver or optimizer without baking release decisions into solver code.
3. **Evaluation layer**  
   Emit stable artifacts so runs remain comparable even when algorithms change.

This is not accidental documentation style. It is the core engineering choice of
the repo.

Why this matters:

- algorithm teams can iterate aggressively
- release criteria stay readable and stable
- field engineers can review evidence without reading solver internals
- multiple methods can coexist under the same acceptance surface

## 2. Module-by-module method summary

| Module | Core method | Why this repo uses it |
| --- | --- | --- |
| `camera` | checkerboard-based intrinsic calibration with reprojection analysis | still the most reliable production baseline for monocular industrial cameras |
| `camera2camera` | target-based stereo bundle adjustment with fixed intrinsics and holdout review | keeps camera↔camera release evidence explicit and debuggable instead of hiding everything inside one opaque solve |
| `lidar2camera` | target-based checkerboard / planar board extraction + PnP / robust optimization | easier to audit and release than targetless methods on vehicle programs |
| `lidar2lidar` | scene-based registration with ICP / GICP, overlap screening, and optional graph refinement | strong practical baseline when the vehicle can drive through feature-rich static scenes |
| `lidar2imu` | staged ground + gravity + motion hand-eye calibration with observability gates | explicit failure modes are easier to manage than one opaque joint solve |

## 3. Why the repo does not use one giant optimizer everywhere

An all-in-one batch optimizer can look elegant on paper, but in production it
often makes failure analysis harder:

- bad data and bad model assumptions are harder to separate
- observability failures look like solver instability
- field iteration becomes slow because everything must be rerun together

The repo instead prefers staged pipelines with stable reporting surfaces:

- `standardized_data.yaml`
- `data_quality.yaml`
- `metrics.yaml`
- `acceptance_report.yaml`
- `visualization_index.yaml`

## 4. Current practical SOTA by calibration type

The phrase "SOTA" means different things in papers and in production. The
guidance below focuses on **practical release-oriented SOTA** rather than
benchmark-only results.

### Camera intrinsic

Practical SOTA today is still:

- high-quality board observations
- accurate lens model selection
- bundle-adjustment-style optimization
- strong coverage diagnostics

In practice, ChArUco / AprilTag-grid style targets often improve robustness over
plain checkerboards, but checkerboards remain a strong baseline when acquisition
is controlled.

### Camera-to-camera

Practical SOTA for release workflows still centers on **target-based** methods:

- pre-calibrated per-camera intrinsics
- repeated multi-pose board capture
- pairwise pose bootstrap plus global bundle adjustment
- holdout / repeatability / coverage review

ChArUco and AprilGrid style targets are often more robust than plain
checkerboards under partial occlusion or weaker lighting, but a strong
checkerboard pipeline remains a legitimate production baseline when acquisition
is controlled.

### LiDAR↔Camera

Practical SOTA for release workflows still centers on **target-based** methods:

- known board geometry
- repeated multi-pose capture
- robust joint optimization
- explicit residual and repeatability review

Targetless and learning-based methods have improved a lot, but in most vehicle
programs they are still better as:

- initialization tools
- monitoring tools
- research / diagnostic branches

than as the only release gate.

### LiDAR-to-LiDAR

Practical SOTA is usually one of these patterns:

1. pairwise registration + graph optimization
2. scan-to-submap / scan-to-map refinement
3. explicit scene-quality and observability checks

Single-frame ICP alone is not enough for high-confidence release. The important
step is not only "better registration", but also stronger graph consistency and
human-visible geometry review.

### LiDAR-to-IMU

Research SOTA often uses continuous-time batch estimation, temporal offset
calibration, and tightly coupled motion models. Those methods are powerful, but
they also demand stronger modeling assumptions and higher-quality excitation.

Practical release-oriented SOTA usually adds:

- explicit motion-quality gating
- observability checks
- holdout validation
- trusted-prior comparison
- clear fallback when `x/y/yaw` are weak

That is the direction `whl-cal` follows.

## 5. How that maps to this repo

### `lidar2lidar`

The repo keeps:

- `scan2scan` as the practical baseline
- `scan2map` as the refinement path
- workflow planning as the topology layer
- loop closure as a graph-consistency layer

This matches good industry practice: keep a conservative baseline alive while a
stronger candidate evolves beside it.

### `lidar2imu`

The repo keeps:

- a conservative `baseline` profile
- a more ambitious `production` profile
- `--planar-motion-policy auto` as the safety mechanism for weak bags

This is a deliberate choice to prevent weak excitation from being reported as a
full-confidence 6DoF success.

### `lidar2camera`

The repo keeps the target-based reference pipeline as the production baseline and
documents targetless / learning-based flows as experimental. That mirrors how
most mature release processes operate today.

### `camera2camera`

The repo should keep a conservative target-based stereo baseline alive, then
evolve toward stronger targets such as ChArUco without breaking the artifact
contract. That mirrors the same “stable review surface, evolving algorithm”
strategy used by the other production-facing modules.

## 6. What "excellent practice" means in this repo

For this repo, excellent practice means:

1. do not trust convergence alone
2. require data-quality evidence before solver evidence
3. require visualization before release
4. keep extraction, algorithm, and evaluation separable
5. preserve a stable artifact contract across iterations

This is why the docs emphasize review order and visualization so heavily.

## 7. When to change the method

Only change the algorithm family when at least one of these is true:

- the current bag class is systematically under-observed
- the accepted method cannot produce stable repeatability
- the new method improves both accuracy and debuggability
- the output contract can remain stable

Do not swap methods just because a paper reports a better benchmark number.

## 8. References worth reading

### Core calibration and optimization

1. Zhengyou Zhang, *A Flexible New Technique for Camera Calibration*, IEEE TPAMI,
   2000.  
   Baseline reference for checkerboard-based camera intrinsic calibration.
2. Richard Hartley and Andrew Zisserman, *Multiple View Geometry in Computer
   Vision*, 2nd edition.  
   Canonical reference for projection geometry and reprojection reasoning.
3. Frank C. Park and Bryan J. Martin, *Robot Sensor Calibration: Solving AX = XB
   on the Euclidean Group*, IEEE Transactions on Robotics and Automation, 1994.  
   Classic hand-eye calibration formulation.
4. Roger Y. Tsai and Reimar K. Lenz, *A New Technique for Fully Autonomous and
   Efficient 3D Robotics Hand/Eye Calibration*, IEEE Transactions on Robotics and
   Automation, 1989.  
   Classic closed-form hand-eye reference.

### Point-cloud registration

5. Paul J. Besl and Neil D. McKay, *A Method for Registration of 3-D Shapes*,
   IEEE TPAMI, 1992.  
   The original ICP reference.
6. Alex Segal, Dirk Haehnel, and Sebastian Thrun, *Generalized-ICP*, RSS, 2009.  
   Standard reference for GICP-style point-cloud alignment.

### LiDAR-IMU and continuous-time calibration

7. Paul Furgale, Joern Rehder, and Roland Siegwart, *Unified Temporal and Spatial
   Calibration for Multi-Sensor Systems*, IROS, 2013.  
   Strong reference for continuous-time and spatiotemporal calibration thinking.
8. Joern Rehder et al., *Extending Kalibr: Calibrating the Extrinsics of Multiple
   IMUs and of Individual Axes*, ICRA Workshop / Kalibr ecosystem references.  
   Useful for observability and practical multi-sensor calibration workflows.

### Practical system references

9. Autoware calibration tools and related design docs.  
   Good reference for production-facing target-based and workflow-oriented sensor
   calibration tooling.
10. Apollo localization and sensor-driver documentation.  
   Relevant for understanding upstream topics, TF assumptions, and record
   acquisition constraints in Apollo-based deployments.

## 9. Related docs in this repo

- bag preparation: [apollo_data_collection.md](apollo_data_collection.md)
- run and review flow: [calibration_review_guide.md](calibration_review_guide.md)
- module entry points: [quickstart_index.md](quickstart_index.md)
- `lidar2lidar` design: [lidar2lidar_design.md](lidar2lidar_design.md)
- `lidar2imu` design: [lidar2imu_design.md](lidar2imu_design.md)
- `lidar2camera` design: [lidar2camera_design.md](lidar2camera_design.md)
