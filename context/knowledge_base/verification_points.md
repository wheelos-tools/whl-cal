# Verification points

This file lists what still needs evidence before conclusions can be promoted.

## lidar2imu

### GRIL/LI-Init data qualification before another solver comparison

1. For Ruantong parking-04, the first shard has measurement-time IMU and
   bidirectional turns without a >50 ms IMU gap, but its `lslidar_main`
   point-index modulo 16/32 does **not** yield plausible rings and the
   decoded points have no explicit ring field. Identify the real sensor
   model/channel ordering or an existing correct decoder before GRIL
   conversion. Then check the remaining three shards for IMU gaps and
   value semantics, LiDAR clocks, static TF, acceleration and
   startup-static support. Do not treat the four-shard family as qualified yet.
2. Recheck Weilan 0612/0827 using the INS measurement-time IMU only as a
   **separate locked-input experiment** if excitation is sufficient. Raw and
   corrected IMU values differ on 0827; earlier raw-topic results must retain
   their provenance and cannot be relabeled.
3. Search the remaining Zhongji 05-07 figure-eight shards for uninterrupted
   INS-IMU and LiDAR segments with useful dual-turn motion. Do not stitch
   derivatives across a gap or reject the entire family because its first
   90 seconds contain gaps.
4. On every eligible capture, compare full-frontend repeated runs, not just
   a deterministic frozen-batch replay. Obtain surveyed extrinsics or an
   independent physical holdout before claiming x/y/yaw accuracy.
5. On Weilan 0827, identify whether the four-window -28/-30/-42/-42 ms
   yaw-rate peak spread is due to clock semantics, LiDAR-only frontend motion,
   or weak offset identifiability. Freeze one new corrected-IMU input for
   **both** methods before comparing to the archived raw-IMU run; do not
   reconcile their timing signs without documenting each convention. Supply
   matched-crop candidate-wise deskewed point-cloud views and independent
   extrinsic evidence before promotion. See
   [0827 decision contract](../../.agents/knowledge/lidar2imu_research.md).
6. For the P2 LI-Calib spike, implement a tested Vanjee PointCloud2 path:
   its present unorganized 0827 scan has 25,291 points, while the upstream
   decoder regenerates VLP-16 time with an 1,824-firing table and ignores
   actual per-point time. For P3 AFLI-Calib, its Velodyne branch does not
   populate scans, and the Hesai branch expects absolute float64 point time,
   unlike 0827's relative float32 time. Verify the motion regime and IMU
   selection before adapting either. Both remain `not_run` on 0827; report
   adapter blockers separately from calibration failures. See
   [P0-P3 method matrix](../../.agents/knowledge/lidar2imu_research.md).
7. For LI-Init P1, the archived 0827 LiDAR-only states reproduce its 0.99
   sufficiency gate failure (pair products 0.02356/0.02374/0.37011 after
   scaling by 800 squared). Raising only `filter_size_surf` to 0.10 m
   removes PCL overflow warnings but gives 0.02712/0.02714/0.36661 and
   still no result. Review LiDAR-only odometry quality and obtain
   independently grounded motion evidence before attributing all of the
   shortfall to the physical vehicle. Keep the gate unchanged; any
   corrected-IMU comparison needs new matched inputs for both P0/P1. See
   [P1 gate reconstruction](../../outputs/gril/research_0827_repeatability/diagnostics/li_init_gate_audit.yaml).

### Highest priority

1. run perturbation / repeatability tests around the initial transform
2. validate on a bag with both left and right turns
3. measure whether `--planar-motion-policy auto` exits freeze mode automatically on a strongly observable bag

### Data-layer follow-up

1. tune motion-window thresholds:
   - minimum window rotation
   - minimum window translation
   - top-k candidates per window
2. decide whether ground extraction should also move from uniform sampling to window + gate
3. compare windowed motion selection across more than one bag

### Acceptance follow-up

1. define variance thresholds for promoting a result from diagnostic to accepted
2. compare multi-bag repeatability of `z/roll/pitch`
3. keep checking whether IMU gravity can ever beat pose gravity on a better bag

## lidar2lidar

### Highest priority for the four-corner raw rig

1. validate the new workflow-yaml planner on more than one rig topology:
   - `tf_tree`
   - explicit loop
   - explicit chain without loop
2. validate the new `scene_sufficiency.yaml` thresholds on more bags:
   - wall-dominant
   - corner-rich
   - open-space weak
   - dynamic traffic contamination
3. validate the new multi-window repeatability thresholds against accepted vs rejected runs
4. validate wall-thickness / ghosting / corner-spread / slice-sharpness metrics against manual review
5. run prepared-dataset rate ablations at `10 Hz`, `5 Hz`, and `2 Hz`

### Additional scan2map follow-up

1. continue right-edge scan2map diagnostics on more bags
2. validate whether constrained scan2map remains stable across perturbations
3. add stronger repeatability / perturbation testing for accepted scan2map candidates

## lidar2camera / camera

1. validate the new intrinsic acceptance gates on multiple real camera models:
   - forced vs native capture modes
   - wide-angle vs narrow-FOV distortion
   - per-view outlier behavior
2. validate the new lidar2camera visual review surfaces on real runs:
   - image_coverage_heatmap
   - pose_diversity_plot
   - geometry_resolution.csv
   - per_pose_reprojection.csv
3. keep measuring whether geometry-resolution warnings correlate with manual board-observability review
4. decide when physical target upgrades become mandatory rather than optional:
   - reflective / coded LiDAR board
   - ChArUco / AprilTag-grid variants
5. continue treating targetless / learning-based calibration as experimental until repeatability is validated against the reference path
