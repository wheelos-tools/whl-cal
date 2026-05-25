# Verification points

This file lists what still needs evidence before conclusions can be promoted.

## lidar2imu

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
