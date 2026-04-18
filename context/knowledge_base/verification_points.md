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

1. continue right-edge scan2map diagnostics on more bags
2. validate whether constrained scan2map remains stable across perturbations
3. add stronger repeatability / perturbation testing for accepted scan2map candidates

## lidar2camera / camera2lidar

1. define the repo-level data -> algorithm -> evaluation split
2. create a stable dataset artifact for image + LiDAR pair selection
3. add window + gate for invalid image / point-cloud / board windows
4. define stable evaluation outputs:
   - reprojection error
   - holdout reprojection error
   - overlay quality
   - drift to initial prior
   - recommendation field
5. decide the current acceptance baseline between:
   - reference-based checkerboard path
   - targetless learning / scene path
