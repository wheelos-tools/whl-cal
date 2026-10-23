# GRIL `wheelos_core` ingress baseline

This baseline records Apollo record ingress and unchanged native GRIL execution
for the Weilan 0827 capture. It is a reproducible software/input regression
reference, not a calibration-accuracy acceptance. The result remains
`review_required`; independent physical validation is still required.

## Frozen inputs and dependencies

| Item | Baseline |
| --- | --- |
| Capture | Weilan 0827 Apollo record, 3 shards |
| LiDAR / IMU topics | `/apollo/sensor/vanjeelidar/up/PointCloud2` / `/apollo/sensor/gnss/imu` |
| Canonical counts | 1,375 scans; 35,239,034 points; 13,741 IMU samples |
| `wheelos_core` module | `1.0.5` |
| `wheelos_core` source revision | `806d5f0b8a136a5d46a0aabc06eb9a9f4321a289` |
| Bzlmod target | `//:record_ingress` |
| `record_ingress` executable SHA-256 | `d2346e6fb87dc9d6c69e8f946c0246e3fb91f082d85a23f0c237e621d240d60c` |
| Bzlmod lockfile SHA-256 | `67c08f3c419fd8d8f0aa982bea87b0c818967c428bcdd10a393efdc2caf99261` |
| GRIL configuration SHA-256 | `a9b16033818209c3d253e1f37def5335a7cded65ed8815fe8e66db2555b6097b` |
| Native full frontend SHA-256 | `14c8167dda104b905340dcc02f6c890ce32b95bdd96993edfde6070a3b144796` |
| Native batch SHA-256 | `089e15942b0f3346582546142a33ce2b29ae489e4ca836b5db6890d366bce7ab` |
| Repository source base | `30cefedae76435cdc37759d9cfaedef9b4f7c713` plus the working-tree changes under review |
| Toolchain | Bazel 7.6.2; Python 3.10.12; GCC 11.4.0; Linux x86_64 |

Build from `tools/gril_record_core_reader` with the local `apollo-lite`
checkout:

```bash
bazel build \
  --override_module=wheelos_core=/path/to/apollo-lite/pkgs/core \
  //:record_ingress
```

The checked-out Core repository had no modified tracked files when recorded.
It contained unrelated untracked files; they are not part of the Core source
revision or this baseline.

## Canonical ingress result

The baseline dataset is stored at
`outputs/gril/0827_wheelos_core_baseline/dataset/`. The adapter comparison
against `outputs/gril/0827_data_layer_record/` returned `equivalent` for
frames, all counts, ordering, timestamps, offsets, rings, point time, XYZ,
intensity, and IMU signals. Maximum numeric errors were zero, and all four
static transforms matched exactly.

| Artifact | SHA-256 |
| --- | --- |
| Canonical LiDAR arrays (`lidar.npz`) | `fc41c27779411670102532f204f0ce9275d198bafad0f7b715b81b759e8545ea` |
| Canonical IMU arrays (`imu.npz`) | `52043895b9f1e1f6db9a70c1ad64d2e1e40a45bd840cd85ef4f378f38554f6e5` |
| Native GRIL boundary (`native_dataset_v1.bin`) | `835d8de08b94a73da32565862aa6bbbd67ddf4e755b1e1f7578e0f141ab565c1` |

## Runtime baseline

Measured sequentially on the same host, without a clean-build cost:

| Stage | Elapsed | CPU utilization | Peak RSS |
| --- | ---: | ---: | ---: |
| C++ record ingress and canonical artifact creation | 195.06 s | 101% | 1,989,424 KB |
| Canonical-to-result native GRIL run | 34.51 s | 250% | 1,185,140 KB |
| End to end | 229.57 s (~3 min 50 s) | sequential stages | per-stage values above |

The native run was not forced to one thread; the measured CPU utilization
indicates that native GRIL used multiple cores. Treat these timings as a host-
and build-specific baseline, not as a single-thread benchmark. The ingress
build was cached/up to date and completed in 3.93 seconds; that time is excluded
from the end-to-end value.

The result and all trace hashes matched the preceding run on the same canonical
boundary, binary, and configuration:

| Native artifact | SHA-256 |
| --- | --- |
| Calibration result | `e11ca09aa4f1d5cb5ec691b7ba3a46f6378159ac4808cd61fcdfefbe5d08e049` |
| Full frontend trace | `189a9c9e86e84c397e81d8270e004a2449a61c31721f2cc6294fff6ae6b891a2` |
| Batch trace | `81ff62ca0650a82bae1e295e197574716781f66d493d26f23651bc9fd4b886b4` |

The estimate is rotation `[0.451236, 0.007481, 86.219815]` degrees,
translation `[-0.049930, 0.236853, 0.611193]` meters, and time offset
`-0.016880` seconds. It is recorded solely as the output of this baseline run;
it is not independently validated ground truth.

## Reproduction and regression gate

The complete local artifacts are in
`outputs/gril/0827_wheelos_core_baseline/`, which is ignored by Git like other
generated GRIL output. Recreate the canonical input and require equivalence
before running the solver:

```bash
gril-migrate prepare \
  --input /mnt/synology/weilan/lidar2imu/0827 \
  --input-type record \
  --core-record-ingress tools/gril_record_core_reader/bazel-bin/record_ingress \
  --lidar-topic /apollo/sensor/vanjeelidar/up/PointCloud2 \
  --imu-topic /apollo/sensor/gnss/imu \
  --scan-lines 16 \
  --output-dir outputs/gril/0827_wheelos_core_baseline/dataset

gril-migrate compare-datasets \
  --left outputs/gril/0827_wheelos_core_baseline/dataset \
  --right outputs/gril/0827_data_layer_record \
  --output outputs/gril/0827_wheelos_core_baseline/canonical_comparison.yaml
```

For later performance or algorithm iterations, keep this dataset, GRIL
configuration, native executable, and native boundary fixed. Compare output
hashes, calibration parameters, traces, elapsed time, and peak RSS. Do not
accept a timing improvement that changes the canonical boundary or weakens
repeatability, and do not interpret successful execution as calibration
accuracy.
