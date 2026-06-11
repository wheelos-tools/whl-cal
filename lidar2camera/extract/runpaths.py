"""Per-run path resolution shared by the pipeline scripts.

A calibration capture lives under  lidar2camera/runs/<id>/  with:
    inputs/             raw video + csv + lidar PCDs
    cam_candidates/     step-1 detections (intermediate)
    calibration_data/   step-2 paired NNNN.png / NNNN.pcd
    calibration_output/ step-3 extrinsic + overlays

config.yaml selects the active run with `run_dir:` and lists the input paths
(relative to the run dir, or absolute). To calibrate another capture, drop its
data under a new runs/<id>/inputs/ and point `run_dir` at it.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent          # lidar2camera/
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RUN = (ROOT / CFG["run_dir"]).resolve()                 # active run directory


def _under_run(p):
    p = Path(p)
    return p if p.is_absolute() else (RUN / p)


# inputs (from config.data, relative to the run dir)
CAMERA_VIDEO = _under_run(CFG["data"]["camera_video"])
CAMERA_CSV = _under_run(CFG["data"]["camera_csv"])
LIVOX_DIR = _under_run(CFG["data"]["livox_pcd_dir"])

# fixed per-run working/output subdirs
CAND_DIR = RUN / "cam_candidates"
PAIR_DIR = RUN / "calibration_data"
OUT_DIR = RUN / "calibration_output"
