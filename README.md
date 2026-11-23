# whl-cal

Calibration workflows for Apollo records, LiDAR point clouds, and camera
images. Start with the [workflow index](docs/quickstart_index.md); it links
the current module guides, data-collection checklist, and result-review guide.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

For a smaller environment, install only the workflow extra you need:
`camera`, `camera2camera`, `lidar2camera`, `lidar2lidar`, or `lidar2imu`.

LiDAR-to-IMU requires separate native GRIL and Apollo record-ingress builds.
The `lidar2imu` extra installs Python dependencies only; see the
[LiDAR-to-IMU quick start](docs/lidar2imu_quickstart.md). GRIL remains a
research workflow, and solver completion alone is not evidence of physical
accuracy.

## Repository guidance

- [Agent entry point](AGENTS.md)
- [Agent knowledge](.agents/knowledge/README.md)
- [Agent notes and investigation records](.agents/notes/README.md)
- [Documentation ownership](docs/documentation_organization.md)
