## Quick Start

### Install dependencies

```bash
pip install opencv-python numpy pyyaml
```

### Prepare the chessboard calibration board

* The default number of inner corners is (9, 6)
* The square size is defined in the YAML file; e.g. if each square is 25 mm then set `square_size = 0.025`

### Run the script

```bash
python camera/intrinsic.py
```

### Capture workflow

Key controls:

| Key | Operation                                                |
| --- | -------------------------------------------------------- |
| S   | Save this frame (when corners are detected successfully) |
| C   | Perform calibration (capture at least 20 frames)         |
| Q   | Quit                                                     |

The UI will display detection status and number of frames captured at the bottom.

---

### Auto-output

#### `calibration_results.yaml`

```yaml
calibration_time: "2024-10-28 14:55:21"
camera_matrix:
  data: [[1051.123, 0, 640.2], [0, 1052.011, 360.3], [0, 0, 1]]
distortion_coefficients:
  data: [0.05, -0.02, 0.001, -0.0005, 0.0]
image_width: 1280
image_height: 720
reprojection_error: 0.1823
```
