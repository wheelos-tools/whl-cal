from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from camera.intrinsic_evaluation import write_review_artifacts


class CalibrationCustomerSummaryTest(unittest.TestCase):
    def test_camera_customer_summary_is_concise_and_keeps_diagnostics(self) -> None:
        sample_records = [
            {
                "sample_id": index + 1,
                "image_size_wh": {"width": 640, "height": 480},
            }
            for index in range(9)
        ]
        coverage = {
            "occupied_cell_count": 9,
            "grid_counts": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "minimum_cell_count": 1,
            "required_samples_per_cell": 1,
            "horizontal_span_ratio": 0.8,
            "vertical_span_ratio": 0.8,
            "edge_corner_coverage": {
                "covered_quadrant_count": 4,
                "required_quadrant_count": 4,
            },
        }
        per_view = [
            {
                "sample_id": index + 1,
                "rms_px": 0.2,
                "p95_px": 0.3,
                "point_count": 4,
            }
            for index in range(9)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.yaml"
            calibration_path.write_text("camera_matrix: {}\n")
            artifacts = write_review_artifacts(
                calibration_path,
                min_total_samples=9,
                sample_records=sample_records,
                capture_runtime_info={
                    "actual_capture_resolution": {"width": 640, "height": 480}
                },
                calibration_target={"type": "chessboard"},
                comparison_view_path=str(root / "comparison_view.png"),
                global_reprojection_rms=0.2,
                solver_reported_rms=0.2,
                per_view_report=per_view,
                coverage=coverage,
                monotonicity_report={
                    "status": "pass",
                    "min_radial_derivative": 0.9,
                },
            )
            summary = yaml.safe_load(Path(artifacts["customer_summary"]).read_text())

        self.assertEqual(summary["verdict"], "accepted")
        self.assertTrue(summary["release_ready"])
        self.assertEqual(summary["key_metrics"]["accepted_samples"], 9)
        self.assertEqual(len(summary["key_metrics"]), 5)
        self.assertNotIn("quality_gates", summary)
        self.assertTrue(artifacts["final_acceptance"]["gates"])


if __name__ == "__main__":
    unittest.main()
