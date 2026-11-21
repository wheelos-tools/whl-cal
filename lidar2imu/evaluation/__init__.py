"""Input contracts, result comparisons, and calibration review reports."""

from lidar2imu.evaluation.customer_summary import (
    build_customer_summary,
    write_customer_summary,
)
from lidar2imu.evaluation.dual_sensor_diagnostics import (
    DualDiagnosticThresholds,
    evaluate_dual_lidar2imu_diagnostics,
    write_colored_dual_pcd,
)
from lidar2imu.evaluation.input_contract import (
    build_input_contract,
    write_input_contract,
)
from lidar2imu.evaluation.quality_gates import (
    QualityGateThresholds,
    evaluate_holdout_trace,
    evaluate_run_quality_gates,
    parse_detailed_gril_result,
    write_quality_gate_report,
)

__all__ = [
    "DualDiagnosticThresholds",
    "QualityGateThresholds",
    "build_customer_summary",
    "build_input_contract",
    "evaluate_dual_lidar2imu_diagnostics",
    "evaluate_holdout_trace",
    "evaluate_run_quality_gates",
    "parse_detailed_gril_result",
    "write_colored_dual_pcd",
    "write_customer_summary",
    "write_input_contract",
    "write_quality_gate_report",
]
