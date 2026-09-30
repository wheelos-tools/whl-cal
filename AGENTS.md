# Agent Guide

## Commands
- Install: `python3 -m venv .venv && source .venv/bin/activate && pip install -e .`
- Lint: `black --check . && isort --check-only . && flake8 .`
- Build: `python -m pip install -e .`
- Test: no unit suite; run the smallest relevant CLI smoke command from the matching skill or quick start.

## Principles & Anti-Patterns
- **DO**: read the relevant source, quick start, knowledge file, and skill before changing behavior.
- **DO**: preserve the extraction → algorithm → evaluation split and stable review artifacts.
- **DO**: validate calibration with metrics, visualization, repeatability, and holdout evidence.
- **DO NOT**: modify unrelated code, invent missing facts, or use configured extrinsics as ground truth.
- **DO NOT**: accept solver convergence, registration fitness, or a bounded trajectory as sufficient evidence.

## Knowledge
- [architecture.md](.agents/knowledge/architecture.md) — package boundaries and data flow.
- [conventions.md](.agents/knowledge/conventions.md) — artifacts, schemas, style, and documentation rules.
- [troubleshooting.md](.agents/knowledge/troubleshooting.md) — failure triage and authoritative diagnostics.
- [lidar2imu_research.md](.agents/knowledge/lidar2imu_research.md) — LiDAR-IMU data gates, 0827 comparison, baseline-preserving iteration, and acceptance rules.
- [camera_intrinsic_research.md](.agents/knowledge/camera_intrinsic_research.md) — intrinsic calibration baseline, validation gaps, and evidence-gated improvement roadmap.
- [lidar2camera_research.md](.agents/knowledge/lidar2camera_research.md) — reference and targetless LiDAR-camera calibration comparison, current implementation limits, and validation roadmap.
- [lidar2lidar_research.md](.agents/knowledge/lidar2lidar_research.md) — scan2scan baseline, semantic/overlap-aware candidates, and controlled comparison roadmap.

## Skills
- [algorithm-research](.agents/skills/algorithm-research/SKILL.md)
- [calibration-algorithm-design](.agents/skills/calibration-algorithm-design/SKILL.md)
- [calibration-algorithm-iteration](.agents/skills/calibration-algorithm-iteration/SKILL.md)
- [calibration-algorithm-validation](.agents/skills/calibration-algorithm-validation/SKILL.md)
- [calibration-benchmark-and-ablation](.agents/skills/calibration-benchmark-and-ablation/SKILL.md)
- [calibration-capture-design](.agents/skills/calibration-capture-design/SKILL.md)
- [calibration-failure-analysis](.agents/skills/calibration-failure-analysis/SKILL.md)
- [calibration-release-gating](.agents/skills/calibration-release-gating/SKILL.md)
- [calibration-sota-upgrade](.agents/skills/calibration-sota-upgrade/SKILL.md)
- [gril-calib-validation](.agents/skills/gril-calib-validation/SKILL.md)
