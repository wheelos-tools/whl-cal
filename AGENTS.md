# Agent Guide

## Repository map

- `docs/`: maintained project documentation, including user guides and design contracts.
- `.agents/skills/`: repeatable workflows and methods.
- `.agents/knowledge/`: concise, durable engineering facts and links to their evidence.
- `.agents/notes/`: dated investigations, experiment records, and temporary follow-ups; verify against current code before relying on them.
- `.github/`: repository governance and automation.

Start with [the knowledge index](.agents/knowledge/README.md) for engineering context and [the documentation index](docs/quickstart_index.md) for supported workflows. Code is authoritative when documentation and implementation disagree.

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
See the [knowledge index](.agents/knowledge/README.md) for package boundaries, conventions, troubleshooting, and module-specific research.

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
