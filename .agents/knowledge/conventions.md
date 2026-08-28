# Conventions

## When
Read before adding APIs, artifacts, transforms, metrics, tests, or documentation.

## Rules
- Keep top-level calibration outputs and diagnostic surfaces stable.
- Use the shared extrinsic I/O helpers and canonical frame/child/transform schema.
- Match existing names, typing, formatting, and error-reporting patterns.
- Update the matching overview, quick start, and design document when behavior changes.
- Use the smallest CLI validation that directly exercises the changed behavior.

## Do NOTs
- Do not silently return success-shaped fallback results.
- Do not duplicate transform parsing or serialization.
- Do not replace explicit skip reasons with fitness-only decisions.
- Do not copy implementation details into agent guidance; link the source of truth.

## Sources
- Extrinsic schema and helpers: `lidar2lidar/extrinsic_io.py`
- Output review contract: `docs/calibration_review_guide.md`
- Documentation responsibilities: `docs/docs_vs_context.md`
- Lint versions and CI entrypoint: `.github/workflows/lint-format.yml`
- Lint implementation: `scripts/ci/apollo_lint.sh`
- User-facing command inventory: `docs/quickstart_index.md`
