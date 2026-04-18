# Calibration iteration rules

## Core principle

Calibration work in this repo follows one rule above all others:

- **算法 + 评测指标持续迭代，以测试数据为准**

In practice, that means:

1. do not accept a result because the optimizer converged
2. do not accept a result because one metric improved in isolation
3. only accept the level of conclusion supported by real-bag evidence

## Required split

Every calibration workflow should stay split into:

1. **data layer**
   - raw record reading
   - sample extraction / window selection
   - gating of invalid or abnormal windows
2. **algorithm layer**
   - solver / refinement logic
   - priors / constraints / robust losses
3. **evaluation layer**
   - stable metrics
   - stable diagnostics
   - acceptance recommendation

## Acceptance rules

### Accept only what is observable

- if the bag only supports `z/roll/pitch`, do not promote `x/y/yaw`
- if scan2map gain comes mostly from `z/pitch/roll` drift, do not call it a full 6DoF gain
- if a bag has one-sided turning, treat planar outputs as weakly observable unless repeated data proves otherwise

### Use windows and gates at the data layer

The preferred data-layer pattern is:

1. build candidate windows
2. summarize each window
3. gate weak or abnormal windows
4. choose valid windows for calibration

This is better than globally sorting all raw candidates because it preserves coverage
while blocking locally bad or abnormal segments.

### Use stable evaluation surfaces

Algorithms may change aggressively, but the following should stay stable:

- `metrics.yaml`
- `diagnostics/*.yaml`
- explicit recommendation fields
- skip / rejection reasons

## Promotion rule

Move a result from “diagnostic” to “accepted” only when:

1. the data layer is healthy
2. observability is sufficient
3. holdout / residual / drift checks agree
4. repeatability across bags or perturbations is acceptable
