# LiDAR-to-LiDAR decisions

- Keep direct `scan2scan` as the production baseline when overlap and scene
  geometry are sufficient.
- Treat `scan2map` as conditional refinement/validation, not a blanket
  replacement.
- Use topology-aware, healthy edges and multi-window evidence; a loop residual,
  registration fitness, or one merged-cloud image is not sufficient by itself.
- Review scene sufficiency, repeatability, transform consistency, and visual
  geometry before acceptance.

See the [quick start](../../docs/lidar2lidar_quickstart.md) and
[design](../../docs/lidar2lidar_design.md) for supported behavior. Historical
playbooks and detailed candidate comparisons are in
[research notes](../notes/lidar2lidar_research.md).
