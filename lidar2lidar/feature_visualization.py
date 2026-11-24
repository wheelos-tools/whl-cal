"""Headless feature galleries backed by the actual semantic solver trace."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import ConvexHull, cKDTree

from lidar2lidar.feature_trace import CHANNELS, geometric_residual_vectors

COLORS = {
    "ground": "#2980b9",
    "facade": "#e67e22",
    "pole": "#8e44ad",
    "intensity": "#16a085",
}
PREFIXES = {"ground": "G", "facade": "F", "pole": "P", "intensity": "I"}
PROJECTIONS = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _sample(size: int, maximum: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, maximum), dtype=int)


def _feature_id(channel: str, index: int) -> str:
    return f"{PREFIXES[channel]}{index:05d}"


def _flatten_arrays(prefix: str, value: Any, arrays: dict[str, np.ndarray]) -> None:
    if isinstance(value, np.ndarray):
        arrays[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten_arrays(f"{prefix}_{key}" if prefix else key, item, arrays)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_arrays(f"{prefix}_{index}", item, arrays)


def _support(side: dict[str, Any], channel: str, members: np.ndarray) -> np.ndarray:
    data = side["features"][channel]
    offsets = data["support_offsets"]
    parts = [data["support_indices"][offsets[i] : offsets[i + 1]] for i in members]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=int)


def group_features(side: dict[str, Any]) -> list[dict[str, Any]]:
    """Diagnostic-only connected planar patches and vertical line candidates."""
    objects = []
    for channel in ("facade", "pole"):
        data = side["features"][channel]
        points, vectors = data["points"], data["vectors"]
        parent = np.arange(len(points))

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for a, b in sorted(cKDTree(points).query_pairs(1.05)):
            delta = points[b] - points[a]
            if abs(float(vectors[a] @ vectors[b])) < 0.96:
                continue
            if channel == "facade":
                close = max(abs(delta @ vectors[a]), abs(delta @ vectors[b])) < 0.12
            else:
                close = (
                    max(
                        np.linalg.norm(delta - (delta @ vectors[a]) * vectors[a]),
                        np.linalg.norm(delta - (delta @ vectors[b]) * vectors[b]),
                    )
                    < 0.25
                )
            if close:
                parent[root(b)] = root(a)
        components: dict[int, list[int]] = {}
        for i in range(len(points)):
            components.setdefault(root(i), []).append(i)
        for members_list in components.values():
            if len(members_list) < (3 if channel == "facade" else 2):
                continue
            members = np.asarray(members_list, dtype=int)
            support = _support(side, channel, members)
            raw = side["frame_points"][support]
            center = raw.mean(axis=0)
            _, singular, axes = np.linalg.svd(raw - center, full_matrices=False)
            if channel == "facade":
                uv = (raw - center) @ axes[:2].T
                if len(uv) < 3 or singular[1] < 1e-8:
                    continue
                boundary = center + uv[ConvexHull(uv).vertices] @ axes[:2]
                boundary = np.vstack([boundary, boundary[0]])
                direction = axes[2]
                residual = (raw - center) @ direction
                kind = "Plane"
            else:
                direction = axes[0]
                along = (raw - center) @ direction
                boundary = (
                    center + np.array([along.min(), along.max()])[:, None] * direction
                )
                delta = raw - center
                residual = np.linalg.norm(
                    delta - (delta @ direction)[:, None] * direction, axis=1
                )
                kind = "PoleCandidate"
            objects.append(
                {
                    "id": f"{kind}_{len(objects) + 1:03d}",
                    "channel": channel,
                    "members": members,
                    "support_count": len(support),
                    "center": center,
                    "direction": direction,
                    "boundary": boundary,
                    "fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
                    "diagnostic_only": True,
                    "cylinder_confirmed": False,
                }
            )
    return objects


def _bounds(trace: dict[str, Any]) -> list[tuple[float, float]]:
    matrices = [trace["T_initial"], trace.get("T_coarse", trace["T_initial"])]
    matrices.extend(s["T_after"] for s in trace["stages"])
    sp = trace["source"]["frame_points"]
    tp = trace["target"]["frame_points"]
    clouds = [tp[np.isfinite(tp).all(axis=1)]]
    finite = sp[np.isfinite(sp).all(axis=1)]
    clouds.extend(_transform(finite, t) for t in matrices)
    points = np.vstack(clouds)
    if len(points) == 0:
        return [(-1.0, 1.0)] * 3
    low, high = points.min(axis=0), points.max(axis=0)
    pad = np.maximum((high - low) * 0.04, 0.5)
    return list(zip(low - pad, high + pad))


def _figure(rows: int = 3, cols: int = 2):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, 10 if rows == 3 else 4.5 * rows), constrained_layout=True)
    FigureCanvasAgg(fig)
    options = {"height_ratios": [1, 0.3, 0.3]} if rows == 3 else None
    return fig, fig.subplots(rows, cols, squeeze=False, gridspec_kw=options)


def _axes(ax: Any, projection: tuple[int, int, str], bounds: Any) -> None:
    x, y, _ = projection
    ax.set_xlim(bounds[x])
    ax.set_ylim(bounds[y])
    ax.set_xlabel(f"{'XYZ'[x]} [m]")
    ax.set_ylabel(f"{'XYZ'[y]} [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def _background(ax: Any, points: np.ndarray, x: int, y: int) -> None:
    finite = points[np.isfinite(points).all(axis=1)]
    sampled = finite[_sample(len(finite), 9000)]
    ax.scatter(sampled[:, x], sampled[:, y], s=0.35, c="#aaaaaa", alpha=0.3)


def _draw_side(
    ax: Any,
    side: dict[str, Any],
    matrix: np.ndarray,
    projection: tuple[int, int, str],
    groups: list[dict[str, Any]],
    decisions: dict[str, Any] | None = None,
    target_side: bool = False,
    channels: tuple[str, ...] = CHANNELS,
    show_huber: bool = False,
) -> None:
    x, y, _ = projection
    _background(ax, _transform(side["frame_points"], matrix), x, y)
    for channel in channels:
        data = side["features"][channel]
        points = _transform(data["points"], matrix)
        color = COLORS[channel]
        ax.scatter(
            points[:, x],
            points[:, y],
            s=7,
            c=color,
            label=f"{channel}: {len(points)}",
            alpha=0.75,
        )
        if decisions is not None:
            accepted = np.zeros(len(points), dtype=bool)
            key = "target_indices" if target_side else "source_indices"
            accepted[decisions[channel][key]] = True
            ax.scatter(
                points[~accepted, x],
                points[~accepted, y],
                s=15,
                facecolors="none",
                edgecolors="#888888" if target_side else "#c0392b",
                linewidths=0.5,
            )
            weights = decisions[channel].get("huber_component_weights")
            if weights is not None and show_huber:
                selected = decisions[channel][key][weights.min(axis=1) < 0.5]
                ax.scatter(
                    points[selected, x],
                    points[selected, y],
                    s=30,
                    facecolors="none",
                    edgecolors="#b8860b",
                    linewidths=0.9,
                )
        subset = _sample(len(points), 24 if channel != "ground" else 6)
        vectors = data["vectors"] @ matrix[:3, :3].T
        ax.quiver(
            points[subset, x],
            points[subset, y],
            vectors[subset, x],
            vectors[subset, y],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=2,
            width=0.002,
        )
        for i in _sample(len(points), 4):
            ax.annotate(
                _feature_id(channel, int(i)),
                (points[i, x], points[i, y]),
                fontsize=6,
                color=color,
            )
    for group in groups:
        if group["channel"] not in channels:
            continue
        boundary = _transform(group["boundary"], matrix)
        ax.plot(
            boundary[:, x],
            boundary[:, y],
            c=COLORS[group["channel"]],
            lw=1.1,
            alpha=0.8,
        )
    annotated: list[np.ndarray] = []
    for group in sorted(groups, key=lambda g: -g["support_count"]):
        if group["channel"] not in channels:
            continue
        center = _transform(group["center"][None], matrix)[0]
        projected = center[[x, y]]
        extent = np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())])
        if any(np.linalg.norm((projected - p) / extent) < 0.12 for p in annotated):
            continue
        ax.annotate(
            group["id"],
            projected,
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )
        annotated.append(projected)
        if len(annotated) == 8:
            break


def _display_pairs(
    points: np.ndarray, residual: np.ndarray, maximum: int
) -> np.ndarray:
    if len(points) <= maximum:
        return np.arange(len(points))
    spatial = np.lexsort((points[:, 1], points[:, 0]))
    by_residual = np.argsort(residual, kind="stable")
    return np.unique(
        np.r_[
            spatial[_sample(len(points), maximum // 2)],
            by_residual[_sample(len(points), maximum - maximum // 2)],
        ]
    )


def _plot_features(
    path: Path,
    trace: dict[str, Any],
    groups: Any,
    bounds: Any,
    *,
    channels: tuple[str, ...] = CHANNELS,
) -> None:
    fig, axes = _figure()
    for row, projection in enumerate(PROJECTIONS):
        for col, side in enumerate(("source", "target")):
            matrix = trace["T_initial"] if side == "source" else np.eye(4)
            _axes(axes[row, col], projection, bounds)
            _draw_side(
                axes[row, col],
                trace[side],
                matrix,
                projection,
                groups[side],
                channels=channels,
            )
            label = trace[side].get("label", side)
            axes[row, col].set_title(f"{label} ({side}) / {projection[2]} / input seed")
            if row == 0:
                axes[row, col].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Detected primitives (arrows: normals / line directions; "
        "groups: diagnostic only)"
    )
    fig.savefig(path, dpi=110)


def _plot_matches(
    path: Path,
    trace: dict[str, Any],
    stage: dict[str, Any],
    bounds: Any,
    *,
    after: bool = False,
    channels: tuple[str, ...] = CHANNELS,
) -> dict[str, int]:
    from matplotlib.patches import ConnectionPatch

    fig, axes = _figure()
    matrix = stage["T_after"] if after else stage["T_matching"]
    counts = {}
    for row, projection in enumerate(PROJECTIONS):
        x, y, _ = projection
        for col, side in enumerate(("source", "target")):
            _axes(axes[row, col], projection, bounds)
            _draw_side(
                axes[row, col],
                trace[side],
                matrix if side == "source" else np.eye(4),
                projection,
                [],
                stage["channels"],
                target_side=(side == "target"),
                channels=channels,
                show_huber=after,
            )
            label = trace[side].get("label", side)
            axes[row, col].set_title(f"{label} ({side}) / {projection[2]}")
        for channel, data in stage["channels"].items():
            if channel not in channels:
                continue
            si, ti = data["source_indices"], data["target_indices"]
            points = _transform(
                trace["source"]["features"][channel]["points"][si], matrix
            )
            target = trace["target"]["features"][channel]["points"][ti]
            vectors = trace["target"]["features"][channel]["vectors"][ti]
            residual = np.linalg.norm(
                geometric_residual_vectors(channel, points - target, vectors),
                axis=1,
            )
            shown = _display_pairs(points, residual, 32)
            counts[channel] = len(shown)
            for k in shown:
                ratio = min(float(residual[k]) / 0.20, 1.0)
                color = (ratio, 0.65 * (1.0 - ratio), 0.1)
                fig.add_artist(
                    ConnectionPatch(
                        xyA=points[k, [x, y]],
                        xyB=target[k, [x, y]],
                        coordsA="data",
                        coordsB="data",
                        axesA=axes[row, 0],
                        axesB=axes[row, 1],
                        color=color,
                        linewidth=0.6,
                        alpha=0.45,
                        linestyle=(
                            ":"
                            if after
                            and "huber_component_weights" in data
                            and data["huber_component_weights"][k].min() < 0.5
                            else "-"
                        ),
                    )
                )
        description = " / ".join(
            f"{c} {counts[c]}/{len(d['source_indices'])}"
            for c, d in stage["channels"].items()
            if c in channels
        )
        axes[row, 0].set_xlabel(
            axes[row, 0].get_xlabel() + "\nshown/total: " + description
        )
    label = "post-solve frozen pairs" if after else "matching-time pairs"
    metric = (
        "post-solve geometric residual" if after else "matching-time geometric residual"
    )
    if not stage["optimized"]:
        metric += " (NOT optimized)"
    weights_note = (
        "gold/dotted: post-solve min Huber<0.5 (display only)"
        if after
        else "NOT object matches"
    )
    fig.suptitle(
        f"Stage {stage['index']} / {label} / color: {metric}, green=0 red>=0.20m\n"
        "Hollow source/target: unmatched; lines sampled by space and residual; "
        + weights_note
    )
    fig.savefig(path, dpi=110)
    return counts


def _plot_line_context(path: Path, trace: Any, bounds: Any) -> None:
    fig, axes = _figure()
    for row, projection in enumerate(PROJECTIONS):
        x, y, _ = projection
        for col, side in enumerate(("source", "target")):
            transform = trace["T_initial"] if side == "source" else np.eye(4)
            raw = trace[side]["frame_points"]
            raw = raw[np.isfinite(raw).all(axis=1)]
            raw = _transform(raw[_sample(len(raw), 12000)], transform)
            ax = axes[row, col]
            ax.scatter(raw[:, x], raw[:, y], s=0.4, c="#bbbbbb", alpha=0.25)
            context = trace[side].get("line_context", {})
            points = _transform(context.get("points", np.empty((0, 3))), transform)
            rejected = context.get("rejected_as_planar", np.empty(0, dtype=bool))
            for mask, color, label in (
                (rejected, "#d73027", "planar context: excluded"),
                (~rejected, "#1a9850", "retained before count cap"),
            ):
                ax.scatter(
                    points[mask, x],
                    points[mask, y],
                    s=12,
                    c=color,
                    label=f"{label} ({int(mask.sum())})",
                )
            _axes(ax, projection, bounds)
            ax.set_title(f"{side} / {projection[2]}")
            ax.legend(fontsize=7)
    fig.suptitle(
        "Line classification before matching: wider non-ground context\n"
        "Red wall-sampling strips do not enter the pole solver; "
        "green is NOT a confirmed cylinder or a match"
    )
    fig.savefig(path, dpi=110)


def _plot_overlays(path: Path, trace: Any, final: np.ndarray, bounds: Any) -> None:
    fig, axes = _figure()
    for row, projection in enumerate(PROJECTIONS):
        x, y, _ = projection
        for col, matrix in enumerate((trace["T_initial"], final)):
            for side, color, transform in (
                ("source", "#00a7bd", matrix),
                ("target", "#d52680", np.eye(4)),
            ):
                points = trace[side]["frame_points"]
                points = points[np.isfinite(points).all(axis=1)]
                points = _transform(points[_sample(len(points), 14000)], transform)
                axes[row, col].scatter(
                    points[:, x], points[:, y], s=0.7, c=color, alpha=0.45, label=side
                )
            _axes(axes[row, col], projection, bounds)
            axes[row, col].set_title(
                ("Before" if col == 0 else "After") + f" / {projection[2]}"
            )
            axes[row, col].legend(fontsize=8)
    fig.suptitle(
        "Same reference frame, scale and samples / cyan: source / magenta: target"
    )
    fig.savefig(path, dpi=110)


def _plot_3d(
    path: Path, trace: Any, final: np.ndarray, bounds: Any, groups: Any
) -> None:
    fig, unused = _figure(1, 2)
    for ax in unused.ravel():
        ax.remove()
    for index, matrix in enumerate((trace["T_initial"], final)):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        for side, color, transform in (
            ("source", "#00a7bd", matrix),
            ("target", "#d52680", np.eye(4)),
        ):
            points = trace[side]["frame_points"]
            points = points[np.isfinite(points).all(axis=1)]
            points = _transform(points[_sample(len(points), 6000)], transform)
            ax.scatter(*points.T, s=0.4, c=color, alpha=0.35)
            for group in groups[side]:
                boundary = _transform(group["boundary"], transform)
                ax.plot(*boundary.T, color=COLORS[group["channel"]], lw=1.4)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_box_aspect([b - a for a, b in bounds])
        ax.view_init(elev=25, azim=-55)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Before" if index == 0 else "After")
    fig.suptitle(
        "3D overlay / boundaries and axes are diagnostic groups, not cylinders"
    )
    fig.savefig(path, dpi=130)


def _plot_residuals(path: Path, stages: Any) -> None:
    fig, axes = _figure(2, 2)
    for channel in CHANNELS:
        optimized = [s for s in stages if s["optimized"]]
        axes[0, 0].plot(
            [s["index"] for s in stages],
            [len(s["channels"][channel]["source_indices"]) for s in stages],
            "-o",
            c=COLORS[channel],
            label=channel,
        )
        axes[0, 1].plot(
            [s["index"] for s in optimized],
            [
                (
                    np.sqrt(np.mean(s["channels"][channel]["residual_norm_m"] ** 2))
                    if len(s["channels"][channel]["source_indices"])
                    else np.nan
                )
                for s in optimized
            ],
            "-o",
            c=COLORS[channel],
            label=channel,
        )
        if optimized:
            data = optimized[-1]["channels"][channel]
            residual = data["residual_norm_m"]
            if len(residual):
                axes[1, 0].hist(
                    residual,
                    bins=35,
                    histtype="step",
                    color=COLORS[channel],
                    label=channel,
                )
                axes[1, 1].hist(
                    data["huber_component_weights"].min(axis=1),
                    bins=20,
                    histtype="step",
                    color=COLORS[channel],
                    label=channel,
                )
    for ax, title in zip(
        axes.ravel(),
        (
            "Actual candidate pairs per stage (failed stages included)",
            "Post-solve geometric norm RMSE [m] (pole: point-to-line norm)",
            "Last optimized stage residual norm [m]",
            "Minimum Huber component weight (not binary rejection)",
        ),
    ):
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Frozen solver correspondences, not post-hoc nearest-neighbor evaluation"
    )
    fig.savefig(path, dpi=110)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _correspondences(trace: Any) -> list[dict[str, Any]]:
    rows = []
    for stage in trace["stages"]:
        for channel, data in stage["channels"].items():
            order = {int(i): k for k, i in enumerate(data["source_indices"])}
            for i, reason in enumerate(data["reason"]):
                target_index = int(data["nearest_target"][i])
                row = {
                    "stage": stage["index"],
                    "channel": channel,
                    "source_id": _feature_id(channel, i),
                    "target_id": (
                        _feature_id(channel, target_index) if target_index >= 0 else ""
                    ),
                    "source_index": i,
                    "target_index": target_index,
                    "reason": str(reason),
                    "candidate_accepted": reason == "accepted",
                    "participated_in_solve": reason == "accepted"
                    and stage["optimized"],
                    "distance_at_matching_m": _plain(data["distance_m"][i]),
                    "residual_norm_m": None,
                    "euclidean_after_m": None,
                    "channel_weight": None,
                    "huber_min_weight": None,
                    "huber_component_weights": "",
                    "checks": json.dumps(
                        _plain({k: v[i] for k, v in data["gates"].items()})
                    ),
                    "measurements": json.dumps(
                        _plain({k: v[i] for k, v in data["values"].items()})
                    ),
                }
                if i in order and stage["optimized"]:
                    j = order[i]
                    row.update(
                        residual_norm_m=float(data["residual_norm_m"][j]),
                        euclidean_after_m=float(data["euclidean_after_m"][j]),
                        channel_weight=data["channel_weight"],
                        huber_min_weight=float(
                            data["huber_component_weights"][j].min()
                        ),
                        huber_component_weights=json.dumps(
                            _plain(data["huber_component_weights"][j])
                        ),
                    )
                rows.append(row)
    return rows


def _object_associations(trace: Any, groups: Any) -> list[dict[str, Any]]:
    lookup = {}
    for side in ("source", "target"):
        lookup[side] = {
            (g["channel"], int(i)): g["id"] for g in groups[side] for i in g["members"]
        }
    rows = []
    for stage in trace["stages"]:
        counts: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        for channel, data in stage["channels"].items():
            for si, ti in zip(data["source_indices"], data["target_indices"]):
                a = lookup["source"].get((channel, int(si)))
                b = lookup["target"].get((channel, int(ti)))
                if a and b:
                    counts.setdefault((channel, a, b), []).append((int(si), int(ti)))
        for (channel, a, b), pairs in counts.items():
            source_group = next(g for g in groups["source"] if g["id"] == a)
            target_group = next(g for g in groups["target"] if g["id"] == b)
            rows.append(
                {
                    "stage": stage["index"],
                    "channel": channel,
                    "source_object": a,
                    "target_object": b,
                    "pair_count": len(pairs),
                    "source_coverage": len({i for i, _ in pairs})
                    / len(source_group["members"]),
                    "target_coverage": len({i for _, i in pairs})
                    / len(target_group["members"]),
                    "optimized": stage["optimized"],
                    "diagnostic_only": True,
                }
            )
    return rows


def _interactive_data(trace: Any, groups: Any, metadata: Any, bounds: Any) -> Any:
    data = {
        "bounds": bounds,
        "metadata": metadata,
        "colors": COLORS,
        "prefixes": PREFIXES,
        "sides": {},
        "stages": [],
    }
    for side in ("source", "target"):
        record = trace[side]
        raw = record["frame_points"]
        finite = raw[np.isfinite(raw).all(axis=1)]
        scan_indices = record.get("scan_indices")
        point_indices = record.get("point_indices")
        if scan_indices is None:
            scan_indices = np.full(len(raw), -1, dtype=int)
        if point_indices is None:
            point_indices = np.arange(len(raw))
        features = {}
        for channel in CHANNELS:
            item = record["features"][channel]
            features[channel] = {k: item[k] for k in ("points", "vectors", "pca")}
            features[channel]["support_count"] = np.diff(item["support_offsets"])
            previews = []
            for i in range(len(item["points"])):
                support = _support(record, channel, np.array([i]))
                subset = support[_sample(len(support), 20)]
                previews.append(
                    {
                        "frame_indices": subset,
                        "scan_indices": scan_indices[subset],
                        "raw_point_indices": point_indices[subset],
                        "points": raw[subset],
                    }
                )
            features[channel]["support_preview"] = previews
        data["sides"][side] = {
            "background": finite[_sample(len(finite), 5000)],
            "features": features,
            "objects": groups[side],
        }
    for stage in trace["stages"]:
        data["stages"].append(
            {
                "index": stage["index"],
                "optimized": stage["optimized"],
                "T_matching": stage["T_matching"],
                "T_after": stage["T_after"],
                "channels": stage["channels"],
            }
        )
    data["T_initial"] = trace["T_initial"]
    return _plain(data)


_VIEWER = r"""
<!doctype html><meta charset="utf-8"><title>L2L feature review</title>
<style>
body{font:14px sans-serif;margin:18px;background:#f6f7f8;color:#17202a}
canvas{width:100%;background:white;border:1px solid #bbb}
pre{white-space:pre-wrap;max-height:480px;overflow:auto;background:white;padding:12px}
img{max-width:100%}select{margin:6px}a{color:#176ca6}
</style>
<h1>L2L detected features and actual solver pairs</h1>
<p id="meta"></p>
<p>Left: source; right: target. Same reference frame and metric scale.
Hollow red: unmatched source; grey target: not selected by this stage.
Lines are sampled; clicking either endpoint reveals all its accepted pairs.
Plane / PoleCandidate groups are diagnostic only; no confirmed cylinders.</p>
<label>Stage <select id="stage"></select></label>
<label>Projection <select id="projection">
<option value="0,1">XY</option><option value="0,2">XZ</option>
<option value="1,2">YZ</option></select></label>
<label>Channel <select id="channel"><option value="all">All</option>
<option>facade</option><option>pole</option><option>ground</option>
<option>intensity</option></select></label>
<label>Pose <select id="pose"><option value="T_matching">Matching time</option>
<option value="T_after">Post-solve frozen pairs</option></select></label>
<p id="counts"></p><canvas id="canvas" width="1600" height="700"></canvas>
<pre id="detail">Click a feature to inspect its ID, support points,
decisions and weights.
Full supports: features.npz. Full candidates: correspondences.csv.
Gate -1: not evaluated.</pre>
<p><a href="metadata.yaml">Metadata</a> |
<a href="features.npz">Full trace and support arrays</a> |
<a href="correspondences.csv">Full correspondence table</a> |
<a href="objects.yaml">Diagnostic groups</a> |
<a href="object_matches.csv">Object association coverage</a></p>
<div id="images"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const C=document.getElementById('canvas'),ctx=C.getContext('2d');
const stage=document.getElementById('stage'),proj=document.getElementById('projection');
const channel=document.getElementById('channel'),pose=document.getElementById('pose');
let hits=[],selected=null;
stage.add(new Option('Detection / initial pose','-1'));
D.stages.forEach((s,i)=>stage.add(new Option('Stage '+s.index+
 (s.optimized?' optimized':' NOT optimized'),String(i))));
if(D.stages.length)stage.value=String(D.stages.length-1);
document.getElementById('meta').textContent=JSON.stringify(D.metadata);
function transform(p,t){return [0,1,2].map(i=>
 t[i][0]*p[0]+t[i][1]*p[1]+t[i][2]*p[2]+t[i][3]);}
function residual(ch,a,b,v){
 const d=a.map((x,i)=>x-b[i]),dot=d.reduce((s,x,i)=>s+x*v[i],0);
 if(ch==='ground'||ch==='facade')return Math.abs(dot);
 if(ch==='pole')return Math.hypot(...d.map((x,i)=>x-dot*v[i]));
 return Math.hypot(...d);
}
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
function redraw(){
 ctx.clearRect(0,0,C.width,C.height);hits=[];
 const axes=proj.value.split(',').map(Number),s=D.stages[Number(stage.value)];
 const t=s?s[pose.value]:D.T_initial;
 const dx=D.bounds[axes[0]][1]-D.bounds[axes[0]][0];
 const dy=D.bounds[axes[1]][1]-D.bounds[axes[1]][0];
 const scale=Math.min(700/dx,590/dy);
 function screen(p,side){return [side*800+400+(p[axes[0]]-
  (D.bounds[axes[0]][0]+D.bounds[axes[0]][1])/2)*scale,
  370-(p[axes[1]]-(D.bounds[axes[1]][0]+D.bounds[axes[1]][1])/2)*scale];}
 function line(a,b,color,width=1){ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
 ctx.font='16px sans-serif';ctx.fillStyle='#333';
 ctx.fillText((D.metadata.source_label||'SOURCE')+' (source to target frame)',40,30);
 ctx.fillText((D.metadata.target_label||'TARGET')+' (target reference frame)',840,30);
 let labels=[];
 ['source','target'].forEach((side,k)=>{
  const rec=D.sides[side],T=k?identity:t;
  ctx.fillStyle='#ccc';rec.background.forEach(p=>{
   const q=screen(transform(p,T),k);ctx.fillRect(q[0],q[1],1,1);});
  Object.entries(rec.features).forEach(([ch,f])=>{
   if(channel.value!=='all'&&channel.value!==ch)return;
   const decisions=s?s.channels[ch]:null;
   const indices=decisions?(k?decisions.target_indices:decisions.source_indices):[];
   const active=new Set(indices);
   f.points.forEach((p,i)=>{
    const q=screen(transform(p,T),k),isActive=active.has(i);
    ctx.beginPath();ctx.arc(q[0],q[1],isActive?3:2.3,0,Math.PI*2);
    ctx.strokeStyle=(!k&&decisions&&!isActive)?'#d33':D.colors[ch];
    ctx.fillStyle=(k&&decisions&&!isActive)?'#aaa':D.colors[ch];
    if(!decisions||isActive||k)ctx.fill();else ctx.stroke();
    hits.push({q,side:k,ch,i});
   });
   if(decisions)labels.push(side+' '+ch+': '+active.size+'/'+f.points.length);
  });
  rec.objects.forEach(g=>{
   if(channel.value!=='all'&&channel.value!==g.channel)return;
   for(let i=1;i<g.boundary.length;i++)line(
    screen(transform(g.boundary[i-1],T),k),screen(transform(g.boundary[i],T),k),
    D.colors[g.channel],1.5);
  });
 });
 if(s)Object.entries(s.channels).forEach(([ch,d])=>{
  if(channel.value!=='all'&&channel.value!==ch)return;
  const n=d.source_indices.length,step=Math.max(1,Math.ceil(n/45));
  d.source_indices.forEach((si,j)=>{
   const ti=d.target_indices[j],chosen=selected&&selected.ch===ch&&
    ((selected.side===0&&selected.i===si)||(selected.side===1&&selected.i===ti));
   if(j%step&&!chosen)return;
   const pa=transform(D.sides.source.features[ch].points[si],t);
   const pb=D.sides.target.features[ch].points[ti];
   const a=screen(pa,0),b=screen(pb,1);
   const r=residual(ch,pa,pb,D.sides.target.features[ch].vectors[ti]);
   const f=Math.min(r/.2,1);
   line(a,b,chosen?'#0066ff':`rgba(${Math.round(255*f)},${Math.round(160*(1-f))},25,.4)`,
    chosen?3:.7);
  });
 });
 document.getElementById('counts').textContent=labels.join(' | ')+
  ' | distances in metres; lines: residual norm (green 0, red >=0.20m)';
}
C.addEventListener('click',e=>{
 const box=C.getBoundingClientRect(),q=[(e.clientX-box.left)*C.width/box.width,
  (e.clientY-box.top)*C.height/box.height];
 let best=null,dist=12;
 hits.forEach(h=>{const d=Math.hypot(h.q[0]-q[0],h.q[1]-q[1]);
  if(d<dist){best=h;dist=d;}});
 if(!best)return;selected=best;
 const side=best.side?'target':'source',f=D.sides[side].features[best.ch],i=best.i;
 const s=D.stages[Number(stage.value)],d=s?s.channels[best.ch]:null;
 const matches=[];
 if(d)d.source_indices.forEach((si,j)=>{
  if((best.side?d.target_indices[j]:si)!==i)return;
  matches.push({source_id:D.prefixes[best.ch]+String(si).padStart(5,'0'),
   target_id:D.prefixes[best.ch]+String(d.target_indices[j]).padStart(5,'0'),
   participated_in_solve:s.optimized,post_solve_residual_norm_m:d.residual_norm_m?.[j],
   channel_weight:d.channel_weight,huber_components:d.huber_component_weights?.[j]});
 });
 const info={side,id:D.prefixes[best.ch]+String(i).padStart(5,'0'),
  point:f.points[i],normal_or_axis:f.vectors[i],pca:f.pca[i],
  support_count:f.support_count[i],support_preview:f.support_preview[i],
  objects:D.sides[side].objects.filter(g=>g.channel===best.ch&&g.members.includes(i)),
  matches};
 if(d&&!best.side){info.reason=d.reason[i];info.nearest_target_index=d.nearest_target[i];
  info.distance_at_matching_m=d.distance_m[i];info.thresholds=d.thresholds;
  info.checks=Object.fromEntries(Object.entries(d.gates).map(([k,v])=>[k,v[i]]));
  info.measurements=Object.fromEntries(Object.entries(d.values).map(([k,v])=>[k,v[i]]));}
 document.getElementById('detail').textContent=JSON.stringify(info,null,2);redraw();
});
[stage,proj,channel,pose].forEach(e=>e.addEventListener('change',()=>{selected=null;redraw();}));
__IMAGES__.forEach(name=>{const h=document.createElement('h2');h.textContent=name;
 const img=document.createElement('img');img.src=name;img.loading='lazy';
 document.getElementById('images').append(h,img);});
redraw();
</script>
"""


def export_feature_window(
    output_dir: Path,
    trace: dict[str, Any],
    result: dict[str, Any],
    metadata: dict[str, Any],
    *,
    edge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for side in ("source", "target"):
        trace[side]["label"] = metadata.get(f"{side}_label", side)
    groups = {side: group_features(trace[side]) for side in ("source", "target")}
    optimized = [s for s in trace["stages"] if s["optimized"]]
    final = result.get("transformation", trace.get("T_coarse", trace["T_initial"]))
    bounds = _bounds(trace)
    rows = _correspondences(trace)
    fields = (
        list(rows[0])
        if rows
        else [
            "stage",
            "channel",
            "source_id",
            "target_id",
            "reason",
            "participated_in_solve",
        ]
    )
    _write_csv(output_dir / "correspondences.csv", rows, fields)
    associations = _object_associations(trace, groups)
    _write_csv(
        output_dir / "object_matches.csv",
        associations,
        [
            "stage",
            "channel",
            "source_object",
            "target_object",
            "pair_count",
            "source_coverage",
            "target_coverage",
            "optimized",
            "diagnostic_only",
        ],
    )
    (output_dir / "objects.yaml").write_text(
        yaml.safe_dump(_plain(groups), sort_keys=False), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {}
    _flatten_arrays("", trace, arrays)
    np.savez_compressed(output_dir / "features.npz", **arrays)
    images = [
        "features.png",
        "overlay_before_after.png",
        "overlay_3d.png",
        "residuals.png",
    ]
    _plot_features(output_dir / "features.png", trace, groups, bounds)
    _plot_features(
        output_dir / "features_facade_pole.png",
        trace,
        groups,
        bounds,
        channels=("facade", "pole"),
    )
    images.append("features_facade_pole.png")
    if any("line_context" in trace[side] for side in ("source", "target")):
        _plot_line_context(output_dir / "line_context.png", trace, bounds)
        images.append("line_context.png")
    _plot_overlays(output_dir / "overlay_before_after.png", trace, final, bounds)
    _plot_3d(output_dir / "overlay_3d.png", trace, final, bounds, groups)
    _plot_residuals(output_dir / "residuals.png", trace["stages"])
    displayed = {}
    for stage in trace["stages"]:
        name = f"matches_stage_{stage['index']:02d}.png"
        displayed[name] = _plot_matches(output_dir / name, trace, stage, bounds)
        images.append(name)
    if trace["stages"]:
        _plot_matches(
            output_dir / "matches_initial.png", trace, trace["stages"][0], bounds
        )
        images.append("matches_initial.png")
    if optimized:
        _plot_matches(
            output_dir / "matches_final.png", trace, optimized[-1], bounds, after=True
        )
        images.append("matches_final.png")
        for channel in ("facade", "pole"):
            name = f"matches_final_{channel}.png"
            _plot_matches(
                output_dir / name,
                trace,
                optimized[-1],
                bounds,
                after=True,
                channels=(channel,),
            )
            images.append(name)
    report = {
        "schema_version": 1,
        **metadata,
        "status": "review-only" if result.get("success") else "failed",
        "solver_success": bool(result.get("success")),
        "failure_reason": result.get("reason"),
        "coordinate_convention": "p_target = T_target_source @ p_source",
        "T_initial": trace["T_initial"],
        "T_coarse": trace.get("T_coarse"),
        "T_final": final,
        "projection_bounds_m": bounds,
        "configs": trace.get("configs", {}),
        "source_stats": trace["source"]["stats"],
        "target_stats": trace["target"]["stats"],
        "source_before_cap": trace["source"]["before_cap"],
        "target_before_cap": trace["target"]["before_cap"],
        "source_voxel_rejections": trace["source"]["voxel_counts"],
        "target_voxel_rejections": trace["target"]["voxel_counts"],
        "channel_matches": result.get("channel_matches", {}),
        "channel_unique_targets": result.get("channel_unique_targets", {}),
        "channel_rmse_m": result.get("channel_rmse_m", {}),
        "localizability": result.get("localizability", {}),
        "directed_overlap": result.get("directed_overlap", {}),
        "joint_edge": {
            k: v
            for k, v in (edge or {}).items()
            if k not in {"info_matrix_6d", "transformation"}
        },
        "last_optimized_stage": optimized[-1]["index"] if optimized else None,
        "displayed_pairs": displayed,
        "stages": [
            {
                "index": s["index"],
                "optimized": s["optimized"],
                "reason": s.get("reason"),
                "T_matching": s["T_matching"],
                "T_after": s["T_after"],
                "max_correspondence_m": s["max_correspondence_m"],
                "sigma_rotation_deg": s["sigma_rotation_deg"],
                "sigma_translation_m": s["sigma_translation_m"],
                "channels": {
                    c: {
                        "thresholds": d["thresholds"],
                        "candidate_counts": dict(
                            zip(*np.unique(d["reason"], return_counts=True))
                        ),
                    }
                    for c, d in s["channels"].items()
                },
            }
            for s in trace["stages"]
        ],
        "notes": [
            "Detected counts are local samples, NOT physical plane/pole counts.",
            "Groups are diagnostic-only; no cylinder fitting or one-to-one assumption.",
            "Gate -1 means not evaluated; unavailable gates are not implied to pass.",
            "Candidate acceptance on an unoptimized stage is NOT solver participation.",
            "Huber weights are component-wise soft weights, not binary inlier labels.",
            "matches_initial is stage-0 at the coarse pose; "
            "T_initial remains the input seed.",
            "Visual agreement alone cannot establish calibration acceptance.",
        ],
        "images": images,
    }
    report = _plain(report)
    (output_dir / "metadata.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
    )
    payload = json.dumps(
        _interactive_data(trace, groups, metadata, bounds), allow_nan=False
    ).replace("<", "\\u003c")
    page = _VIEWER.replace("__DATA__", payload).replace(
        "__IMAGES__", json.dumps(images)
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return report


def export_feature_index(
    output_dir: Path,
    windows: list[dict[str, Any]],
    screening: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in windows:
        joint = window.get("joint_edge", {})
        rows.append(
            {
                "window_id": window["window_id"],
                "rel_time_s": window.get("rel_time_s"),
                "status": window["status"],
                "solver_attempted": not window.get("screening_rejected", False),
                "failure_reason": window.get("failure_reason"),
                "sync_dt_ms": window.get("sync_dt_ms"),
                "facade_matches": window.get("channel_matches", {}).get("facade", 0),
                "pole_matches": window.get("channel_matches", {}).get("pole", 0),
                "ground_matches": window.get("channel_matches", {}).get("ground", 0),
                "intensity_matches": window.get("channel_matches", {}).get(
                    "intensity", 0
                ),
                **{
                    f"{channel}_unique_targets": window.get(
                        "channel_unique_targets", {}
                    ).get(channel)
                    for channel in CHANNELS
                },
                "source_facade_detected": window.get("source_stats", {}).get(
                    "facade", 0
                ),
                "target_facade_detected": window.get("target_stats", {}).get(
                    "facade", 0
                ),
                "source_pole_detected": window.get("source_stats", {}).get("pole", 0),
                "target_pole_detected": window.get("target_stats", {}).get("pole", 0),
                "downweighted_by_gravity_gate": joint.get(
                    "downweighted_by_gravity_gate", False
                ),
                "downweighted_by_weak_support": joint.get(
                    "downweighted_by_weak_support", False
                ),
                "consensus_outlier": joint.get("consensus_outlier", False),
                "consensus_weight": joint.get("consensus_weight", 1.0),
            }
        )
    fields = list(rows[0]) if rows else ["window_id", "status", "failure_reason"]
    _write_csv(output_dir / "summary.csv", rows, fields)
    rejected = [r for r in screening if not r.get("accepted")]
    successful = [w for w in windows if w["solver_success"]]
    transforms = [np.asarray(w["T_final"]) for w in successful]
    spread = {}
    if transforms:
        from scipy.spatial.transform import Rotation

        translations, rotations = [], []
        for i, a in enumerate(transforms):
            for b in transforms[i + 1 :]:
                delta = np.linalg.inv(a) @ b
                translations.append(float(np.linalg.norm(delta[:3, 3])))
                rotations.append(
                    float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))
                )
        spread = {
            "window_count": len(transforms),
            "max_pairwise_translation_m": max(translations) if translations else None,
            "max_pairwise_rotation_deg": max(rotations) if rotations else None,
            "interpretation": "repeatability diagnostic, not ground truth",
        }
    report = {
        "schema_version": 1,
        "status": "review-only",
        **(metadata or {}),
        "candidate_count": len(screening),
        "screening_accepted": len(screening) - len(rejected),
        "screening_accepted_ratio": (
            (len(screening) - len(rejected)) / len(screening) if screening else None
        ),
        "solver_success_count": len(successful),
        "solver_attempt_count": sum(r["solver_attempted"] for r in rows),
        "screening_rejected": rejected,
        "window_transform_spread": spread,
        "holdout_verified": False,
        "seed_sensitivity_verified": False,
        "windows": rows,
    }
    (output_dir / "visualization_index.yaml").write_text(
        yaml.safe_dump(_plain(report), sort_keys=False), encoding="utf-8"
    )
    links = ""
    for window, row in zip(windows, rows):
        name = html.escape(str(window["window_id"]), quote=True)
        link = (
            f'<a href="{name}/index.html">{name}</a>'
            if window.get("has_gallery", True)
            else name + " (input not read)"
        )
        factor = (
            ", ".join(
                k
                for k in (
                    "downweighted_by_gravity_gate",
                    "downweighted_by_weak_support",
                    "consensus_outlier",
                )
                if row[k]
            )
            or "no downweight flag"
        )
        links += (
            f"<tr><td>{link}</td><td>{row['rel_time_s']}</td>"
            f"<td>{row['facade_matches']} / "
            f"{row['source_facade_detected']}:{row['target_facade_detected']}</td>"
            f"<td>{row['pole_matches']} / "
            f"{row['source_pole_detected']}:{row['target_pole_detected']}</td>"
            f"<td>{html.escape(row['status'])}</td><td>{html.escape(factor)}</td>"
            f"<td>{html.escape(str(row['failure_reason'] or ''))}</td></tr>"
        )
    rejected_html = html.escape(yaml.safe_dump(_plain(rejected), sort_keys=False))
    evidence = "".join(
        f'<a href="{html.escape(str(path), quote=True)}">'
        f"{html.escape(str(label))}</a> | "
        for label, path in (metadata or {}).get("validation_artifacts", {}).items()
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>L2L feature gallery</title>'
        "<style>body{font:14px sans-serif;margin:24px}"
        "td{padding:8px;border:1px solid #ddd}"
        "pre{white-space:pre-wrap}</style><h1>L2L feature gallery: review-only</h1>"
        "<p>Click a window for detected features, actual matches "
        "and interactive details. "
        "Detection is not physical object recognition. Visuals are not acceptance.</p>"
        '<p><a href="summary.csv">Summary CSV</a> | '
        '<a href="visualization_index.yaml">Data quality and repeatability</a></p>'
        f"<p>{evidence}</p>"
        "<table><tr><th>Window</th><th>Time [s]</th>"
        "<th>Facade pairs / detected S:T</th><th>Pole pairs / detected S:T</th>"
        "<th>Status</th><th>Joint factor flags</th><th>Reason</th></tr>"
        f"{links}</table>"
        "<h2>Screening rejected candidates (not sent to solver)</h2>"
        f"<pre>{rejected_html}</pre>"
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
