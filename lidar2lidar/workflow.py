from __future__ import annotations

import copy
from pathlib import Path

import yaml


DEFAULT_WORKFLOW = {
    "version": 1,
    "selection": {
        "target_topic": None,
        "topics": None,
        "source_topics": None,
    },
    "planner": {
        "mode": "target_star",
        "enable_global_optimization": False,
        "relations": [],
    },
    "scene_sufficiency": {
        "max_windows_per_relation": 5,
        "min_valid_windows_per_relation": 2,
        "min_overlap_ratio": None,
        "dynamic_distance_threshold_m": 0.40,
        "max_dynamic_unmatched_ratio": 0.65,
        "min_wall_plane_count": 1,
        "min_corner_pair_count": 1,
    },
    "repeatability": {
        "max_windows_per_edge": 5,
        "pass_translation_p95_m": 0.25,
        "pass_rotation_p95_deg": 2.0,
    },
    "visualization": {
        "enabled": False,
        "save_merged_clouds": False,
        "downsample_voxel_size": 0.10,
        "plane_distance_threshold": 0.08,
        "max_planes": 4,
        "min_plane_points": 600,
        "corner_angle_tolerance_deg": 20.0,
        "corner_distance_threshold_m": 0.12,
        "slice_bin_size_m": 0.25,
        "min_slice_points": 120,
        "max_wall_double_edge_m": 0.20,
        "max_corner_spread_p95_m": 0.15,
        "min_slice_sharpness_score": 6.0,
    },
}


def _deep_merge(base: dict, override: dict | None) -> dict:
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_workflow_config(path: str | None) -> dict | None:
    if path is None:
        return None
    workflow_path = Path(path)
    with workflow_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow YAML must be a mapping: {path}")
    return payload


def _selected_topics_from_config(
    workflow: dict,
    pointcloud_topics: list[str],
    default_target_topic: str,
    cli_source_topics: list[str] | None,
) -> list[str]:
    selection = workflow["selection"]
    configured_topics = selection.get("topics")
    configured_sources = selection.get("source_topics")
    if configured_topics:
        selected_topics = list(dict.fromkeys(configured_topics))
    elif configured_sources:
        selected_topics = [default_target_topic, *configured_sources]
    elif cli_source_topics:
        selected_topics = [default_target_topic, *cli_source_topics]
    else:
        selected_topics = list(pointcloud_topics)
    return list(dict.fromkeys(topic for topic in selected_topics if topic))


def _relation_defaults(entry: dict) -> dict:
    role = str(entry.get("role", "primary"))
    return {
        "relation_id": entry.get("relation_id"),
        "source_topic": entry["source_topic"],
        "target_topic": entry["target_topic"],
        "role": role,
        "required": bool(entry.get("required", role == "primary")),
        "use_for_solution": bool(
            entry.get("use_for_solution", role in {"primary", "supporting"})
        ),
        "use_for_optimization": bool(
            entry.get("use_for_optimization", role in {"primary", "supporting"})
        ),
        "use_for_evaluation": bool(entry.get("use_for_evaluation", True)),
        "inferred_from": entry.get("inferred_from", "explicit"),
    }


def _build_explicit_relations(
    relation_entries: list[dict],
) -> list[dict]:
    relations = []
    for entry in relation_entries:
        if "source_topic" not in entry or "target_topic" not in entry:
            raise ValueError("Each workflow relation must provide source_topic and target_topic.")
        relation = _relation_defaults(entry)
        if relation["relation_id"] is None:
            relation["relation_id"] = (
                f"{relation['source_topic']}__to__{relation['target_topic']}"
            )
        relations.append(relation)
    return relations


def _frame_topic_map(topic_infos: dict[str, dict], selected_topics: list[str]) -> dict[str, str]:
    frame_to_topic = {}
    for topic in selected_topics:
        frame_id = topic_infos[topic]["frame_id"]
        if frame_id and frame_id not in frame_to_topic:
            frame_to_topic[frame_id] = topic
    return frame_to_topic


def _build_tf_tree_relations(
    *,
    topic_infos: dict[str, dict],
    selected_topics: list[str],
    tf_edges: list,
) -> list[dict]:
    frame_to_topic = _frame_topic_map(topic_infos, selected_topics)
    relations = []
    for edge in tf_edges:
        parent_topic = frame_to_topic.get(edge.parent_frame)
        child_topic = frame_to_topic.get(edge.child_frame)
        if not parent_topic or not child_topic:
            continue
        relations.append(
            {
                "relation_id": f"{child_topic}__to__{parent_topic}",
                "source_topic": child_topic,
                "target_topic": parent_topic,
                "role": "primary",
                "required": True,
                "use_for_solution": True,
                "use_for_optimization": True,
                "use_for_evaluation": True,
                "inferred_from": "tf_tree_direct_edge",
            }
        )
    relations.sort(key=lambda item: (item["target_topic"], item["source_topic"]))
    return relations


def _build_target_star_relations(
    *,
    selected_topics: list[str],
    target_topic: str,
) -> list[dict]:
    relations = []
    for topic in selected_topics:
        if topic == target_topic:
            continue
        relations.append(
            {
                "relation_id": f"{topic}__to__{target_topic}",
                "source_topic": topic,
                "target_topic": target_topic,
                "role": "primary",
                "required": True,
                "use_for_solution": True,
                "use_for_optimization": True,
                "use_for_evaluation": True,
                "inferred_from": "target_star",
            }
        )
    return relations


def _build_complete_relations(
    *,
    selected_topics: list[str],
    target_topic: str,
) -> list[dict]:
    relations = []
    for index, topic_a in enumerate(selected_topics):
        for topic_b in selected_topics[index + 1 :]:
            if target_topic in {topic_a, topic_b}:
                source_topic = topic_b if topic_a == target_topic else topic_a
                relation_target = target_topic
            else:
                source_topic, relation_target = sorted([topic_a, topic_b])
            relations.append(
                {
                    "relation_id": f"{source_topic}__to__{relation_target}",
                    "source_topic": source_topic,
                    "target_topic": relation_target,
                    "role": "supporting",
                    "required": False,
                    "use_for_solution": True,
                    "use_for_optimization": True,
                    "use_for_evaluation": True,
                    "inferred_from": "complete_pair_graph",
                }
            )
    return relations


def resolve_workflow_plan(
    *,
    workflow_config: dict | None,
    workflow_path: str | None,
    pointcloud_topics: list[str],
    topic_infos: dict[str, dict],
    tf_edges: list,
    default_target_topic: str,
    cli_source_topics: list[str] | None,
    default_min_overlap: float,
    default_enable_global_optimization: bool,
    default_save_visuals: bool,
) -> dict:
    workflow = _deep_merge(DEFAULT_WORKFLOW, workflow_config)
    selection = workflow["selection"]
    planner = workflow["planner"]
    scene_sufficiency = workflow["scene_sufficiency"]
    repeatability = workflow["repeatability"]
    visualization = workflow["visualization"]

    target_topic = selection.get("target_topic") or default_target_topic
    if target_topic not in topic_infos:
        raise ValueError(f"Unknown workflow target topic: {target_topic}")

    explicit_relation_entries = planner.get("relations") or []
    if explicit_relation_entries and not selection.get("topics") and not selection.get(
        "source_topics"
    ):
        selected_topics = [target_topic]
    else:
        selected_topics = _selected_topics_from_config(
            workflow,
            pointcloud_topics,
            target_topic,
            cli_source_topics,
        )
    if target_topic not in selected_topics:
        selected_topics.insert(0, target_topic)
    for topic in selected_topics:
        if topic not in topic_infos:
            raise ValueError(f"Unknown workflow topic: {topic}")

    mode = str(planner.get("mode", "target_star"))
    relation_entries = explicit_relation_entries
    if relation_entries:
        relations = _build_explicit_relations(relation_entries)
        resolved_mode = "explicit"
    elif mode == "tf_tree":
        relations = _build_tf_tree_relations(
            topic_infos=topic_infos,
            selected_topics=selected_topics,
            tf_edges=tf_edges,
        )
        resolved_mode = "tf_tree"
        if not relations:
            relations = _build_target_star_relations(
                selected_topics=selected_topics,
                target_topic=target_topic,
            )
            resolved_mode = "tf_tree_fallback_target_star"
    elif mode == "complete":
        relations = _build_complete_relations(
            selected_topics=selected_topics,
            target_topic=target_topic,
        )
        resolved_mode = "complete"
    else:
        relations = _build_target_star_relations(
            selected_topics=selected_topics,
            target_topic=target_topic,
        )
        resolved_mode = "target_star"

    relation_ids = set()
    resolved_relations = []
    for relation in relations:
        normalized = _relation_defaults(relation)
        if normalized["relation_id"] is None:
            normalized["relation_id"] = (
                f"{normalized['source_topic']}__to__{normalized['target_topic']}"
            )
        if normalized["relation_id"] in relation_ids:
            raise ValueError(f"Duplicate workflow relation_id: {normalized['relation_id']}")
        relation_ids.add(normalized["relation_id"])
        if normalized["source_topic"] == normalized["target_topic"]:
            raise ValueError(
                f"Workflow relation {normalized['relation_id']} has identical source and target topic."
            )
        for topic_key in ("source_topic", "target_topic"):
            if normalized[topic_key] not in topic_infos:
                raise ValueError(
                    f"Workflow relation {normalized['relation_id']} references unknown topic {normalized[topic_key]}."
                )
        resolved_relations.append(
            {
                **normalized,
                "source_frame": topic_infos[normalized["source_topic"]]["frame_id"],
                "target_frame": topic_infos[normalized["target_topic"]]["frame_id"],
            }
        )
        if normalized["source_topic"] not in selected_topics:
            selected_topics.append(normalized["source_topic"])
        if normalized["target_topic"] not in selected_topics:
            selected_topics.append(normalized["target_topic"])

    scene_sufficiency["min_overlap_ratio"] = (
        float(scene_sufficiency["min_overlap_ratio"])
        if scene_sufficiency.get("min_overlap_ratio") is not None
        else float(default_min_overlap)
    )
    visualization["enabled"] = bool(
        visualization.get("enabled", False) or default_save_visuals
    )

    return {
        "source": workflow_path or "cli_defaults",
        "target_topic": target_topic,
        "selected_topics": selected_topics,
        "planner_mode": resolved_mode,
        "enable_global_optimization": bool(
            planner.get("enable_global_optimization", default_enable_global_optimization)
        ),
        "relations": resolved_relations,
        "scene_sufficiency": scene_sufficiency,
        "repeatability": repeatability,
        "visualization": visualization,
        "summary": {
            "selected_topic_count": len(selected_topics),
            "relation_count": len(resolved_relations),
            "required_relation_count": int(
                sum(1 for relation in resolved_relations if relation["required"])
            ),
            "solution_relation_count": int(
                sum(1 for relation in resolved_relations if relation["use_for_solution"])
            ),
            "optimization_relation_count": int(
                sum(
                    1 for relation in resolved_relations if relation["use_for_optimization"]
                )
            ),
        },
    }
