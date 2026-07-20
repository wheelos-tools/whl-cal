import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const SENSOR_DOT_CLASS = {
  left_front: "green",
  left_back: "red",
  right_front: "orange",
  right_back: "blue",
};

const SENSOR_LABEL = {
  left_front: "左前",
  left_back: "左后",
  right_front: "右前",
  right_back: "右后",
};

const DEFAULT_ALIGN_EDGES = [
  {
    relation_id: "lf_rf",
    label: "右前→左前",
    source_frame: "right_front",
    target_frame: "left_front",
  },
  {
    relation_id: "rf_rb",
    label: "右后→右前",
    source_frame: "right_back",
    target_frame: "right_front",
  },
  {
    relation_id: "rb_lb",
    label: "左后→右后",
    source_frame: "left_back",
    target_frame: "right_back",
  },
  {
    relation_id: "lb_lf",
    label: "左后→左前",
    source_frame: "left_back",
    target_frame: "left_front",
  },
];

const state = {
  session: null,
  sensors: {},
  activeFrameId: null,
  transformMode: "translate",
  pointSize: 0.08,
  syncingUi: false,
  visibility: {},
};

const ui = {
  frameIndex: document.getElementById("frame-index"),
  reloadBtn: document.getElementById("reload-btn"),
  sessionInfo: document.getElementById("session-info"),
  sensorSelect: document.getElementById("sensor-select"),
  modeTranslate: document.getElementById("mode-translate"),
  modeRotate: document.getElementById("mode-rotate"),
  visibilityList: document.getElementById("visibility-list"),
  showAllBtn: document.getElementById("show-all-btn"),
  hideAllBtn: document.getElementById("hide-all-btn"),
  pointSize: document.getElementById("point-size"),
  tx: document.getElementById("tx"),
  ty: document.getElementById("ty"),
  tz: document.getElementById("tz"),
  yaw: document.getElementById("yaw"),
  pitch: document.getElementById("pitch"),
  roll: document.getElementById("roll"),
  resetSensorBtn: document.getElementById("reset-sensor-btn"),
  resetAllBtn: document.getElementById("reset-all-btn"),
  alignEdgeList: document.getElementById("align-edge-list"),
  alignAllBtn: document.getElementById("align-all-btn"),
  autoAlignResult: document.getElementById("auto-align-result"),
  exportBtn: document.getElementById("export-btn"),
  exportResult: document.getElementById("export-result"),
  statusBar: document.getElementById("status-bar"),
};

ui.alignEdgeButtons = [];

const host = document.getElementById("canvas-host");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101418);

const camera = new THREE.PerspectiveCamera(
  60,
  host.clientWidth / host.clientHeight,
  0.05,
  500,
);
camera.position.set(0, -12, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(host.clientWidth, host.clientHeight);
host.appendChild(renderer.domElement);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0, 0);
orbit.update();

const transformControls = new TransformControls(camera, renderer.domElement);
transformControls.setSpace("local");
scene.add(transformControls);

const grid = new THREE.GridHelper(40, 40, 0x334155, 0x1e293b);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const axes = new THREE.AxesHelper(2);
scene.add(axes);

transformControls.addEventListener("dragging-changed", (event) => {
  orbit.enabled = !event.value;
});

transformControls.addEventListener("objectChange", () => {
  if (!state.activeFrameId || state.syncingUi) {
    return;
  }
  updateSensorFromGroup(state.activeFrameId);
  syncUiFromActiveSensor();
});

function setStatus(text) {
  ui.statusBar.textContent = text;
}

function matrixFromRows(rows) {
  const matrix = new THREE.Matrix4();
  const flat = rows.flat();
  if (rows.length === 4 && Array.isArray(rows[0])) {
    matrix.set(
      rows[0][0], rows[0][1], rows[0][2], rows[0][3],
      rows[1][0], rows[1][1], rows[1][2], rows[1][3],
      rows[2][0], rows[2][1], rows[2][2], rows[2][3],
      rows[3][0], rows[3][1], rows[3][2], rows[3][3],
    );
  } else {
    matrix.fromArray(flat);
  }
  return matrix;
}

function rowsFromMatrix(matrix) {
  const elements = matrix.elements;
  return [
    [elements[0], elements[4], elements[8], elements[12]],
    [elements[1], elements[5], elements[9], elements[13]],
    [elements[2], elements[6], elements[10], elements[14]],
    [elements[3], elements[7], elements[11], elements[15]],
  ];
}

function decomposeToUi(matrix) {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);
  const euler = new THREE.Euler().setFromQuaternion(quaternion, "ZYX");
  return {
    tx: position.x,
    ty: position.y,
    tz: position.z,
    yaw: THREE.MathUtils.radToDeg(euler.z),
    pitch: THREE.MathUtils.radToDeg(euler.y),
    roll: THREE.MathUtils.radToDeg(euler.x),
  };
}

function matrixFromUi(values) {
  const position = new THREE.Vector3(values.tx, values.ty, values.tz);
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(values.roll),
    THREE.MathUtils.degToRad(values.pitch),
    THREE.MathUtils.degToRad(values.yaw),
    "ZYX",
  );
  const quaternion = new THREE.Quaternion().setFromEuler(euler);
  return new THREE.Matrix4().compose(
    position,
    quaternion,
    new THREE.Vector3(1, 1, 1),
  );
}

function buildPointCloud(positions, colorRgb, size) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  const material = new THREE.PointsMaterial({
    color: new THREE.Color(...colorRgb),
    size,
    sizeAttenuation: true,
  });
  return new THREE.Points(geometry, material);
}

function fitCameraToScene() {
  const box = new THREE.Box3();
  Object.values(state.sensors).forEach((sensor) => {
    if (!sensor.group.visible) {
      return;
    }
    box.expandByObject(sensor.group);
  });
  if (box.isEmpty()) {
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 4);
  orbit.target.copy(center);
  camera.position.set(center.x, center.y - radius * 1.4, center.z + radius * 0.8);
  orbit.update();
}

function sensorDisplayName(meta) {
  return SENSOR_LABEL[meta.frame_id] || meta.sensor_name || meta.frame_id;
}

function setSensorVisible(frameId, visible) {
  const sensor = state.sensors[frameId];
  if (!sensor) {
    return;
  }
  state.visibility[frameId] = Boolean(visible);
  sensor.group.visible = Boolean(visible);
  if (!visible && state.activeFrameId === frameId) {
    transformControls.detach();
  } else if (visible && state.activeFrameId === frameId) {
    transformControls.attach(sensor.group);
  }
  syncVisibilityUi();
}

function setAllSensorsVisible(visible) {
  Object.keys(state.sensors).forEach((frameId) => {
    setSensorVisible(frameId, visible);
  });
}

function syncVisibilityUi() {
  ui.visibilityList.querySelectorAll(".visibility-item").forEach((item) => {
    const frameId = item.dataset.frameId;
    const checkbox = item.querySelector('input[type="checkbox"]');
    const isVisible = Boolean(state.visibility[frameId]);
    checkbox.checked = isVisible;
    item.classList.toggle("is-hidden", !isVisible);
  });
}

function renderVisibilityControls(sensors) {
  ui.visibilityList.innerHTML = "";
  sensors.forEach((meta) => {
    const visible = state.visibility[meta.frame_id] ?? true;
    state.visibility[meta.frame_id] = visible;

    const item = document.createElement("label");
    item.className = "visibility-item";
    item.dataset.frameId = meta.frame_id;
    item.classList.toggle("is-hidden", !visible);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = visible;
    checkbox.addEventListener("change", () => {
      setSensorVisible(meta.frame_id, checkbox.checked);
    });

    const dot = document.createElement("span");
    dot.className = `dot ${SENSOR_DOT_CLASS[meta.frame_id] || ""}`;

    const text = document.createElement("span");
    text.textContent = meta.fixed
      ? `${sensorDisplayName(meta)} (固定)`
      : sensorDisplayName(meta);

    item.append(checkbox, dot, text);
    ui.visibilityList.appendChild(item);
  });
}

async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function normalizeSession(session) {
  if (!session.align_edges?.length) {
    session.align_edges = DEFAULT_ALIGN_EDGES;
  }
  if (!session.align_settings) {
    session.align_settings = { registration_voxel_size: 0.04 };
  }
  return session;
}

function formatFrameOption(frame) {
  const syncLabel = frame.strict_sync ? "严格同步" : "最近邻";
  return `帧 ${frame.index} · ${frame.timestamp_ns} · ${syncLabel}`;
}

async function refreshFrameOptions(selectedIndex) {
  const frames = await fetchJson("/api/frames");
  ui.frameIndex.innerHTML = "";
  const available = frames.available_frames || [];
  if (available.length === 0) {
    const option = document.createElement("option");
    option.value = "0";
    option.textContent = "无可用帧";
    ui.frameIndex.appendChild(option);
    ui.frameIndex.disabled = true;
    throw new Error("Record 中没有可用的目标雷达帧。");
  }
  ui.frameIndex.disabled = false;
  for (const frame of available) {
    const option = document.createElement("option");
    option.value = String(frame.index);
    option.textContent = formatFrameOption(frame);
    ui.frameIndex.appendChild(option);
  }
  const targetIndex = selectedIndex ?? frames.default_frame_index ?? available[0].index;
  ui.frameIndex.value = String(targetIndex);
  return frames;
}

async function loadSession() {
  setStatus("加载点云…");
  ui.sessionInfo.textContent = "加载中…";
  ui.sessionInfo.className = "info-box";

  const selectedIndex = Number(ui.frameIndex.value);
  const frames = await refreshFrameOptions(
    Number.isFinite(selectedIndex) ? selectedIndex : undefined,
  );
  const frameIndex = Number(ui.frameIndex.value || frames.default_frame_index || 0);
  const session = normalizeSession(await fetchJson(`/api/session?frame_index=${frameIndex}`));
  state.session = session;

  const previousVisibility = { ...state.visibility };
  for (const sensor of Object.values(state.sensors)) {
    scene.remove(sensor.group);
    sensor.points.geometry.dispose();
    sensor.points.material.dispose();
  }
  state.sensors = {};

  for (const meta of session.sensors) {
    const pointsPayload = await fetchJson(`/api/points/${meta.frame_id}`);
    const group = new THREE.Group();
    const points = buildPointCloud(
      pointsPayload.positions,
      meta.color_rgb,
      state.pointSize,
    );
    group.add(points);

    const matrix = matrixFromRows(meta.current_transform);
    group.matrix.copy(matrix);
    group.matrixAutoUpdate = false;
    group.updateMatrix();
    group.matrixWorldNeedsUpdate = true;

    state.sensors[meta.frame_id] = {
      meta,
      group,
      points,
      initialTransform: matrixFromRows(meta.initial_transform),
      currentTransform: matrix.clone(),
    };
    const visible = previousVisibility[meta.frame_id] ?? true;
    state.visibility[meta.frame_id] = visible;
    group.visible = visible;
    scene.add(group);
  }

  renderVisibilityControls(session.sensors);

  const movable = session.sensors.filter((item) => !item.fixed);
  ui.sensorSelect.innerHTML = "";
  for (const sensor of movable) {
    const option = document.createElement("option");
    option.value = sensor.frame_id;
    option.textContent = `${sensor.sensor_name} (${sensor.frame_id})`;
    ui.sensorSelect.appendChild(option);
  }

  if (movable.length > 0) {
    selectSensor(movable[0].frame_id);
  } else {
    transformControls.detach();
  }

  ui.sessionInfo.className = session.sync_warnings?.length
    ? "info-box error"
    : "info-box";
  ui.sessionInfo.textContent = [
    `Record: ${session.record_files[0]}`,
    `基准: ${session.target_frame}`,
    `可用帧: ${(session.available_frame_indices || []).join(", ")}`,
    `当前帧: ${session.frame_index} / ${Math.max((session.target_frame_count || 1) - 1, 0)}`,
    `时间戳: ${session.reference_timestamp_ns}`,
    session.sync_mode === "nearest_fallback" ? "同步: 最近邻回退" : "同步: 严格",
    session.workflow_path ? `Workflow: ${session.workflow_path}` : "",
    session.conf_dir ? `Conf: ${session.conf_dir}` : "Conf: (none)",
    ...(session.sync_warnings || []),
  ].filter(Boolean).join("\n");

  renderAlignEdgeButtons(session);
  fitCameraToScene();
  setStatus(`已加载 ${session.sensors.length} 个雷达`);
}

function selectSensor(frameId) {
  const sensor = state.sensors[frameId];
  if (!sensor || sensor.meta.fixed) {
    return;
  }
  state.activeFrameId = frameId;
  ui.sensorSelect.value = frameId;
  if (sensor.group.visible) {
    transformControls.attach(sensor.group);
  } else {
    transformControls.detach();
  }
  transformControls.setMode(state.transformMode);
  syncUiFromActiveSensor();
  setStatus(`编辑: ${sensor.meta.sensor_name}`);
}

function syncUiFromActiveSensor() {
  const sensor = state.sensors[state.activeFrameId];
  if (!sensor) {
    return;
  }
  state.syncingUi = true;
  sensor.group.updateMatrix();
  sensor.currentTransform.copy(sensor.group.matrix);
  const values = decomposeToUi(sensor.currentTransform);
  ui.tx.value = values.tx.toFixed(3);
  ui.ty.value = values.ty.toFixed(3);
  ui.tz.value = values.tz.toFixed(3);
  ui.yaw.value = values.yaw.toFixed(2);
  ui.pitch.value = values.pitch.toFixed(2);
  ui.roll.value = values.roll.toFixed(2);
  state.syncingUi = false;
}

function updateSensorFromGroup(frameId) {
  const sensor = state.sensors[frameId];
  if (!sensor) {
    return;
  }
  sensor.group.updateMatrix();
  sensor.currentTransform.copy(sensor.group.matrix);
}

function applyUiToActiveSensor() {
  if (state.syncingUi || !state.activeFrameId) {
    return;
  }
  const sensor = state.sensors[state.activeFrameId];
  if (!sensor) {
    return;
  }
  const matrix = matrixFromUi({
    tx: Number(ui.tx.value),
    ty: Number(ui.ty.value),
    tz: Number(ui.tz.value),
    yaw: Number(ui.yaw.value),
    pitch: Number(ui.pitch.value),
    roll: Number(ui.roll.value),
  });
  updateSensorMatrix(sensor, matrix);
  scene.updateMatrixWorld(true);
}

function resetActiveSensor() {
  const sensor = state.sensors[state.activeFrameId];
  if (!sensor) {
    return;
  }
  updateSensorMatrix(sensor, sensor.initialTransform);
  scene.updateMatrixWorld(true);
  syncUiFromActiveSensor();
}

function resetAllSovableSensors() {
  Object.values(state.sensors).forEach((sensor) => {
    if (sensor.meta.fixed) {
      return;
    }
    updateSensorMatrix(sensor, sensor.initialTransform);
  });
  scene.updateMatrixWorld(true);
  syncUiFromActiveSensor();
}

function collectTransforms() {
  const transforms = {};
  Object.values(state.sensors).forEach((sensor) => {
    transforms[sensor.meta.frame_id] = rowsFromMatrix(sensor.currentTransform);
  });
  return transforms;
}

function updateSensorMatrix(sensor, matrix) {
  sensor.group.matrix.copy(matrix);
  sensor.group.updateMatrix();
  sensor.group.matrixWorldNeedsUpdate = true;
  sensor.currentTransform.copy(matrix);
}

function applyTransforms(transforms) {
  if (!transforms || typeof transforms !== "object") {
    return;
  }
  transformControls.detach();
  Object.entries(transforms).forEach(([frameId, rows]) => {
    const sensor = state.sensors[frameId];
    if (!sensor || sensor.meta.fixed) {
      return;
    }
    updateSensorMatrix(sensor, matrixFromRows(rows));
    sensor.meta.current_transform = rows;
  });
  if (state.session?.sensors) {
    for (const meta of state.session.sensors) {
      const rows = transforms[meta.frame_id];
      if (rows) {
        meta.current_transform = rows;
      }
    }
  }
  scene.updateMatrixWorld(true);
  if (state.activeFrameId) {
    const active = state.sensors[state.activeFrameId];
    if (active && active.group.visible && !active.meta.fixed) {
      transformControls.attach(active.group);
    }
  }
  syncUiFromActiveSensor();
  fitCameraToScene();
  renderer.render(scene, camera);
}

function setAlignButtonsDisabled(disabled) {
  for (const button of ui.alignEdgeButtons) {
    button.disabled = disabled;
  }
  if (ui.alignAllBtn) {
    ui.alignAllBtn.disabled = disabled;
  }
}

function formatEdgeLabel(label) {
  return String(label || "").replace(/→/g, " → ");
}

function findAlignEdge(relationId) {
  return (state.session?.align_edges || []).find((edge) => edge.relation_id === relationId);
}

function renderAlignEdgeButtons(session) {
  if (!ui.alignEdgeList) {
    return;
  }
  ui.alignEdgeList.innerHTML = "";
  ui.alignEdgeButtons = [];
  for (const edge of session.align_edges || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.edge = edge.relation_id;
    button.textContent = formatEdgeLabel(edge.label);
    button.addEventListener("click", () => {
      alignEdge(edge.relation_id, button).catch((error) => {
        ui.autoAlignResult.className = "info-box error";
        ui.autoAlignResult.textContent = String(error.message || error);
        setStatus("自动对齐失败");
        setAlignButtonsDisabled(false);
      });
    });
    ui.alignEdgeList.appendChild(button);
    ui.alignEdgeButtons.push(button);
  }
}

function formatEdgeResult(item) {
  const label = formatEdgeLabel(item.label)
    || `${item.source_frame} -> ${item.target_frame}`;
  const overlapLine = item.seed_overlap_ratio !== undefined
    ? `overlap ${Number(item.seed_overlap_ratio).toFixed(4)} → ${Number(item.overlap_ratio || 0).toFixed(4)}`
    : "";
  const coarseLine = item.coarse_overlap_ratio != null
    ? `coarse ${Number(item.coarse_overlap_ratio).toFixed(3)}`
    : "";
  const windowLine = item.frame_index != null
    ? `frame ${item.frame_index}`
    : "";
  if (!item.success) {
    const reason = item.reason === "insufficient_overlap_improvement"
      ? "重叠度提升不足，已保持当前位置"
      : item.reason === "excessive_registration_delta"
        ? "配准位移/旋转过大，已保持当前位置"
        : (item.reason || "失败");
    return [label, reason, overlapLine, coarseLine, windowLine, item.attempt ? `策略: ${item.attempt}` : ""]
      .filter(Boolean)
      .join("\n");
  }
  const metricLine = item.fitness != null
    ? `fitness=${Number(item.fitness).toFixed(4)} rmse=${Number(item.inlier_rmse).toFixed(4)}`
    : "";
  const deltaLine = item.delta_translation_m != null
    ? `ΔT=${Number(item.delta_translation_m).toFixed(3)}m ΔR=${Number(item.delta_rotation_deg || 0).toFixed(2)}°`
    : "";
  return [label, item.attempt ? `策略: ${item.attempt}` : "", overlapLine, coarseLine, windowLine, deltaLine, metricLine]
    .filter(Boolean)
    .join("\n");
}

function buildAlignRequestBody(relationId, loopClosure = false) {
  return {
    relation_id: relationId,
    method: 2,
    registration_voxel_size: state.session?.align_settings?.registration_voxel_size,
    loop_closure: loopClosure,
    seed_transforms: collectTransforms(),
  };
}

async function alignEdge(relationId, buttonEl) {
  const edgeMeta = findAlignEdge(relationId);
  const label = formatEdgeLabel(edgeMeta?.label) || relationId;
  const originalLabel = buttonEl.textContent;
  setAlignButtonsDisabled(true);
  buttonEl.textContent = "对齐中…";
  ui.autoAlignResult.textContent = `${label}\nPCL GICP 配准当前帧中，约需 30–90 秒…`;
  ui.autoAlignResult.className = "info-box";
  ui.autoAlignResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  setStatus(`${label} 对齐中…`);
  try {
    const payload = await fetchJson("/api/auto-align", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildAlignRequestBody(relationId, false)),
    });
    applyTransforms(payload.transforms);
    const edge = (payload.edge_results || [])[0];
    if (edge?.success) {
      const sourceFrame = edgeMeta?.source_frame || edge.source_frame;
      if (sourceFrame && state.sensors[sourceFrame]) {
        selectSensor(sourceFrame);
      }
    }
    ui.autoAlignResult.className = edge?.success ? "info-box success" : "info-box error";
    ui.autoAlignResult.textContent = edge
      ? formatEdgeResult(edge)
      : `${label}: 未返回结果`;
    setStatus(edge?.success ? `${label} 对齐完成` : `${label} 对齐失败`);
  } catch (error) {
    ui.autoAlignResult.className = "info-box error";
    ui.autoAlignResult.textContent = `${label}\n${String(error.message || error)}`;
    setStatus(`${label} 对齐失败`);
  } finally {
    buttonEl.textContent = originalLabel;
    setAlignButtonsDisabled(false);
  }
}

async function alignAllEdges() {
  if (!ui.alignAllBtn) {
    return;
  }
  const originalLabel = ui.alignAllBtn.textContent;
  setAlignButtonsDisabled(true);
  ui.alignAllBtn.textContent = "全部对齐中…";
  ui.autoAlignResult.className = "info-box";
  ui.autoAlignResult.textContent = "按 workflow 顺序配准当前帧 + 闭环优化，约需 2–4 分钟…";
  setStatus("全部对齐中…");
  try {
    const payload = await fetchJson("/api/auto-align", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        method: 2,
        registration_voxel_size: state.session?.align_settings?.registration_voxel_size,
        loop_closure: state.session?.align_settings?.enable_loop_closure ?? true,
        seed_transforms: collectTransforms(),
      }),
    });
    applyTransforms(payload.transforms);
    const lines = (payload.edge_results || []).map((edge) => formatEdgeResult(edge));
    if (payload.summary?.loop_closure_applied) {
      lines.push("闭环优化: 已应用");
    } else if (payload.loop_closure?.reason === "insufficient_loop_edges") {
      lines.push("闭环优化: 可用边不足，已跳过");
    }
    ui.autoAlignResult.className = payload.summary?.aligned_count
      ? "info-box success"
      : "info-box error";
    ui.autoAlignResult.textContent = lines.filter(Boolean).join("\n\n");
    setStatus(`全部对齐完成 ${payload.summary?.aligned_count || 0}/${payload.summary?.edge_count || 0}`);
  } catch (error) {
    ui.autoAlignResult.className = "info-box error";
    ui.autoAlignResult.textContent = String(error.message || error);
    setStatus("全部对齐失败");
  } finally {
    ui.alignAllBtn.textContent = originalLabel;
    setAlignButtonsDisabled(false);
  }
}

async function exportResults() {
  ui.exportBtn.disabled = true;
  ui.exportResult.textContent = "导出中…";
  ui.exportResult.className = "info-box";
  try {
    const payload = await fetchJson("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transforms: collectTransforms() }),
    });
    ui.exportResult.className = "info-box success";
    ui.exportResult.textContent = [
      "导出成功:",
      `merged_cloud: ${payload.merged_cloud}`,
      `calibrated_tf: ${payload.calibrated_tf}`,
      `points: ${payload.point_count}`,
    ].join("\n");
    setStatus("导出完成");
  } catch (error) {
    ui.exportResult.className = "info-box error";
    ui.exportResult.textContent = String(error.message || error);
    setStatus("导出失败");
  } finally {
    ui.exportBtn.disabled = false;
  }
}

function setTransformMode(mode) {
  state.transformMode = mode;
  transformControls.setMode(mode);
  ui.modeTranslate.classList.toggle("active", mode === "translate");
  ui.modeRotate.classList.toggle("active", mode === "rotate");
}

function updatePointSize(size) {
  state.pointSize = size;
  Object.values(state.sensors).forEach((sensor) => {
    sensor.points.material.size = size;
    sensor.points.material.needsUpdate = true;
  });
}

function bindClick(element, handler) {
  if (element) {
    element.addEventListener("click", handler);
  }
}

function reportInitError(error) {
  const message = String(error?.message || error);
  if (ui.sessionInfo) {
    ui.sessionInfo.className = "info-box error";
    ui.sessionInfo.textContent = `页面初始化失败: ${message}`;
  }
  if (ui.autoAlignResult) {
    ui.autoAlignResult.className = "info-box error";
    ui.autoAlignResult.textContent = `自动对齐不可用: ${message}`;
  }
  setStatus("页面初始化失败，请强制刷新 (Ctrl+Shift+R)");
  console.error(error);
}
try {
  bindClick(ui.reloadBtn, () => {
    loadSession().catch((error) => {
      ui.sessionInfo.className = "info-box error";
      ui.sessionInfo.textContent = String(error.message || error);
      setStatus("加载失败");
    });
  });

  if (ui.sensorSelect) {
    ui.sensorSelect.addEventListener("change", () => {
      selectSensor(ui.sensorSelect.value);
    });
  }

  bindClick(ui.modeTranslate, () => setTransformMode("translate"));
  bindClick(ui.modeRotate, () => setTransformMode("rotate"));
  bindClick(ui.showAllBtn, () => setAllSensorsVisible(true));
  bindClick(ui.hideAllBtn, () => setAllSensorsVisible(false));

  if (ui.pointSize) {
    ui.pointSize.addEventListener("input", () => {
      updatePointSize(Number(ui.pointSize.value));
    });
  }

  for (const input of [ui.tx, ui.ty, ui.tz, ui.yaw, ui.pitch, ui.roll]) {
    if (input) {
      input.addEventListener("input", applyUiToActiveSensor);
    }
  }

  bindClick(ui.resetSensorBtn, resetActiveSensor);
  bindClick(ui.resetAllBtn, resetAllSovableSensors);
  bindClick(ui.alignAllBtn, () => {
    alignAllEdges().catch((error) => {
      ui.autoAlignResult.className = "info-box error";
      ui.autoAlignResult.textContent = String(error.message || error);
      setStatus("全部对齐失败");
      setAlignButtonsDisabled(false);
    });
  });
  bindClick(ui.exportBtn, exportResults);

  loadSession().catch((error) => {
    ui.sessionInfo.className = "info-box error";
    ui.sessionInfo.textContent = String(error.message || error);
    setStatus("加载失败");
  });
} catch (error) {
  reportInitError(error);
}

window.addEventListener("resize", () => {
  camera.aspect = host.clientWidth / host.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(host.clientWidth, host.clientHeight);
});

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

animate();
