/**
 * visit-it viewer v0.9 — the Phase 1 shell walkthrough.
 *
 * Loads one `scene.json` and its `shell.glb`, and gives you three ways to look at
 * a flat: a waypoint walkthrough, a dollhouse overview, and a floor-plan minimap
 * that stays in sync with both. No splats — those are Phase 2, and the room
 * records already carry the `splats` field they will fill.
 *
 * The one rule the front end enforces rather than assumes: **the only area shown
 * is the advertised one** (ARCHITECTURE §5). Per-room areas exist in the scene and
 * are visible only behind `?dev=1`, because publishing a competing measurement of
 * someone's flat is not a thing we do.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { applyHonesty, legendHtml } from "./honesty";
import { Minimap } from "./minimap";
import { WaypointNav } from "./nav";
import type { Room, Scene, ViewMode } from "./types";

const params = new URLSearchParams(location.search);
const DEV = params.get("dev") === "1";
const SCENE_URL = params.get("scene") ?? "./fixtures/scene.json";

const app = document.getElementById("app") as HTMLDivElement;
const hud = document.getElementById("hud") as HTMLDivElement;
const minimapCanvas = document.getElementById("minimap") as HTMLCanvasElement;
const statusEl = document.getElementById("status") as HTMLDivElement;

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(app.clientWidth, app.clientHeight);
renderer.setClearColor(0x11131a);
app.appendChild(renderer.domElement);

const scene3 = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, app.clientWidth / app.clientHeight, 0.05, 200);
const dollCamera = new THREE.PerspectiveCamera(50, app.clientWidth / app.clientHeight, 0.1, 400);

// Lighting an untextured interior shell is its own small problem. Seen from a
// waypoint you are *inside* the geometry, so every surface you look at is a back
// face whose normal points away from you, and a hemisphere light alone leaves the
// room near black. Three lights, each doing one job:
scene3.add(new THREE.AmbientLight(0xffffff, 0.55));      // nothing is ever unlit
scene3.add(new THREE.HemisphereLight(0xf2efe6, 0x3a3c44, 1.1));  // up/down shape
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(4, 9, 6);
scene3.add(sun);
// A soft lamp carried at head height, so walls fall off with distance and the
// room reads as a room rather than as flat paint.
const torch = new THREE.PointLight(0xfff4e2, 14, 16, 1.6);
camera.add(torch);
scene3.add(camera);

renderer.localClippingEnabled = true;
// Dollhouse means dollhouse: the ceilings come off. A clipping plane does it
// without a second mesh, and without the shell needing per-face role tags.
const ceilingCut = new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.0);

let mode: ViewMode = "walkthrough";
let nav: WaypointNav | null = null;
let minimap: Minimap | null = null;
let orbit: OrbitControls | null = null;
let scene: Scene | null = null;
let activeRoom: Room | null = null;
let dragging = false;
let lastX = 0;
let lastY = 0;

function setStatus(msg: string, kind: "info" | "error" = "info"): void {
  statusEl.textContent = msg;
  statusEl.className = kind;
  statusEl.style.display = msg ? "block" : "none";
}

async function load(): Promise<void> {
  setStatus("Loading scene…");
  const t0 = performance.now();
  const res = await fetch(SCENE_URL);
  if (!res.ok) throw new Error(`scene.json: ${res.status} ${res.statusText}`);
  scene = (await res.json()) as Scene;
  if (scene.schema !== "scene/v1") {
    throw new Error(`unsupported scene schema ${String(scene.schema)}`);
  }

  if (scene.shell) {
    const base = SCENE_URL.replace(/[^/]*$/, "");
    const gltf = await new GLTFLoader().loadAsync(base + scene.shell.uri);
    const present = applyHonesty(gltf.scene);
    scene3.add(gltf.scene);
    document.getElementById("legend")!.innerHTML = legendHtml(scene, present);
  }

  nav = new WaypointNav(scene, camera);
  minimap = new Minimap(minimapCanvas, scene, (room) => {
    const w = nav?.waypointForRoom(room.room_id);
    if (w) nav?.goTo(w.waypoint_id, performance.now());
  });

  const [[minx, miny], [maxx, maxy]] = scene.minimap.bounds_m;
  const span = Math.max(maxx - minx, maxy - miny, 4);
  ceilingCut.constant = Math.max(...scene.rooms.map((r) => r.height_m)) - 0.35;
  dollCamera.position.set((minx + maxx) / 2, span * 1.5, -(miny + maxy) / 2 + span * 1.15);
  orbit = new OrbitControls(dollCamera, renderer.domElement);
  orbit.target.set((minx + maxx) / 2, 1.2, -(miny + maxy) / 2);
  orbit.enableDamping = true;
  orbit.enabled = false;
  orbit.update();

  renderHud();
  const ms = Math.round(performance.now() - t0);
  // G1 budgets first-room-interactive: measured here so it is a number, not a hope.
  setStatus(DEV ? `ready in ${ms} ms · ${scene.shell?.triangles ?? 0} triangles` : "");
  if (!DEV) window.setTimeout(() => setStatus(""), 1200);
}

function renderHud(): void {
  if (!scene || !nav) return;
  const room = activeRoomOf();
  const area = scene.advertised_area_m2;
  const flags = scene.qa_flags.length;
  hud.innerHTML = `
    <div class="title">${scene.address ?? scene.listing_id}</div>
    <div class="sub">
      ${area ? `${area} m² (advertised)` : "area not stated"} ·
      ${scene.rooms.length} rooms ·
      tier ${scene.tier}${scene.tier === "B" ? " — arrangement inferred" : ""}
    </div>
    <div class="room">${room ? room.display_name : "—"}
      ${DEV && room?.area_m2 ? `<span class="dev">${room.area_m2} m² · conf ${room.confidence}</span>` : ""}
    </div>
    <div class="nav">${nav
      .neighboursOf(nav.current.waypoint_id)
      .map((w) => `<button data-goto="${w.waypoint_id}">${w.label}</button>`)
      .join("")}</div>
    <div class="modes">
      <button data-mode="walkthrough" class="${mode === "walkthrough" ? "on" : ""}">Walkthrough</button>
      <button data-mode="dollhouse" class="${mode === "dollhouse" ? "on" : ""}">Dollhouse</button>
    </div>
    ${DEV && flags ? `<details class="qa"><summary>${flags} QA flags</summary>
       <ul>${scene.qa_flags.map((f) => `<li><code>${f}</code></li>`).join("")}</ul></details>` : ""}
  `;
  hud.querySelectorAll<HTMLButtonElement>("[data-goto]").forEach((b) =>
    b.addEventListener("click", () => nav?.goTo(b.dataset.goto!, performance.now())),
  );
  hud.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode as ViewMode)),
  );
}

function activeRoomOf(): Room | null {
  if (!scene || !nav) return null;
  const id = nav.current.room_id;
  return scene.rooms.find((r) => r.room_id === id) ?? null;
}

function setMode(next: ViewMode): void {
  mode = next;
  if (orbit) orbit.enabled = next === "dollhouse";
  renderer.clippingPlanes = next === "dollhouse" ? [ceilingCut] : [];
  torch.visible = next === "walkthrough";
  renderHud();
}

// Look around by dragging. Deliberately yaw+limited pitch: from a waypoint you
// turn your head, you do not fly.
renderer.domElement.addEventListener("pointerdown", (e) => {
  if (mode !== "walkthrough") return;
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  renderer.domElement.setPointerCapture(e.pointerId);
});
renderer.domElement.addEventListener("pointerup", (e) => {
  dragging = false;
  renderer.domElement.releasePointerCapture(e.pointerId);
});
renderer.domElement.addEventListener("pointermove", (e) => {
  if (!dragging || mode !== "walkthrough") return;
  camera.rotation.y -= (e.clientX - lastX) * 0.005;
  camera.rotation.x = THREE.MathUtils.clamp(
    camera.rotation.x - (e.clientY - lastY) * 0.004, -0.9, 0.9);
  lastX = e.clientX;
  lastY = e.clientY;
});

window.addEventListener("keydown", (e) => {
  if (!nav) return;
  if (e.key === "d") setMode(mode === "dollhouse" ? "walkthrough" : "dollhouse");
  const ns = nav.neighboursOf(nav.current.waypoint_id);
  const n = Number(e.key);
  if (n >= 1 && n <= ns.length) nav.goTo(ns[n - 1].waypoint_id, performance.now());
});

window.addEventListener("resize", () => {
  const w = app.clientWidth;
  const h = app.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = dollCamera.aspect = w / h;
  camera.updateProjectionMatrix();
  dollCamera.updateProjectionMatrix();
});

let lastRoomId: string | null = null;
function frame(now: number): void {
  requestAnimationFrame(frame);
  if (nav) {
    nav.update(now);
    const room = activeRoomOf();
    if (room?.room_id !== lastRoomId) {
      lastRoomId = room?.room_id ?? null;
      activeRoom = room;
      renderHud();
    }
    minimap?.draw(nav.metricPosition, nav.heading, activeRoom?.room_id ?? null);
  }
  if (mode === "dollhouse") orbit?.update();
  renderer.render(scene3, mode === "dollhouse" ? dollCamera : camera);
}

load()
  .then(() => requestAnimationFrame(frame))
  .catch((e: unknown) => {
    console.error(e);
    setStatus(`Could not load the scene: ${e instanceof Error ? e.message : String(e)}`, "error");
  });
