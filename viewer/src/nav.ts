/**
 * Waypoint navigation (AD-1/AD-9).
 *
 * Teleport-between-waypoints is load-bearing, not a limitation we are apologising
 * for: the scene only has to look right *from the waypoints and along the
 * sightlines between them*, which is exactly what sparse listing photos can
 * support. Free roam would promise a fidelity the input does not contain.
 */
import * as THREE from "three";
import type { Scene, Waypoint } from "./types";

const TRANSITION_MS = 520;

export class WaypointNav {
  readonly byId = new Map<string, Waypoint>();
  private neighbours = new Map<string, string[]>();
  current: Waypoint;
  private from = new THREE.Vector3();
  private to = new THREE.Vector3();
  private t = 1;
  private startedAt = 0;

  constructor(private scene: Scene, private camera: THREE.PerspectiveCamera) {
    for (const w of scene.waypoints) {
      this.byId.set(w.waypoint_id, w);
      this.neighbours.set(w.waypoint_id, []);
    }
    for (const [a, b] of scene.waypoint_edges) {
      this.neighbours.get(a)?.push(b);
      this.neighbours.get(b)?.push(a);
    }
    // Arrive in the biggest room, not the first one in the file. It is the room a
    // person would walk into, and it is the one most likely to be reconstructed
    // well — small rooms are where the geometry and the assignment both struggle.
    const area = new Map(scene.rooms.map((r) => [r.room_id, r.area_m2 ?? 0]));
    const centres = scene.waypoints.filter((w) => w.kind === "room_centre");
    centres.sort((a, b) => (area.get(b.room_id ?? "") ?? 0) - (area.get(a.room_id ?? "") ?? 0));
    const first = centres[0] ?? scene.waypoints[0];
    if (!first) throw new Error("scene has no waypoints");
    this.current = first;
    this.camera.position.set(...this.toWorld(first));
    this.camera.rotation.order = "YXZ";
    // Metric heading (CCW from +x) to three's yaw (CW from -z).
    this.camera.rotation.y = THREE.MathUtils.degToRad(first.look_deg) - Math.PI / 2;
  }

  /** Metric frame (z up) to three.js world (y up). */
  private toWorld(w: Waypoint): [number, number, number] {
    return [w.position_m[0], w.position_m[2], -w.position_m[1]];
  }

  neighboursOf(id: string): Waypoint[] {
    return (this.neighbours.get(id) ?? [])
      .map((n) => this.byId.get(n))
      .filter((w): w is Waypoint => !!w);
  }

  /** Nearest waypoint whose room is `roomId`; used by minimap click-through. */
  waypointForRoom(roomId: string): Waypoint | undefined {
    return this.scene.waypoints.find((w) => w.room_id === roomId && w.kind === "room_centre");
  }

  goTo(id: string, now: number): void {
    const target = this.byId.get(id);
    if (!target || target.waypoint_id === this.current.waypoint_id) return;
    this.from.copy(this.camera.position);
    this.to.set(...this.toWorld(target));
    this.current = target;
    this.camera.rotation.y = THREE.MathUtils.degToRad(target.look_deg) - Math.PI / 2;
    this.camera.rotation.x = 0;
    this.t = 0;
    this.startedAt = now;
  }

  update(now: number): void {
    if (this.t >= 1) return;
    this.t = Math.min(1, (now - this.startedAt) / TRANSITION_MS);
    // Ease-in-out. A linear teleport reads as a jump cut and makes people lose
    // track of where they were, which defeats the point of the waypoint graph.
    const e = this.t < 0.5 ? 2 * this.t * this.t : 1 - Math.pow(-2 * this.t + 2, 2) / 2;
    this.camera.position.lerpVectors(this.from, this.to, e);
  }

  get moving(): boolean {
    return this.t < 1;
  }

  /** Position in the metric frame, for the minimap. */
  get metricPosition(): [number, number] {
    return [this.camera.position.x, -this.camera.position.z];
  }

  get heading(): number {
    return this.camera.rotation.y;
  }
}
