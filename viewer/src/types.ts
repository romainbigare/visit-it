/**
 * The scene contract. Mirrors `schemas/scene.json` — the viewer's ONLY input.
 *
 * Keeping this file honest matters more than it looks: stream E develops against
 * hand-authored fixture scenes, and the contract (not the pipeline) is the
 * interface between the two. If a field appears here that stage 9 does not emit,
 * the fixture will pass and the real listing will not.
 */

export type Provenance = "photographed" | "reconstructed" | "inferred" | "generated";

export interface Room {
  room_id: string;
  label: string | null;
  display_name: string;
  polygon_m: [number, number][];
  centroid_m: [number, number];
  height_m: number;
  area_m2: number | null;
  provenance: Provenance;
  confidence: number;
  photo_ids: string[];
  splats: unknown | null;
}

export interface Waypoint {
  waypoint_id: string;
  room_id: string | null;
  position_m: [number, number, number];
  look_deg: number;
  kind: "room_centre" | "doorway" | "photo_pose";
  label: string;
}

export interface Scene {
  schema: "scene/v1";
  listing_id: string;
  generated_at: string;
  tier: "A" | "B";
  units: "metres";
  profile?: string;
  /** ARCHITECTURE §5: the ONLY area the viewer may show. Never our own measurement. */
  advertised_area_m2: number | null;
  address: string | null;
  shell: { uri: string; bytes: number; triangles: number } | null;
  rooms: Room[];
  waypoints: Waypoint[];
  waypoint_edges: [string, string][];
  minimap: { bounds_m: [number, number][]; footprint_m: [number, number][] | null };
  provenance_legend: Record<string, string>;
  provenance_colours?: Record<string, number[]>;
  confidence: number;
  qa_flags: string[];
}

export type ViewMode = "walkthrough" | "dollhouse";
