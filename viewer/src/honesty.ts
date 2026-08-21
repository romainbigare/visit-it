/**
 * Honesty rendering (AD-15, ARCHITECTURE §10).
 *
 * The shell's glTF already carries one material per provenance class, so the
 * treatment is a material tweak rather than a shader: photographed and
 * reconstructed surfaces render normally, inferred surfaces are desaturated and
 * slightly transparent, and the legend says which is which in plain words.
 *
 * This is kept for debuggability, not compliance. When a room looks wrong the
 * first question is always "is that surface real?", and a viewer that renders
 * invented geometry identically to measured geometry cannot answer it.
 */
import * as THREE from "three";
import type { Provenance, Scene } from "./types";

const TREATMENT: Record<Provenance, { opacity: number; desaturate: number }> = {
  photographed: { opacity: 1.0, desaturate: 0.0 },
  reconstructed: { opacity: 1.0, desaturate: 0.15 },
  inferred: { opacity: 0.82, desaturate: 0.65 },
  generated: { opacity: 0.7, desaturate: 0.8 },
};

export function applyHonesty(root: THREE.Object3D): Set<Provenance> {
  const seen = new Set<Provenance>();
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) {
      const mat = m as THREE.MeshStandardMaterial;
      const name = (mat.name || "reconstructed") as Provenance;
      const t = TREATMENT[name] ?? TREATMENT.reconstructed;
      seen.add(name);
      if (t.desaturate > 0) {
        const hsl = { h: 0, s: 0, l: 0 };
        mat.color.getHSL(hsl);
        mat.color.setHSL(hsl.h, hsl.s * (1 - t.desaturate), hsl.l);
      }
      if (t.opacity < 1) {
        mat.transparent = true;
        mat.opacity = t.opacity;
      }
      mat.side = THREE.DoubleSide;
      mat.flatShading = true;
      mat.needsUpdate = true;
    }
  });
  return seen;
}

export function legendHtml(scene: Scene, present: Set<Provenance>): string {
  const rows = (Object.keys(scene.provenance_legend) as Provenance[])
    .filter((k) => present.has(k))
    .map((k) => {
      const t = TREATMENT[k] ?? TREATMENT.reconstructed;
      const swatch = k === "inferred" ? "swatch inferred" : "swatch";
      return `<li><span class="${swatch}" style="opacity:${t.opacity}"></span>
        <b>${k}</b> — ${scene.provenance_legend[k]}</li>`;
    })
    .join("");
  return `<ul class="legend">${rows}</ul>`;
}
