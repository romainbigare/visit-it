/**
 * The floor-plan minimap, drawn straight from `scene.json`.
 *
 * It is a 2D canvas rather than a second 3D view on purpose: a plan is the thing
 * people already know how to read, and it stays legible at 140 px on a phone,
 * which an orthographic 3D render does not.
 */
import type { Provenance, Room, Scene } from "./types";

const FILL: Record<Provenance, string> = {
  photographed: "#c9c3b6",
  reconstructed: "#b6c2cc",
  // Inferred content renders differently everywhere it appears (ARCHITECTURE §10):
  // a map that hides which rooms are guesses is one we cannot debug.
  inferred: "#9a9a9e",
  generated: "#8d84a0",
};

export class Minimap {
  private ctx: CanvasRenderingContext2D;
  private scale = 1;
  private ox = 0;
  private oy = 0;
  private hot: Room | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private scene: Scene,
    private onPick: (room: Room) => void,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("minimap: no 2d context");
    this.ctx = ctx;
    this.fit();
    canvas.addEventListener("click", (e) => {
      const r = this.roomAt(e);
      if (r) this.onPick(r);
    });
    canvas.addEventListener("mousemove", (e) => {
      const r = this.roomAt(e);
      if (r !== this.hot) {
        this.hot = r;
        canvas.style.cursor = r ? "pointer" : "default";
      }
    });
    window.addEventListener("resize", () => this.fit());
  }

  private fit(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth || 220;
    const h = this.canvas.clientHeight || 220;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [[minx, miny], [maxx, maxy]] = this.scene.minimap.bounds_m;
    const pad = 10;
    const sx = (w - pad * 2) / Math.max(maxx - minx, 0.5);
    const sy = (h - pad * 2) / Math.max(maxy - miny, 0.5);
    this.scale = Math.min(sx, sy);
    this.ox = pad + (w - pad * 2 - (maxx - minx) * this.scale) / 2 - minx * this.scale;
    this.oy = pad + (h - pad * 2 - (maxy - miny) * this.scale) / 2 + maxy * this.scale;
  }

  /** Metres to canvas pixels. Y flips: metric y is up, canvas y is down. */
  private toPx(x: number, y: number): [number, number] {
    return [this.ox + x * this.scale, this.oy - y * this.scale];
  }

  private roomAt(e: MouseEvent): Room | null {
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    for (const room of this.scene.rooms) {
      this.ctx.beginPath();
      room.polygon_m.forEach(([x, y], i) => {
        const [cx, cy] = this.toPx(x, y);
        i === 0 ? this.ctx.moveTo(cx, cy) : this.ctx.lineTo(cx, cy);
      });
      this.ctx.closePath();
      if (this.ctx.isPointInPath(px, py)) return room;
    }
    return null;
  }

  draw(position: [number, number], headingRad: number, activeRoomId: string | null): void {
    const { ctx } = this;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    for (const room of this.scene.rooms) {
      ctx.beginPath();
      room.polygon_m.forEach(([x, y], i) => {
        const [cx, cy] = this.toPx(x, y);
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      });
      ctx.closePath();
      const active = room.room_id === activeRoomId;
      ctx.fillStyle = active ? "#e0d5a8" : FILL[room.provenance];
      ctx.globalAlpha = room.provenance === "inferred" ? 0.68 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = active ? 2 : 1;
      ctx.strokeStyle = active ? "#8a7a2e" : "#5c5c5c";
      ctx.stroke();

      const [lx, ly] = this.toPx(room.centroid_m[0], room.centroid_m[1]);
      ctx.fillStyle = "#2b2b2b";
      ctx.font = "10px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(room.display_name, lx, ly + 3);
    }

    const [px, py] = this.toPx(position[0], position[1]);
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-headingRad);
    ctx.beginPath();
    ctx.moveTo(0, -8);
    ctx.lineTo(5, 5);
    ctx.lineTo(0, 2);
    ctx.lineTo(-5, 5);
    ctx.closePath();
    ctx.fillStyle = "#c0392b";
    ctx.fill();
    ctx.restore();
  }
}
