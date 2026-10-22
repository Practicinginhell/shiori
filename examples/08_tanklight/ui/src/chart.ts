// A plain SVG timeline: estimated vs true chlorine, the 0.2 mg/L line, tank level, and a
// strip showing which light was on. The <desc> carries a text summary for screen readers.

import type { StateName, TimelinePoint } from "./types.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const W = 720, H = 240, LEFT = 44, RIGHT = 44, TOP = 28, PLOT_BOTTOM = 196, STRIP_TOP = 206, STRIP_H = 14;
const FC_MAX = 1.0;

function node(tag: string, attrs: Record<string, string | number>, text?: string): SVGElement {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  if (text !== undefined) el.textContent = text;
  return el;
}

export function renderChart(svg: SVGSVGElement, desc: SVGDescElement, points: TimelinePoint[]): void {
  svg.querySelectorAll("[data-plot]").forEach((el) => el.remove());
  if (points.length < 2) {
    const empty = node("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "empty" },
      "No readings yet: press Play or +1 hour");
    empty.setAttribute("data-plot", "");
    svg.append(empty);
    desc.textContent = "No readings yet. Press Play or +1 hour to start the timeline.";
    return;
  }
  const t0 = points[0].h, t1 = points[points.length - 1].h;
  const x = (h: number) => LEFT + ((h - t0) / Math.max(1, t1 - t0)) * (W - LEFT - RIGHT);
  const yFc = (fc: number) => PLOT_BOTTOM - (Math.min(fc, FC_MAX) / FC_MAX) * (PLOT_BOTTOM - TOP);
  const yLevel = (pct: number) => PLOT_BOTTOM - (pct / 100) * (PLOT_BOTTOM - TOP);
  const add = (el: SVGElement) => { el.setAttribute("data-plot", ""); svg.append(el); };

  const area = `M${x(t0)},${PLOT_BOTTOM} ` + points.map((p) => `L${x(p.h)},${yLevel(p.level_pct)}`).join(" ")
    + ` L${x(t1)},${PLOT_BOTTOM} Z`;
  add(node("path", { d: area, class: "level" }));
  add(node("line", { x1: LEFT, x2: W - RIGHT, y1: PLOT_BOTTOM, y2: PLOT_BOTTOM, class: "axis" }));
  add(node("line", { x1: LEFT, x2: W - RIGHT, y1: yFc(0.2), y2: yFc(0.2), class: "min" }));

  add(node("path", { d: "M" + points.map((p) => `${x(p.h)},${yFc(p.fc_true)}`).join(" L"), class: "true" }));
  const est = points.filter((p) => p.fc_est !== null);
  if (est.length > 1) add(node("path", { d: "M" + est.map((p) => `${x(p.h)},${yFc(p.fc_est as number)}`).join(" L"), class: "fc" }));

  for (const v of [0, 0.5, 1.0]) add(node("text", { x: 6, y: yFc(v) + 4 }, `${v.toFixed(1)}`));
  add(node("text", { x: 6, y: 14 }, "chlorine, mg/L"));
  add(node("text", { x: W - 6, y: 14, "text-anchor": "end" }, "tank level"));
  for (const pct of [0, 50, 100]) add(node("text", { x: W - RIGHT + 6, y: yLevel(pct) + 4 }, `${pct}%`));

  // State strip, merged into runs so the SVG stays small.
  let start = 0;
  for (let i = 1; i <= points.length; i++) {
    if (i === points.length || points[i].state !== points[start].state) {
      const state: StateName = points[start].state;
      const rect = node("rect", { x: x(points[start].h), y: STRIP_TOP, height: STRIP_H,
        width: Math.max(1, x(points[i - 1].h) - x(points[start].h) + 1), class: `s-${state}` });
      rect.append(node("title", {}, state));
      add(rect);
      start = i;
    }
  }
  add(node("text", { x: LEFT, y: H - 4 }, `Day ${Math.floor(t0 / 24) + 1}`));
  add(node("text", { x: W - RIGHT - 44, y: H - 4 }, `Day ${Math.floor(t1 / 24) + 1}`));

  const last = points[points.length - 1];
  const notGreen = points.filter((p) => p.state !== "protected").length / points.length;
  desc.textContent = `Over the last ${Math.round(t1 - t0)} hours: estimated chlorine now ` +
    `${last.fc_est === null ? "unknown" : last.fc_est.toFixed(2) + " mg/L"}, tank ${Math.round(last.level_pct)}% full; ` +
    `the light was not green ${Math.round(notGreen * 100)}% of the time.`;
}
