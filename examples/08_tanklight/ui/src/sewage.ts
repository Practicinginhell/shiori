// The sewage tank (outside or under the house): a second, smaller status on the kitchen
// display, and the sewage truck's pump-out plan. Kept apart from the drinking-water light:
// a full sewage tank is urgent, but it doesn't make the drinking water unsafe.

import { sewageDetails, sewageMessage, sewageTitle, sewageWord } from "./i18n.js";
import type { Lang, SewageStatus, Snapshot } from "./types.js";

// Reuse the four light colours: ok → green, soon → amber, urgent → red, service → grey.
const LIGHT: Record<SewageStatus["state"], string> = { ok: "protected", soon: "check", urgent: "boil", service: "service" };

export function renderSewage(box: HTMLElement, label: HTMLElement, word: HTMLElement, text: HTMLElement,
                             details: HTMLElement, snap: Snapshot, lang: Lang): void {
  const sewage = snap.featured.sewage;
  label.textContent = sewageTitle(lang);
  box.dataset.state = LIGHT[sewage.state];
  word.textContent = sewageWord(sewage.state, lang);
  text.textContent = sewageMessage(sewage, lang);
  details.textContent = sewageDetails(sewage, lang);
}

export function renderPumpOuts(tbody: HTMLTableSectionElement, snap: Snapshot): void {
  const planned = snap.policy !== "priority" ? []
    : snap.pump_outs.filter((r) => !r.can_skip).slice(0, snap.pump_outs_per_day).map((r) => r.home_id);
  tbody.replaceChildren(...snap.pump_outs.map((row, i) => {
    const tr = document.createElement("tr");
    if (row.can_skip) tr.classList.add("skip-row");
    if (row.home_id === snap.featured.home_id) tr.classList.add("featured");
    const cells: [string, string][] = [
      ["Order", String(i + 1)],
      ["Home", row.home_id + (planned.includes(row.home_id) ? " · today" : "")],
      ["Why", row.why + (row.can_skip ? " (skip)" : "")],
      ["Sewage tank", row.level_pct === null ? "?" : `${Math.round(row.level_pct)}%`],
      ["Full in", row.days_to_full === null ? "?" : `${row.days_to_full.toFixed(1)} days`],
    ];
    for (const [label, value] of cells) {
      const td = document.createElement("td");
      td.dataset.label = label;
      td.textContent = value;
      tr.append(td);
    }
    return tr;
  }));
}
