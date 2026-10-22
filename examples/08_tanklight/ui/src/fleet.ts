import { icon } from "./icons.js";
import { stateWord } from "./i18n.js";
import type { Lang, Snapshot } from "./types.js";

// data-label is shown as the field name when a row becomes a card on phones.
function cell(label: string, text = ""): HTMLTableCellElement {
  const td = document.createElement("td");
  td.dataset.label = label;
  td.textContent = text;
  return td;
}

export function renderFleet(tbody: HTMLTableSectionElement, stats: HTMLElement, policyLabel: HTMLElement,
                            snap: Snapshot, lang: Lang): void {
  const byId = new Map(snap.homes.map((h) => [h.home_id, h]));
  // Only priority dispatch follows this order; the fixed loop's stops are not these.
  const planned = snap.policy !== "priority" ? []
    : snap.fleet.filter((r) => !r.can_skip).slice(0, snap.visits_per_day).map((r) => r.home_id);
  tbody.replaceChildren(...snap.fleet.map((row, i) => {
    const home = byId.get(row.home_id);
    const tr = document.createElement("tr");
    if (row.can_skip) tr.classList.add("skip-row");
    if (row.home_id === snap.featured.home_id) tr.classList.add("featured");
    tr.append(cell("Order", String(i + 1)));

    const name = cell("Home", row.home_id);
    if (planned.includes(row.home_id)) {
      const today = document.createElement("span");
      today.className = "today";
      today.innerHTML = `${icon("truck")}<span>today</span>`;
      name.append(today);
    }
    tr.append(name);

    const light = cell("Light");
    if (home?.state) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.dataset.state = home.state;
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.setAttribute("aria-hidden", "true");
      pill.append(dot, stateWord(home.state, lang));
      light.append(pill);
    }
    tr.append(light);

    const why = cell("Why", row.why);
    if (row.can_skip) {
      const badge = document.createElement("span");
      badge.className = "badge badge-skip";
      badge.textContent = "skip";
      why.append(" ", badge);
    }
    tr.append(why);
    tr.append(cell("Tank", row.level_pct === null ? "?" : `${Math.round(row.level_pct)}%`));
    tr.append(cell("Water left", row.days_left === null ? "?" : `${row.days_left.toFixed(1)} days`));
    tr.append(cell("People", String(row.people ?? "?")));
    return tr;
  }));

  policyLabel.textContent = snap.policy === "priority"
    ? `· priority dispatch, ${snap.visits_per_day} stops a day`
    : `· fixed loop, ${snap.visits_per_day} stops a day (order shown is TankLight's advice)`;
  const s = snap.stats;
  const perVisit = s.visits ? Math.round(s.litres_delivered / s.visits) : 0;
  stats.textContent = `Since the demo started: ${s.dry_home_hours} home-hours without water, ${s.runouts} run-outs, ` +
    `${s.visits} truck stops, ${perVisit} L per stop.`;
}
