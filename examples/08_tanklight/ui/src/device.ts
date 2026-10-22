// The kitchen display: colour, shape and words all carry the state, so it reads without
// colour vision and without reading (the shapes and the spoken status).

import { alsoMessage, details, INUKTITUT_PENDING, message, sewageMessage, stateWord } from "./i18n.js";
import type { Lang, Snapshot, StateName } from "./types.js";

const ICONS: Record<StateName, string> = {
  protected: '<path d="M16 33 L27 44 L48 21" fill="none" stroke="#fff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>',
  check: '<path d="M32 14 V38" stroke="#fff" stroke-width="7" stroke-linecap="round"/><circle cx="32" cy="50" r="4.5" fill="#fff"/>',
  boil: '<path d="M20 20 L44 44 M44 20 L20 44" stroke="#fff" stroke-width="7" stroke-linecap="round"/>',
  service: '<path d="M40 14a10 10 0 0 0-9 14L15 44l5 5 16-16a10 10 0 0 0 14-9l-6 6-6-2-2-6z" fill="#fff"/>',
};

export interface DeviceElements {
  panel: HTMLElement;
  icon: SVGSVGElement;
  word: HTMLElement;
  message: HTMLElement;
  also: HTMLElement;
  details: HTMLUListElement;
  homeLabel: HTMLElement;
  langNote: HTMLElement;
  engineNote: HTMLElement;
}

export function renderDevice(els: DeviceElements, snap: Snapshot, lang: Lang): void {
  const status = snap.featured.status;
  els.panel.dataset.state = status.state;
  els.icon.innerHTML = ICONS[status.state];
  els.word.textContent = stateWord(status.state, lang);
  els.message.textContent = message(status, lang);
  const also = alsoMessage(status, lang);
  els.also.hidden = also === null;
  els.also.textContent = also ?? "";
  els.details.replaceChildren(...details(status, lang).map((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    return li;
  }));
  els.homeLabel.textContent = `· ${snap.featured.home_id}, ${snap.featured.people} people, ${snap.featured.capacity_l} L tank`;
  els.langNote.hidden = lang !== "iu";
  els.langNote.textContent = lang === "iu" ? INUKTITUT_PENDING : "";
  els.engineNote.textContent = snap.featured.engine === "graphflow"
    ? "This home's readings run through the graphflow household pipeline."
    : "This home's readings run through the direct rules path.";
}

export function spokenText(snap: Snapshot, lang: Lang): string {
  const status = snap.featured.status;
  const also = alsoMessage(status, lang);
  const sewage = snap.featured.sewage;
  const sewageLines = sewage.state === "ok" ? [] : [sewageMessage(sewage, lang)];
  return [stateWord(status.state, lang), message(status, lang), ...(also ? [also] : []), ...details(status, lang),
    ...sewageLines].join(". ");
}
