import { act, getState } from "./api.js";
import { renderChart } from "./chart.js";
import { type DeviceElements, renderDevice, spokenText } from "./device.js";
import { renderFleet } from "./fleet.js";
import { INUKTITUT_PENDING, speechLang } from "./i18n.js";
import { icon, mountIcons } from "./icons.js";
import { renderInbox, setupInbox } from "./inbox.js";
import { setupOffice } from "./office.js";
import { renderPumpOuts, renderSewage } from "./sewage.js";
import { type Action, setupTour } from "./tour.js";
import type { Lang, Snapshot } from "./types.js";

function byId<T extends HTMLElement | SVGElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
}

const device: DeviceElements = {
  panel: byId<HTMLElement>("panel"),
  icon: byId<SVGSVGElement>("light-icon"),
  word: byId<HTMLElement>("state-word"),
  message: byId<HTMLElement>("message"),
  also: byId<HTMLElement>("also"),
  details: byId<HTMLUListElement>("details"),
  homeLabel: byId<HTMLElement>("home-label"),
  langNote: byId<HTMLElement>("lang-note"),
  engineNote: byId<HTMLElement>("engine-note"),
};
const errorBox: HTMLElement = byId("error");
const playButton: HTMLButtonElement = byId("play");
const controls: HTMLElement = byId("controls");

let snapshot: Snapshot | null = null;
let lang: Lang = "en";
let playTimer: number | undefined;
let busy = false;

function render(): void {
  if (!snapshot) return;
  const snap = snapshot;
  renderDevice(device, snap, lang);
  renderChart(byId<SVGSVGElement>("chart"), byId<SVGDescElement>("chart-desc"), snap.featured.timeline);
  renderFleet(byId<HTMLTableSectionElement>("fleet"), byId("fleet-stats"), byId("policy-label"), snap, lang);
  renderSewage(byId("sewage"), byId("sewage-label"), byId("sewage-word"), byId("sewage-message"), byId("sewage-details"), snap, lang);
  renderPumpOuts(byId<HTMLTableSectionElement>("pump-outs"), snap);
  renderInbox(byId<HTMLUListElement>("inbox-list"), snap);
  byId<HTMLElement>("pump-label").textContent = `· ${snap.pump_outs_per_day} stops a day`;
  byId<HTMLElement>("pump-stats").textContent = `Since the demo started: ${snap.stats.sewage_backup_hours} ` +
    `home-hours with a full sewage tank, ${snap.stats.pump_outs} pump-outs.`;
  byId<HTMLElement>("clock").textContent = snap.clock;
  byId<HTMLElement>("season-label").textContent = snap.season_label;
  byId<HTMLElement>("storm-badge").hidden = !snap.storm_today;
  document.querySelectorAll<HTMLButtonElement>("button[data-policy]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.policy === snap.policy));
  });
}

// While a request is in flight the controls say so (aria-busy) and can't be pressed twice.
function setBusy(value: boolean): void {
  busy = value;
  controls.setAttribute("aria-busy", String(value));
  controls.querySelectorAll<HTMLButtonElement>("button[data-action]").forEach((b) => { b.disabled = value; });
}

async function run(request: () => Promise<Snapshot>): Promise<void> {
  if (busy) return;
  setBusy(true);
  try {
    snapshot = await request();
    errorBox.hidden = true;
    render();
  } catch (err) {
    errorBox.hidden = false;
    errorBox.textContent = `Could not reach the demo server: ${(err as Error).message}`;
    stopPlaying();
  } finally {
    setBusy(false);
  }
}

function actionFrom(button: HTMLButtonElement): Record<string, string | number> {
  const { action, hours, days, pct, season, policy } = button.dataset;
  const body: Record<string, string | number> = { action: action as string };
  if (hours) body.hours = Number(hours);
  if (days) body.days = Number(days);
  if (pct) body.pct = Number(pct);
  if (season) body.season = season;
  if (policy) body.policy = policy;
  return body;
}

function stopPlaying(): void {
  if (playTimer !== undefined) window.clearInterval(playTimer);
  playTimer = undefined;
  playButton.setAttribute("aria-pressed", "false");
  playButton.innerHTML = `${icon("play")}<span class="label">Play</span>`;
}

playButton.addEventListener("click", () => {
  if (playTimer !== undefined) return stopPlaying();
  playButton.setAttribute("aria-pressed", "true");
  playButton.innerHTML = `${icon("pause")}<span class="label">Pause</span>`;
  playTimer = window.setInterval(() => void run(() => act({ action: "advance", hours: 1 })), 800);
});

document.querySelectorAll<HTMLButtonElement>("button[data-action]").forEach((button) => {
  button.addEventListener("click", () => void run(() => act(actionFrom(button))));
});

document.querySelectorAll<HTMLInputElement>("input[name=lang]").forEach((input) => {
  input.addEventListener("change", () => {
    lang = input.value as Lang;
    // Screen readers pick their voice from lang; Inuktitut text isn't shown yet, so English stays.
    document.documentElement.lang = lang === "fr" ? "fr" : "en";
    render();
  });
});

byId<HTMLButtonElement>("speak").addEventListener("click", () => {
  if (!snapshot) return;
  if (!("speechSynthesis" in window)) {
    errorBox.hidden = false;
    errorBox.textContent = "This browser cannot speak; the device plays recorded clips instead.";
    return;
  }
  const utterance = new SpeechSynthesisUtterance(spokenText(snapshot, lang));
  utterance.lang = speechLang(lang);
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
  if (lang === "iu") device.langNote.textContent = INUKTITUT_PENDING;
});

// The tour's "Show me" buttons run a short sequence of demo actions, then render once.
async function perform(actions: Action[]): Promise<void> {
  stopPlaying();
  await run(async () => {
    let latest: Snapshot | null = null;
    for (const action of actions) latest = await act(action);
    return latest ?? getState();
  });
}

// Kitchen mode: only what the family sees (the light and the sewage line), large, for the
// first moment of a demo or a wall-mounted screen. ?kitchen opens in it.
const kitchenButton: HTMLButtonElement = byId("kitchen-toggle");
function setKitchen(on: boolean): void {
  document.body.classList.toggle("kitchen", on);
  kitchenButton.setAttribute("aria-pressed", String(on));
  kitchenButton.querySelector(".label")!.textContent = on ? "Show the demo" : "Kitchen mode";
}
kitchenButton.addEventListener("click", () => setKitchen(!document.body.classList.contains("kitchen")));

const tour = setupTour({
  perform,
  reducedMotion: () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
});
byId<HTMLButtonElement>("tour-start").addEventListener("click", () => { setKitchen(false); tour.start(); });

mountIcons();
setupOffice();
setupInbox((snap) => { snapshot = snap; render(); });
void run(getState).then(() => {
  const params = new URLSearchParams(location.search);
  if (params.has("kitchen")) setKitchen(true);
  else if (params.has("tour")) tour.start();
});
