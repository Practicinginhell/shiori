// Messages from homes: a resident's text, triaged by the local classifier (inbox.py).
// The list shows where each message went and why, including the model's yes/no scores.

import { sendMessage } from "./api.js";
import type { InboxDecision, Snapshot } from "./types.js";

const ROUTE: Record<InboxDecision["route"], string> = {
  water_quality: "Bad water",
  pump_out: "Sewage truck",
  water_delivery: "Water truck",
  device: "Device",
  person: "A person reads it",
};
const SHORT: Record<string, string> = {
  water_quality: "bad water", pump_out: "sewage", water_delivery: "water", device: "device", emergency: "emergency",
};

function scores(d: InboxDecision): string {
  if (d.error) return `The model failed (${d.error}), so a person reads it.`;
  if (!d.model) return "No local model installed, so a person reads every message.";
  const parts = Object.entries(d.probabilities).map(([k, p]) => `${SHORT[k] ?? k} ${p.toFixed(2)}`);
  return `${d.model}: ${parts.join(", ")}`;
}

export function renderInbox(list: HTMLUListElement, snap: Snapshot): void {
  list.replaceChildren(...snap.inbox.map((d) => {
    const li = document.createElement("li");
    li.dataset.route = d.route;
    const head = document.createElement("p");
    head.className = "inbox-head";
    const route = document.createElement("strong");
    route.textContent = ROUTE[d.route];
    head.append(route);
    if (d.urgent) {
      const badge = document.createElement("span");
      badge.className = "badge badge-urgent";
      badge.textContent = "urgent";
      head.append(" ", badge);
    }
    const text = document.createElement("p");
    text.className = "inbox-text";
    text.textContent = `${d.home_id}: “${d.text}”`;
    const action = document.createElement("p");
    action.textContent = d.action;
    const detail = document.createElement("p");
    detail.className = "muted inbox-scores";
    detail.textContent = scores(d);
    li.append(head, text, action, detail);
    return li;
  }));
}

export function setupInbox(onSnapshot: (snap: Snapshot) => void): void {
  const form = document.getElementById("inbox-form") as HTMLFormElement;
  const box = document.getElementById("inbox-text") as HTMLTextAreaElement;
  const send = document.getElementById("inbox-send") as HTMLButtonElement;
  const status = document.getElementById("inbox-status") as HTMLElement;
  form.querySelectorAll<HTMLButtonElement>("button[data-example]").forEach((b) => {
    b.addEventListener("click", () => { box.value = b.dataset.example ?? ""; box.focus(); });
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!box.value.trim()) return;
    send.disabled = true;
    status.textContent = "Sorting the message… (the first one loads the model)";
    try {
      const { decision, snapshot } = await sendMessage(box.value);
      onSnapshot(snapshot);
      status.textContent = `Sorted: ${ROUTE[decision.route]}. ${decision.action}`;
      box.value = "";
    } catch (err) {
      status.textContent = `Could not send the message: ${(err as Error).message}`;
    } finally {
      send.disabled = false;
    }
  });
}
