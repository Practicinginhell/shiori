// The water office card: the 7-day forecast and the note, from the local models (office.py).
// Nothing in the kitchen display depends on it.

import { makePlan } from "./api.js";
import type { OfficePlan } from "./types.js";

function byId<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function sourceText(plan: OfficePlan): string {
  if (plan.source === "llm") return `Note reworded by ${plan.llm}; every home and number was checked against the plan.`;
  if (plan.writer_error) return `${plan.llm ?? "The local LLM"} is not running, so this is the plain template.`;
  if (plan.draft) return `${plan.llm}'s draft dropped or changed a fact, so the check sent the plain template instead.`;
  return "Plain template (no LLM).";
}

function render(plan: OfficePlan): void {
  const scores = byId<HTMLUListElement>("plan-scores");
  scores.replaceChildren(...Object.entries(plan.scores).map(([name, mae]) => {
    const li = document.createElement("li");
    li.textContent = `${name}: off by ${mae.toFixed(2)} homes a day${name === plan.model ? " (used)" : ""}`;
    return li;
  }));
  if (plan.model_error) scores.textContent = `TimesFM failed (${plan.model_error}), so this uses the simple guess.`;
  else if (!Object.keys(plan.scores).length) scores.textContent = "Not enough history to score the models yet.";
  const max = Math.max(1, ...plan.predicted);
  byId("plan-bars").replaceChildren(...plan.predicted.map((v, k) => {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.setProperty("--h", `${Math.round((v / max) * 100)}%`);
    bar.innerHTML = `<span class="bar-fill"></span><span class="bar-value">${v.toFixed(0)}</span><span class="bar-day">+${k + 1}d</span>`;
    return bar;
  }));
  byId("plan-bars").setAttribute("aria-label", `Homes needing urgent water, next 7 days: ${plan.predicted.map((v) => v.toFixed(0)).join(", ")}`);
  byId("plan-model").textContent = plan.model;
  byId("plan-source").textContent = sourceText(plan);
  // The small model writes Markdown bold; show plain text.
  byId("plan-note").textContent = plan.note.replace(/\*\*/g, "");
  byId("plan-out").hidden = false;
}

export function setupOffice(): void {
  const button = byId<HTMLButtonElement>("make-plan");
  const status = byId("plan-status");
  button.addEventListener("click", async () => {
    button.disabled = true;
    status.textContent = "Running the local models… (the first run loads TimesFM and takes longer)";
    try {
      render(await makePlan());
      status.textContent = "";
    } catch (err) {
      status.textContent = `Could not make the plan: ${(err as Error).message}`;
    } finally {
      button.disabled = false;
    }
  });
}
