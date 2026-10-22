import type { InboxDecision, OfficePlan, Snapshot } from "./types.js";

async function parse(response: Response): Promise<Snapshot> {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `Request failed (${response.status})`);
  return body as Snapshot;
}

export async function getState(): Promise<Snapshot> {
  return parse(await fetch("/api/state"));
}

export async function act(action: Record<string, string | number>): Promise<Snapshot> {
  return parse(await fetch("/api/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(action),
  }));
}

// The office plan runs the local models (seconds, longer the first time), so it has its own call.
export async function makePlan(): Promise<OfficePlan> {
  const response = await fetch("/api/plan", { method: "POST", body: "{}" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `Request failed (${response.status})`);
  return body as OfficePlan;
}

export async function sendMessage(text: string): Promise<{ decision: InboxDecision; snapshot: Snapshot }> {
  const response = await fetch("/api/message", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `Request failed (${response.status})`);
  return body;
}
