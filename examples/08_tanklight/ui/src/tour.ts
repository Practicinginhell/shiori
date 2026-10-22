// A guided tour of the UI. Each step highlights one part, explains what it does, and can run
// that part's demo ("Show me"), so the tour doubles as a live demo script.
// ui-ux-pro-max rules: progress shown, Back and Skip always available, Esc closes, focus moves
// to the step and returns to the start button, no keyboard trap (the page stays usable).

export type Action = Record<string, string | number>;

export interface TourStep {
  target: string;           // element id to highlight
  title: string;
  body: string;
  show?: { label: string; actions: Action[] };
}

// Every "Show me" starts from a clean jar: probe in, nearly empty tank, fresh delivery.
const FRESH_TANK: Action[] = [
  { action: "attach_probe" }, { action: "drain", pct: 10 }, { action: "deliver_now" }, { action: "advance", hours: 2 },
];

export const STEPS: TourStep[] = [
  {
    target: "panel",
    title: "The kitchen display",
    body: "This is the one thing a family looks at. The state is shown four ways at once: colour, shape " +
      "(✓ ! ✕ wrench), the pulse speed and words. It reads without colour vision and without reading. " +
      "Green: protected. Amber: check. Red: boil first. Grey: the device needs service and never pretends to be green.",
    show: { label: "Fill a nearly empty tank", actions: FRESH_TANK },
  },
  {
    target: "details",
    title: "The numbers behind the light",
    body: "Hours of chlorine protection left (until it falls below 0.2 mg/L, the regulated minimum), days of " +
      "water left, the chance of running out before the next truck (from this home's own delivery history), " +
      "and the chlorine the sensor estimates.",
  },
  {
    target: "langs",
    title: "Language",
    body: "English and French now. Inuktitut is deliberately empty: it will be written and recorded with a " +
      "fluent speaker from the community, never machine-translated.",
  },
  {
    target: "speak",
    title: "Hear the status",
    body: "Reads the light, the message and the numbers aloud. In the browser this uses speech synthesis; " +
      "the device would play clips recorded by a community member.",
  },
  {
    target: "grp-time",
    title: "Time",
    body: "The town is simulated in half-hour steps. Play runs an hour every 0.8 seconds; the other buttons jump ahead. " +
      "Chlorine fades and water gets used as time passes.",
    show: { label: "Jump 6 hours", actions: [{ action: "advance", hours: 6 }] },
  },
  {
    target: "grp-water",
    title: "Water in this home",
    body: "The same tests as the live jar demo. \"Use most of the water\" sets up a nearly empty tank. Vitamin C " +
      "removes chlorine (the light turns red); clay makes the water cloudy (red). The light waits for two " +
      "readings in a row before changing, so one noisy reading doesn't cause an alarm.",
    show: { label: "Add vitamin C to a fresh tank", actions: [...FRESH_TANK, { action: "vitamin_c" }, { action: "advance", hours: 1.5 }] },
  },
  {
    target: "grp-device",
    title: "Device faults",
    body: "Pull the probe out and the light turns grey: Needs service. A crashed sensor or program can never " +
      "leave the light green. Put the probe back and it recovers after three good readings.",
    show: { label: "Pull the probe out", actions: [...FRESH_TANK, { action: "pull_probe" }, { action: "advance", hours: 0.5 }] },
  },
  {
    target: "grp-weather",
    title: "Blizzards and seasons",
    body: "A blizzard stops the trucks. Chlorine fades while the water sits, so the light changes; with a day " +
      "and a half of water left it adds \"Also: ask for a delivery\", well before the tank runs dry. Restart in " +
      "July: warmer water loses chlorine faster, so protection lasts fewer hours.",
    show: { label: "Blizzard, then 2½ days", actions: [...FRESH_TANK, { action: "storm", days: 3 }, { action: "advance", hours: 60 }] },
  },
  {
    target: "sewage",
    title: "The sewage tank outside",
    body: "Every litre the family uses ends up in the sewage tank outside or under the house. This line shows how " +
      "full it is, when it will be full, and its temperature. If the tank heater fails in January, it warns of " +
      "freezing before the tank can't be pumped. It is separate from the water light: a full sewage tank is " +
      "urgent, but it doesn't make the drinking water unsafe.",
    show: { label: "The tank heater fails in January", actions: [{ action: "fresh_sewage_tank" }, { action: "heater_off" },
      { action: "advance", hours: 6 }, { action: "heater_on" }] },
  },
  {
    target: "grp-trucks",
    title: "Truck dispatch",
    body: "Fixed loop: trucks visit every house in turn, full or not. Priority: trucks follow each tank's forecast " +
      "(empty first, full tanks skipped). In the simulation this cuts the hours homes spend without water with " +
      "the same number of stops.",
  },
  {
    target: "chart-card",
    title: "The last 7 days",
    body: "Solid line: the chlorine the device estimates. Dashed grey: the true value, which only the simulator " +
      "knows (it shows the estimate is honest). Red dashes: the 0.2 mg/L minimum. Blue area: tank level. The " +
      "strip underneath: which light was on.",
  },
  {
    target: "fleet-card",
    title: "Today's truck plan",
    body: "Every home in delivery order with the reason. \"Skip\" means the tank is full enough. Some homes say " +
      "\"ask the water office to check the tank\": when a tank is mostly old water, a top-up can't restore " +
      "chlorine or clear cloudy water, so a truck stop would be wasted.",
  },
  {
    target: "inbox-card",
    title: "Messages from homes",
    body: "Families text the water office. A small local model (Laya) answers one yes/no question per need and " +
      "sorts the message: water truck, sewage truck, bad water, device, or a person reads it when it isn't sure. " +
      "A health report always comes first, and a request moves that home up the truck plan. Try an example.",
  },
  {
    target: "office-card",
    title: "Water office: the local AI",
    body: "The only place AI is used, and it's for planning, not safety. On the office laptop, offline, TimesFM " +
      "forecasts how many homes will need urgent water next week, with the blizzard forecast as an input. It's " +
      "used only if it beat a simple guess on the last 4 weeks. A small LLM (270M parameters) rewords the plan, " +
      "and a check throws the note away if a single home or number changed.",
  },
  {
    target: "kitchen-toggle",
    title: "Kitchen mode",
    body: "Hides the demo and shows only what the family would see, large enough to read across the room: the " +
      "light, the message and the sewage line. Press it again to come back.",
  },
  {
    target: "foot",
    title: "What this is, honestly",
    body: "All data is simulated with placeholder numbers. TankLight tracks chlorine protection; it doesn't " +
      "detect E. coli or metals, and official boil-water advisories always come first.",
  },
];

export interface TourDeps {
  perform: (actions: Action[]) => Promise<void>;
  reducedMotion: () => boolean;
}

export function setupTour(deps: TourDeps): { start: () => void } {
  const panel = document.getElementById("tour") as HTMLElement;
  const title = document.getElementById("tour-title") as HTMLElement;
  const body = document.getElementById("tour-body") as HTMLElement;
  const count = document.getElementById("tour-count") as HTMLElement;
  const bar = document.getElementById("tour-progress") as HTMLProgressElement;
  const back = document.getElementById("tour-back") as HTMLButtonElement;
  const next = document.getElementById("tour-next") as HTMLButtonElement;
  const show = document.getElementById("tour-show") as HTMLButtonElement;
  const close = document.getElementById("tour-close") as HTMLButtonElement;
  const opener = document.getElementById("tour-start") as HTMLButtonElement;
  let index = 0;
  let highlighted: HTMLElement | null = null;

  function render(): void {
    const step = STEPS[index];
    highlighted?.classList.remove("tour-target");
    highlighted = document.getElementById(step.target);
    highlighted?.classList.add("tour-target");
    highlighted?.scrollIntoView({ block: "start", behavior: deps.reducedMotion() ? "auto" : "smooth" });
    title.textContent = step.title;
    body.textContent = step.body;
    count.textContent = `Step ${index + 1} of ${STEPS.length}`;
    bar.max = STEPS.length;
    bar.value = index + 1;
    back.disabled = index === 0;
    next.textContent = index === STEPS.length - 1 ? "Finish" : "Next";
    show.hidden = !step.show;
    show.textContent = step.show ? `Show me: ${step.show.label}` : "";
    title.focus();
  }

  function stop(): void {
    panel.hidden = true;
    highlighted?.classList.remove("tour-target");
    highlighted = null;
    opener.focus();
  }

  back.addEventListener("click", () => { if (index > 0) { index--; render(); } });
  next.addEventListener("click", () => { if (index < STEPS.length - 1) { index++; render(); } else stop(); });
  close.addEventListener("click", stop);
  show.addEventListener("click", async () => {
    const step = STEPS[index];
    if (!step.show) return;
    show.disabled = true;
    try {
      await deps.perform(step.show.actions);
    } finally {
      show.disabled = false;
    }
    // Every demo changes the light, so bring the kitchen display into view to show the result.
    document.getElementById("device")?.scrollIntoView({ block: "start", behavior: deps.reducedMotion() ? "auto" : "smooth" });
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) stop(); });

  return {
    start: () => {
      index = 0;
      panel.hidden = false;
      render();
    },
  };
}
