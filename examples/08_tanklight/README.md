# TankLight: household water protection tracker (demo core)

A hackathon demo for **Hack for Humanity Ottawa 2026**. The challenge (from Inukjuak, Nunavik):
make water quality visible, understandable and accessible in homes that get their water by
truck and store it in a tank.

TankLight is a **protection tracker, not a water tester**. It watches whether the chlorine
that protects stored water is still there, how long it will last, and whether the tank will
run dry before the next truck. The answer is one light a family can read across the room:

| Light | Shape | Means |
|---|---|---|
| Protected | ✓ | Chlorine above 0.2 mg/L, clear water, enough water |
| Check | ! | Protection fading, water running low, or water a bit cloudier |
| Boil first | ✕ | No chlorine protection, cloudy water, or the tank nearly empty |
| Needs service | wrench | A sensor is unsure: **never shown as green** |

Under the light, a second line covers the **sewage tank** outside or under the house: how full
it is, days until full, and a freezing warning when its heater fails. It is kept separate from
the drinking-water light, because a full sewage tank is urgent but doesn't make the water unsafe.

**Everything here is simulated.** Chlorine decay, storms and water use are placeholder models,
not measurements from Inukjuak. TankLight does not detect E. coli or metals, and official
boil-water advisories always come first.

## Try it

```bash
./scripts/dev 08_tanklight                     # terminal demo: delivery, vitamin C, clay, probe, blizzard, seasons
./scripts/dev 08_tanklight backtest            # warnings and truck dispatch on a simulated town (January)
./scripts/dev 08_tanklight backtest --season july
./scripts/dev 08_tanklight plan                # the water office's 7-day plan (local TimesFM + small LLM)
./scripts/dev 08_tanklight inbox               # residents' messages sorted by a local classifier (Laya)
./scripts/dev 08_tanklight serve               # builds the TypeScript UI (Node >= 22.13), opens on http://127.0.0.1:8008
```

In the UI, **Kitchen mode** (or `/?kitchen`) shows only what the family sees. **Take the tour** (or open `/?tour`) walks through every part, and each step's
"Show me" runs that part's demo. The 3-minute pitch built on it is in [PITCH.md](PITCH.md). Answers to judge questions are in [QA.md](QA.md).

`serve` needs Node.js 22.13 or newer and no npm packages: `ui/build.mjs` uses Node's built-in
type stripping. It does not type-check; add the `typescript` package and run `tsc --noEmit`
if you want that.

## What each part does

| Part | File | Proved by (`tests/test_tanklight.py`) |
|---|---|---|
| Rules core | `tanklight/rules.py` | ORP + pH → free chlorine, with a field-calibration fit from DPD readings; temperature-aware decay and a "hours protected" forecast at 0.2 mg/L; run-out risk from the same month's delivery gaps; sensor faults → Service; worst state wins; a light that doesn't flicker on one noisy reading |
| Household pipeline | `tanklight/pipeline.py` | graphflow workflow: `validate → parallel(quality, supply) → combine`. It gives the same status as the direct path, and a branch that crashes gives Service, not green. `TankMonitor` learns fills, delivery gaps and water use |
| Simulated town | `tanklight/simulate.py` | 12 homes, trucks with 5 stops a day, blizzard days; demo scenarios; January protection lasts longer than July |
| Backtest | `tanklight/backtest.py` | Warnings ahead of run-outs and chlorine loss; fixed truck loop vs delivery priority |
| Truck priority | `tanklight/fleet.py` | Most urgent first; skips full tanks; says when a truck can't fix the problem. `pump_out_priority` does the same for the sewage truck (full or freezing tanks first) |
| Sewage tank | `rules.assess_sewage`, `simulate.py` | Fills with every litre used; warns 1.5 days before full and when the heater fails in January; priority pump-outs cut backup hours |
| Server | `tanklight/server.py` | JSON API; demo actions are whitelisted; static files come from a fixed list (no path traversal); binds to 127.0.0.1 |
| UI | `ui/` (TypeScript) | Builds with Node; kitchen display, demo controls, 7-day chart, truck plan |

`rules.py` imports only `math` and avoids annotations and dataclasses so it can be copied onto
an ESP32 running MicroPython. It was written to that subset but **has not been run on a device**.

## Making the trucks more efficient

A fixed loop visits every house in turn, full or not. `fleet.delivery_priority` uses each
tank's own forecast instead:

1. **Tank nearly empty, or will run out before the next truck:** first.
2. **Less than a day and a half of water, or a high run-out risk with under 3 days left:** next.
3. **Chlorine low in a low tank:** a delivery replaces most of the water, so it restores protection.
4. **Full enough (3 or more days of water for this household):** skip today, which frees the stop
   for a home that needs it. Days, not tank percent: 70% full lasts about 2 days for 8 people.
5. **Chlorine gone or cloudy water in a full tank:** skip the truck. A top-up is mostly old water
   and can't fix it (the demo shows this), so the plan asks the water office to check the tank.

Backtest (`./scripts/dev 08_tanklight backtest`), same town, same storms, same number of stops:

| Season | Fixed loop: hours homes had no water | Priority | L per stop, fixed → priority |
|---|---|---|---|
| January (storms 15% of days) | 1,486 | 1,282 | 833 → 861 |
| July | 534 | 195 | 792 → 826 |

The sewage truck (4 stops a day) gets the same treatment. Over 30 days, with the same number of
pump-outs, priority cut the hours homes had a full sewage tank by 1–45% (seeds 7 and 11,
January and July; for example July seed 7: 434 → 240).

In both, every run-out was warned at least 24 h ahead. Counting run-out *events* is misleading
when trucks are short (every home runs dry sometimes); **hours without water** is what dispatch
changes, so the test checks that across three seeds.

More gains that aren't modelled here: ordering stops to cut driving distance, topping every home
up before a forecast blizzard, and giving the water office a tank-check list instead of wasted stops.

## The water office plan (local AI)

AI is used in one place: planning at the water office. The kitchen light never depends on it.

```
Community.daily ──▶ forecast.forecast_week ──▶ office.plan_facts ──▶ graphflow: compose ▶ draft (LLM) ▶ check ──▶ note
                    (TimesFM vs baseline)       (code decides)                  (template)                 (facts must match)
```

- **Forecast** (`tanklight/forecast.py`): how many homes will need urgent water each day next
  week (the top two delivery tiers). TimesFM 3.0 runs locally on Apple silicon (MLX), with
  blizzard days as a known past-and-future covariate. It is scored against "the mean of the last
  7 days" on the past 4 weeks, and the plan uses whichever did better. Simulated results, 92 days
  of history: January seed 11, 0.92 vs 1.16 homes a day; seed 3, 1.18 vs 1.30; July about even
  (0.68 vs 0.66). With only 3 weeks of history TimesFM did much worse (4.98 vs 0.89), so the
  demo server warms up 70 days.
- **Plan** (`tanklight/office.py`): code decides the facts: blizzard days, which homes to top up
  and pump out first with the stops before it (most urgent first), how many will still run
  short, and tanks that need a check.
- **Note:** a graphflow workflow. A small local LLM (`gemma3:270m` through Ollama, localhost only)
  rewords the template; `check` keeps its note only when it names exactly the same homes and
  numbers, otherwise the template goes out. In testing, it rejected about 1 draft in 5.

Optional and offline once downloaded. Without them, `plan` and the UI card fall back to the
baseline and the template.

```bash
python -m pip install "timesfm[mlx]"   # Apple silicon; downloads the 1.3 GB checkpoint on first use
ollama pull gemma3:270m                # about 290 MB
./scripts/dev 08_tanklight plan --season july --storm-in 0
```

TimesFM's weights are under Google's non-commercial licence: fine for this demo, not for a product.

## Messages from homes (local classifier)

Families text the water office; `tanklight/inbox.py` sorts each message with
[Laya](https://huggingface.co/convaiinnovations/laya), a small local classifier that answers yes/no
questions with calibrated probabilities instead of generating text. One question per need: water
truck, sewage truck, bad water, device problem, plus "is anyone at risk right now?".

- The strongest need routes the message. A strong bad-water report (0.8 or more) always leads; a
  weaker one still marks the message urgent and flags it to the water office.
- Nothing above 0.5, or a model that fails or times out (graphflow's `timeout` and `on_error`):
  a person reads it. The classifier only sorts the inbox; it never closes a message.
- Requests feed the plans: "we're out of water" makes that home a water-truck stop even when the
  sensor says full (the family can see the tank); a reported backup goes to the top tier of the
  sewage plan. Requests clear when the truck comes.

Measured with the real model: on the 12 labelled examples in `inbox.EXAMPLES` (which the routing
rule was tuned on), 10 right and 2 left for a person. On 10 messages it had never seen, 6 right and
4 left for a person: none sent to the wrong place. Its misses (brown water, a blank display, French
"tastes strange") are slower, not wrong.

```bash
python -m pip install laya            # pulls PyTorch and Transformers (about 2 GB); Apache 2.0
./scripts/dev 08_tanklight inbox
```

## What the simulation says (and why it matters)

- **The light is not green 82–94% of the time.** With truck water at 0.3–0.8 mg/L (Cambridge
  Bay trucks averaged 0.48) mixed into what's left in the tank, chlorine falls under 0.2 mg/L
  within a day or two. That matches studies that found low chlorine in many household tanks,
  but it depends on the placeholder decay rate. On real data, this is either evidence for a
  higher truck dose (SWOT-style targets) or a sign that thresholds need tuning with the health
  board. Either way it has to be settled before a pilot, because a light that is always red
  gets ignored.
- **A top-up can't restore chlorine in a mostly-full tank.** Protection comes back only when
  most of the tank is new water.

## UI and accessibility

Designed with the `ui-ux-pro-max` skill (claude-skills-hub): calm cyan and health-green brand
tokens, kept separate from the four status colours. Checked against the skill's critical rules:

- The state is carried by colour, shape, motion speed and words, never colour alone.
- Visible focus rings, a skip link, real buttons of 44 px or more with visible text, and pressed
  and busy feedback (`aria-busy`, disabled while updating).
- The status is announced (`role="status"`), and `<html lang>` follows the language choice.
- Reduced motion is respected. The pulse is kept otherwise because it is a colour-free urgency signal.
- One inline SVG icon family (no emoji), and local fonts only (Noto Sans, Noto Sans Canadian
  Aboriginal or Euphemia UCAS for syllabics), because the product story is offline.
- No horizontal scrolling at 375 px: on phones the truck plan becomes labelled cards, and the
  chart scrolls inside its card. Light and dark themes.

**Inuktitut is deliberately not machine-translated.** The ᐃᓄᒃᑎᑐᑦ option says the text and
audio will be written and recorded with a fluent speaker from the community.

## Not done (roadmap)

Real sewage-tank sensors and heater data, Truck Relay / LoRa, community reports for council, pipeline-connection
version, real DPD calibration, running `rules.py` on an ESP32, and testing with Inukjuak residents.
