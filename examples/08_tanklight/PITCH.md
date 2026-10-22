# TankLight: 3-minute pitch

Script for Hack for Humanity Ottawa 2026. About 330 spoken words: 2:15 at a calm 150 a minute, leaving about 40 seconds for clicks,
the vitamin C moment and pauses. Lines marked *(cut if short)* go first if a rehearsal runs over 3:00.

## Before you go on stage

- [ ] `./scripts/dev 08_tanklight serve` is running. Open **http://127.0.0.1:8008/?kitchen**: only the kitchen display, large.
- [ ] Press **Restart in January** once (in "Show the demo") so the town starts fresh, then go back to Kitchen mode.
- [ ] Zoom the browser to 125%; use light mode on a projector.
- [ ] On the table: two jars of tap water and a vitamin C tablet (the prop for the vitamin C moment).
- [ ] Backup: a terminal with `./scripts/dev 08_tanklight` already run (the same scenarios as text), and screenshots.
- [ ] Confirm with Amenda that it's OK to mention the pipeline plan (it came from a draft resolution).
- [ ] Press "Make the office plan" once before going on: the first run loads TimesFM (about 10 seconds).
- [ ] Rehearse twice out loud with a timer; aim for 2:50.

## Timing

| Time | Section | On screen |
|---|---|---|
| 0:00–0:25 | Hook | Kitchen mode (the light alone) |
| 0:25–0:50 | The problem | Kitchen mode |
| 0:50–1:55 | Demo: four moments | "Show the demo", tour steps 1, 6, 7, 9 ("Show me") |
| 1:55–2:35 | Trucks and the office AI | Tour step 13: press "Make the office plan" first |
| 2:35–3:00 | Finding, honesty, close | Kitchen mode |

Delivery: slow down on the numbers, and stop talking while the light changes. The silence is
the demo.

## Script

### 0:00 Hook
> Seventy-six dollars. That's what 24 bottles of water cost in Inukjuak.
> There are no pipes here: rock and permafrost. A truck fills a tank inside each home, and another
> pumps the sewage tank outside. The water leaves the plant safe. Then it sits in the tank, and
> nobody can see what happens next.

### 0:25 The problem
> Chlorine is what keeps stored water safe. In Cambridge Bay, researchers found the trucks were
> fine, but the tanks were low. A Nunavut study scored household tanks: "no barriers, no
> monitoring". *(cut if short)* In Kangiqsualujjuaq, a third of homes ran out of water in one week.

### 0:50 Demo: four moments
> This is TankLight: a sensor at the tank, and one light at the kitchen tap, where you decide
> whether to drink.

**[Press "Show the demo", "Take the tour", then "Show me: Fill a nearly empty tank"]** Green.
> Green: protected. Colour, shape, pulse and words, so anyone can read it: hours of protection
> left, days of water, and the chance you run out before the truck.

**[Hold up the jar, drop in the vitamin C tablet. Next to step 6, "Show me: Add vitamin C"]** Red.
> Vitamin C removes chlorine. *(pause)* Red: boil first.

**[Next to step 7, "Show me: Pull the probe out"]** Grey.
> Pull the probe out: grey. TankLight never shows green when it isn't sure.

**[Next to step 9, "Show me: The tank heater fails in January"]** The sewage line turns red.
> And the sewage tank outside: if its heater fails in January, it warns before it freezes.

### 1:55 Trucks and the office AI
**[Next to step 13. Press "Make the office plan" now; it takes about 6 seconds]**
> Now the trucks. Instead of a fixed loop, trucks go where tanks are about to run dry. In our
> simulated July, with the same trucks, homes went from 534 hours without water to 195.
> For the water office, AI runs on this laptop, offline. TimesFM forecasts next week's demand
> around the blizzard, and a tiny language model writes the plan, but a check throws it out if a
> single number changes.
> AI plans the trucks. It never decides whether your water is safe.

### 2:35 Finding, honesty, close
**[Kitchen mode]**
> Our biggest finding: in simulation, truck chlorine fades below the minimum within a day or two
> in most tanks. That's evidence the health board can act on.
> It's simulated, it's not an E. coli test, and the Inuktitut audio will be recorded with the
> community.
> Trucks today, pipes tomorrow. TankLight: know your water is protected before you drink it.

## If the demo fails

Say "here's the same run from the device logic" and show the terminal output of
`./scripts/dev 08_tanklight`: each scenario and the light it gives, in text. Don't debug on stage.

## Likely judge questions

The most likely ones are below. The full set, grouped by topic, is in [QA.md](QA.md).

| Question | Answer |
|---|---|
| How do you measure chlorine cheaply? | An ORP and pH probe, calibrated against a DPD test kit (the one water operators already use). For a real pilot, a reagent-free amperometric sensor that works in still water. |
| How does it read the tank level? | An ultrasonic sensor on the tank lid measures the distance to the water, so nothing touches the drinking water (waterproof modules cost about $10–20). At installation we record "empty" and "full" and the tank's shape, which turns distance into litres. Rises over 10% are deliveries, which gives each home's delivery history; the falls in between are the family's own daily use, which gives days of water left. A missing or impossible reading turns the light grey, never a guess. In the demo the sensor is simulated, with noise. The sewage tank uses the same idea, rated for −30 °C. |
| How do you know priority dispatch is better? | A backtest: the same simulated town, same storms, same number of truck stops, run for 30 days with a fixed loop and with TankLight's priority. Hours homes spent without water: January 1,486 → 1,282, July 534 → 195, with fewer run-outs too. Priority skips homes with 3 or more days of water, so stops go to the homes that are shortest. It's simulated; real delivery logs would confirm it. Run it: `backtest` (and `--season july`). |
| Does it detect E. coli? | No, and we say so. Pair it with a monthly field test that needs no electricity (Aquagenx CBT). TankLight covers the gap between those tests. |
| Won't the light be red all the time? | In our simulation it's often not green (82–94% of the time), because truck chlorine fades fast in a tank. That's the finding, not a bug: it's evidence for a higher truck dose, or thresholds set with the health board. They decide the thresholds, not us, and a pilot must settle it first, because a light that is always red gets ignored. |
| Why does the sewage tank need pumping out? | There are no sewer pipes. Everything that goes down the sink, shower or toilet collects in a holding tank outside or under the house, and a sewage truck empties it. It fills about as fast as the family uses water: a family of 5 fills an 1,100 L tank in about 4 days. A full tank backs up into the house, so the family has to stop using water; in a blizzard the trucks stop, and if the tank heater fails it can freeze and can't be pumped at all. |
| What about sewage? | The sewage tank outside has its own line on the display: how full, days until full, and a freezing warning if the tank heater fails. The sewage truck gets a pump-out plan; in simulation, 1–45% fewer hours with a full tank for the same stops. |
| Where does it go in the house? | Two parts. The sensor unit sits at the indoor drinking-water tank: probes in the water line leaving the tank, a level sensor on top. The display goes on the wall beside the kitchen tap, at adult eye height, facing the room, connected by a short cable or low-power radio. The sewage sensor sits on the outdoor tank and shows as the second line on the display. Mains power with a battery for blizzard outages. We'd confirm the spot with families: wherever they actually get drinking water. |
| What does it cost? | Prototype parts about $200–250 CAD; the target is under $100 at volume (an estimate). |
| Who maintains it? | Standard parts, a swappable probe cartridge, and local "water keepers" doing a monthly check. Spares go on the fall sealift. |
| Is the data real? | No, everything is simulated with placeholder numbers. Next step: real delivery logs and DPD readings with the municipality and KRG. |
| Where's the AI? | In the water office, not the kitchen. TimesFM 3.0 (Google's forecasting model, run locally on Apple silicon with MLX) forecasts next week's urgent-water demand, with blizzard days as a known input. It is scored against a "last 7 days" guess on the past 4 weeks and used only when it wins: in our simulated January it did (average error 1.21 vs 1.38 homes a day); in steadier July it didn't, so the plan used the simple guess. A 270M-parameter LLM (gemma3 through Ollama) rewords the plan in a graphflow workflow; a check keeps its note only if every home and number matches, otherwise the plain template goes out. The safety light stays plain rules, on purpose. |
| Why such small models? | They run offline on one office laptop, and the job is small: forecast one number a day, reword six lines. The check means a small model can't do harm. TimesFM's weights are non-commercial: fine for a demo; a real deployment needs Google's permission or another model. |
| Privacy? | Readings stay in the home. Community reports would be opt-in and combined, and the data belongs to the community. |

## Sources for the numbers

- $76.29 for 24 × 500 mL: photo in the challenge slides (Amenda Soucy).
- 0.2 mg/L minimum; "no barriers, no monitoring": [risk matrices, IJCH 2025](https://doi.org/10.1080/22423982.2025.2450877).
- Trucks fine, tanks low: [Cambridge Bay, Water Research 2025](https://pubmed.ncbi.nlm.nih.gov/41072346/).
- 33% ran out in a week, 37% don't drink tap water: [Kangiqsualujjuaq, J. Water & Health 2024](https://doi.org/10.2166/wh.2024.246).
- Inukjuak advisory, E. coli at the loading arm: [Nunatsiaq News, June 2026](https://nunatsiaq.com/stories/article/inukjuak-under-boil-water-advisory-after-e-coli-detected-at-water-plant/).
- SWOT: [York University Dahdaleh Institute](https://www.yorku.ca/dighr/project/safe-water-optimization-tool/).
- 534 → 195 hours without water (July; January 1,486 → 1,282): TankLight backtest, **simulated**.
- 1–45% fewer sewage-backup hours with priority pump-outs: **simulated**, 30 days, seeds 7 and 11, January and July.
