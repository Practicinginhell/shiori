# TankLight: judge questions and answers

Short answers you can say out loud, grouped by topic. The first sentence of each is the answer;
the rest is backup if they ask more. "Simulated" means our model town, not measurements from
Inukjuak. If you don't know, say so and say how you'd find out: honesty scores better than a
guess.

The five most likely: [is it safe to trust](#safety-and-trust), [won't it always be red](#the-light),
[where's the AI](#the-ai), [what does it cost](#cost-and-funding), [did you talk to residents](#people-and-community).

## The problem

**Why not just fix the water plant?**
The plant isn't the gap: water leaves it safe. What nobody sees is the days it sits in a tank at
home, where chlorine fades. Cambridge Bay researchers found truck water was fine but many
household tanks were low.

**Why not just build pipes?**
Rock and permafrost make buried pipes slow and very expensive, and Inukjuak is only starting to
discuss a pipeline. Trucks will serve homes for years; TankLight is for those years, and its records
can help plan the pipeline.

**How big is the problem?**
A Nunavut risk study scored household tanks "no barriers, no monitoring". In Kangiqsualujjuaq a
third of households ran out of water in one week and 37% don't drink their tap water. Inukjuak
had a boil-water advisory in June 2026.

**Who is it for?**
The family at the kitchen tap first; then the water office and truck drivers, who plan deliveries
and pump-outs.

## The light

**What do the four states mean?**
Green, Protected: chlorine above 0.2 mg/L, clear water, enough water. Amber, Check: protection
fading, water running low, or a bit cloudier. Red, Boil first: no chlorine protection, cloudy
water, or a nearly empty tank. Grey, Needs service: a sensor is unsure, never shown as green.

**Won't it be red all the time?**
In our simulation it's not green 80–94% of the time, because truck chlorine fades below the
minimum within a day or two. That's the finding, not a bug: it's evidence for a higher truck dose,
or for thresholds set with the health board. A light that is always red gets ignored, so this has
to be settled before a pilot.

**Why not just show the chlorine number?**
A number needs you to know what 0.2 mg/L means. The light gives the decision, and the numbers are
underneath for anyone who wants them.

**What if someone is colour-blind or can't read?**
Every state is shown four ways: colour, shape (✓ ! ✕ wrench), pulse speed and words. It can also
read itself aloud.

**Won't it flicker or cry wolf?**
It changes to a worse state only after two readings agree, and back to a better one only after
three, so one noisy reading neither alarms nor reassures.

**What does "Needs service" do for the family?**
It says: don't rely on me, follow the municipality's advice. It never pretends to be green.

## Safety and trust

**Can a family trust green?**
Green means the chlorine that protects stored water is still there. It doesn't mean the water was
tested for E. coli or metals, and official boil-water advisories always come first.

**Does it detect E. coli?**
No, and we say so on the screen. Pair it with a monthly field test that needs no electricity
(Aquagenx CBT). TankLight covers the days between those tests.

**What if the device is wrong and someone gets sick?**
It's designed to fail safe: any sensor fault or crash turns it grey, never green. A real
deployment would be run with the health board, with their thresholds and their advice on screen.

**Who decides the thresholds?**
The health board and water office, not us. We used the regulated 0.2 mg/L minimum, and warn 24
hours before protection runs out.

**What if the water is protected but tastes bad?**
Taste and smell aren't what it measures. A cloudiness sensor catches sediment; anything else is a
reason to call the water office.

## The technology

**How do you measure chlorine cheaply?**
An ORP and pH probe estimates free chlorine, calibrated against a DPD test kit, the one water
operators already use. For a pilot we'd use a reagent-free amperometric chlorine sensor, which
works better in still water.

**How accurate is it?**
We haven't measured that; it's a simulation. Accuracy depends on calibration, so the design fits
the curve to real DPD readings in each home.

**How does it know how long protection will last?**
Chlorine fades at a rate that depends on temperature. The device fits that rate from its own
readings since the last delivery and forecasts the hours until it drops below 0.2 mg/L.

**How does it read the tank level?**
An ultrasonic sensor on the tank lid measures the distance to the water, so nothing touches the
drinking water. At installation we record "empty", "full" and the tank shape. Rises over 10% are
deliveries; the falls in between are the family's own daily use.

**How does it know when the truck will come?**
It learns this home's own delivery gaps in the same month, and gives the chance the water runs
out before the next truck.

**Does it need the internet?**
No. Everything runs on the device. The water office plan runs on one office laptop, also offline.

**What about power cuts?**
Mains power with a battery, because blizzards are when both the power and the trucks stop.

**What hardware does it run on?**
The rules are written to run on a small microcontroller (an ESP32 running MicroPython). They're
written for it but haven't been run on one yet.

**What's graphflow?**
Our own workflow framework. The household checks run as one graphflow workflow: validate the
reading, then check water quality and supply in parallel, then combine. If one branch crashes,
the result is grey, not green.

**How did you test it?**
64 automated tests for TankLight, a simulated town of 12 homes with blizzards and two seasons, and a
backtest comparing truck strategies. Everything in the demo is simulated.

## The sewage tank

**Why does the sewage tank need pumping out?**
There are no sewer pipes. Everything down the sink, shower or toilet collects in a tank outside,
and a truck empties it. A family of 5 fills an 1,100 L tank in about 4 days; a full tank backs up
into the house.

**Why is the sewage status separate from the light?**
A full sewage tank is urgent, but it doesn't make the drinking water unsafe. Mixing them would
turn the water light red for the wrong reason.

**What does it warn about?**
Full within a day and a half, full now, and near freezing when the tank heater fails in winter,
before the tank can't be pumped.

## Trucks and operations

**How does it make the trucks more efficient?**
Instead of a fixed loop, trucks go to the homes with the least water first and skip homes with 3
or more days left. Same trucks, same stops.

**How do you know that's better?**
A 30-day backtest on the same simulated town and storms: hours homes spent without water went from
1,486 to 1,282 in January and 534 to 195 in July, with fewer run-outs. Sewage backups fell 1–45%.
It's simulated; real delivery logs would confirm it.

**Why doesn't a top-up fix low chlorine?**
A top-up of a mostly full tank is mostly old water, so the chlorine stays low. The plan asks the
water office to check that tank instead of wasting a truck stop.

**What happens in a blizzard?**
Trucks stop. The light warns a day and a half before the water runs out, and the office plan says
which homes to top up first before the storm, with the stops available.

**Do drivers need a new app?**
No. The plan is a short list; it can be printed or sent as a text. It works on any phone.

**Does it route the trucks?**
Not yet. It decides who needs a stop; ordering stops to cut driving is a next step.

## The AI

**Where's the AI?**
In the water office, not the kitchen. TimesFM, Google's forecasting model, runs on a laptop
offline and forecasts next week's urgent-water demand, with the blizzard forecast as an input. A
tiny language model writes the plan as a note. The safety light stays plain rules, on purpose.

**Is the AI actually better than a simple guess?**
Sometimes, and we check every time. Each run, TimesFM is scored against "the average of the last 7
days" on the past 4 weeks, and the plan uses whichever did better. In our simulated January it won
(1.21 vs 1.38 homes a day off); in steadier July it didn't, so the plan used the simple guess.

**What if the language model makes something up?**
The code decides every fact. The model only rewords, and a check keeps its note only if every home,
number and action matches line by line; otherwise the plain version goes out.

**Why such small models?**
They run offline on one laptop, and the job is small: forecast one number a day, reword a few
lines. The check means a small model can't do harm.

**Can it run on a phone?**
The small language model could; TimesFM would need converting. But phones don't need either: they
just receive the plan.

**What happens when a family texts the water office?**
A small local classifier (Laya) sorts the message: water truck, sewage truck, bad water, device
problem, or a person reads it. It answers one yes/no question per need with a calibrated
probability; anything unclear goes to a person, and it never closes a message on its own. A
request moves that home onto the truck plan, even if the sensor says the tank is full.

**How accurate is the message sorting?**
On 10 messages it had never seen: 6 sorted right, 4 left for a person, none sent to the wrong
place. Being unsure costs time, not safety. A bad-water report always counts as urgent.

**Why a classifier and not a chatbot?**
It can't write anything, so it can't invent an answer. It only picks from fixed choices with a
probability, in about a tenth of a second, offline, and reads French and other languages.

**Is the AI free to use?**
TimesFM's weights are under a non-commercial licence: fine for this demo, but a real deployment
needs Google's permission or another model.

## Cost and funding

**What does it cost?**
Prototype parts about $200–250 CAD; the target is under $100 at volume. Those are estimates, not
quotes.

**Who pays?**
Likely the municipality or the regional water program, as part of the water service: fewer wasted
truck stops and fewer emergencies offset the cost. We'd need to confirm that with them.

**Is it cheaper than bottled water?**
One case of 24 bottles is $76.29. A device that helps a family trust their tap for a year costs
about as much as a few cases.

## Maintenance and deployment

**Who maintains it?**
Local "water keepers" do a monthly check and calibration with the DPD kit. It uses standard parts
and a swappable probe cartridge.

**What about spare parts?**
Spares go north on the fall sealift, so one year's supply is planned in advance.

**How long does a probe last?**
Probes need replacing; the exact life depends on the sensor we pick for the pilot. The monthly
check catches drift, and a faulty probe turns the light grey.

**How would you roll it out?**
A pilot in a few homes with the municipality and the health board, with real DPD readings and
delivery logs, before any wider use.

**Could it work in other communities?**
Yes, anywhere with trucked water and household tanks, which includes most of Nunavik and Nunavut.
Each community would set its own thresholds.

## People and community

**Did you talk to residents?**
Answer honestly with what you have. If not yet: our challenge came from Amenda Soucy in Inukjuak,
and a pilot starts with families choosing the words, the placement and the thresholds with us.

**Where does the device go in the house?**
The sensor at the indoor tank; the display on the wall beside the kitchen tap, where people decide
whether to drink. We'd confirm the spot with each family.

**What about Inuktitut?**
The Inuktitut text and audio will be written and recorded with a fluent speaker from the community,
never machine-translated. Until then the option says so, instead of showing bad text.

**Who owns the data?**
The community. Readings stay in the home; community reports would be opt-in and combined.

**Could this be used against families, for example to blame them for using too much water?**
It's designed for the family and the water service, not for enforcement. What gets shared and with
whom is the community's decision.

## Limits and next steps

**What's the weakest part?**
Everything is simulated: chlorine decay, storms and water use are placeholder numbers. And there's
no working hardware yet.

**What would you do with more time?**
Build one real probe reading a jar, get real delivery logs and DPD readings from the municipality,
record the Inuktitut audio, and test the light with families.

**What did you learn?**
That the tank at home is the blind spot, that a top-up can't restore chlorine, and that a safety
device has to be honest when it isn't sure.

**What makes this different from what exists?**
Chlorine monitors exist in treatment plants, and SWOT sets chlorine targets for refugee camps, but
nothing watches the tank inside an Arctic home, for the family, offline, and links it to the
trucks.
