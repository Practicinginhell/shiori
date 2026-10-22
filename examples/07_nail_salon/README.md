# Nail salon AI receptionist (phases 0–3)

A virtual receptionist for a nail salon, built on graphflow. It answers questions
about services, prices, hours and policies, checks real availability, and books,
reschedules and cancels appointments.

The core rule: **the AI handles the conversation; deterministic code handles the
salon's operations.** The model never decides whether a time is free, what
something costs, or who the customer is.

## Try it

```bash
# Phase 2 on its own, no AI: the engine's bookable times
python examples/07_nail_salon/app.py slots gel_manicure

# The whole flow offline (a scripted model; the tools, engine and database are real)
python examples/07_nail_salon/app.py demo

# Talk to it with a real model
export OPENAI_API_KEY=...        # or OPENAI_BASE_URL=http://localhost:8000/v1 for a local server
python examples/07_nail_salon/app.py chat --model openai:gpt-4.1 -v
```

In `chat`, `-v` shows every tool call, and `--db salon.db` keeps bookings
between runs. `--phone` sets who the customer is: the channel supplies it, just
as WhatsApp will.

## What each phase delivers

| Phase | File | End condition (all automated in `tests/test_nail_salon.py`) |
|---|---|---|
| 0. Salon data | `salon.yaml`, `nail_salon/config.py` | The whole salon is structured data; inconsistent data (a service nobody can do, hours that close before they open, ...) is rejected when loading; service, price, duration and technician questions are answered from it |
| 1. Database | `nail_salon/db.py` | Customers, services, technicians and appointments are created, read, updated and cancelled without an LLM; a database trigger rejects overlapping appointments even if the engine had a bug |
| 2. Appointment engine | `nail_salon/scheduling.py` | Availability combines service duration, business hours, technician hours, skills, existing appointments (with cleanup time) and booking rules; the plan's example (Emily 09:00–17:00, 10:00–11:00 taken, gives 09:00, 11:00, …, 16:00) passes; eight simultaneous bookings of one slot produce exactly one appointment |
| 3. AI receptionist | `nail_salon/receptionist.py` | A new customer goes from conversation to service, availability, time, confirmation and booking, with no human involved |

## How the receptionist stays reliable

- **Facts come from tools.** `search_services`, `get_service_details`,
  `get_salon_info`, `check_availability` and `customer_lookup` read the database;
  the instructions tell the model never to state a fact it didn't get from them.
- **Bookings use slot ids, not times.** `check_availability` returns slot ids such
  as `gel_manicure@sarah@2026-09-26T14:00`. `create_booking` takes one, and the
  engine recalculates availability inside the database transaction. A time the
  engine didn't offer (or that someone else just took) can't be booked.
- **Identity comes from the channel.** The customer's phone number arrives with
  every message as graphflow's run context (`context={"customer_phone": ...}`).
  Tools receive it as an injected argument the model can't see or set, so it
  can't act for someone else.
- **The customer confirms every change.** Booking, rescheduling and cancelling
  pause the agent (graphflow's tool approval). The approval policy
  (`ChangeReview`) writes the summary the customer sees from the database, not
  from the model's wording, and the customer taps yes or no. A change the engine
  would refuse is declined without asking the customer, and every change is
  checked again when the customer answers, in case the slot was taken meanwhile.
  This lives in the agent, so any channel gets it.
- **Policies are enforced in code.** Cancelling within 24 hours is flagged as a
  late cancellation (and shown in the confirmation) by the engine, not the model.

## Adapting it to a real salon

Edit `salon.yaml`: services (price, duration, required skills), technicians
(skills, working hours), opening hours, closures, booking rules, policies and FAQs.
Which technicians can do a service is worked out from skills, so there is one
place to change. Loading fails with a clear message if the data doesn't add up.

## Not in this version

These are the plan's later phases: customer preferences and "same as last time"
(Phase 4), design recommendations (5), notifications and reminders (7), the owner
agent (8), and human handoff (9). WhatsApp comes after that, as a thin adapter
that calls `Conversation.say` / `Conversation.answer` and renders confirmations
as reply buttons.

`Conversation` is stateless: the conversation and any pending confirmation live
in the checkpointer (graphflow's `app.thread(...)`), so each webhook call can
build a new one. Passing the channel's message id (`say(text, message_id=...)`)
skips redelivered messages, and one customer's messages are handled one at a time.
For WhatsApp, use a persistent checkpointer and store, and route each customer to
one worker process (the per-conversation lock is per process).
