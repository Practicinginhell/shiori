"""Example 7: a nail salon AI receptionist (phases 0-3), built on graphflow.

    python examples/07_nail_salon/app.py slots gel_manicure 2026-09-26
        Phase 2 without any AI: the engine's bookable times for a service and date.

    python examples/07_nail_salon/app.py demo
        Offline, no API key: a scripted model plays the receptionist through a full
        booking (question -> availability -> booking -> confirmation -> cancellation),
        using the real tools and database.

    python examples/07_nail_salon/app.py chat --model openai:gpt-4.1
        Talk to the receptionist with a real model. Needs the provider's API key
        (e.g. OPENAI_API_KEY; OPENAI_BASE_URL for an OpenAI-compatible local server).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../..")))
sys.path.insert(0, HERE)

from nail_salon import BookingError, SalonDB, Scheduler, load_salon  # noqa: E402
from nail_salon.receptionist import Conversation, Reply, build_app  # noqa: E402

from graphflow import GraphflowError, MockModel, ModelResponse, ToolCall, ToolCalled  # noqa: E402

SALON_YAML = os.path.join(HERE, "salon.yaml")


def open_salon(db_path: str = ":memory:") -> Scheduler:
    """The salon's database (seeded from salon.yaml) and its scheduler."""
    return Scheduler(SalonDB.from_config(load_salon(SALON_YAML), path=db_path))


def next_weekday(scheduler: Scheduler, weekday: int) -> date:
    today = scheduler.now().date()
    return today + timedelta(days=(weekday - today.weekday() - 1) % 7 + 1)


def seed_existing_bookings(scheduler: Scheduler) -> None:
    """A few regulars already booked on the coming Saturday, so availability has gaps."""
    saturday = next_weekday(scheduler, 5)
    db = scheduler.db
    if db.get_customer(phone="+15550199"):
        return
    regular = db.create_customer(phone="+15550199", name="Grace")
    for service, tech, hhmm in [("gel_manicure", "emily", "13:00"), ("gel_manicure", "sarah", "12:00"),
                                ("acrylic_full_set", "mia", "12:00")]:
        slot = f"{service}@{tech}@{saturday.isoformat()}T{hhmm}"
        try:
            scheduler.book_appointment(regular.id, slot)
        except BookingError:  # seeding is best effort (e.g. that time already passed)
            pass


# --------------------------------------------------------------------------- #
# slots: the engine alone
# --------------------------------------------------------------------------- #

def show_slots(service: str, day: str) -> None:
    scheduler = open_salon()
    seed_existing_bookings(scheduler)
    names = {t.id: t.name for t in scheduler.db.list_technicians()}
    slots = scheduler.check_availability(service, day)
    print(f"{scheduler.find_service(service).name} on {day}: {len(slots)} bookable slots")
    for s in slots:
        print(f"  {s.start:%H:%M}-{s.end:%H:%M}  {names[s.technician_id]:<6}  {s.slot_id}")


# --------------------------------------------------------------------------- #
# demo: a scripted model, real tools
# --------------------------------------------------------------------------- #

def scripted_receptionist(saturday: str):
    """Plays the receptionist deterministically; every fact it states comes from a tool result."""

    def call(name: str, **args):
        return ModelResponse(tool_calls=[ToolCall(id=f"{name}-{len(args)}", name=name, args=args)])

    def policy(messages):
        last = messages[-1]
        if last["role"] == "tool":
            content = last["content"]
            if last["name"] == "get_service_details":
                name, price, minutes = re.match(r"(.+?) \(id \w+\): (\$\d+), (\d+) min", content).groups()
                return f"A {name} is {price} and takes about {minutes} minutes. Would you like to book one?"
            if last["name"] == "customer_lookup" and "New customer" in content:
                return call("check_availability", service="gel manicure", date=saturday, after="12:00")
            if last["name"] == "customer_lookup":
                appointment_id = int(re.search(r"Upcoming appointment (\d+)", content).group(1))
                return call("cancel_booking", appointment_id=appointment_id)
            if last["name"] == "check_availability":
                times = re.findall(r"- (\d\d:\d\d) with (\w+)", content)[:3]
                listed = ", ".join(f"{t} with {n}" for t, n in times)
                return f"On Saturday afternoon I have {listed}. Which would you like? And may I have your name?"
            if "rejected" in content:
                return "No problem, nothing was changed. What would you like instead?"
            return content.replace("Booked appointment", "You're booked! Appointment").replace(
                "Cancelled appointment", "Done, I've cancelled appointment")
        text = last["content"].lower()
        if "how much" in text:
            return call("get_service_details", service="gel manicure")
        if "saturday" in text or "cancel" in text:
            return call("customer_lookup")
        if "i'm" in text:
            # The time and technician the customer picked, matched against what the engine offered.
            offered = [m for m in messages if m.get("name") == "check_availability"][-1]["content"]
            hour, minute, meridiem = re.search(r"(\d{1,2})(?::(\d\d))?\s*(am|pm)?", text).groups()
            hhmm = f"{int(hour) + (12 if meridiem == 'pm' and int(hour) < 12 else 0):02d}:{minute or '00'}"
            for line in re.findall(rf"- {hhmm} with (\w+) \(slot_id: (\S+)\)", offered):
                if line[0].lower() in text:
                    name = re.search(r"i'm (\w+)", text).group(1).title()
                    return call("create_booking", slot_id=line[1], customer_name=name)
            return "Sorry, that time isn't one I have. Which of the times I listed works?"
        return "Hi! I can help with our services, prices and bookings."

    return policy


def print_tool_calls(event) -> None:
    if isinstance(event, ToolCalled):
        args = ", ".join(f"{k}={v!r}" for k, v in event.tool_args.items())
        print(f"    · {event.tool_name}({args})")


def show(reply: Reply) -> None:
    if reply.confirmation:
        print("  Receptionist: Please confirm:")
        for line in reply.confirmation.summary.splitlines():
            print(f"    ┃ {line}")
        print("    ┃ [ Yes ]  [ No ]")
    else:
        print(f"  Receptionist: {reply.text}")


def demo() -> None:
    scheduler = open_salon()
    seed_existing_bookings(scheduler)
    saturday = next_weekday(scheduler, 5).isoformat()
    app = build_app(scheduler, MockModel(scripted_receptionist(saturday)), listeners=[print_tool_calls])
    chat = Conversation(app, scheduler, phone="+15550123")
    print("Scripted model (no API key); the tools, scheduling engine and database are real.\n")

    for customer_says, confirm in [
        ("Hi! How much is a gel manicure?", None),
        ("Great. Can I come this Saturday afternoon?", None),
        ("2 PM with Sarah, please. I'm Anna", True),
        ("Sorry, something came up. Can you cancel it?", True),
    ]:
        print(f"  Customer: {customer_says}")
        reply = chat.say(customer_says)
        show(reply)
        if confirm is not None and reply.confirmation:
            print(f"  Customer: [{'Yes' if confirm else 'No'}]")
            show(chat.answer(confirm))
        print()

    customer = scheduler.db.get_customer(phone="+15550123")
    print("Database afterwards:")
    for a in scheduler.db.get_customer_appointments(customer.id):
        print(f"  appointment {a.id}: {customer.name}, {a.service_id} with {a.technician_id}, {a.start:%a %Y-%m-%d %H:%M}, {a.status}")


# --------------------------------------------------------------------------- #
# chat: a real model
# --------------------------------------------------------------------------- #

def chat(model: str, phone: str, db_path: str, verbose: bool) -> None:
    from graphflow.models.registry import resolve_model

    scheduler = open_salon(db_path)
    seed_existing_bookings(scheduler)
    app = build_app(scheduler, resolve_model(model), listeners=[print_tool_calls] if verbose else None)
    conversation = Conversation(app, scheduler, phone=phone)
    print(f"{scheduler.salon.name} receptionist ({model}), chatting as {phone}. "
          f"Salon time: {scheduler.now():%A %Y-%m-%d %H:%M}. Ctrl-D to quit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        try:
            reply = conversation.say(text)
            while reply.confirmation:
                show(reply)
                answer = input("Confirm? [y/n, or type a message]: ").strip()
                if answer.lower() in ("y", "yes"):
                    reply = conversation.answer(True)
                elif answer.lower() in ("n", "no"):
                    reply = conversation.answer(False)
                else:
                    reply = conversation.say(answer)
        except GraphflowError as e:  # e.g. missing API key, model unreachable
            print(f"  (the receptionist could not answer: {e.__cause__ or e})")
            continue
        show(reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("slots", help="bookable times for a service and date, without AI")
    s.add_argument("service")
    s.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: the coming Saturday)")
    sub.add_parser("demo", help="offline scripted conversation")
    c = sub.add_parser("chat", help="talk to the receptionist with a real model")
    c.add_argument("--model", required=True, help='e.g. "openai:gpt-4.1" or "anthropic:claude-sonnet-5"')
    c.add_argument("--phone", default="+15550123", help="the customer's phone number (their identity)")
    c.add_argument("--db", default=":memory:", help="SQLite file to keep bookings between runs")
    c.add_argument("-v", "--verbose", action="store_true", help="show tool calls")
    args = parser.parse_args()
    if args.command == "slots":
        show_slots(args.service, args.date or next_weekday(open_salon(), 5).isoformat())
    elif args.command == "demo":
        demo()
    else:
        chat(args.model, args.phone, args.db, args.verbose)
