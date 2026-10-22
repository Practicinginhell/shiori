"""Nail salon agent example (examples/07_nail_salon), phases 0-3.

Each phase's end condition from the project plan is a test here:
  0  the salon is structured data; questions are answered from it
  1  core data is created, read, updated and cancelled without an LLM
  2  availability is computed deterministically; double booking is impossible
  3  a new customer goes from conversation to a confirmed booking, no human involved
"""

import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "07_nail_salon"
sys.path.insert(0, str(EXAMPLE))

from nail_salon import (  # noqa: E402
    BookingError,
    DoubleBookingError,
    SalonConfigError,
    SalonDB,
    Scheduler,
    load_salon,
)
from nail_salon.receptionist import Conversation, build_app, make_tools  # noqa: E402

from graphflow import MockModel, ModelResponse, ToolCall  # noqa: E402

NOW = datetime(2026, 9, 24, 8, 0)  # Thursday, before opening
SATURDAY = "2026-09-26"


@pytest.fixture
def config():
    return load_salon(EXAMPLE / "salon.yaml")


@pytest.fixture
def db(config):
    return SalonDB.from_config(config)


@pytest.fixture
def scheduler(db):
    return Scheduler(db, clock=lambda: NOW)


def tools_by_name(scheduler):
    return {t.name: t for t in make_tools(scheduler)}


def starts(slots):
    return [s.start.strftime("%H:%M") for s in slots]


def single_tech_salon(**rules):
    """The plan's Phase 2 example: Emily works 09:00-17:00, gel manicure is 60 minutes."""
    return load_salon({
        "salon": {"name": "Test", "timezone": "UTC"},
        "business_hours": {"saturday": "08:00-20:00"},
        "booking_rules": {"slot_interval_minutes": 60, "buffer_minutes": 0, "min_notice_minutes": 0, **rules},
        "services": [{"id": "gel_manicure", "name": "Gel Manicure", "price": 45, "duration_minutes": 60,
                      "required_skills": ["gel"]}],
        "technicians": [{"id": "emily", "name": "Emily", "skills": ["gel"],
                         "working_hours": {"saturday": "09:00-17:00"}}],
    })


# =========================================================================== #
# Phase 0: the salon as structured data
# =========================================================================== #

def test_salon_yaml_loads_completely(config):
    assert config.name == "Luxe Nails Studio"
    assert {s.id for s in config.services} >= {"gel_manicure", "acrylic_full_set", "spa_pedicure", "nail_art"}
    assert {t.id for t in config.technicians} == {"emily", "sarah", "mia"}
    assert {p.id for p in config.policies} == {"cancellation", "late_arrival", "no_show", "booking"}
    assert config.faqs and config.rules.cancellation_notice_hours == 24


def test_qualified_technicians_come_from_skills(config):
    assert [t.name for t in config.qualified_technicians("nail_art")] == ["Emily", "Mia"]
    assert [t.name for t in config.qualified_technicians("spa_pedicure")] == ["Sarah", "Mia"]


def test_data_questions_are_answered_from_the_data(scheduler):
    tools = tools_by_name(scheduler)
    offered = tools["search_services"].execute(query="")
    assert "Gel Manicure" in offered and "Spa Pedicure" in offered       # What services do you offer?
    assert "$45" in tools["get_service_details"].execute(service="gel manicure")  # How much is a gel manicure?
    assert "90 min" in tools["get_service_details"].execute(service="acrylic")    # How long does acrylic take?
    nail_art = tools["get_service_details"].execute(service="nail art")  # Which technicians do nail art?
    assert "Emily" in nail_art and "Mia" in nail_art and "Sarah" not in nail_art
    policies = tools["get_salon_info"].execute(topic="policies")
    assert "24 hours" in policies
    assert "Free parking" in tools["get_salon_info"].execute(topic="faq")


def test_unknown_service_is_reported_not_invented(scheduler):
    out = tools_by_name(scheduler)["get_service_details"].execute(service="eyelash extensions")
    assert "no service" in out.lower()


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["services"][0].update(required_skills=["laser"]), "no technician"),
    (lambda d: d["technicians"][0]["working_hours"].update(saturday="17:00-09:00"), "hours"),
    (lambda d: d["business_hours"].update(funday="09:00-17:00"), "weekday"),
    (lambda d: d["services"].append(dict(d["services"][0])), "duplicate"),
    (lambda d: d["services"][0].update(duration_minutes=0), "duration"),
])
def test_inconsistent_salon_data_is_rejected(mutate, message):
    import yaml

    data = yaml.safe_load((EXAMPLE / "salon.yaml").read_text())
    mutate(data)
    with pytest.raises(SalonConfigError, match=message):
        load_salon(data)


# =========================================================================== #
# Phase 1: database operations, no LLM
# =========================================================================== #

def test_customer_crud(db):
    created = db.create_customer(phone="+15550001", name="Sarah")
    assert db.get_customer(created.id) == created
    assert db.get_customer(phone="+15550001") == created
    assert db.get_customer(phone="+19999999") is None
    with pytest.raises(BookingError, match="already"):
        db.create_customer(phone="+15550001", name="Other")


def test_service_and_technician_crud(db, config):
    from nail_salon.config import Service, Technician

    db.create_service(Service(id="dip", name="Dip Powder", description="", price_cents=5000,
                              duration_minutes=60, required_skills=("dip",)))
    assert db.get_service("dip").price_cents == 5000
    db.create_technician(Technician(id="zoe", name="Zoe", skills=("dip",), working_hours={0: ("09:00", "17:00")}))
    assert db.get_technician("zoe").skills == ("dip",)
    assert db.get_service("gel_manicure").name == "Gel Manicure"  # seeded from salon.yaml
    assert len(db.list_services()) == len(config.services) + 1


def test_appointment_crud(db):
    customer = db.create_customer(phone="+15550001", name="Sarah")
    appt = db.create_appointment(customer.id, "gel_manicure", "emily",
                                 datetime(2026, 9, 26, 10), datetime(2026, 9, 26, 11), price_cents=4500)
    assert db.get_appointment(appt.id) == appt and appt.status == "booked"
    moved = db.update_appointment(appt.id, start=datetime(2026, 9, 26, 12), end=datetime(2026, 9, 26, 13))
    assert moved.start == datetime(2026, 9, 26, 12)
    assert [a.id for a in db.get_customer_appointments(customer.id)] == [appt.id]
    cancelled = db.cancel_appointment(appt.id)
    assert cancelled.status == "cancelled"


def test_database_rejects_overlapping_appointments_even_without_the_engine(db):
    customer = db.create_customer(phone="+15550001", name="Sarah")
    db.create_appointment(customer.id, "gel_manicure", "emily",
                          datetime(2026, 9, 26, 10), datetime(2026, 9, 26, 11), price_cents=4500)
    with pytest.raises(DoubleBookingError):
        db.create_appointment(customer.id, "gel_manicure", "emily",
                              datetime(2026, 9, 26, 10, 30), datetime(2026, 9, 26, 11, 30), price_cents=4500)
    # another technician at the same time is fine
    db.create_appointment(customer.id, "gel_manicure", "sarah",
                          datetime(2026, 9, 26, 10), datetime(2026, 9, 26, 11), price_cents=4500)


# =========================================================================== #
# Phase 2: scheduling engine, no LLM
# =========================================================================== #

def test_plan_example_availability():
    db = SalonDB.from_config(single_tech_salon())
    scheduler = Scheduler(db, clock=lambda: NOW)
    customer = db.create_customer(phone="+1", name="A")
    db.create_appointment(customer.id, "gel_manicure", "emily",
                          datetime(2026, 9, 26, 10), datetime(2026, 9, 26, 11), price_cents=4500)
    slots = scheduler.check_availability(service="gel_manicure", date=SATURDAY)
    assert starts(slots) == ["09:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]


def test_only_qualified_technicians_are_offered(scheduler):
    monday = scheduler.check_availability(service="spa_pedicure", date="2026-09-28")
    assert {s.technician_id for s in monday} == {"mia"}  # Emily works Monday but doesn't do pedicures


def test_business_hours_and_technician_hours_intersect(scheduler):
    friday = scheduler.check_availability(service="gel_manicure", date="2026-09-25", technician="emily")
    assert starts(friday)[0] == "11:00"  # salon opens 09:00, Emily starts 11:00
    assert friday[-1].end <= datetime(2026, 9, 25, 20, 0)


def test_closed_days_have_no_slots(scheduler):
    assert scheduler.check_availability(service="gel_manicure", date="2026-09-27") == []  # Sunday
    december = Scheduler(scheduler.db, clock=lambda: datetime(2026, 12, 20, 8, 0))
    assert december.check_availability(service="gel_manicure", date="2026-12-25") == []  # closure (a Friday)
    assert december.check_availability(service="gel_manicure", date="2026-12-24")      # the day before is open


def test_min_notice_and_booking_window(db):
    thursday_10am = Scheduler(db, clock=lambda: datetime(2026, 9, 24, 10, 0))
    assert starts(thursday_10am.check_availability(service="gel_manicure", date="2026-09-24"))[0] == "11:00"
    with pytest.raises(BookingError, match="past"):
        thursday_10am.check_availability(service="gel_manicure", date="2026-09-23")
    with pytest.raises(BookingError, match="60 days"):
        thursday_10am.check_availability(service="gel_manicure", date="2026-12-31")


def test_buffer_between_appointments(scheduler, db):
    customer = db.create_customer(phone="+1", name="A")
    db.create_appointment(customer.id, "gel_manicure", "emily",
                          datetime(2026, 9, 26, 10), datetime(2026, 9, 26, 11), price_cents=4500)
    emily = starts(scheduler.check_availability(service="gel_manicure", date=SATURDAY, technician="emily"))
    assert "09:00" not in emily  # would end at 10:00 with no cleanup time
    assert "11:00" not in emily and "11:30" in emily


def test_time_of_day_filters(scheduler):
    afternoon = scheduler.check_availability(service="gel_manicure", date=SATURDAY, after="12:00", before="15:00")
    assert afternoon and all("12:00" <= s.start.strftime("%H:%M") < "15:00" for s in afternoon)


def test_service_lookup_by_name(scheduler):
    assert scheduler.check_availability(service="Gel Manicure", date=SATURDAY)
    with pytest.raises(BookingError, match="no service"):
        scheduler.check_availability(service="eyelash lift", date=SATURDAY)


def test_booking_removes_the_slot_and_cannot_be_repeated(scheduler, db):
    customer = db.create_customer(phone="+1", name="A")
    slot = scheduler.check_availability(service="gel_manicure", date=SATURDAY, technician="emily")[0]
    appt = scheduler.book_appointment(customer.id, slot.slot_id)
    assert (appt.start, appt.technician_id, appt.price_cents) == (slot.start, "emily", 4500)
    assert slot.slot_id not in {s.slot_id for s in scheduler.check_availability(service="gel_manicure", date=SATURDAY)}
    with pytest.raises(BookingError, match="no longer available"):
        scheduler.book_appointment(customer.id, slot.slot_id)


@pytest.mark.parametrize("slot_id, message", [
    ("gel_manicure@emily@2026-09-26T14:07", "not available"),    # off the slot grid
    ("spa_pedicure@emily@2026-09-26T14:00", "not qualified"),    # Emily doesn't do pedicures
    ("gel_manicure@emily@2026-09-27T14:00", "not available"),    # Sunday
    ("facial@emily@2026-09-26T14:00", "no service"),
    ("gel_manicure@nobody@2026-09-26T14:00", "no technician"),
    ("tomorrow at 2", "slot id"),
])
def test_invented_slots_are_rejected(scheduler, db, slot_id, message):
    customer = db.create_customer(phone="+1", name="A")
    with pytest.raises(BookingError, match=message):
        scheduler.book_appointment(customer.id, slot_id)


def test_concurrent_bookings_of_one_slot_produce_one_appointment(scheduler, db):
    slot = scheduler.check_availability(service="gel_manicure", date=SATURDAY, technician="emily")[0]
    customers = [db.create_customer(phone=f"+1{i}", name=f"C{i}") for i in range(8)]
    results = []

    def attempt(customer_id):
        try:
            results.append(scheduler.book_appointment(customer_id, slot.slot_id))
        except BookingError:
            results.append(None)

    threads = [threading.Thread(target=attempt, args=(c.id,)) for c in customers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r is not None for r in results) == 1


def test_reschedule_moves_the_appointment(scheduler, db):
    customer = db.create_customer(phone="+1", name="A")
    first = scheduler.check_availability(service="gel_manicure", date=SATURDAY, technician="emily")[0]
    appt = scheduler.book_appointment(customer.id, first.slot_id)
    overlapping_own_time = f"gel_manicure@emily@{SATURDAY}T{(first.start.replace(minute=30)).strftime('%H:%M')}"
    moved = scheduler.reschedule_appointment(appt.id, overlapping_own_time, customer_id=customer.id)
    assert moved.id == appt.id and moved.start.minute == 30  # may overlap its own old time
    assert [a.id for a in db.get_customer_appointments(customer.id)] == [appt.id]


def test_reschedule_can_change_service_and_price(scheduler, db):
    customer = db.create_customer(phone="+1", name="A")
    slot = scheduler.check_availability(service="gel_manicure", date=SATURDAY, technician="emily")[0]
    appt = scheduler.book_appointment(customer.id, slot.slot_id)
    new = scheduler.check_availability(service="acrylic_full_set", date=SATURDAY, technician="mia")[0]
    moved = scheduler.reschedule_appointment(appt.id, new.slot_id, customer_id=customer.id)
    assert (moved.service_id, moved.technician_id, moved.price_cents) == ("acrylic_full_set", "mia", 6500)


def test_cancel_frees_the_slot_and_applies_the_policy(db):
    early = Scheduler(db, clock=lambda: NOW)
    customer = db.create_customer(phone="+1", name="A")
    slot = early.check_availability(service="gel_manicure", date=SATURDAY, technician="emily")[0]
    appt = early.book_appointment(customer.id, slot.slot_id)
    result = early.cancel_appointment(appt.id, customer_id=customer.id)
    assert result.appointment.status == "cancelled" and not result.late
    assert slot.slot_id in {s.slot_id for s in early.check_availability(service="gel_manicure", date=SATURDAY)}
    with pytest.raises(BookingError, match="already cancelled"):
        early.cancel_appointment(appt.id, customer_id=customer.id)

    appt2 = early.book_appointment(customer.id, slot.slot_id)
    late = Scheduler(db, clock=lambda: datetime(2026, 9, 25, 12, 0))  # less than 24h before
    assert late.cancel_appointment(appt2.id, customer_id=customer.id).late


def test_customers_cannot_touch_each_others_appointments(scheduler, db):
    alice = db.create_customer(phone="+1", name="Alice")
    bob = db.create_customer(phone="+2", name="Bob")
    slot = scheduler.check_availability(service="gel_manicure", date=SATURDAY)[0]
    appt = scheduler.book_appointment(alice.id, slot.slot_id)
    with pytest.raises(BookingError, match="no appointment"):
        scheduler.cancel_appointment(appt.id, customer_id=bob.id)
    later = scheduler.check_availability(service="gel_manicure", date=SATURDAY)[-1]
    with pytest.raises(BookingError, match="no appointment"):
        scheduler.reschedule_appointment(appt.id, later.slot_id, customer_id=bob.id)
    assert db.get_appointment(appt.id).status == "booked"


def test_next_available_date_skips_closed_days(scheduler):
    assert scheduler.next_available_date("gel_manicure", "2026-09-27") == "2026-09-28"  # Sunday -> Monday


# =========================================================================== #
# Phase 3: the AI receptionist
# =========================================================================== #

def slot_ids(text):
    return re.findall(r"slot_id: ([\w@:\-]+)", text)


def scripted_receptionist(messages):
    """A stand-in model: calls the salon tools and answers only from their results."""
    last = messages[-1]
    if last["role"] == "tool":
        content = last["content"]
        if last["name"] == "check_availability":
            ids = slot_ids(content)
            times = re.findall(r"- (\d\d:\d\d) with (\w+)", content)
            return "I have " + " and ".join(f"{t} with {n}" for t, n in times[:2]) + ". Which works?" if ids else content
        if last["name"] == "create_booking":
            return "Sorry, I couldn't book that." if content.startswith("Tool call") else f"All set! {content}"
        return content
    text = last["content"].lower()
    if "saturday afternoon" in text:
        return ModelResponse(tool_calls=[ToolCall("c1", "check_availability", {
            "service": "gel manicure", "date": SATURDAY, "after": "12:00"})])
    if "invent" in text:
        return ModelResponse(tool_calls=[ToolCall("c3", "create_booking", {
            "slot_id": f"gel_manicure@emily@{SATURDAY}T12:07", "customer_name": "Anna"})])
    if "cancel appointment" in text:
        appointment_id = int(re.findall(r"\d+", text)[-1])
        return ModelResponse(tool_calls=[ToolCall("c4", "cancel_booking", {"appointment_id": appointment_id})])
    if "first one" in text:
        offered = [m for m in messages if m.get("name") == "check_availability"][-1]["content"]
        return ModelResponse(tool_calls=[ToolCall("c2", "create_booking", {
            "slot_id": slot_ids(offered)[0], "customer_name": "Anna"})])
    return "How can I help?"


def conversation(scheduler, phone="+15550100"):
    app = build_app(scheduler, MockModel(scripted_receptionist))
    return Conversation(app, scheduler, phone=phone)


def test_new_customer_books_end_to_end(scheduler, db):
    chat = conversation(scheduler)
    offer = chat.say("Hi, I'd like a gel manicure this Saturday afternoon")
    assert offer.confirmation is None and "Which works?" in offer.text
    first_offered = re.search(r"(\d\d:\d\d) with (\w+)", offer.text).groups()
    real = scheduler.check_availability(service="gel_manicure", date=SATURDAY, after="12:00")
    assert (real[0].start.strftime("%H:%M"), db.get_technician(real[0].technician_id).name) == first_offered

    pending = chat.say("The first one please, I'm Anna")
    assert pending.confirmation is not None
    summary = pending.confirmation.summary
    assert "Gel Manicure" in summary and "Saturday, September 26" in summary and "$45" in summary
    assert db.get_customer(phone="+15550100") is None  # nothing is written before the customer says yes

    done = chat.answer(True)
    assert "All set!" in done.text
    customer = db.get_customer(phone="+15550100")
    [appt] = db.get_customer_appointments(customer.id)
    assert (customer.name, appt.start, appt.status) == ("Anna", real[0].start, "booked")


def test_declining_the_confirmation_books_nothing(scheduler, db):
    chat = conversation(scheduler)
    chat.say("Hi, I'd like a gel manicure this Saturday afternoon")
    chat.say("The first one please, I'm Anna")
    chat.answer(False, "actually I need to check my calendar")
    assert db.get_customer(phone="+15550100") is None


def test_a_slot_the_engine_never_offered_is_refused_without_asking_the_customer(scheduler, db):
    model = MockModel(scripted_receptionist)
    chat = Conversation(build_app(scheduler, model), scheduler, phone="+15550100")
    reply = chat.say("please invent a time")
    assert reply.confirmation is None
    assert db.get_customer(phone="+15550100") is None
    tool_messages = [m for m in model.history[-1] if m["role"] == "tool"]
    assert "not available" in tool_messages[-1]["content"]


def test_a_customer_cannot_cancel_someone_elses_appointment(scheduler, db):
    alice = db.create_customer(phone="+1", name="Alice")
    slot = scheduler.check_availability(service="gel_manicure", date=SATURDAY)[0]
    appt = scheduler.book_appointment(alice.id, slot.slot_id)
    bob = conversation(scheduler, phone="+2")
    reply = bob.say(f"cancel appointment {appt.id}")
    assert reply.confirmation is None
    assert db.get_appointment(appt.id).status == "booked"


def test_the_model_is_told_the_salon_date(scheduler):
    model = MockModel("Hello!")
    Conversation(build_app(scheduler, model), scheduler, phone="+1").say("hi")
    system_text = " ".join(m["content"] for m in model.history[0] if m["role"] == "system")
    assert "Thursday 2026-09-24" in system_text and "Saturday 2026-09-26" in system_text


def test_booking_tools_need_a_customer_from_the_channel(scheduler):
    tools = tools_by_name(scheduler)
    assert "phone" not in tools["create_booking"].args_schema.model_fields  # the model never sees it
    slot = scheduler.check_availability(service="gel_manicure", date=SATURDAY)[0]
    with pytest.raises(Exception, match="phone"):
        tools["create_booking"].execute(slot_id=slot.slot_id, customer_name="Anna")
    assert "Booked" in tools["create_booking"].execute(slot_id=slot.slot_id, customer_name="Anna", phone="+15550100")


def test_conversation_memory_across_turns(scheduler):
    model = MockModel(["Hi!", "Sure."])
    chat = Conversation(build_app(scheduler, model), scheduler, phone="+1")
    chat.say("first message")
    chat.say("second message")
    contents = [m["content"] for m in model.history[1]]
    assert "first message" in contents and "Hi!" in contents and "second message" in contents


@pytest.mark.skipif(not os.environ.get("NAIL_SALON_LIVE_MODEL"), reason="set NAIL_SALON_LIVE_MODEL to test a real model")
def test_live_model_books_end_to_end(scheduler, db):
    from graphflow.models.registry import resolve_model

    chat = Conversation(build_app(scheduler, resolve_model(os.environ["NAIL_SALON_LIVE_MODEL"])), scheduler, phone="+1")
    reply = chat.say("Hi, I'm Anna. I'd like a gel manicure this Saturday at 2pm with anyone.")
    for _ in range(3):
        if reply.confirmation:
            break
        reply = chat.say("Yes, book 2pm please.")
    assert reply.confirmation is not None, reply.text
    chat.answer(True)
    assert db.get_customer_appointments(db.get_customer(phone="+1").id)


def test_badly_formatted_times_get_a_clear_message(scheduler):
    with pytest.raises(BookingError, match="HH:MM"):
        scheduler.check_availability(service="gel_manicure", date=SATURDAY, after="afternoon")
    out = tools_by_name(scheduler)["check_availability"].execute(service="gel manicure", date="next saturday")
    assert "YYYY-MM-DD" in out


def test_reopening_a_database_applies_changes_to_salon_yaml(tmp_path, db):
    import yaml

    path = str(tmp_path / "salon.db")
    data = yaml.safe_load((EXAMPLE / "salon.yaml").read_text())
    first = SalonDB.from_config(load_salon(data), path=path)
    customer = first.create_customer(phone="+1", name="A")
    first.create_appointment(customer.id, "gel_manicure", "emily",
                             datetime(2026, 9, 26, 10), datetime(2026, 9, 26, 11), price_cents=4500)

    data["services"][0]["price"] = 48                          # the salon raises the gel manicure price
    data["technicians"][1]["skills"].append("nail_art")        # Sarah learns nail art
    reopened = SalonDB.from_config(load_salon(data), path=path)
    assert reopened.get_service("gel_manicure").price_cents == 4800
    assert "nail_art" in reopened.get_technician("sarah").skills
    assert len(reopened.get_customer_appointments(customer.id)) == 1  # existing bookings are kept


def test_any_worker_can_answer_a_confirmation(scheduler, db):
    # Each webhook call builds a new Conversation; nothing is kept in memory between them.
    app = build_app(scheduler, MockModel(scripted_receptionist))
    Conversation(app, scheduler, phone="+15550100").say("Hi, I'd like a gel manicure this Saturday afternoon")
    pending = Conversation(app, scheduler, phone="+15550100").say("The first one please, I'm Anna")
    assert pending.confirmation is not None
    Conversation(app, scheduler, phone="+15550100").answer(True)
    assert db.get_customer_appointments(db.get_customer(phone="+15550100").id)


def test_a_slot_taken_before_the_customer_answers_is_refused(scheduler, db):
    app = build_app(scheduler, MockModel(scripted_receptionist))
    chat = Conversation(app, scheduler, phone="+15550100")
    chat.say("Hi, I'd like a gel manicure this Saturday afternoon")
    pending = chat.say("The first one please, I'm Anna")
    slot_id = pending.confirmation.actions[0]["args"]["slot_id"]
    someone_else = db.create_customer(phone="+19999", name="Zoe")
    scheduler.book_appointment(someone_else.id, slot_id)   # taken while Anna was deciding
    reply = chat.answer(True)
    assert db.get_customer(phone="+15550100") is None      # nothing booked for Anna
    assert reply.text == "Sorry, I couldn't book that."


def test_the_agent_itself_checks_changes_before_asking(scheduler, db):
    """The checks live in the agent's approval policy, so any channel gets them, not only Conversation."""
    model = MockModel(scripted_receptionist)
    thread = build_app(scheduler, model).thread("t", context={"customer_phone": "+15550100"})
    result = thread.send("please invent a time")
    assert not result.interrupted and thread.pending == []
    [refusal] = [m["content"] for m in model.history[-1] if m["role"] == "tool"]
    assert refusal.startswith("Tool call refused before running:") and "not available" in refusal
    assert "human" not in refusal and "customer" not in refusal


def test_the_confirmation_summary_comes_from_the_database(scheduler, db):
    app = build_app(scheduler, MockModel(scripted_receptionist))
    thread = app.thread("t", context={"customer_phone": "+15550100"})
    thread.send("Hi, I'd like a gel manicure this Saturday afternoon")
    thread.send("The first one please, I'm Anna")
    [action] = thread.pending[0].actions
    assert action["summary"].startswith("Book Gel Manicure with ") and "$45" in action["summary"]


def test_a_repeated_webhook_delivery_is_ignored(scheduler, db):
    model = MockModel(scripted_receptionist)
    chat = Conversation(build_app(scheduler, model), scheduler, phone="+15550100")
    chat.say("Hi, I'd like a gel manicure this Saturday afternoon", message_id="wamid.1")
    calls_before = model.call_count
    again = chat.say("Hi, I'd like a gel manicure this Saturday afternoon", message_id="wamid.1")
    assert again.duplicate and model.call_count == calls_before
