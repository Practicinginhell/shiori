"""Phase 3: the AI receptionist, on graphflow.

The model handles the conversation; tools backed by the scheduling engine do the
salon operations. Three rules keep critical operations deterministic:

* **Facts come from tools.** Services, prices, hours, policies and availability
  are only ever read from the salon's data.
* **Identity comes from the channel.** The customer's phone number arrives with
  every run as the graphflow run context (from WhatsApp, SMS, ...). Tools receive
  it as an injected argument (``FromContext("customer_phone")``) that the model
  can neither see nor set, so it cannot act for another customer.
* **The customer confirms every change.** ``create_booking``, ``reschedule_booking``
  and ``cancel_booking`` pause the agent (graphflow tool approval). The approval
  policy's ``describe`` writes the summary the customer sees from the database,
  not from the model's words; its ``check`` refuses a change the engine would
  refuse (a time it never offered, someone else's appointment) without asking the
  customer, and checks again when they answer, in case the slot was taken meanwhile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Annotated, Any

from graphflow import (
    Agent,
    AgentMiddleware,
    Application,
    Field,
    FromContext,
    HistoryLimit,
    InMemoryCheckpointStore,
    State,
    Tool,
    ToolDecision,
    ToolRequest,
    Workflow,
)
from nail_salon.db import Customer
from nail_salon.scheduling import BookingError, Scheduler, Slot

WRITE_TOOLS = ("create_booking", "reschedule_booking", "cancel_booking")
MAX_SLOTS_SHOWN = 12
MAX_PROMPT_MESSAGES = 40  # a chat thread never ends; the model sees its recent part

INSTRUCTIONS = """You are the receptionist of {salon}, a nail salon. You help customers \
with questions about the salon and with booking, rescheduling and cancelling appointments.

Rules:
- Take every fact (services, prices, durations, technicians, hours, policies, available \
times, the customer's appointments) from your tools. Never guess or invent one. If the \
tools don't have the answer, say you'll check with the staff.
- Only offer times that check_availability returned, and book with the exact slot_id it gave.
- Call customer_lookup at the start of a booking conversation. For a new customer, ask \
for their name before booking.
- create_booking, reschedule_booking and cancel_booking ask the customer to confirm \
before anything changes. Call them once you know exactly what the customer wants; if \
the customer declines, ask what they would like instead.
- Resolve dates like "tomorrow" or "Saturday" with the salon calendar you are given.
- Keep replies short and friendly, as in a chat app. Offer at most three times at once.
- Only help with this salon. Politely decline unrelated requests."""


class ReceptionState(State):
    messages: list = Field.reducer("append")
    response: str = ""
    output: str = ""


# The customer's phone number, from the run context (set by the channel, never by the model).
CustomerPhone = Annotated[str, FromContext("customer_phone")]


class SalonCalendar(AgentMiddleware):
    """Tells the model the salon's current date and the dates of the coming week."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def before_model(self, request):
        now = self.scheduler.now()
        week = ", ".join(f"{d:%A} {d:%Y-%m-%d}" for d in (now + timedelta(days=i) for i in range(1, 8)))
        note = {
            "role": "system",
            "content": f"Salon calendar: now is {now:%A %Y-%m-%d %H:%M} ({self.scheduler.salon.timezone}). "
                       f"Coming days: {week}.",
        }
        return request.override(messages=[request.messages[0], note, *request.messages[1:]])


# ---------------------------------------------------------------------- #
# Formatting
# ---------------------------------------------------------------------- #

def _sentence(error: Exception) -> str:
    text = str(error)
    return text if text.endswith((".", "?", "!")) else f"{text}."


def when(dt: datetime) -> str:
    """'Saturday, September 26 at 2:00 PM'."""
    return f"{dt:%A, %B} {dt.day} at {dt.strftime('%I:%M %p').lstrip('0')}"


def describe_slot(scheduler: Scheduler, slot: Slot) -> str:
    service = scheduler.db.get_service(slot.service_id)
    technician = scheduler.db.get_technician(slot.technician_id)
    return f"{service.name} with {technician.name} on {when(slot.start)} ({service.duration_minutes} min, {service.price})"


def describe_appointment(scheduler: Scheduler, appt: Any) -> str:
    return describe_slot(scheduler, Slot(appt.service_id, appt.technician_id, appt.start, appt.end))


# ---------------------------------------------------------------------- #
# Tools
# ---------------------------------------------------------------------- #

def make_tools(scheduler: Scheduler) -> list[Tool]:
    """The receptionist's tools. Every answer is read from the salon's data."""
    db = scheduler.db

    def customer(phone: str) -> Customer | None:
        return db.get_customer(phone=phone)

    def search_services(query: str = "") -> str:
        """List the salon's services with price and duration. Optionally filter by a word, e.g. 'gel'."""
        services = db.list_services()
        words = query.lower().split()
        matches = [s for s in services if all(w in f"{s.name} {s.description}".lower() for w in words)]
        if not matches:
            return f"No service matches '{query}'. Services offered: {', '.join(s.name for s in services)}."
        return "\n".join(
            f"- {s.name} (id {s.id}): {s.price}, {s.duration_minutes} min. {s.description}" for s in matches
        )

    def get_service_details(service: str) -> str:
        """Price, duration, description and qualified technicians of one service (by name or id)."""
        try:
            svc = scheduler.find_service(service)
        except BookingError as e:
            return f"{_sentence(e)} Services offered: {', '.join(s.name for s in db.list_services())}."
        technicians = [t.name for t in db.list_technicians() if t.can_perform(svc)]
        return (
            f"{svc.name} (id {svc.id}): {svc.price}, {svc.duration_minutes} min. {svc.description}\n"
            f"Technicians who do it: {', '.join(technicians)}."
        )

    def get_salon_info(topic: str = "all") -> str:
        """Salon information. topic: hours | policies | faq | technicians | contact | all."""
        salon = scheduler.salon
        weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        sections = {
            "contact": f"{salon.name}, {salon.address}. Phone {salon.phone}.",
            "hours": "Opening hours: " + "; ".join(
                f"{weekdays[d]} {salon.business_hours[d][0]}-{salon.business_hours[d][1]}"
                if d in salon.business_hours else f"{weekdays[d]} closed" for d in range(7)
            ) + (". Closed on " + ", ".join(d.isoformat() for d in salon.closures) if salon.closures else "") + ".",
            "technicians": "\n".join(
                f"- {t.name}: " + ", ".join(s.name for s in salon.services if t.can_perform(s))
                for t in salon.technicians
            ),
            "policies": "\n".join(f"- {p.title}: {p.text}" for p in salon.policies),
            "faq": "\n".join(f"- Q: {f.question} A: {f.answer}" for f in salon.faqs),
        }
        if topic in sections:
            return sections[topic]
        if topic != "all":
            return f"Unknown topic '{topic}'. Use one of: {', '.join(sections)}, all."
        return "\n\n".join(sections.values())

    def check_availability(service: str, date: str, technician: str = "", after: str = "", before: str = "") -> str:
        """Bookable times for a service on a date (YYYY-MM-DD). Optional: a technician's name, and a
        start-time range after='HH:MM' / before='HH:MM' (e.g. afternoon: after='12:00')."""
        try:
            slots = scheduler.check_availability(service, date, technician or None, after or None, before or None)
            svc = scheduler.find_service(service)
        except BookingError as e:
            return f"Cannot check: {_sentence(e)}"
        day = datetime.fromisoformat(date).date()
        header = f"{svc.name} ({svc.duration_minutes} min, {svc.price}) on {day:%A %Y-%m-%d}"
        if not slots:
            next_day = scheduler.next_available_date(svc.id, day + timedelta(days=1))
            hint = f" The next day with openings is {next_day}." if next_day else ""
            return f"No available times for {header}.{hint}"
        names = {t.id: t.name for t in db.list_technicians()}
        lines = [f"- {s.start:%H:%M} with {names[s.technician_id]} (slot_id: {s.slot_id})" for s in slots[:MAX_SLOTS_SHOWN]]
        more = len(slots) - MAX_SLOTS_SHOWN
        if more > 0:
            lines.append(f"({more} more; narrow down with after/before or a technician)")
        return f"Available for {header}:\n" + "\n".join(lines)

    def customer_lookup(phone: CustomerPhone) -> str:
        """The current customer's name, upcoming appointments (with ids) and past services."""
        found = customer(phone)
        if found is None:
            return "New customer: no record for this phone number. Ask for their name before booking."
        now = scheduler.now()
        appts = db.get_customer_appointments(found.id)
        upcoming = [a for a in appts if a.status == "booked" and a.start >= now]
        past = [a for a in appts if a.status in ("booked", "completed") and a.start < now]
        lines = [f"Customer: {found.name}."]
        lines += [f"- Upcoming appointment {a.id}: {describe_appointment(scheduler, a)}" for a in upcoming] or [
            "- No upcoming appointments."]
        if past:
            lines.append("Past services: " + ", ".join(db.get_service(a.service_id).name for a in past[-5:]) + ".")
        return "\n".join(lines)

    def create_booking(slot_id: str, phone: CustomerPhone, customer_name: str = "") -> str:
        """Book a slot_id from check_availability for the current customer. customer_name is
        required for a new customer. The customer is asked to confirm first."""
        try:
            found = customer(phone)
            if found is None:
                if not customer_name.strip():
                    return "Not booked: this is a new customer; ask for their name first."
                found = db.create_customer(phone=phone, name=customer_name.strip())
            appt = scheduler.book_appointment(found.id, slot_id)
        except BookingError as e:
            return f"Not booked: {_sentence(e)}"
        return f"Booked appointment {appt.id}: {describe_appointment(scheduler, appt)}."

    def reschedule_booking(appointment_id: int, new_slot_id: str, phone: CustomerPhone) -> str:
        """Move one of the current customer's appointments to a new slot_id from check_availability.
        The customer is asked to confirm first."""
        try:
            found = customer(phone)
            appt = scheduler.reschedule_appointment(appointment_id, new_slot_id, customer_id=found.id if found else None)
        except BookingError as e:
            return f"Not rescheduled: {_sentence(e)}"
        return f"Rescheduled appointment {appt.id}: {describe_appointment(scheduler, appt)}."

    def cancel_booking(appointment_id: int, phone: CustomerPhone) -> str:
        """Cancel one of the current customer's appointments. The customer is asked to confirm first."""
        try:
            found = customer(phone)
            result = scheduler.cancel_appointment(appointment_id, customer_id=found.id if found else None)
        except BookingError as e:
            return f"Not cancelled: {_sentence(e)}"
        note = f" This is a late cancellation. {result.policy}" if result.late else ""
        return f"Cancelled appointment {appointment_id}: {describe_appointment(scheduler, result.appointment)}.{note}"

    return [
        Tool(f) for f in (search_services, get_service_details, get_salon_info, check_availability,
                          customer_lookup, create_booking, reschedule_booking, cancel_booking)
    ]


# ---------------------------------------------------------------------- #
# Confirming changes
# ---------------------------------------------------------------------- #

class ChangeReview:
    """The approval policy of the write tools: checked against the database, described from it."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def check(self, request: ToolRequest) -> str | None:
        """Why the engine would refuse this change, or None. Runs before asking and on the answer."""
        return self._review(request)[1]

    def describe(self, request: ToolRequest) -> str:
        """The summary the customer confirms."""
        return self._review(request)[0] or ""

    def _review(self, request: ToolRequest) -> tuple[str | None, str | None]:
        """(summary, reason to refuse)."""
        scheduler, db = self.scheduler, self.scheduler.db
        tool, args = request.tool_call.name, request.tool_call.args
        customer = db.get_customer(phone=(request.context or {}).get("customer_phone", ""))
        try:
            if tool == "create_booking":
                slot = scheduler.validate_slot(str(args.get("slot_id", "")))
                if customer is None and not str(args.get("customer_name", "")).strip():
                    return None, "this is a new customer; ask for their name first."
                return f"Book {describe_slot(scheduler, slot)}", None
            appt = scheduler.owned_appointment(int(args.get("appointment_id", -1)), customer.id if customer else None)
            if tool == "reschedule_booking":
                slot = scheduler.validate_slot(str(args.get("new_slot_id", "")), exclude_appointment=appt.id)
                technician = db.get_technician(slot.technician_id).name
                return f"Move {describe_appointment(scheduler, appt)} to {when(slot.start)} with {technician}", None
            if tool == "cancel_booking":
                late = scheduler.is_late_cancellation(appt)
                policy = next((p.text for p in scheduler.salon.policies if p.id == "cancellation"), "")
                warning = f"\nThis is a late cancellation. {policy}" if late else ""
                return f"Cancel {describe_appointment(scheduler, appt)}{warning}", None
        except (BookingError, TypeError, ValueError) as e:
            return None, _sentence(e)
        return f"{tool}({args})", None


# ---------------------------------------------------------------------- #
# Agent and application
# ---------------------------------------------------------------------- #

def build_receptionist(scheduler: Scheduler, model: Any) -> Agent:
    review = ChangeReview(scheduler)
    policy = {"allowed_decisions": ["approve", "reject"], "check": review.check, "describe": review.describe}
    return Agent(
        name="receptionist",
        model=model,
        instructions=INSTRUCTIONS.format(salon=scheduler.salon.name),
        tools=make_tools(scheduler),
        middleware=[SalonCalendar(scheduler), HistoryLimit(max_messages=MAX_PROMPT_MESSAGES)],
        interrupt_on=dict.fromkeys(WRITE_TOOLS, policy),
        max_tool_iterations=8,
    )


def build_app(
    scheduler: Scheduler, model: Any, checkpointer: Any = None, listeners: list[Any] | None = None
) -> Application:
    """One conversation per customer: run it with ``thread_id`` = the customer's channel id."""
    workflow = Workflow("reception", state_schema=ReceptionState).then(build_receptionist(scheduler, model))
    app = Application("nail-salon", checkpointer=checkpointer or InMemoryCheckpointStore(), listeners=listeners)
    return app.register(workflow)


# ---------------------------------------------------------------------- #
# Conversation driver (what a chat channel such as WhatsApp calls)
# ---------------------------------------------------------------------- #

@dataclass
class Confirmation:
    """A change waiting for the customer's yes or no; `summary` is built from the database."""
    summary: str
    actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Reply:
    text: str
    confirmation: Confirmation | None = None
    duplicate: bool = False  # a message id already handled (e.g. a webhook delivered twice): send nothing


class Conversation:
    """One customer's conversation, identified by their phone number.

    Stateless: it wraps graphflow's ``app.thread(...)``, so the conversation and any
    pending confirmation live in the checkpointer. Every webhook call can build a
    new ``Conversation`` for the incoming message. The phone number is the run
    context; the agent's approval policy (``ChangeReview``) checks and describes
    every change.
    """

    def __init__(self, app: Application, scheduler: Scheduler, phone: str):
        self.scheduler = scheduler
        self.phone = phone
        self.thread = app.thread(f"customer:{phone}", context={"customer_phone": phone})

    def say(self, text: str, *, message_id: str | None = None) -> Reply:
        """The customer sends a message (`message_id`: the channel's id, to skip redeliveries)."""
        if self.thread.pending:
            # A new message instead of yes/no: decline the pending change, passing on what they said.
            return self.answer(False, f"The customer replied instead of confirming: {text}")
        return self._reply(self.thread.send(text, message_id=message_id))

    def answer(self, yes: bool, feedback: str = "") -> Reply:
        """The customer answers the pending confirmation."""
        if not self.thread.pending:
            raise RuntimeError("nothing is waiting for confirmation")
        decision = ToolDecision.approve() if yes else ToolDecision.reject(feedback or "The customer said no.")
        return self._reply(self.thread.respond(decision))

    def _reply(self, result: Any) -> Reply:
        if result.duplicate:
            return Reply(text="", duplicate=True)
        if result.interrupted:
            actions = result.interrupts[0].actions
            summary = "\n".join(a.get("summary", "") for a in actions)
            return Reply(text=f"Please confirm:\n{summary}", confirmation=Confirmation(summary, list(actions)))
        return Reply(text=result.get("output", ""))
