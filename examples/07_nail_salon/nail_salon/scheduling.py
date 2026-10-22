"""Phase 2: the deterministic appointment engine.

Availability = service duration + business hours + technician working hours +
technician skills + existing appointments (with cleanup buffer) + booking rules
(slot grid, minimum notice, how far ahead, closures).

Booking takes a ``slot_id`` from ``check_availability`` and re-computes
availability inside the database transaction: a slot is bookable only if the
same calculation still offers it. The caller (for example a language model)
never decides whether a time is free.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from nail_salon.config import SalonConfig, Service, Technician
from nail_salon.db import Appointment, BookingError, DoubleBookingError, SalonDB

_SLOT_TIME = "%Y-%m-%dT%H:%M"


@dataclass(frozen=True)
class Slot:
    """A bookable time: one service, with one technician, starting at `start`."""
    service_id: str
    technician_id: str
    start: datetime
    end: datetime

    @property
    def slot_id(self) -> str:
        return f"{self.service_id}@{self.technician_id}@{self.start.strftime(_SLOT_TIME)}"


@dataclass(frozen=True)
class CancelResult:
    appointment: Appointment
    late: bool          # inside the cancellation notice period
    policy: str         # the cancellation policy text, for the customer


def parse_slot_id(slot_id: str) -> tuple[str, str, datetime]:
    parts = slot_id.split("@")
    if len(parts) != 3:
        raise BookingError(f"'{slot_id}' is not a slot id; use one returned by check_availability")
    try:
        start = datetime.strptime(parts[2], _SLOT_TIME)
    except ValueError:
        raise BookingError(f"'{slot_id}' is not a slot id; use one returned by check_availability") from None
    return parts[0], parts[1], start


def _clock(t: str) -> time:
    try:
        return time.fromisoformat(t)
    except ValueError:
        raise BookingError(f"'{t}' is not a time; use HH:MM, e.g. 14:00") from None


class Scheduler:
    """Availability and bookings for one salon.

    Args:
        db: The salon's database.
        clock: Returns the salon's current local time (naive); defaults to now in
            the salon's timezone. Tests pass a fixed clock.
    """

    def __init__(self, db: SalonDB, clock: Callable[[], datetime] | None = None):
        self.db = db
        self._clock = clock

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    @property
    def salon(self) -> SalonConfig:
        return self.db.salon()

    def now(self) -> datetime:
        if self._clock is not None:
            return self._clock()
        return datetime.now(ZoneInfo(self.salon.timezone)).replace(tzinfo=None, second=0, microsecond=0)

    def find_service(self, query: str) -> Service:
        """A service by id or (case-insensitive, partial) name."""
        wanted = query.strip().lower().replace("_", " ")
        services = self.db.list_services()
        for match in (
            lambda s: s.id == query or s.name.lower() == wanted,
            lambda s: s.id.replace("_", " ") == wanted,
            lambda s: wanted and s.name.lower().startswith(wanted),
            lambda s: wanted and wanted in s.name.lower(),
        ):
            found = [s for s in services if match(s)]
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                names = ", ".join(s.name for s in found)
                raise BookingError(f"'{query}' matches several services ({names}); which one?")
        raise BookingError(f"there is no service called '{query}'")

    def find_technician(self, query: str) -> Technician:
        wanted = query.strip().lower()
        for t in self.db.list_technicians():
            if t.id == query or t.name.lower() == wanted:
                return t
        raise BookingError(f"there is no technician called '{query}'")

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #

    def _day(self, value: str | date) -> date:
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(value)
        except ValueError:
            raise BookingError(f"'{value}' is not a date; use YYYY-MM-DD") from None

    def _check_window(self, day: date, salon: SalonConfig) -> None:
        today = self.now().date()
        if day < today:
            raise BookingError(f"{day.isoformat()} is in the past")
        if day > today + timedelta(days=salon.rules.max_days_ahead):
            raise BookingError(f"bookings open only {salon.rules.max_days_ahead} days ahead")

    def _slots_for(
        self, salon: SalonConfig, service: Service, technician: Technician, day: date, exclude_id: int | None
    ) -> list[Slot]:
        weekday = day.weekday()
        if day in salon.closures or weekday not in salon.business_hours or weekday not in technician.working_hours:
            return []
        opens = max(_clock(salon.business_hours[weekday][0]), _clock(technician.working_hours[weekday][0]))
        closes = min(_clock(salon.business_hours[weekday][1]), _clock(technician.working_hours[weekday][1]))
        rules = salon.rules
        duration = timedelta(minutes=service.duration_minutes)
        buffer = timedelta(minutes=rules.buffer_minutes)
        earliest = self.now() + timedelta(minutes=rules.min_notice_minutes)
        booked = self.db.booked_for_technician(technician.id, day, exclude_id=exclude_id)

        slots = []
        start, last_end = datetime.combine(day, opens), datetime.combine(day, closes)
        while start + duration <= last_end:
            end = start + duration
            free = all(end + buffer <= a.start or a.end + buffer <= start for a in booked)
            if start >= earliest and free:
                slots.append(Slot(service.id, technician.id, start, end))
            start += timedelta(minutes=rules.slot_interval_minutes)
        return slots

    def check_availability(
        self,
        service: str,
        date: str | date,
        technician: str | None = None,
        after: str | None = None,
        before: str | None = None,
        *,
        exclude_appointment: int | None = None,
    ) -> list[Slot]:
        """Every bookable slot for `service` on `date`, optionally for one technician and a time range.

        `after` / `before` ("HH:MM") bound the start time: after <= start < before.
        """
        salon = self.salon
        svc = self.find_service(service)
        day = self._day(date)
        self._check_window(day, salon)
        technicians = [t for t in salon.technicians if t.can_perform(svc)]
        if technician:
            chosen = self.find_technician(technician)
            if not chosen.can_perform(svc):
                raise BookingError(f"{chosen.name} is not qualified for {svc.name}")
            technicians = [chosen]
        slots = [s for t in technicians for s in self._slots_for(salon, svc, t, day, exclude_appointment)]
        if after:
            slots = [s for s in slots if s.start.time() >= _clock(after)]
        if before:
            slots = [s for s in slots if s.start.time() < _clock(before)]
        return sorted(slots, key=lambda s: (s.start, s.technician_id))

    def next_available_date(self, service: str, after: str | date, days: int = 14) -> str | None:
        """The first date from `after` (inclusive) with any slot for `service`, within `days`."""
        day = self._day(after)
        for offset in range(days):
            candidate = day + timedelta(days=offset)
            try:
                if self.check_availability(service, candidate):
                    return candidate.isoformat()
            except BookingError:
                return None  # past the booking window
        return None

    def validate_slot(self, slot_id: str, exclude_appointment: int | None = None) -> Slot:
        """The slot `slot_id` names, if the engine offers it right now; BookingError otherwise."""
        service_id, technician_id, start = parse_slot_id(slot_id)
        service = self.find_service(service_id)
        technician = self.db.get_technician(technician_id)
        if technician is None:
            raise BookingError(f"there is no technician '{technician_id}'")
        if not technician.can_perform(service):
            raise BookingError(f"{technician.name} is not qualified for {service.name}")
        offered = self.check_availability(service.id, start.date(), technician=technician.id,
                                          exclude_appointment=exclude_appointment)
        for slot in offered:
            if slot.start == start:
                return slot
        raise BookingError(
            f"{service.name} with {technician.name} at {start:%A %d %B %H:%M} is not available"
            " (or no longer available); check availability again"
        )

    # ------------------------------------------------------------------ #
    # Bookings
    # ------------------------------------------------------------------ #

    def owned_appointment(self, appointment_id: int, customer_id: int | None) -> Appointment:
        appt = self.db.get_appointment(appointment_id)
        if appt is None or customer_id is None or appt.customer_id != customer_id:
            raise BookingError(f"there is no appointment {appointment_id} for this customer")
        return appt

    def book_appointment(self, customer_id: int, slot_id: str) -> Appointment:
        with self.db.transaction():
            slot = self.validate_slot(slot_id)
            price = self.db.get_service(slot.service_id).price_cents  # type: ignore[union-attr]
            try:
                return self.db.create_appointment(customer_id, slot.service_id, slot.technician_id,
                                                  slot.start, slot.end, price)
            except DoubleBookingError:
                raise BookingError("that time is no longer available; check availability again") from None

    def reschedule_appointment(self, appointment_id: int, new_slot_id: str, customer_id: int) -> Appointment:
        """Moves an appointment (possibly to another service or technician)."""
        with self.db.transaction():
            appt = self.owned_appointment(appointment_id, customer_id)
            if appt.status != "booked":
                raise BookingError(f"appointment {appointment_id} is {appt.status}")
            slot = self.validate_slot(new_slot_id, exclude_appointment=appointment_id)
            price = self.db.get_service(slot.service_id).price_cents  # type: ignore[union-attr]
            return self.db.update_appointment(
                appointment_id, service_id=slot.service_id, technician_id=slot.technician_id,
                start=slot.start, end=slot.end, price_cents=price,
            )

    def is_late_cancellation(self, appt: Appointment) -> bool:
        notice = timedelta(hours=self.salon.rules.cancellation_notice_hours)
        return appt.start - self.now() < notice

    def cancel_appointment(self, appointment_id: int, customer_id: int) -> CancelResult:
        with self.db.transaction():
            appt = self.owned_appointment(appointment_id, customer_id)
            if appt.status == "cancelled":
                raise BookingError(f"appointment {appointment_id} is already cancelled")
            if appt.status != "booked" or appt.start <= self.now():
                raise BookingError(f"appointment {appointment_id} can no longer be cancelled")
            late = self.is_late_cancellation(appt)
            cancelled = self.db.cancel_appointment(appointment_id, late=late)
        policy = next((p.text for p in self.salon.policies if p.id == "cancellation"), "")
        return CancelResult(cancelled, late, policy)
