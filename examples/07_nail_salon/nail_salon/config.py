"""Phase 0: the salon as structured data, loaded from ``salon.yaml`` and validated.

Which technicians can perform a service is derived from skills (a technician
needs every skill the service requires), so the two can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Any

import yaml

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

Hours = dict[int, tuple[str, str]]  # weekday (0 = Monday) -> ("HH:MM", "HH:MM")


class SalonConfigError(ValueError):
    """The salon data is incomplete or inconsistent."""


@dataclass(frozen=True)
class Service:
    id: str
    name: str
    description: str
    price_cents: int
    duration_minutes: int
    required_skills: tuple[str, ...]

    @property
    def price(self) -> str:
        return f"${self.price_cents / 100:.0f}" if self.price_cents % 100 == 0 else f"${self.price_cents / 100:.2f}"


@dataclass(frozen=True)
class Technician:
    id: str
    name: str
    skills: tuple[str, ...]
    working_hours: Hours

    def can_perform(self, service: Service) -> bool:
        return set(service.required_skills) <= set(self.skills)


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class FAQ:
    question: str
    answer: str


@dataclass(frozen=True)
class BookingRules:
    slot_interval_minutes: int = 30
    buffer_minutes: int = 10
    min_notice_minutes: int = 60
    max_days_ahead: int = 60
    cancellation_notice_hours: int = 24


@dataclass(frozen=True)
class SalonConfig:
    name: str
    timezone: str
    phone: str = ""
    address: str = ""
    business_hours: Hours = field(default_factory=dict)
    closures: tuple[date, ...] = ()
    rules: BookingRules = field(default_factory=BookingRules)
    services: tuple[Service, ...] = ()
    technicians: tuple[Technician, ...] = ()
    policies: tuple[Policy, ...] = ()
    faqs: tuple[FAQ, ...] = ()

    def service(self, service_id: str) -> Service:
        return next(s for s in self.services if s.id == service_id)

    def qualified_technicians(self, service_id: str) -> list[Technician]:
        service = self.service(service_id)
        return [t for t in self.technicians if t.can_perform(service)]


# ---------------------------------------------------------------------- #
# Loading and validation
# ---------------------------------------------------------------------- #

def _parse_time(text: str, where: str) -> time:
    try:
        return time.fromisoformat(text)
    except (TypeError, ValueError):
        raise SalonConfigError(f"{where}: '{text}' is not a HH:MM time") from None


def _hours(raw: dict[str, str] | None, where: str) -> Hours:
    hours: Hours = {}
    for day, span in (raw or {}).items():
        if day not in WEEKDAYS:
            raise SalonConfigError(f"{where}: '{day}' is not a weekday")
        opens, sep, closes = str(span).partition("-")
        if not sep:
            raise SalonConfigError(f"{where}.{day}: expected 'HH:MM-HH:MM', got '{span}'")
        if _parse_time(opens, f"{where}.{day}") >= _parse_time(closes, f"{where}.{day}"):
            raise SalonConfigError(f"{where}.{day}: hours '{span}' close before they open")
        hours[WEEKDAYS.index(day)] = (opens.strip(), closes.strip())
    return hours


def _unique(items: list[Any], kind: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise SalonConfigError(f"duplicate {kind} id '{item.id}'")
        seen.add(item.id)


def _cents(amount: Any, where: str) -> int:
    try:
        cents = round(float(amount) * 100)
    except (TypeError, ValueError):
        raise SalonConfigError(f"{where}: price '{amount}' is not a number") from None
    if cents < 0:
        raise SalonConfigError(f"{where}: price must not be negative")
    return cents


def load_salon(source: str | Path | dict[str, Any]) -> SalonConfig:
    """Loads and validates the salon from a YAML file path or an already-parsed dict."""
    data = source if isinstance(source, dict) else yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    salon = data.get("salon") or {}
    if not salon.get("name") or not salon.get("timezone"):
        raise SalonConfigError("salon.name and salon.timezone are required")

    services = []
    for raw in data.get("services") or []:
        where = f"service '{raw.get('id')}'"
        if int(raw.get("duration_minutes") or 0) <= 0:
            raise SalonConfigError(f"{where}: duration_minutes must be positive")
        services.append(Service(
            id=raw["id"],
            name=raw["name"],
            description=raw.get("description", ""),
            price_cents=_cents(raw.get("price"), where),
            duration_minutes=int(raw["duration_minutes"]),
            required_skills=tuple(raw.get("required_skills") or ()),
        ))
    technicians = [
        Technician(
            id=raw["id"],
            name=raw["name"],
            skills=tuple(raw.get("skills") or ()),
            working_hours=_hours(raw.get("working_hours"), f"technician '{raw.get('id')}'.working_hours"),
        )
        for raw in data.get("technicians") or []
    ]
    policies = [Policy(p["id"], p["title"], p["text"]) for p in data.get("policies") or []]
    _unique(services, "service")
    _unique(technicians, "technician")
    _unique(policies, "policy")
    if not services or not technicians:
        raise SalonConfigError("the salon needs at least one service and one technician")
    for service in services:
        if not any(t.can_perform(service) for t in technicians):
            raise SalonConfigError(
                f"service '{service.id}' needs skills {list(service.required_skills)} that no technician has"
            )

    try:
        rules = BookingRules(**(data.get("booking_rules") or {}))
    except TypeError as e:
        raise SalonConfigError(f"booking_rules: {e}") from None
    try:
        closures = tuple(date.fromisoformat(str(d)) for d in data.get("closures") or ())
    except ValueError as e:
        raise SalonConfigError(f"closures: {e}") from None

    return SalonConfig(
        name=salon["name"],
        timezone=salon["timezone"],
        phone=str(salon.get("phone", "")),
        address=str(salon.get("address", "")),
        business_hours=_hours(data.get("business_hours"), "business_hours"),
        closures=closures,
        rules=rules,
        services=tuple(services),
        technicians=tuple(technicians),
        policies=tuple(policies),
        faqs=tuple(FAQ(f["question"], f["answer"]) for f in data.get("faqs") or ()),
    )
