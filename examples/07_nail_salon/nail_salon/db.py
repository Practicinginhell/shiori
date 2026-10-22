"""Phase 1: the salon's data in SQLite (standard library only).

Tables: salon (profile, hours, rules, policies, FAQs), services, technicians,
customers, appointments. A trigger rejects any booked appointment that overlaps
another booked appointment of the same technician, so even a bug in the
scheduling engine cannot create a double booking.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

from nail_salon.config import FAQ, BookingRules, Policy, SalonConfig, Service, Technician


class BookingError(ValueError):
    """A booking operation that can't be done; the message is safe to show the customer."""


class DoubleBookingError(BookingError):
    """The technician already has an appointment at that time."""


@dataclass(frozen=True)
class Customer:
    id: int
    phone: str
    name: str
    notes: str = ""


@dataclass(frozen=True)
class Appointment:
    id: int
    customer_id: int
    service_id: str
    technician_id: str
    start: datetime
    end: datetime
    price_cents: int
    status: str = "booked"  # booked | cancelled | completed | no_show
    late_cancellation: bool = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS salon (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS services (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
    duration_minutes INTEGER NOT NULL CHECK (duration_minutes > 0),
    required_skills TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS technicians (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    skills TEXT NOT NULL,
    working_hours TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    phone TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    service_id TEXT NOT NULL REFERENCES services(id),
    technician_id TEXT NOT NULL REFERENCES technicians(id),
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked'
        CHECK (status IN ('booked', 'cancelled', 'completed', 'no_show')),
    late_cancellation INTEGER NOT NULL DEFAULT 0,
    CHECK (start_at < end_at)
);
CREATE INDEX IF NOT EXISTS appointments_by_technician ON appointments (technician_id, start_at);

CREATE TRIGGER IF NOT EXISTS no_double_booking_insert
BEFORE INSERT ON appointments WHEN NEW.status = 'booked'
BEGIN
    SELECT RAISE(ABORT, 'double booking') WHERE EXISTS (
        SELECT 1 FROM appointments a
        WHERE a.technician_id = NEW.technician_id AND a.status = 'booked'
          AND a.start_at < NEW.end_at AND NEW.start_at < a.end_at);
END;
CREATE TRIGGER IF NOT EXISTS no_double_booking_update
BEFORE UPDATE OF technician_id, start_at, end_at, status ON appointments WHEN NEW.status = 'booked'
BEGIN
    SELECT RAISE(ABORT, 'double booking') WHERE EXISTS (
        SELECT 1 FROM appointments a
        WHERE a.id != NEW.id AND a.technician_id = NEW.technician_id AND a.status = 'booked'
          AND a.start_at < NEW.end_at AND NEW.start_at < a.end_at);
END;
"""

_TIME_FORMAT = "%Y-%m-%dT%H:%M"


def _ts(value: datetime) -> str:
    return value.strftime(_TIME_FORMAT)


def _dt(text: str) -> datetime:
    return datetime.strptime(text, _TIME_FORMAT)


class SalonDB:
    """The salon's data store. ``path=":memory:"`` gives a throwaway database."""

    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._depth = 0

    @classmethod
    def from_config(cls, config: SalonConfig, path: str = ":memory:") -> SalonDB:
        """A database holding the salon's profile, services and technicians from `config`.

        Reopening an existing database applies the config's current values (prices,
        hours, skills) and keeps customers and appointments. Services and technicians
        removed from the config stay in the database, since appointments refer to them.
        """
        db = cls(path)
        with db.transaction():
            db.save_salon(config)
            for service in config.services:
                db._upsert("services", db._service_row(service))
            for technician in config.technicians:
                db._upsert("technicians", db._technician_row(technician))
        return db

    def _upsert(self, table: str, row: dict[str, object]) -> None:
        columns = ", ".join(row)
        updates = ", ".join(f"{c} = excluded.{c}" for c in row if c != "id")
        self._execute(
            f"INSERT INTO {table} ({columns}) VALUES ({', '.join('?' * len(row))})"
            f" ON CONFLICT(id) DO UPDATE SET {updates}",  # table and columns are fixed by this module
            tuple(row.values()),
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """One atomic unit of work; nested calls join the outer transaction.

        ``BEGIN IMMEDIATE`` takes SQLite's write lock up front, so a check-then-insert
        inside a transaction cannot interleave with another writer (thread or process).
        """
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outermost:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outermost:
                self._conn.execute("COMMIT")

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Runs a write; the returned cursor is only used for ``lastrowid``."""
        with self._lock:
            return self._conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Runs a read and fetches every row while holding the lock (the connection is shared by threads)."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ #
    # Salon profile
    # ------------------------------------------------------------------ #

    def save_salon(self, config: SalonConfig) -> None:
        data = {
            "name": config.name,
            "timezone": config.timezone,
            "phone": config.phone,
            "address": config.address,
            "business_hours": config.business_hours,
            "closures": [d.isoformat() for d in config.closures],
            "rules": asdict(config.rules),
            "policies": [asdict(p) for p in config.policies],
            "faqs": [asdict(f) for f in config.faqs],
        }
        self._execute("INSERT OR REPLACE INTO salon (id, data) VALUES (1, ?)", (json.dumps(data),))

    def salon(self) -> SalonConfig:
        """The salon as currently stored (services and technicians from their tables)."""
        row = self._query_one("SELECT data FROM salon WHERE id = 1")
        if row is None:
            raise BookingError("the salon has not been set up")
        data = json.loads(row["data"])
        return SalonConfig(
            name=data["name"],
            timezone=data["timezone"],
            phone=data["phone"],
            address=data["address"],
            business_hours={int(k): tuple(v) for k, v in data["business_hours"].items()},
            closures=tuple(date.fromisoformat(d) for d in data["closures"]),
            rules=BookingRules(**data["rules"]),
            services=tuple(self.list_services()),
            technicians=tuple(self.list_technicians()),
            policies=tuple(Policy(**p) for p in data["policies"]),
            faqs=tuple(FAQ(**f) for f in data["faqs"]),
        )

    # ------------------------------------------------------------------ #
    # Services and technicians
    # ------------------------------------------------------------------ #

    @staticmethod
    def _service_row(service: Service) -> dict[str, object]:
        return {"id": service.id, "name": service.name, "description": service.description,
                "price_cents": service.price_cents, "duration_minutes": service.duration_minutes,
                "required_skills": json.dumps(list(service.required_skills))}

    def create_service(self, service: Service) -> Service:
        row = self._service_row(service)
        self._execute(f"INSERT INTO services ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                      tuple(row.values()))
        return service

    @staticmethod
    def _service(row: sqlite3.Row) -> Service:
        return Service(row["id"], row["name"], row["description"], row["price_cents"], row["duration_minutes"],
                       tuple(json.loads(row["required_skills"])))

    def get_service(self, service_id: str) -> Service | None:
        row = self._query_one("SELECT * FROM services WHERE id = ?", (service_id,))
        return self._service(row) if row else None

    def list_services(self) -> list[Service]:
        return [self._service(r) for r in self._query("SELECT * FROM services ORDER BY rowid")]

    @staticmethod
    def _technician_row(technician: Technician) -> dict[str, object]:
        return {"id": technician.id, "name": technician.name, "skills": json.dumps(list(technician.skills)),
                "working_hours": json.dumps(technician.working_hours)}

    def create_technician(self, technician: Technician) -> Technician:
        row = self._technician_row(technician)
        self._execute(f"INSERT INTO technicians ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                      tuple(row.values()))
        return technician

    @staticmethod
    def _technician(row: sqlite3.Row) -> Technician:
        hours = {int(k): tuple(v) for k, v in json.loads(row["working_hours"]).items()}
        return Technician(row["id"], row["name"], tuple(json.loads(row["skills"])), hours)

    def get_technician(self, technician_id: str) -> Technician | None:
        row = self._query_one("SELECT * FROM technicians WHERE id = ?", (technician_id,))
        return self._technician(row) if row else None

    def list_technicians(self) -> list[Technician]:
        return [self._technician(r) for r in self._query("SELECT * FROM technicians ORDER BY rowid")]

    # ------------------------------------------------------------------ #
    # Customers
    # ------------------------------------------------------------------ #

    def create_customer(self, phone: str, name: str, notes: str = "") -> Customer:
        try:
            cur = self._execute("INSERT INTO customers (phone, name, notes) VALUES (?, ?, ?)", (phone, name, notes))
        except sqlite3.IntegrityError:
            raise BookingError(f"a customer with phone {phone} already exists") from None
        return Customer(cur.lastrowid, phone, name, notes)

    def get_customer(self, customer_id: int | None = None, *, phone: str | None = None) -> Customer | None:
        if customer_id is not None:
            row = self._query_one("SELECT * FROM customers WHERE id = ?", (customer_id,))
        else:
            row = self._query_one("SELECT * FROM customers WHERE phone = ?", (phone,))
        return Customer(row["id"], row["phone"], row["name"], row["notes"]) if row else None

    # ------------------------------------------------------------------ #
    # Appointments
    # ------------------------------------------------------------------ #

    @staticmethod
    def _appointment(row: sqlite3.Row) -> Appointment:
        return Appointment(
            id=row["id"],
            customer_id=row["customer_id"],
            service_id=row["service_id"],
            technician_id=row["technician_id"],
            start=_dt(row["start_at"]),
            end=_dt(row["end_at"]),
            price_cents=row["price_cents"],
            status=row["status"],
            late_cancellation=bool(row["late_cancellation"]),
        )

    def create_appointment(
        self, customer_id: int, service_id: str, technician_id: str, start: datetime, end: datetime, price_cents: int
    ) -> Appointment:
        try:
            cur = self._execute(
                "INSERT INTO appointments (customer_id, service_id, technician_id, start_at, end_at, price_cents)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (customer_id, service_id, technician_id, _ts(start), _ts(end), price_cents),
            )
        except sqlite3.IntegrityError as e:
            if "double booking" in str(e):
                raise DoubleBookingError("that technician already has an appointment at that time") from None
            raise
        return Appointment(cur.lastrowid, customer_id, service_id, technician_id, start, end, price_cents)

    def get_appointment(self, appointment_id: int) -> Appointment | None:
        row = self._query_one("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
        return self._appointment(row) if row else None

    _UPDATABLE = frozenset({"service_id", "technician_id", "start", "end", "price_cents", "status", "late_cancellation"})

    def update_appointment(self, appointment_id: int, **changes: object) -> Appointment:
        unknown = set(changes) - self._UPDATABLE
        if unknown:
            raise ValueError(f"cannot update appointment field(s) {sorted(unknown)}")
        columns = {"start": "start_at", "end": "end_at"}
        values = {columns.get(k, k): (_ts(v) if isinstance(v, datetime) else v) for k, v in changes.items()}
        if values:
            assignments = ", ".join(f"{column} = ?" for column in values)
            try:
                self._execute(f"UPDATE appointments SET {assignments} WHERE id = ?",  # column names come from _UPDATABLE
                              (*values.values(), appointment_id))
            except sqlite3.IntegrityError as e:
                if "double booking" in str(e):
                    raise DoubleBookingError("that technician already has an appointment at that time") from None
                raise
        updated = self.get_appointment(appointment_id)
        if updated is None:
            raise BookingError(f"no appointment {appointment_id}")
        return updated

    def cancel_appointment(self, appointment_id: int, late: bool = False) -> Appointment:
        return self.update_appointment(appointment_id, status="cancelled", late_cancellation=int(late))

    def get_customer_appointments(self, customer_id: int) -> list[Appointment]:
        rows = self._query("SELECT * FROM appointments WHERE customer_id = ? ORDER BY start_at", (customer_id,))
        return [self._appointment(r) for r in rows]

    def booked_for_technician(self, technician_id: str, day: date, exclude_id: int | None = None) -> list[Appointment]:
        """The technician's booked appointments that touch `day`."""
        start, end = datetime.combine(day, datetime.min.time()), datetime.combine(day + timedelta(days=1),
                                                                                  datetime.min.time())
        rows = self._query(
            "SELECT * FROM appointments WHERE technician_id = ? AND status = 'booked'"
            " AND start_at < ? AND end_at > ? AND id != ? ORDER BY start_at",
            (technician_id, _ts(end), _ts(start), exclude_id if exclude_id is not None else -1),
        )
        return [self._appointment(r) for r in rows]
