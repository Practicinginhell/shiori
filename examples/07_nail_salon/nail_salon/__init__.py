"""A nail salon receptionist built on graphflow (phases 0-3 of the project plan).

    config.py        Phase 0: the salon as validated structured data (salon.yaml)
    db.py            Phase 1: SQLite store and CRUD operations
    scheduling.py    Phase 2: deterministic availability and bookings
    receptionist.py  Phase 3: the graphflow agent, its tools, and a conversation driver
"""

from nail_salon.config import SalonConfig, SalonConfigError, Service, Technician, load_salon
from nail_salon.db import Appointment, BookingError, Customer, DoubleBookingError, SalonDB
from nail_salon.scheduling import CancelResult, Scheduler, Slot

__all__ = [
    "Appointment",
    "BookingError",
    "CancelResult",
    "Customer",
    "DoubleBookingError",
    "SalonConfig",
    "SalonConfigError",
    "SalonDB",
    "Scheduler",
    "Service",
    "Slot",
    "Technician",
    "load_salon",
]
