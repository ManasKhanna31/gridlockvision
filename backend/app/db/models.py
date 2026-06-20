"""
GridlockVision AI — Database models (SQLAlchemy ORM, SQLite by default,
swappable to PostgreSQL by changing DATABASE_URL in core/config.py).
"""
import uuid
import datetime as dt

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, ForeignKey, Text, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.core.config import DATABASE_URL

Base = declarative_base()


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Vehicle(Base):
    """A vehicle identified primarily by its OCR'd plate number.

    A plate number may legitimately be re-seen across many violations —
    this table lets us aggregate "repeat offender" history per plate,
    which feeds the Risk Score (see violations/risk_engine.py).
    """
    __tablename__ = "vehicles"

    id = Column(String, primary_key=True, default=gen_uuid)
    plate_number = Column(String, index=True, nullable=True)  # nullable: OCR can fail
    vehicle_type = Column(String, nullable=False)
    first_seen = Column(DateTime, default=dt.datetime.utcnow)
    last_seen = Column(DateTime, default=dt.datetime.utcnow)

    violations = relationship("Violation", back_populates="vehicle")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(String, primary_key=True, default=gen_uuid)
    violation_id = Column(String, unique=True, index=True, default=gen_uuid)

    vehicle_id = Column(String, ForeignKey("vehicles.id"), nullable=True)
    vehicle = relationship("Vehicle", back_populates="violations")

    vehicle_type = Column(String, nullable=False)
    plate_number = Column(String, nullable=True, index=True)
    violation_type = Column(String, nullable=False, index=True)  # e.g. HELMET, TRIPLE_RIDING
    confidence = Column(Float, nullable=False)

    timestamp = Column(DateTime, default=dt.datetime.utcnow, index=True)
    camera_id = Column(String, nullable=False)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)

    evidence_image_path = Column(String, nullable=True)
    evidence_json_path = Column(String, nullable=True)

    risk_score = Column(Float, nullable=True)
    risk_category = Column(String, nullable=True)   # LOW / MEDIUM / HIGH / CRITICAL
    enforcement_priority = Column(String, nullable=True)

    notes = Column(Text, nullable=True)


class Camera(Base):
    """Static camera metadata + the ROI/lane config used by violation rules.

    In production each physical camera has its own stop-line, no-parking
    zone, and lane-direction vectors. For the hackathon demo, one camera
    profile is preconfigured (see violations/roi_config.py) but the schema
    supports many.
    """
    __tablename__ = "cameras"

    id = Column(String, primary_key=True)  # human-readable camera id, e.g. CAM-DEMO-01
    location_name = Column(String, nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    roi_config_json = Column(Text, nullable=True)  # serialized ROI polygons / lines


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
