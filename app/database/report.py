from datetime import datetime
from enum import Enum
from typing import Any

from sqlmodel import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    Text,
)


class ReportStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    REJECTED = "rejected"


class ReportType(str, Enum):
    USER = "user"
    BEATMAP = "beatmap"
    SCORE = "score"
    COMMENT = "comment"
    OTHER = "other"


class Report(SQLModel, table=True):
    __tablename__: str = "reports"

    id: int = Field(primary_key=True, index=True, default=None)
    reporter_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="CASCADE"), index=True)
    )
    reported_user_id: int | None = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="SET NULL"), index=True), default=None
    )
    report_type: ReportType = Field(index=True)
    target_type: str = Field(index=True)  # e.g., "user", "beatmap", "score"
    target_id: int = Field(sa_column=Column(BigInteger, index=True))
    reason: str = Field(max_length=255)
    description: str = Field(sa_column=Column(Text))
    status: ReportStatus = Field(default=ReportStatus.PENDING, index=True)
    resolved_by: int | None = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="SET NULL")), default=None
    )
    resolution_action: str | None = Field(default=None, max_length=100)  # e.g., "close", "ban", "warn"
    resolution_notes: str | None = Field(sa_column=Column(Text), default=None)
    created_at: datetime = Field(sa_column=Column(DateTime, index=True), default_factory=datetime.utcnow)
    resolved_at: datetime | None = Field(sa_column=Column(DateTime, index=True), default=None)
    additional_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))

class ReportCreate(SQLModel):
    reporter_id: int
    reported_user_id: int | None = None
    report_type: ReportType
    target_type: str
    target_id: int
    reason: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportUpdate(SQLModel):
    status: ReportStatus | None = None
    resolved_by: int | None = None
    resolution_action: str | None = None
    resolution_notes: str | None = None
    resolved_at: datetime | None = None


class ReportResponse(SQLModel):
    id: int
    reporter_id: int
    reporter_username: str | None = None
    reported_user_id: int | None = None
    reported_username: str | None = None
    report_type: ReportType
    target_type: str
    target_id: int
    reason: str
    description: str
    status: ReportStatus
    resolved_by: int | None = None
    resolved_by_username: str | None = None
    resolution_action: str | None = None
    resolution_notes: str | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    additional_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))
