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


class RankRequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class RankRequest(SQLModel, table=True):
    __tablename__: str = "beatmap_rank_requests"

    id: int = Field(primary_key=True, index=True, default=None)
    beatmapset_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("beatmapsets.id", ondelete="CASCADE"), index=True)
    )
    requester_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="CASCADE"), index=True)
    )
    status: RankRequestStatus = Field(default=RankRequestStatus.PENDING, index=True)
    reason: str = Field(sa_column=Column(Text))
    reviewed_by: int | None = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="SET NULL")), default=None
    )
    review_notes: str | None = Field(sa_column=Column(Text), default=None)
    rejection_reason: str | None = Field(sa_column=Column(Text), default=None)
    created_at: datetime = Field(sa_column=Column(DateTime, index=True), default_factory=datetime.utcnow)
    reviewed_at: datetime | None = Field(sa_column=Column(DateTime, index=True), default=None)
    request_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class RankRequestCreate(SQLModel):
    beatmapset_id: int
    requester_id: int
    reason: str
    request_metadata: dict[str, Any] = Field(default_factory=dict)


class RankRequestUpdate(SQLModel):
    status: RankRequestStatus | None = None
    reviewed_by: int | None = None
    review_notes: str | None = None
    rejection_reason: str | None = None
    reviewed_at: datetime | None = None


class RankRequestResponse(SQLModel):
    id: int
    beatmapset_id: int
    beatmapset_title: str | None = None
    beatmapset_artist: str | None = None
    requester_id: int
    requester_username: str | None = None
    status: RankRequestStatus
    reason: str
    reviewed_by: int | None = None
    reviewed_by_username: str | None = None
    review_notes: str | None = None
    rejection_reason: str | None = None
    created_at: datetime
    reviewed_at: datetime | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
