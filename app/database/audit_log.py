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


class AuditActionType(str, Enum):
    USER_BAN = "USER_BAN"
    USER_UNBAN = "USER_UNBAN"
    USER_ROLE_CHANGE = "USER_ROLE_CHANGE"
    USER_WIPE = "USER_WIPE"
    USER_UPDATE = "USER_UPDATE"
    BEATMAP_DELETE = "BEATMAP_DELETE"
    BEATMAP_RANK = "BEATMAP_RANK"
    BEATMAP_UNRANK = "BEATMAP_UNRANK"
    BEATMAP_LOVE = "BEATMAP_LOVE"
    BEATMAP_UNLOVE = "BEATMAP_UNLOVE"
    SCORE_DELETE = "SCORE_DELETE"
    TEAM_DISBAND = "TEAM_DISBAND"
    TEAM_CREATE = "TEAM_CREATE"
    TEAM_UPDATE = "TEAM_UPDATE"
    SETTINGS_CHANGE = "SETTINGS_CHANGE"
    ANNOUNCEMENT_CREATE = "ANNOUNCEMENT_CREATE"
    ANNOUNCEMENT_UPDATE = "ANNOUNCEMENT_UPDATE"
    ANNOUNCEMENT_DELETE = "ANNOUNCEMENT_DELETE"
    BADGE_CREATE = "BADGE_CREATE"
    BADGE_UPDATE = "BADGE_UPDATE"
    BADGE_DELETE = "BADGE_DELETE"
    REPORT_RESOLVE = "REPORT_RESOLVE"
    MAINTENANCE_MODE = "MAINTENANCE_MODE"
    RECALCULATION_TRIGGERED = "RECALCULATION_TRIGGERED"


class TargetType(str, Enum):
    USER = "USER"
    BEATMAP = "BEATMAP"
    BEATMAPSET = "BEATMAPSET"
    SCORE = "SCORE"
    TEAM = "TEAM"
    ANNOUNCEMENT = "ANNOUNCEMENT"
    BADGE = "BADGE"
    REPORT = "REPORT"
    SYSTEM = "SYSTEM"


class AuditLog(SQLModel, table=True):
    __tablename__: str = "audit_logs"

    id: int = Field(primary_key=True, index=True, default=None)
    actor_id: int = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="CASCADE"), index=True)
    )
    actor_username: str = Field(max_length=255, index=True)
    action_type: AuditActionType = Field(index=True)
    target_type: TargetType = Field(index=True)
    target_id: int | None = Field(sa_column=Column(BigInteger, index=True), default=None)
    target_name: str | None = Field(max_length=255, default=None)
    reason: str | None = Field(sa_column=Column(Text), default=None)
    additional_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    ip_address: str | None = Field(max_length=45, default=None)  # IPv6 compatible
    created_at: datetime = Field(sa_column=Column(DateTime, index=True), default_factory=datetime.utcnow)


class AuditLogCreate(SQLModel):
    actor_id: int
    actor_username: str
    action_type: AuditActionType
    target_type: TargetType
    target_id: int | None = None
    target_name: str | None = None
    reason: str | None = None
    additional_metadata: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None


class AuditLogResponse(SQLModel):
    id: int
    actor_id: int
    actor_username: str
    actor_avatar_url: str | None = None
    action_type: AuditActionType
    target_type: TargetType
    target_id: int | None = None
    target_name: str | None = None
    reason: str | None = None
    additional_metadata: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    created_at: datetime
