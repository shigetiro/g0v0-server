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


class RecalculationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecalculationType(str, Enum):
    USER = "user"
    BEATMAP = "beatmap"
    OVERALL = "overall"
    LEADERBOARD = "leaderboard"


class SystemSetting(SQLModel, table=True):
    __tablename__: str = "system_settings"

    id: int = Field(primary_key=True, index=True, default=None)
    key: str = Field(max_length=255, unique=True, index=True)
    value: str = Field(sa_column=Column(Text))
    value_type: str = Field(default="string", max_length=50)  # string, int, bool, json
    description: str | None = Field(max_length=512, default=None)
    updated_by: int | None = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="SET NULL")), default=None
    )
    updated_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)


class RecalculationTask(SQLModel, table=True):
    __tablename__: str = "recalculation_tasks"

    id: int = Field(primary_key=True, index=True, default=None)
    task_type: RecalculationType = Field(index=True)
    target_id: int | None = Field(sa_column=Column(BigInteger, index=True), default=None)  # user_id or beatmap_id
    status: RecalculationStatus = Field(default=RecalculationStatus.PENDING, index=True)
    priority: int = Field(default=0, index=True)  # Higher priority tasks run first
    progress: float = Field(default=0.0)  # 0.0 to 1.0
    total_items: int | None = Field(default=None)
    processed_items: int | None = Field(default=None)
    error_message: str | None = Field(sa_column=Column(Text), default=None)
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_by: int = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="CASCADE"), index=True)
    )
    created_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    updated_at: datetime = Field(
        sa_column=Column(DateTime, onupdate=datetime.utcnow),
        default_factory=datetime.utcnow
    )
    started_at: datetime | None = Field(sa_column=Column(DateTime), default=None)
    completed_at: datetime | None = Field(sa_column=Column(DateTime), default=None)


class RecalculationTaskCreate(SQLModel):
    task_type: RecalculationType
    target_id: int | None = None
    priority: int = 0


class RecalculationTaskResponse(SQLModel):
    id: int
    task_type: RecalculationType
    target_id: int | None = None
    status: RecalculationStatus
    priority: int
    progress: float
    total_items: int | None = None
    processed_items: int | None = None
    error_message: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    created_by: int
    created_by_username: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
