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


class ClientLogType(str, Enum):
    CRASH = "CRASH"
    ERROR = "ERROR"
    WARNING = "WARNING"
    PERFORMANCE = "PERFORMANCE"
    INFO = "INFO"


class ClientLog(SQLModel, table=True):
    __tablename__: str = "client_logs"

    id: int = Field(primary_key=True, index=True, default=None)
    user_id: int | None = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id", ondelete="SET NULL"), index=True), default=None
    )
    username: str | None = Field(max_length=255, default=None, index=True)
    user_avatar_url: str | None = Field(max_length=512, default=None)
    client_version: str = Field(max_length=255, index=True)
    client_hash: str | None = Field(max_length=64, default=None, index=True)
    os_version: str | None = Field(max_length=255, default=None)
    log_type: ClientLogType = Field(default=ClientLogType.INFO, index=True)
    message: str = Field(sa_column=Column(Text))
    stack_trace: str | None = Field(sa_column=Column(Text), default=None)
    client_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    created_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)


class ClientLogCreate(SQLModel):
    user_id: int | None = None
    username: str | None = None
    user_avatar_url: str | None = None
    client_version: str
    client_hash: str | None = None
    os_version: str | None = None
    log_type: ClientLogType = ClientLogType.INFO
    message: str
    stack_trace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClientLogResponse(SQLModel):
    id: int
    user_id: int | None = None
    username: str | None = None
    user_avatar_url: str | None = None
    client_version: str
    client_hash: str | None = None
    os_version: str | None = None
    log_type: ClientLogType
    message: str
    stack_trace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
