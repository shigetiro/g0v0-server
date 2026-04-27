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


class AnnouncementType(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    EVENT = "event"
    MAINTENANCE = "maintenance"


class Announcement(SQLModel, table=True):
    __tablename__: str = "announcements"

    id: int = Field(primary_key=True, index=True, default=None)
    title: str = Field(max_length=255)
    content: str = Field(sa_column=Column(Text))
    type: AnnouncementType = Field(default=AnnouncementType.INFO, index=True)
    target_roles: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    start_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    end_at: datetime | None = Field(sa_column=Column(DateTime), default=None)
    is_active: bool = Field(default=True, index=True)
    is_pinned: bool = Field(default=False, index=True)
    show_in_client: bool = Field(default=True)
    show_on_website: bool = Field(default=True)
    created_by: int = Field(
        sa_column=Column(BigInteger, ForeignKey("lazer_users.id"), index=True)
    )
    created_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    updated_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    announcement_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class AnnouncementCreate(SQLModel):
    title: str
    content: str
    type: AnnouncementType = AnnouncementType.INFO
    target_roles: list[str] = Field(default_factory=list)
    start_at: datetime | None = None
    end_at: datetime | None = None
    is_active: bool = True
    is_pinned: bool = False
    show_in_client: bool = True
    show_on_website: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnouncementUpdate(SQLModel):
    title: str | None = None
    content: str | None = None
    type: AnnouncementType | None = None
    target_roles: list[str] | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    is_active: bool | None = None
    is_pinned: bool | None = None
    show_in_client: bool | None = None
    show_on_website: bool | None = None
    metadata: dict[str, Any] | None = None


class AnnouncementResponse(SQLModel):
    id: int
    title: str
    content: str
    type: AnnouncementType
    target_roles: list[str]
    start_at: datetime
    end_at: datetime | None
    is_active: bool
    is_pinned: bool
    show_in_client: bool
    show_on_website: bool
    created_by: int
    created_by_username: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
