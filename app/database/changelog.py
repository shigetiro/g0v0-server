from datetime import datetime
from enum import Enum
from typing import Any

from sqlmodel import (
    JSON,
    Column,
    DateTime,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    Text,
)


class ChangelogStream(SQLModel, table=True):
    __tablename__: str = "changelog_streams"

    id: int = Field(primary_key=True, index=True, default=None)
    name: str = Field(max_length=50, unique=True, index=True)
    display_name: str = Field(max_length=100)
    is_featured: bool = Field(default=False)
    user_count: int = Field(default=0)
    created_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    updated_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)


class ChangelogBuild(SQLModel, table=True):
    __tablename__: str = "changelog_builds"

    id: int = Field(primary_key=True, index=True, default=None)
    stream_id: int = Field(
        sa_column=Column(Integer, ForeignKey("changelog_streams.id"), index=True)
    )
    version: str = Field(max_length=50)
    display_version: str = Field(max_length=100)
    users: int = Field(default=0)
    github_url: str | None = Field(max_length=500, default=None)
    created_at: datetime = Field(sa_column=Column(DateTime, index=True), default_factory=datetime.utcnow)
    updated_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)

    entries: list["ChangelogEntry"] = Field(default_factory=list, sa_column=Column(JSON), alias="entries")
    update_stream: "ChangelogStream" = Field(default=None, sa_column=Column(JSON), alias="update_stream")

    class Config:
        arbitrary_types_allowed = True


class ChangeType(str, Enum):
    ADD = "add"
    FIX = "fix"
    MISC = "misc"
    REMOVE = "remove"


class ChangelogCategory(str, Enum):
    CLIENT = "client"
    UI = "ui"
    PP = "pp"
    NETWORK = "network"
    TOOLBAR = "toolbar"
    DOWNLOAD = "download"
    SERVER = "server"
    OTHER = "other"


class ChangelogEntry(SQLModel, table=True):
    __tablename__: str = "changelog_entries"

    id: int = Field(primary_key=True, index=True, default=None)
    build_id: int = Field(
        sa_column=Column(Integer, ForeignKey("changelog_builds.id"), index=True)
    )
    repository: str = Field(max_length=100, default="torii-osu")
    github_pull_request_id: int | None = Field(default=None)
    github_url: str = Field(max_length=500, default="https://github.com/shikkesora")
    url: str | None = Field(max_length=500, default=None)
    type: str = Field(default="misc", max_length=20, index=True)
    category: str = Field(default="other", max_length=20, index=True)
    title: str = Field(max_length=500)
    message_html: str = Field(sa_column=Column(Text), default="")
    major: bool = Field(default=False)
    created_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    updated_at: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.utcnow)
    github_user: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    class Config:
        arbitrary_types_allowed = True


class ChangelogBuildCreate(SQLModel):
    stream_id: int
    version: str
    display_version: str
    users: int = 0
    created_at: datetime | None = None
    github_url: str | None = None


class ChangelogBuildUpdate(SQLModel):
    version: str | None = None
    display_version: str | None = None
    users: int | None = None


class ChangelogEntryCreate(SQLModel):
    build_id: int
    repository: str = "torii-osu"
    github_pull_request_id: int | None = None
    github_url: str | None = None
    url: str | None = None
    type: str = "misc"
    category: str = "other"
    title: str
    message_html: str = ""
    major: bool = False
    github_user: dict[str, Any] | None = None


class ChangelogEntryUpdate(SQLModel):
    repository: str | None = None
    github_pull_request_id: int | None = None
    github_url: str | None = None
    url: str | None = None
    type: ChangeType | None = None
    category: ChangelogCategory | None = None
    title: str | None = None
    message_html: str | None = None
    major: bool | None = None
    github_user: dict[str, Any] | None = None


class ChangelogStreamCreate(SQLModel):
    name: str
    display_name: str
    is_featured: bool = False
    user_count: int = 0


class ChangelogStreamUpdate(SQLModel):
    name: str | None = None
    display_name: str | None = None
    is_featured: bool | None = None
    user_count: int | None = None
