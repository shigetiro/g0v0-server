from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from app.models.model import UTCBaseModel

from sqlmodel import (
    Column,
    Date,
    Field,
    Integer,
    SQLModel,
    Text,
    select,
)
from sqlmodel.ext.asyncio.session import AsyncSession

if TYPE_CHECKING:
    from .beatmap import Beatmap


class DailyChallengeBase(SQLModel):
    date: datetime = Field(sa_column=Column(Date, nullable=False, unique=True))
    beatmap_id: int = Field(sa_column=Column(Integer, nullable=False))
    ruleset_id: int = Field(sa_column=Column(Integer, nullable=False))
    required_mods: str = Field(sa_column=Column(Text, nullable=False, default='[]'))
    allowed_mods: str = Field(sa_column=Column(Text, nullable=False, default='[]'))
    room_id: int | None = Field(default=None, sa_column=Column(Integer, nullable=True, unique=True))
    max_attempts: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    time_limit: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))


class DailyChallenge(DailyChallengeBase, table=True):
    __tablename__: str = "daily_challenges"

    id: int | None = Field(default=None, primary_key=True)


class DailyChallengeCreate(SQLModel):
    date: str
    beatmap_id: int
    ruleset_id: int
    required_mods: str = "[]"
    allowed_mods: str = "[]"
    room_id: int | None = Field(default=None)
    max_attempts: int | None = Field(default=None)
    time_limit: int | None = Field(default=None)


class DailyChallengeUpdate(SQLModel):
    beatmap_id: int | None = Field(default=None)
    ruleset_id: int | None = Field(default=None)
    required_mods: str | None = Field(default=None)
    allowed_mods: str | None = Field(default=None)
    room_id: int | None = Field(default=None)
    max_attempts: int | None = Field(default=None)
    time_limit: int | None = Field(default=None)


class DailyChallengeResponse(DailyChallengeBase):
    id: int
    beatmap: dict | None = None
