from datetime import datetime
from typing import TYPE_CHECKING

from app.models.model import UTCBaseModel

from sqlmodel import (
    BigInteger,
    Column,
    DateTime,
    Field,
    ForeignKey,
    Relationship,
    SQLModel,
    Text,
    select,
)
from sqlmodel.ext.asyncio.session import AsyncSession

if TYPE_CHECKING:
    from .user import User


class UserBadgeBase(SQLModel, UTCBaseModel):
    description: str = Field(sa_column=Column(Text, nullable=False))
    image_url: str = Field(sa_column=Column(Text, nullable=False))
    image_2x_url: str = Field(sa_column=Column(Text, nullable=False))
    url: str = Field(sa_column=Column(Text, nullable=False))
    awarded_at: datetime = Field(sa_column=Column(DateTime, nullable=False))


class UserBadge(UserBadgeBase, table=True):
    __tablename__: str = "user_badges"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int | None = Field(
        default=None,
        sa_column=Column(
            BigInteger,
            ForeignKey("lazer_users.id"),
            nullable=True,
            index=True,
        ),
    )
    user: "User" = Relationship(back_populates="user_badges")


class UserBadgeCreate(SQLModel):
    description: str
    image_url: str
    image_2x_url: str | None = None
    url: str | None = None
    awarded_at: datetime | None = None
    user_id: int | None = None


class UserBadgeUpdate(SQLModel):
    description: str | None = None
    image_url: str | None = None
    image_2x_url: str | None = None
    url: str | None = None
    awarded_at: datetime | None = None
    user_id: int | None = None


class UserBadgeResponse(UserBadgeBase):
    id: int
    user_id: int | None = None