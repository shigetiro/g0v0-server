from typing import Optional
from datetime import datetime
from app.database._base import DatabaseModel
from sqlmodel import Field

class RhythiaBeatmap(DatabaseModel, table=True):
    __tablename__ = "rhythia_beatmaps"

    id: int = Field(default=None, primary_key=True)
    beatmapset_id: int = Field(default=0)
    difficulty_rating: float = Field(default=0.0)
    mode: str = Field(default="SPACE")
    version: str = Field(default="Standard")
    total_length: int = Field(default=0)
    bpm: float = Field(default=0.0)
    cs: float = Field(default=0.0)
    drain: float = Field(default=0.0)
    accuracy: float = Field(default=0.0)
    max_combo: int = Field(default=0)
    count_circles: int = Field(default=0)
    count_sliders: int = Field(default=0)
    count_spinners: int = Field(default=0)
    title: str = Field(default="Unknown")
    artist: str = Field(default="Unknown")
