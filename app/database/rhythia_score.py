from typing import Optional, List, Dict
from datetime import datetime
from app.database._base import DatabaseModel
from sqlmodel import Field, Column, JSON, Relationship
from app.database.rhythia_beatmap import RhythiaBeatmap

class RhythiaScore(DatabaseModel, table=True):
    __tablename__ = "rhythia_scores"

    id: int = Field(default=None, primary_key=True)
    beatmap_id: int = Field(index=True, foreign_key="rhythia_beatmaps.id")
    beatmap: Optional[RhythiaBeatmap] = Relationship()
    user_id: int = Field(index=True)
    mode: str = Field(default="space")
    mode_int: int = Field(default=727)
    score: int = Field(default=0)
    max_combo: int = Field(default=0)
    accuracy: float = Field(default=0.0)
    rank: str = Field(default="F")
    mods: List[str] = Field(default=[], sa_column=Column(JSON))
    statistics: Dict[str, int] = Field(default={}, sa_column=Column(JSON))
    pp: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
