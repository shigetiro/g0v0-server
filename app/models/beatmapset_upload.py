from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

class BeatmapSubmissionTarget(str, Enum):
    WIP = "WIP"
    Pending = "Pending"

class PutBeatmapSetRequest(BaseModel):
    beatmapset_id: int | None = Field(None, description="Beatmapset ID if updating existing")
    beatmaps_to_create: int = Field(0, description="Number of new beatmaps to create")
    beatmaps_to_keep: list[int] = Field(default_factory=list, description="IDs of beatmaps to keep")
    target: BeatmapSubmissionTarget = Field(BeatmapSubmissionTarget.WIP)
    notify_on_discussion_replies: bool = Field(False)
    artist: str | None = Field(None)
    title: str | None = Field(None)

class BeatmapSetFile(BaseModel):
    filename: str
    sha2_hash: str

class PutBeatmapSetResponse(BaseModel):
    beatmapset_id: int
    beatmap_ids: list[int] = Field(default_factory=list)
    files: list[BeatmapSetFile] = Field(default_factory=list)
