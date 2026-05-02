from __future__ import annotations

from datetime import date as dt_date
import random
from typing import TYPE_CHECKING

from app.database.beatmap import Beatmap, BeatmapRankStatus
from app.database.daily_challenge_model import DailyChallenge
from app.models.score import GameMode

from sqlalchemy import func, select
from sqlmodel import col

if TYPE_CHECKING:
    from sqlmodel.ext.asyncio.session import AsyncSession


class DailyChallengeService:
    @staticmethod
    async def get_random_beatmap(
        session: AsyncSession,
        ruleset_id: int | None = None,
        min_difficulty: float | None = None,
        max_difficulty: float | None = None,
        excluded_dates: list[dt_date] | None = None,
    ) -> Beatmap | None:
        valid_statuses = [
            BeatmapRankStatus.RANKED,
            BeatmapRankStatus.APPROVED,
            BeatmapRankStatus.QUALIFIED,
            BeatmapRankStatus.LOVED,
        ]

        query = select(Beatmap).where(col(Beatmap.beatmap_status).in_(valid_statuses))

        if ruleset_id is not None:
            query = query.where(col(Beatmap.mode) == GameMode.from_int(ruleset_id))

        if min_difficulty is not None:
            query = query.where(Beatmap.difficulty_rating >= min_difficulty)

        if max_difficulty is not None:
            query = query.where(Beatmap.difficulty_rating <= max_difficulty)

        if excluded_dates:
            excluded_beatmap_ids = select(DailyChallenge.beatmap_id).where(
                col(DailyChallenge.date).in_(excluded_dates)
            )
            query = query.where(~col(Beatmap.id).in_(excluded_beatmap_ids))

        count_query = select(func.count()).select_from(query.subquery())
        total_count = (await session.exec(count_query)).one()[0]

        if total_count == 0:
            return None

        random_offset = random.randint(0, max(0, total_count - 1))
        query = query.offset(random_offset).limit(1)

        return (await session.exec(query)).scalars().first()

    @staticmethod
    async def get_random_beatmap_filtered(
        session: AsyncSession,
        ruleset_id: int | None = None,
        min_difficulty: float | None = None,
        max_difficulty: float | None = None,
    ) -> Beatmap | None:
        valid_statuses = [
            BeatmapRankStatus.RANKED,
            BeatmapRankStatus.APPROVED,
            BeatmapRankStatus.QUALIFIED,
            BeatmapRankStatus.LOVED,
        ]

        query = select(Beatmap).where(col(Beatmap.beatmap_status).in_(valid_statuses))

        if ruleset_id is not None:
            query = query.where(col(Beatmap.mode) == GameMode.from_int(ruleset_id))

        if min_difficulty is not None:
            query = query.where(Beatmap.difficulty_rating >= min_difficulty)

        if max_difficulty is not None:
            query = query.where(Beatmap.difficulty_rating <= max_difficulty)

        count_query = select(func.count()).select_from(query.subquery())
        total_count = (await session.exec(count_query)).one()[0]

        if total_count == 0:
            return None

        random_offset = random.randint(0, max(0, total_count - 1))
        query = query.offset(random_offset).limit(1)

        return (await session.exec(query)).scalars().first()
