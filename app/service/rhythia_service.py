import logging
from typing import Any

from app.calculator import pre_fetch_and_calculate_pp

import httpx
from sqlmodel import select

logger = logging.getLogger(__name__)

class RhythiaService:
    BASE_URL = "https://production.rhythia.com/api"

    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30)

    async def search_beatmaps(
        self,
        page: int = 1,
        text_filter: str = "",
        author_filter: str = "",
        tags_filter: str = "",
        min_stars: float = 0,
        max_stars: float = 20,
        status: str = "RANKED"
    ) -> list[dict[str, Any]]:
        """
        Search for beatmaps on Rhythia API.
        """
        payload = {
            "page": page,
            "textFilter": text_filter,
            "authorFilter": author_filter,
            "tagsFilter": tags_filter,
            "minStars": min_stars,
            "maxStars": max_stars,
            "status": status,
            "session": "guest"  # Required by API, can be empty for public access
        }

        try:
            logger.info(f"Sending Rhythia search request to {self.BASE_URL}/getBeatmaps with payload: {payload}")
            response = await self.http_client.post(
                f"{self.BASE_URL}/getBeatmaps",
                json=payload
            )
            logger.info(f"Rhythia API response status: {response.status_code}")
            logger.info(f"Rhythia API response headers: {response.headers}")
            logger.info(f"Rhythia API response body: {response.text}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error fetching Rhythia maps: {e}")
            logger.error(f"Request payload: {payload}")
            # Try to get response info from the exception
            response_obj = getattr(e, "response", None)
            if response_obj is not None:
                logger.error(f"Response status: {response_obj.status_code}")
                logger.error(f"Response body: {response_obj.text}")
            raise

    async def get_beatmap_details(self, map_id: int) -> dict[str, Any]:
        """
        Get details for a specific beatmap by ID.
        """
        payload = {
            "id": map_id,
            "session": "guest"
        }

        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/getBeatmapPage",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error fetching Rhythia map details for {map_id}: {e}")
            raise

    async def register_beatmap(self, session: Any, beatmap_data: dict[str, Any], map_id: int | None = None) -> Any:
        from app.database.rhythia_beatmap import RhythiaBeatmap

        bid = map_id if map_id is not None else beatmap_data.get("id")
        if not bid:
            raise ValueError("Beatmap ID is required")

        # Check if exists
        stmt = select(RhythiaBeatmap).where(RhythiaBeatmap.id == bid)
        result = await session.exec(stmt)
        existing = result.first()

        if existing:
            return existing

        # Create new beatmap entry
        # Map fields from Rhythia API to RhythiaBeatmap
        beatmap = RhythiaBeatmap(
            id=bid,
            beatmapset_id=beatmap_data.get("setID", 0),
            difficulty_rating=float(beatmap_data.get("stars", 0) or 0),
            mode="SPACE",
            version=beatmap_data.get("difficulty", "Standard"),
            total_length=int(beatmap_data.get("length", 0) or 0),
            bpm=float(beatmap_data.get("bpm", 0) or 0),
            max_combo=int(beatmap_data.get("maxCombo", 0) or 0),
            count_circles=int(beatmap_data.get("notes", 0) or 0),
            title=beatmap_data.get("name", "Unknown"),
            artist=beatmap_data.get("author", "Unknown")
        )

        session.add(beatmap)
        await session.commit()
        await session.refresh(beatmap)
        return beatmap

    async def get_leaderboard(self, session: Any, map_id: int, mode: str = None, limit: int = 50) -> list[Any]:
        # Placeholder for leaderboard retrieval
        # In a real implementation, this would query RhythiaScore table
        try:
            from app.database.rhythia_score import RhythiaScore
            from sqlmodel import desc, select

            stmt = (
                select(RhythiaScore)
                .where(RhythiaScore.beatmap_id == map_id)
                .order_by(desc(RhythiaScore.score))
                .limit(limit)
            )

            if mode is not None:
                from app.models.score import GameMode
                try:
                    parsed = GameMode.parse(mode)
                    mode_int = int(parsed) if parsed is not None else 727
                except Exception:
                    mode_int = 727
                stmt = stmt.where(RhythiaScore.mode_int == mode_int)

            result = await session.exec(stmt)
            return result.all()
        except ImportError:
            return []

    async def submit_score(self, session: Any, map_id: int, user_id: int, score_data: dict[str, Any]) -> Any:
        """Submit a score to Rhythia."""
        try:
            from app.database.rhythia_score import RhythiaScore
            from app.models.score import GameMode

            # Debug: Log the incoming data
            logger.info(f"submit_score called with user_id: {user_id}, map_id: {map_id}")
            logger.info(f"score_data contents: {score_data}")

            # Validate incoming data
            if user_id == 0:
                logger.warning("submit_score received user_id=0, this will result in incorrect score attribution")

            # Extract values with detailed logging
            mode = score_data.get("mode", "osu")
            mode_int = score_data.get("mode_int", 0)
            score_value = score_data.get("score", 0)
            max_combo = score_data.get("max_combo", 0)
            accuracy = score_data.get("accuracy", 0.0)
            rank = score_data.get("rank", "F")
            mods = score_data.get("mods", [])
            statistics = score_data.get("statistics", {})
            pp = score_data.get("pp") or 0.0

            # Map long ruleset names to short database-compatible names
            if mode == "osuspaceruleset":
                mode = "space"
                logger.info("Mapped mode 'osuspaceruleset' to 'space' for database compatibility")

            # Validate mode_int for SPACE mode
            if mode == GameMode.SPACE and mode_int != 727:
                logger.warning(f"Mode SPACE should have mode_int=727, but got {mode_int}. Auto-correcting.")
                mode_int = 727

            logger.info(
                f"Extracted values - user_id: {user_id}, mode: {mode}, "
                f"mode_int: {mode_int}, score: {score_value}"
            )

            score = RhythiaScore(
                beatmap_id=map_id,
                user_id=user_id,
                mode=mode,
                mode_int=mode_int,
                score=score_value,
                max_combo=max_combo,
                accuracy=accuracy,
                rank=rank,
                mods=mods,
                statistics=statistics,
                pp=pp,
                created_at=score_data.get("created_at"),
                updated_at=score_data.get("updated_at")
            )
            logger.info(
                f"Created RhythiaScore with user_id: {score.user_id}, "
                f"mode_int: {score.mode_int}, score: {score.score}"
            )
            session.add(score)
            await session.commit()
            await session.refresh(score)
            return score
        except ImportError:
            raise NotImplementedError("RhythiaScore model not available")

    async def process_rhythia_score_pp(self, score: Any, session: Any, redis: Any, fetcher: Any) -> None:
        """Process PP calculation for Rhythia scores, similar to regular scores."""
        try:
            from app.models.score import GameMode
            from app.database.rhythia_beatmap import RhythiaBeatmap

            if score.pp != 0:
                logger.debug(
                    f"Skipping PP calculation for Rhythia score {score.id} | already set {score.pp:.2f}"
                )
                return

            # Check if score passed and can get PP
            # For Rhythia, we assume all scores are ranked and can get PP
            passed = score.rank != "F"  # Assuming F rank means failed
            if not passed:
                logger.debug(
                    f"Skipping PP calculation for Rhythia score {score.id} | passed={passed}"
                )
                return

            # Convert Rhythia score to regular Score format for PP calculation
            # Create a temporary Score object with the necessary fields
            from app.database.score import Score

            # Get statistics from RhythiaScore or use defaults
            stats = score.statistics or {}
            logger.info(f"Rhythia score statistics: {stats}")

            # Convert Rhythia statistics (perfect/miss) to osu! format
            perfect = stats.get("perfect", 0)
            miss = stats.get("miss", 0)

            # Simple conversion: Rhythia "perfect" -> osu! 300, "miss" -> osu! miss
            # This is the most straightforward mapping given limited Rhythia data
            n300 = perfect
            nmiss = miss
            n100 = 0  # Rhythia doesn't distinguish between 300/100/50
            n50 = 0   # Rhythia doesn't distinguish between 300/100/50
            ngeki = 0  # Rhythia doesn't have geki
            nkatu = 0  # Rhythia doesn't have katu

            logger.info(f"Simple Rhythia->osu! conversion: n300={n300}, n100={n100}, n50={n50}, nmiss={nmiss}, perfect={perfect}, miss={miss}")

            temp_score = Score(
                id=score.id,
                beatmap_id=score.beatmap_id,
                user_id=score.user_id,
                gamemode=GameMode.SPACE,  # Rhythia uses SPACE mode (mode_int=727)
                accuracy=score.accuracy,
                max_combo=score.max_combo,
                mods=score.mods,
                passed=passed,
                total_score=score.score,
                ranked=True,  # Rhythia scores are assumed to be ranked
                pp=0,  # Start with 0 PP
                # Add statistics fields required for PP calculation
                n300=n300,
                n100=n100,
                n50=n50,
                nmiss=nmiss,
                ngeki=ngeki,
                nkatu=nkatu,
                nlarge_tick_hit=0,  # Rhythia doesn't have these
                nlarge_tick_miss=0,
                nsmall_tick_hit=0,
                nslider_tail_hit=0,
            )

            # Calculate PP using the existing calculator
            pp, success = await pre_fetch_and_calculate_pp(temp_score, session, redis, fetcher)

            if success:
                score.pp = pp
                session.add(score)
                await session.commit()
                logger.info(
                    f"Calculated PP for Rhythia score {score.id} | pp={pp:.2f}"
                )
            else:
                logger.warning(
                    f"Failed to calculate PP for Rhythia score {score.id}"
                )

        except Exception as e:
            logger.error(f"Error processing PP for Rhythia score {score.id}: {e}")
            logger.exception(e)

    async def get_user_best_scores(self, session: Any, user_id: int, limit: int = 50) -> list[Any]:
        try:
            from app.database.rhythia_score import RhythiaScore

            from sqlmodel import desc, select

            stmt = (
                select(RhythiaScore)
                .where(RhythiaScore.user_id == user_id)
                .order_by(desc(RhythiaScore.pp))
                .limit(limit)
            )
            result = await session.exec(stmt)
            return result.all()
        except ImportError:
            return []

    async def close(self):
        await self.http_client.aclose()

# Global instance
rhythia_service = RhythiaService()
