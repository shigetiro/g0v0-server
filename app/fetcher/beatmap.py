from app.database.beatmap import BeatmapDict, BeatmapModel
from app.log import fetcher_logger

from ._base import BaseFetcher

from pydantic import TypeAdapter

logger = fetcher_logger("BeatmapFetcher")
adapter = TypeAdapter(
    BeatmapModel.generate_typeddict(
        (
            "checksum",
            "accuracy",
            "ar",
            "bpm",
            "convert",
            "count_circles",
            "count_sliders",
            "count_spinners",
            "cs",
            "deleted_at",
            "drain",
            "hit_length",
            "is_scoreable",
            "last_updated",
            "mode_int",
            "ranked",
            "url",
            "max_combo",
            "beatmapset",
        )
    )
)


class BeatmapFetcher(BaseFetcher):
    async def get_beatmap(self, beatmap_id: int | None = None, beatmap_checksum: str | None = None) -> BeatmapDict:
        if beatmap_id:
            params = {"id": beatmap_id}
        elif beatmap_checksum:
            params = {"checksum": beatmap_checksum}
        else:
            raise ValueError("Either beatmap_id or beatmap_checksum must be provided.")
        logger.opt(colors=True).debug(f"get_beatmap: <y>{params}</y>")

        from httpx import HTTPStatusError

        try:
            return adapter.validate_python(  # pyright: ignore[reportReturnType]
                await self.request_api(
                    "https://osu.ppy.sh/api/v2/beatmaps/lookup",
                    params=params,
                )
            )
        except HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(
                    f"Beatmap {params} not found on osu.ppy.sh, trying mirrors..."
                )
                # Try mirrors if official API fails with 404
                # Note: Lookup by checksum might not be supported by all mirrors
                if beatmap_id:
                    mirrors = [
                        f"https://api.nerinyan.moe/beatmap/{beatmap_id}",
                        f"https://catboy.best/api/v2/b/{beatmap_id}",
                        f"https://storage.ripple.moe/api/b/{beatmap_id}",
                    ]
                    for mirror_url in mirrors:
                        try:
                            logger.debug(f"Trying mirror: {mirror_url}")
                            from httpx import AsyncClient

                            async with AsyncClient(timeout=10.0) as client:
                                resp = await client.get(mirror_url)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    try:
                                        return adapter.validate_python(data)
                                    except Exception as val_err:
                                        logger.warning(
                                            f"Mirror {mirror_url} returned incompatible data: {val_err}"
                                        )
                                        continue
                        except Exception as mirror_err:
                            logger.warning(
                                f"Failed to fetch from mirror {mirror_url}: {mirror_err}"
                            )
                            continue
            raise
