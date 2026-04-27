from app.dependencies.rate_limit import LIMITERS

from fastapi import APIRouter

router = APIRouter(prefix="/api/private", dependencies=LIMITERS)

# 导入并包含子路由
from .admin import router as admin_router
from .audio_proxy import router as audio_proxy_router

# Include admin router with explicit prefix
router.include_router(admin_router, prefix="")
router.include_router(audio_proxy_router)
