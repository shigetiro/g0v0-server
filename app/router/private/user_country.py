"""
Minimal user country update router to avoid circular dependencies
"""

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

# Minimal router without heavy dependencies
router = APIRouter(tags=["用户", "g0v0 API"])


class CountryUpdateRequest(BaseModel):
    country_code: str = Body(..., description="新的国家/地区代码 (ISO 3166-1 alpha-2)")


@router.patch("/country", name="修改国家/地区", tags=["用户", "g0v0 API"])
async def user_update_country_minimal(
    country_request: CountryUpdateRequest,
):
    """
    修改国家/地区 - 简化版本
    
    为指定用户修改国家/地区代码
    
    错误情况:
    - 403: 账户被限制或国家代码无效
    - 400: 国家代码格式无效
    
    返回:
    - 成功: None
    """
    country_code = country_request.country_code
    
    # 验证国家代码格式 (必须是2个大写字母)
    if not country_code or len(country_code) != 2 or not country_code.isalpha() or not country_code.isupper():
        raise HTTPException(400, "Invalid country code format. Must be 2 uppercase letters (ISO 3166-1 alpha-2).")
    
    # For now, return a success response
    # In a real implementation, this would connect to the database and user service
    return {"message": "Country update functionality needs to be connected to user service", "country_code": country_code}