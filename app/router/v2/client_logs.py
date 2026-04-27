"""Client log submission endpoints for the game client to report logs."""

from typing import Annotated

from app.database import User
from app.database.client_log import ClientLog, ClientLogCreate, ClientLogType
from app.dependencies.database import Database
from app.dependencies.user import ClientUser, get_client_user
from app.log import log
from .router import router

from fastapi import Body, HTTPException, Security, status
from pydantic import BaseModel

logger = log("ClientLogs")


# Response model
class ClientLogSubmittedResponse(BaseModel):
    success: bool
    message: str


@router.post(
    "/logs/client",
    tags=["日志"],
    name="Submit client log",
    description="Submit a log entry from the game client (crashes, errors, performance logs, etc.)",
    response_model=ClientLogSubmittedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_client_log(
    db: Database,
    current_user: Annotated[User, Security(get_client_user)],
    log_data: Annotated[ClientLogCreate, Body(...)],
) -> ClientLogSubmittedResponse:
    """
    Submit a client log entry from the game client.

    Used for:
    - Crash reports
    - Error logs
    - Performance logs
    - General info logging

    The client version is automatically captured from the authenticated session.
    """
    try:
        # Create the log entry
        log_entry = ClientLog(
            user_id=current_user.id,
            username=current_user.username,
            user_avatar_url=current_user.avatar_url,
            client_version=log_data.client_version,
            client_hash=log_data.client_hash,
            os_version=log_data.os_version,
            log_type=log_data.log_type,
            message=log_data.message,
            stack_trace=log_data.stack_trace,
            client_metadata=log_data.metadata,
        )

        db.add(log_entry)
        await db.commit()
        await db.refresh(log_entry)

        logger.debug(
            f"Client log submitted: user={current_user.id}, type={log_data.log_type.value}, "
            f"version={log_data.client_version}"
        )

        return ClientLogSubmittedResponse(
            success=True,
            message="Log submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit client log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit log",
        )


@router.post(
    "/logs/client/batch",
    tags=["日志"],
    name="Submit batch client logs",
    description="Submit multiple log entries from the game client in a single request",
    response_model=ClientLogSubmittedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_client_logs_batch(
    db: Database,
    current_user: Annotated[User, Security(get_client_user)],
    logs: Annotated[list[ClientLogCreate], Body(...)],
) -> ClientLogSubmittedResponse:
    """
    Submit multiple client log entries in a single request.

    Useful for batching logs to reduce API calls.
    """
    try:
        log_entries = []
        for log_data in logs:
            log_entry = ClientLog(
                user_id=current_user.id,
                username=current_user.username,
                user_avatar_url=current_user.avatar_url,
                client_version=log_data.client_version,
                client_hash=log_data.client_hash,
                os_version=log_data.os_version,
                log_type=log_data.log_type,
                message=log_data.message,
                stack_trace=log_data.stack_trace,
                client_metadata=log_data.metadata,
            )
            log_entries.append(log_entry)

        db.add_all(log_entries)
        await db.commit()

        logger.debug(
            f"Batch client logs submitted: user={current_user.id}, count={len(logs)}"
        )

        return ClientLogSubmittedResponse(
            success=True,
            message=f"{len(logs)} logs submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit batch client logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit logs",
        )
