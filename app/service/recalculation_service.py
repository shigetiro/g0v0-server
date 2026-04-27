"""
Recalculation service with concurrency prevention and status checking.
"""

from typing import Any

from sqlalchemy import select, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from app.database.system_settings import RecalculationTask, RecalculationType, RecalculationStatus
from app.log import logger
from datetime import datetime, timedelta


async def is_recalculation_running(session: AsyncSession, task_type: RecalculationType, target_id: int | None = None) -> bool:
    """
    Check if a recalculation is currently running.

    Args:
        session: Database session
        task_type: Type of recalculation
        target_id: Target ID (user_id or beatmap_id)

    Returns:
        True if a task is currently running
    """
    query = select(RecalculationTask).where(
        RecalculationTask.status == RecalculationStatus.RUNNING
    )

    # Check specific task type
    if target_id is not None and task_type != RecalculationType.OVERALL:
        query = query.where(
            RecalculationTask.task_type == task_type,
            RecalculationTask.target_id == target_id
        )
    else:
        query = query.where(RecalculationTask.task_type == task_type)

    result = await session.exec(query)
    return result.first() is not None


async def has_pending_or_running_task(session: AsyncSession, task_type: RecalculationType, target_id: int | None = None) -> bool:
    """
    Check if there's a pending or running task for the given type and target.

    Args:
        session: Database session
        task_type: Type of recalculation
        target_id: Target ID (user_id or beatmap_id)

    Returns:
        True if a task exists
    """
    query = select(RecalculationTask).where(
        or_(
            RecalculationTask.status == RecalculationStatus.PENDING,
            RecalculationTask.status == RecalculationStatus.RUNNING,
        )
    )

    if target_id is not None and task_type != RecalculationType.OVERALL:
        query = query.where(
            RecalculationTask.task_type == task_type,
            RecalculationTask.target_id == target_id
        )
    else:
        query = query.where(RecalculationTask.task_type == task_type)

    result = await session.exec(query)
    return result.first() is not None


async def check_concurrent_limit(session: AsyncSession) -> bool:
    """
    Check if maximum concurrent tasks limit is reached.

    Args:
        session: Database session

    Returns:
        True if a task is currently running
    """
    query = select(RecalculationTask).where(
        RecalculationTask.status == RecalculationStatus.RUNNING
    )

    result = await session.exec(query)
    # Allow only 1 running task at a time for queueing
    return len(result.all()) >= 1


async def get_current_task_status(session: AsyncSession) -> dict[str, Any]:
    """
    Get the current status of all recalculation tasks.

    Args:
        session: Database session

    Returns:
        Dictionary with task status information
    """
    from sqlalchemy import func

    # Get running task
    running_query = select(RecalculationTask).where(
        RecalculationTask.status == RecalculationStatus.RUNNING
    ).order_by(RecalculationTask.started_at.desc())
    running_result = await session.exec(running_query)
    running_task = running_result.first()

    # Get pending count
    pending_count_result = await session.exec(
        select(func.count()).select_from(RecalculationTask).where(
            RecalculationTask.status == RecalculationStatus.PENDING
        )
    )
    pending_count = pending_count_result.scalar_one() or 0

    # Get running count
    running_count_result = await session.exec(
        select(func.count()).select_from(RecalculationTask).where(
            RecalculationTask.status == RecalculationStatus.RUNNING
        )
    )
    running_count = running_count_result.scalar_one() or 0

    # Get completed count from last 24 hours
    completed_count_result = await session.exec(
        select(func.count()).select_from(RecalculationTask).where(
            RecalculationTask.status == RecalculationStatus.COMPLETED,
            RecalculationTask.completed_at >= datetime.utcnow() - timedelta(hours=24)
        )
    )
    completed_count = completed_count_result.scalar_one() or 0

    status = {
        "running": bool(running_task),
        "running_task": None,
        "running_count": running_count,
        "pending_count": pending_count,
        "completed_24h": completed_count,
        "queue_size": pending_count + running_count,
    }

    if running_task is not None:
        # Handle both Row and model instance cases
        task_dict = running_task._asdict() if hasattr(running_task, '_asdict') else running_task.__dict__
        if hasattr(running_task, '_asdict'):
            # It's a Row, access via dict
            status["running_task"] = {
                "id": task_dict.get('id'),
                "type": task_dict.get('task_type', '').value if task_dict.get('task_type') else None,
                "target_id": task_dict.get('target_id'),
                "progress": task_dict.get('progress') or 0.0,
                "started_at": task_dict.get('started_at').isoformat() if task_dict.get('started_at') else None,
            }
        else:
            # It's a model instance
            status["running_task"] = {
                "id": running_task.id,
                "type": running_task.task_type.value if running_task.task_type else None,
                "target_id": running_task.target_id,
                "progress": running_task.progress or 0.0,
                "started_at": running_task.started_at.isoformat() if running_task.started_at else None,
            }

    return status
