"""
Recalculation task worker that processes pending recalculation tasks from the database.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import select

from app.database.system_settings import RecalculationTask, RecalculationStatus, RecalculationType
from app.dependencies.database import with_db
from app.dependencies.scheduler import get_scheduler
from app.log import logger


# Check if running on Windows
_IS_WINDOWS = sys.platform == "win32"


# Track running tasks to prevent duplicates
_running_tasks: dict[str, asyncio.Task] = {}


def _get_task_key(task_type: RecalculationType, target_id: int | None) -> str:
    """Generate a unique key for a task based on its type and target."""
    if target_id is not None:
        return f"{task_type.value}_{target_id}"
    return task_type.value


def _now() -> datetime:
    """Get current UTC datetime (timezone-naive for database compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _get_user_score_count(session: Any, user_id: int) -> int:
    """Get approximate score count for a user."""
    try:
        from app.database.score import Score
        result = await session.exec(
            select(Score).where(Score.user_id == user_id)
        )
        return len(result.all())
    except Exception as e:
        logger.warning(f"Failed to get score count for user {user_id}: {e}")
        return 100  # Default estimate


async def _get_total_user_count(session: Any) -> int:
    """Get total user count."""
    try:
        from app.database.user import User
        result = await session.exec(select(User))
        return len(result.all())
    except Exception as e:
        logger.warning(f"Failed to get user count: {e}")
        return 1000  # Default estimate


async def _execute_recalculation_task(task: RecalculationTask, session: Any) -> dict[str, Any]:
    """Execute a recalculation task using the recalculate.py script."""
    task_key = _get_task_key(task.task_type, task.target_id)

    # Build command arguments using absolute path
    base_dir = Path(__file__).parent.parent.parent
    recalculate_script = base_dir / "tools" / "recalculate.py"

    if not recalculate_script.exists():
        raise FileNotFoundError(f"Recalculate script not found: {recalculate_script}")

    cmd = [sys.executable, "-u", str(recalculate_script)]  # -u for unbuffered output

    if task.task_type == RecalculationType.USER:
        if task.target_id is None:
            raise ValueError("User recalculation task must have target_id")
        cmd.extend(["performance", "--user-id", str(task.target_id)])
        # Estimate items
        score_count = await _get_user_score_count(session, task.target_id)
        task.total_items = score_count or 1
        # Don't commit here - let caller handle it
    elif task.task_type == RecalculationType.BEATMAP:
        if task.target_id is None:
            raise ValueError("Beatmap recalculation task must have target_id")
        cmd.extend(["rating", "--beatmap-id", str(task.target_id)])
        task.total_items = 1
        # Don't commit here - let caller handle it
    elif task.task_type == RecalculationType.OVERALL:
        cmd.extend(["performance", "--all"])
        user_count = await _get_total_user_count(session)
        task.total_items = user_count
        # Don't commit here - let caller handle it
    elif task.task_type == RecalculationType.LEADERBOARD:
        cmd.extend(["leaderboard", "--all"])
        user_count = await _get_total_user_count(session)
        task.total_items = user_count
        # Don't commit here - let caller handle it
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")

    # Add timeout for the subprocess (2 hours for overall, 10 minutes for user, 5 minutes for beatmap)
    if task.task_type == RecalculationType.OVERALL:
        timeout_seconds = 7200
    elif task.task_type == RecalculationType.LEADERBOARD:
        timeout_seconds = 7200
    elif task.task_type == RecalculationType.USER:
        timeout_seconds = 600
    else:
        timeout_seconds = 300

    logger.info(f"Task {task.id} starting recalculation script...")

    # On Windows, use synchronous subprocess with DEVNULL to avoid pipe deadlock
    # On Unix, use asyncio subprocess with proper pipe handling
    if _IS_WINDOWS:
        logger.info(f"Task {task.id} using Windows-compatible subprocess mode")
        return await _execute_sync_subprocess(cmd, base_dir, timeout_seconds, task.id)

    return await _execute_async_subprocess(cmd, base_dir, timeout_seconds, task.id)


async def _execute_sync_subprocess(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    task_id: int
) -> dict[str, Any]:
    """Execute subprocess using sync subprocess in a thread pool (Windows-compatible)."""
    result = {"returncode": None, "stdout": "", "stderr": ""}

    def _run_subprocess():
        """Run subprocess in thread pool."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                text=True,
                errors='replace',
            )
            logger.info(f"Task {task_id} sync subprocess started, pid={process.pid}")
            try:
                _, stderr = process.communicate(timeout=timeout)
                logger.info(f"Task {task_id} sync subprocess completed, returncode={process.returncode}")
                return process.returncode, "", stderr
            except subprocess.TimeoutExpired:
                logger.error(f"Task {task_id} sync subprocess timed out")
                process.kill()
                process.wait()
                return -1, "", f"Process timed out after {timeout} seconds"
        except Exception as e:
            logger.exception(f"Task {task_id} sync subprocess error")
            return -1, "", str(e)

    try:
        loop = asyncio.get_running_loop()
        returncode, stdout, stderr = await asyncio.wait_for(
            loop.run_in_executor(None, _run_subprocess),
            timeout=timeout + 30,  # Extra buffer time
        )
        result["returncode"] = returncode
        result["stdout"] = stdout
        result["stderr"] = stderr
    except asyncio.TimeoutError:
        logger.error(f"Task {task_id} executor timed out")
        result["returncode"] = -1
        result["stderr"] = f"Execution timed out after {timeout + 30} seconds"
    except Exception as e:
        logger.exception(f"Task {task_id} executor failed: {e}")
        result["returncode"] = -1
        result["stderr"] = str(e)[:2000]

    return result


async def _execute_async_subprocess(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    task_id: int
) -> dict[str, Any]:
    """Execute subprocess using asyncio (Unix/Linux compatible)."""
    result = {"returncode": None, "stdout": "", "stderr": ""}
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    process: asyncio.subprocess.Process | None = None

    async def read_stream(stream, store_list: list[str], name: str, tid: int):
        """Read output stream."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                try:
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                except Exception:
                    decoded = line.decode('latin-1', errors='replace').rstrip()
                store_list.append(decoded)
                logger.debug(f"Task {tid} [{name}]: {decoded[:200]}")
        except Exception as e:
            logger.warning(f"Task {tid} stream reading error: {e}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        logger.info(f"Task {task_id} asyncio subprocess created, pid={process.pid}")

        stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_lines, "out", task_id))
        stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_lines, "err", task_id))

        logger.info(f"Task {task_id} waiting for completion (timeout={timeout}s)...")
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, process.wait()),
            timeout=timeout
        )
        result["returncode"] = process.returncode
        logger.info(f"Task {task_id} completed, returncode={process.returncode}")

        result["stdout"] = '\n'.join(stdout_lines)
        result["stderr"] = '\n'.join(stderr_lines)

    except asyncio.TimeoutError:
        logger.error(f"Task {task_id} timed out after {timeout}s")
        if process:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=10)
            except Exception:
                pass
        result["returncode"] = -1
        result["stderr"] = f"Process timed out after {timeout} seconds"
    except Exception as e:
        logger.exception(f"Task {task_id} subprocess execution failed: {e}")
        if process and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        result["returncode"] = -1
        result["stderr"] = str(e)[:2000]

    return result


async def _process_single_task(task_id: int) -> None:
    """Process a single recalculation task with its own database session."""
    async with with_db() as session:
        # Load the task from database fresh
        task = await session.get(RecalculationTask, task_id)
        if not task:
            logger.error(f"Task {task_id} not found in database")
            return

        task_key = _get_task_key(task.task_type, task.target_id)

        try:
            # Update task status to RUNNING
            task.status = RecalculationStatus.RUNNING
            task.started_at = _now()
            task.updated_at = _now()
            task.progress = 0.0
            task.processed_items = 0
            task.error_message = None
            await session.flush()
            await session.commit()
            await session.refresh(task)

            logger.info(f"Processing recalculation task {task.id} ({task_key})")

            # Execute the recalculation
            result = await _execute_recalculation_task(task, session)

            # Flush to save total_items set by _execute_recalculation_task
            await session.flush()
            await session.commit()

            # Reload task after subprocess execution
            await session.refresh(task)

            if result["returncode"] == 0:
                # Mark as completed
                task.status = RecalculationStatus.COMPLETED
                task.completed_at = _now()
                task.progress = 1.0
                task.processed_items = task.total_items or 1
                task.error_message = None
                task.result = {
                    "success": True,
                    "returncode": 0,
                }
                logger.success(f"Recalculation task {task.id} completed successfully")
            else:
                # Mark as failed
                task.status = RecalculationStatus.FAILED
                task.completed_at = _now()
                task.error_message = result["stderr"][-2000:] if result["stderr"] else f"Exit code {result['returncode']}"
                task.result = {
                    "success": False,
                    "error": task.error_message,
                    "returncode": result["returncode"],
                }
                logger.error(f"Recalculation task {task.id} failed with exit code {result['returncode']}")

        except Exception as e:
            logger.exception(f"Recalculation task {task.id} failed: {e}")
            task.status = RecalculationStatus.FAILED
            task.completed_at = _now()
            task.error_message = str(e)[:2000]
            task.result = {"success": False, "error": str(e)}

        finally:
            # Ensure changes are committed
            task.updated_at = _now()
            try:
                await session.flush()
                await session.commit()
                logger.info(
                    f"Task {task.id} finalized: "
                    f"status={task.status.value}, "
                    f"progress={task.progress:.1%}"
                )
            except Exception as commit_err:
                logger.error(f"Failed to commit task {task.id} status: {commit_err}")

            # Remove from running tasks
            _running_tasks.pop(task_key, None)


async def process_pending_recalculation_tasks() -> None:
    """Check for pending recalculation tasks and process them."""
    try:
        async with with_db() as session:
            # Check for any currently running tasks
            running_result = await session.exec(
                select(RecalculationTask).where(
                    RecalculationTask.status == RecalculationStatus.RUNNING
                )
            )
            running_tasks = running_result.all()
            running_count = len(running_tasks)

            if running_count >= 1:
                # Check if any running tasks have been stuck for too long
                now = _now()
                for task in running_tasks:
                    if task.started_at:
                        runtime = (now - task.started_at).total_seconds()
                        # If task has been running for >30 minutes, mark as failed
                        if runtime > 1800:
                            logger.warning(f"Task {task.id} has been running too long ({runtime}s), marking as failed")
                            task.status = RecalculationStatus.FAILED
                            task.error_message = f"Task timed out after running for {runtime} seconds"
                            task.completed_at = now
                            task.updated_at = now
                await session.commit()

                if running_count >= 1:
                    logger.debug(f"Found {running_count} running tasks, skipping new task processing")
                    return

            # Find the highest priority pending task
            pending_task = (
                await session.exec(
                    select(RecalculationTask)
                    .where(RecalculationTask.status == RecalculationStatus.PENDING)
                    .order_by(RecalculationTask.priority.desc())
                    .order_by(RecalculationTask.created_at)
                    .limit(1)
                )
            ).first()

            if not pending_task:
                return  # No pending tasks

            # Check if already running
            task_key = _get_task_key(pending_task.task_type, pending_task.target_id)
            if task_key in _running_tasks:
                logger.debug(f"Task {task_key} is already running, skipping duplicate")
                return

            # Start processing the task
            logger.info(f"Starting processing of recalculation task {pending_task.id}")
            task_coro = asyncio.create_task(_process_single_task(pending_task.id))
            _running_tasks[task_key] = task_coro

    except Exception as e:
        logger.exception(f"Error in process_pending_recalculation_tasks: {e}")


@get_scheduler().scheduled_job(IntervalTrigger(seconds=30), id="process_recalculation_tasks")
async def scheduled_recalculation_worker() -> None:
    """Scheduled job that runs every 30 seconds to process pending recalculation tasks."""
    await process_pending_recalculation_tasks()


async def reset_stuck_tasks() -> None:
    """Reset any RUNNING tasks to FAILED on startup (handles crashes/restarts)."""
    try:
        async with with_db() as session:
            # Find all stuck RUNNING tasks
            running_result = await session.exec(
                select(RecalculationTask).where(
                    RecalculationTask.status == RecalculationStatus.RUNNING
                )
            )
            stuck_tasks = running_result.all()

            if stuck_tasks:
                for task in stuck_tasks:
                    task.status = RecalculationStatus.FAILED
                    task.completed_at = _now()
                    task.updated_at = _now()
                    task.error_message = "Task was in RUNNING state when server restarted"
                    task.result = {"success": False, "error": "Server restarted while task was running"}
                    logger.warning(f"Reset stuck task {task.id} ({task.task_type.value} {task.target_id}) to FAILED")

                await session.commit()
                logger.info(f"Reset {len(stuck_tasks)} stuck tasks from previous session")
    except Exception as e:
        logger.exception(f"Failed to reset stuck tasks: {e}")


def start_recalculation_worker() -> None:
    """Start the recalculation worker scheduled job."""
    # Clear any stuck tasks from previous runs
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(reset_stuck_tasks())
        else:
            asyncio.run(reset_stuck_tasks())
    except Exception as e:
        logger.warning(f"Could not reset stuck tasks: {e}")

    scheduler = get_scheduler()
    if not scheduler.get_job("process_recalculation_tasks"):
        scheduler.add_job(
            scheduled_recalculation_worker,
            IntervalTrigger(seconds=30),
            id="process_recalculation_tasks",
            replace_existing=True,
        )
        logger.info("Recalculation worker scheduled job started (runs every 30 seconds)")


def stop_recalculation_worker() -> None:
    """Stop the recalculation worker scheduled job."""
    scheduler = get_scheduler()
    job = scheduler.get_job("process_recalculation_tasks")
    if job:
        scheduler.remove_job("process_recalculation_tasks")
        logger.info("Recalculation worker scheduled job stopped")
