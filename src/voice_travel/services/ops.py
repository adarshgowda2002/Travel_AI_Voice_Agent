"""Agent task + audit logging."""
from typing import Any

from sqlalchemy import select

from voice_travel.infra.postgres import AgentTask, AuditLog, Database
from voice_travel.services.errors import service_call
from voice_travel.utils import new_id, utcnow


class OpsLogService:
    def __init__(self, db: Database) -> None:
        self._db = db

    @service_call
    def log_agent_task(self, session_id: str, agent: str, tool: str, request: dict[str, Any],
                       result: dict[str, Any], status: str = "success") -> AgentTask:
        task = AgentTask(id=new_id("AT"), session_id=session_id, agent=agent, tool=tool, status=status,
                         request=request, result=result, created_at=utcnow())
        with self._db.session() as s:
            s.add(task)
        return task

    @service_call
    def log_event(self, event_type: str, user_id: str | None, details: dict[str, Any]) -> AuditLog:
        event = AuditLog(id=new_id("AL"), event_type=event_type, user_id=user_id, details=details,
                         timestamp=utcnow())
        with self._db.session() as s:
            s.add(event)
        return event

    @service_call
    def failed_tasks(self, limit: int = 20) -> list[AgentTask]:
        """Most recent failed tool calls (drives alerting; served by the partial index)."""
        with self._db.session() as s:
            q = (select(AgentTask).where(AgentTask.status == "failed")
                 .order_by(AgentTask.created_at.desc()).limit(limit))
            return list(s.scalars(q))
