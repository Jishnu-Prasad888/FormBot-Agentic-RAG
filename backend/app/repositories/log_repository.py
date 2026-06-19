from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.models import RetrievalLog, EvaluationRun, QueryLog



class LogRepository:
    async def create_retrieval_log(self, db: AsyncSession, data: dict) -> RetrievalLog:
        log = RetrievalLog(**data)
        db.add(log)
        await db.commit()
        await db.refresh(log)
        return log

    async def create_evaluation_run(self, db: AsyncSession, data: dict) -> EvaluationRun:
        run = EvaluationRun(**data)
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return run

    async def create_query_log(self, db: AsyncSession, data: dict) -> QueryLog:
        log = QueryLog(**data)
        db.add(log)
        await db.commit()
        await db.refresh(log)
        return log

    async def list_retrieval_logs(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[RetrievalLog]:
        result = await db.execute(
            select(RetrievalLog).offset(skip).limit(limit).order_by(RetrievalLog.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_evaluation_runs(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[EvaluationRun]:
        result = await db.execute(
            select(EvaluationRun).offset(skip).limit(limit).order_by(EvaluationRun.created_at.desc())
        )
        return list(result.scalars().all())


log_repo = LogRepository()
