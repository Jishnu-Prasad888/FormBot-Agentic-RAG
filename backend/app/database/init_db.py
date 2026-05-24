from app.database.session import engine
from app.database.base import Base
from app.database import models  # noqa: F401 - registers all models
from app.core.logging import get_logger

logger = get_logger("init_db")


async def init_db():
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")


async def drop_db():
    logger.warning("Dropping all database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All tables dropped.")
