from app.database.session import engine
from app.database.base import Base
from app.database import models  # noqa: F401 - registers all models



async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
