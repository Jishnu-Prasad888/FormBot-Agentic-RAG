from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings


def _connect_args(url: str) -> dict:
    # SQLite needs check_same_thread; Postgres/others do not.
    return {"check_same_thread": False} if url.startswith("sqlite") else {}


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    connect_args=_connect_args(settings.DATABASE_URL),
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)
