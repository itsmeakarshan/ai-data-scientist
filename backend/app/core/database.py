"""
AutoDS Database Connection & Session Management
Supports both Async Engine (FastAPI handlers) and Sync Engine (ML/Tools/Alembic).
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from backend.app.core.config import settings


# Declarative Base for ORM Models
class Base(DeclarativeBase):
    pass


# Async Engine & Session (for FastAPI routes)
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# Sync Engine & Session (for deterministic ML tools, background workers, scripts)
# If using SQLite async URL, convert to sync for sync_engine
sync_url = settings.DATABASE_SYNC_URL
if sync_url.startswith("sqlite+aiosqlite"):
    sync_url = sync_url.replace("sqlite+aiosqlite", "sqlite")
elif sync_url.startswith("postgresql+asyncpg"):
    sync_url = sync_url.replace("postgresql+asyncpg", "postgresql")

sync_engine = create_engine(
    sync_url,
    echo=False,
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """Context manager / generator for synchronous database sessions in tools/ML tasks."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def init_db():
    """Create all database tables if they do not exist."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
