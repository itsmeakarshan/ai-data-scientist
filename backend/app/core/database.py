"""
AutoDS Database Connection & Session Management
Supports both Async Engine (FastAPI handlers) and Sync Engine (ML/Tools/Alembic)
with robust SQLite WAL mode, 60s busy timeout, and resilient concurrency handling.
"""

import sqlite3
import time
from typing import AsyncGenerator, Callable, TypeVar

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.core.config import settings
from backend.app.core.logging import logger


# Declarative Base for ORM Models
class Base(DeclarativeBase):
    pass


# -----------------------------------------------------------------------------
# Async Engine & Session (for FastAPI routes)
# -----------------------------------------------------------------------------
async_connect_args = (
    {"check_same_thread": False, "timeout": 60} if "sqlite" in settings.DATABASE_URL else {}
)

async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=async_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# -----------------------------------------------------------------------------
# Sync Engine & Session (for deterministic ML tools, background workers, scripts)
# -----------------------------------------------------------------------------
sync_url = settings.DATABASE_SYNC_URL
if sync_url.startswith("sqlite+aiosqlite"):
    sync_url = sync_url.replace("sqlite+aiosqlite", "sqlite")
elif sync_url.startswith("postgresql+asyncpg"):
    sync_url = sync_url.replace("postgresql+asyncpg", "postgresql")

sync_connect_args = (
    {"check_same_thread": False, "timeout": 60} if "sqlite" in sync_url else {}
)

sync_engine = create_engine(
    sync_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=sync_connect_args,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


# -----------------------------------------------------------------------------
# SQLite Connection PRAGMA Configuration (WAL Mode & Concurrency)
# -----------------------------------------------------------------------------
if "sqlite" in sync_url:
    @event.listens_for(sync_engine, "connect")
    def set_sqlite_sync_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.close()
        except Exception as e:
            logger.debug(f"Could not set SQLite sync PRAGMAs: {e}")

if "sqlite" in settings.DATABASE_URL:
    @event.listens_for(async_engine.sync_engine, "connect")
    def set_sqlite_async_pragma(dbapi_connection, connection_record):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.close()
        except Exception as e:
            logger.debug(f"Could not set SQLite async PRAGMAs: {e}")


# -----------------------------------------------------------------------------
# Session Lifecycle Dependencies
# -----------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an asynchronous database session.
    
    Safe session lifecycle: Mutating endpoints explicitly call await db.commit().
    Read-only requests are not forced to write-commit, preventing transaction locks.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# -----------------------------------------------------------------------------
# Resilient Retry Helper for SQLite Locking
# -----------------------------------------------------------------------------
T = TypeVar("T")


def with_db_retry(
    fn: Callable[[], T],
    max_retries: int = 5,
    initial_delay: float = 0.1,
    backoff: float = 2.0,
) -> T:
    """Execute a database callable with bounded exponential backoff on transient SQLite locking errors."""
    delay = initial_delay
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (OperationalError, DBAPIError, sqlite3.OperationalError) as err:
            err_msg = str(err).lower()
            if "locked" in err_msg or "busy" in err_msg:
                last_err = err
                logger.warning(
                    f"SQLite database busy/locked on attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
                delay *= backoff
            else:
                raise
        except Exception:
            raise
    if last_err:
        raise last_err
    raise RuntimeError("with_db_retry failed without capturing exception.")


# -----------------------------------------------------------------------------
# Database Initialization
# -----------------------------------------------------------------------------
async def init_db():
    """Create all database tables and enable WAL mode on initialization."""
    async with async_engine.begin() as conn:
        if "sqlite" in settings.DATABASE_URL:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=60000"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
        await conn.run_sync(Base.metadata.create_all)
