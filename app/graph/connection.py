"""Neo4j driver management — exposes both sync and async drivers.

Sync driver  → startup tasks (schema creation, APOC detection).
Async driver → FastAPI request handlers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from neo4j import AsyncGraphDatabase, AsyncSession, GraphDatabase, Session

from app.config import settings

logger = logging.getLogger(__name__)

# ── Singleton drivers ──────────────────────────────────────────────────────────

_sync_driver = None
_async_driver = None


def get_sync_driver():
    """Return (and lazily create) the synchronous Neo4j driver."""
    global _sync_driver
    if _sync_driver is None:
        _sync_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        _sync_driver.verify_connectivity()
        logger.info("Sync Neo4j driver connected to %s", settings.neo4j_uri)
    return _sync_driver


def get_async_driver():
    """Return (and lazily create) the asynchronous Neo4j driver."""
    global _async_driver
    if _async_driver is None:
        _async_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Async Neo4j driver created for %s", settings.neo4j_uri)
    return _async_driver


# ── Session helpers ────────────────────────────────────────────────────────────


@contextmanager
def sync_session() -> Generator[Session, None, None]:
    """Yield a synchronous Neo4j session."""
    driver = get_sync_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an asynchronous Neo4j session."""
    driver = get_async_driver()
    session = driver.session()
    try:
        yield session
    finally:
        await session.close()


# ── APOC detection ─────────────────────────────────────────────────────────────

_apoc_extended_available: bool | None = None


def detect_apoc_extended() -> bool:
    """Check whether APOC Extended procedures are available."""
    global _apoc_extended_available
    if _apoc_extended_available is not None:
        return _apoc_extended_available

    try:
        with sync_session() as session:
            result = session.run(
                "CALL apoc.help('apoc.neighbors.byhop') YIELD name RETURN count(name) AS cnt"
            )
            record = result.single()
            _apoc_extended_available = record is not None and record["cnt"] > 0
    except Exception:
        _apoc_extended_available = False

    logger.info("APOC Extended available: %s", _apoc_extended_available)
    return _apoc_extended_available


def has_apoc_extended() -> bool:
    """Return cached APOC Extended availability flag."""
    if _apoc_extended_available is None:
        return detect_apoc_extended()
    return _apoc_extended_available


# ── Cleanup ────────────────────────────────────────────────────────────────────


async def close_drivers() -> None:
    """Close both drivers — call on app shutdown."""
    global _sync_driver, _async_driver
    if _async_driver is not None:
        await _async_driver.close()
        _async_driver = None
    if _sync_driver is not None:
        _sync_driver.close()
        _sync_driver = None
    logger.info("Neo4j drivers closed.")
