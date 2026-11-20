"""Entrypoint wrapper running Alembic migrations before starting gateway."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess

logger = logging.getLogger("start_gateway")


def run_migrations_if_configured() -> None:
    """Run Alembic migrations to upgrade the database schema to head if Postgres is configured.

    Purpose:
        Apply database schema migrations automatically on gateway container startup.

    Inputs/Outputs:
        Reads LANDAUER_PG_DSN environment variable. None returned.

    Key Logic:
        If LANDAUER_PG_DSN is non-empty, executes the Alembic CLI runner in the isolated
        migration virtualenv at /opt/migrate/bin/alembic. If unset or empty, logs a note and
        skips migration so that the gateway can run in environments without PostgreSQL
        (e.g. unit tests).

    Side Effects:
        Modifies database schema via Alembic. Raises FileNotFoundError if /opt/migrate/bin/alembic
        is missing, or subprocess.CalledProcessError on migration execution failure.
    """
    dsn = os.environ.get("LANDAUER_PG_DSN")
    if not dsn or not dsn.strip():
        logger.info("LANDAUER_PG_DSN is unset or empty; skipping Alembic migrations.")
        return

    alembic_bin = Path("/opt/migrate/bin/alembic")
    if not alembic_bin.is_file():
        raise FileNotFoundError(
            f"Alembic binary not found at '{alembic_bin}' with LANDAUER_PG_DSN configured"
        )

    logger.info("Running database migrations (alembic upgrade head)...")
    subprocess.run([str(alembic_bin), "upgrade", "head"], check=True, cwd="/app")


def main() -> None:
    """Run migrations and replace current process with Uvicorn gateway server."""
    logging.basicConfig(level=logging.INFO)
    run_migrations_if_configured()
    os.execvp("uvicorn", ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8080"])


if __name__ == "__main__":
    main()
