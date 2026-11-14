"""Alembic migration environment configured for Landauer's Limit."""

from logging.config import fileConfig
import os

from sqlalchemy import create_engine, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def get_db_url() -> str:
    """Read and format the database DSN from LANDAUER_PG_DSN environment variable.

    Purpose:
        Dynamically obtain PostgreSQL connection string from environment without
        storing sensitive credentials in alembic.ini or logging them.

    Outputs:
        str: Database URL configured with psycopg dialect (postgresql+psycopg://).

    Errors:
        RuntimeError: If LANDAUER_PG_DSN environment variable is missing or empty.
    """
    raw_dsn = os.environ.get("LANDAUER_PG_DSN")
    if not raw_dsn or not raw_dsn.strip():
        raise RuntimeError("LANDAUER_PG_DSN environment variable is missing or empty")

    dsn = raw_dsn.strip()
    if dsn.startswith("postgresql://"):
        dsn = "postgresql+psycopg://" + dsn[len("postgresql://") :]
    elif dsn.startswith("postgres://"):
        dsn = "postgresql+psycopg://" + dsn[len("postgres://") :]
    return dsn


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    raise NotImplementedError(
        "Offline migration mode is not supported; run in online mode with LANDAUER_PG_DSN"
    )


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    db_url = get_db_url()
    connectable = create_engine(db_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
