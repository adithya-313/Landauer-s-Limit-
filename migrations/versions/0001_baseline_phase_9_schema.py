"""baseline phase 9 schema

Revision ID: 0001
Revises:
Create Date: 2026-11-14 23:32:12.724297

"""

from pathlib import Path
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply baseline schema matching frozen Phase 9 specification.

    Purpose:
        Execute db/init/001_schema.sql to initialize telemetry_events and tenants tables.

    Key Logic:
        Resolves path to 001_schema.sql relative to migrations/ directory to ensure
        independent execution regardless of current working directory, and executes
        the SQL statements using op.execute().
        WHY: One source of truth, so the baseline can never drift from the frozen
        Phase 9 contract.
    """
    migrations_dir = Path(__file__).resolve().parent.parent
    sql_path = migrations_dir.parent / "db" / "init" / "001_schema.sql"
    sql_content = sql_path.read_text(encoding="utf-8")
    op.execute(sql_content)


def downgrade() -> None:
    """Drop baseline tables."""
    # Indexes (idx_telemetry_events_timestamp, idx_telemetry_events_engine_timestamp)
    # are dropped automatically when telemetry_events table is dropped.
    op.drop_table("tenants")
    op.drop_table("telemetry_events")
