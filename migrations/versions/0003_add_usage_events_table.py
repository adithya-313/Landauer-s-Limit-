"""add usage_events table

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-19 23:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create usage_events table to persist token and energy consumption per tenant."""
    op.create_table(
        "usage_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), primary_key=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("engine", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("energy_cost", postgresql.DOUBLE_PRECISION(), nullable=True),
        sa.Column("currency", sa.CHAR(3), nullable=True),
    )
    op.create_index(
        "idx_usage_events_tenant_created_at",
        "usage_events",
        ["tenant_id", "created_at"],
    )


def downgrade() -> None:
    """Drop usage_events index and table."""
    op.drop_index("idx_usage_events_tenant_created_at", table_name="usage_events")
    op.drop_table("usage_events")
