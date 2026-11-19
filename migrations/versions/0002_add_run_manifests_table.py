"""add run_manifests table

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-14 23:34:38.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create run_manifests table to persist execution metadata and cryptographic hashes."""
    op.create_table(
        "run_manifests",
        sa.Column("run_id", sa.Text(), primary_key=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("git_commit", sa.Text(), nullable=False),
        sa.Column("engine", sa.Text(), nullable=False),
        sa.Column("quantization", sa.Text(), nullable=False),
        sa.Column("concurrency", sa.Integer(), nullable=False),
        sa.Column("manifest_sha256", sa.CHAR(64), nullable=False),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Drop run_manifests table."""
    op.drop_table("run_manifests")
