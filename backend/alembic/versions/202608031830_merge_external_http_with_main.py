"""Merge external HTTP channel migration with current main.

Revision ID: merge_external_http_with_main
Revises: add_external_http_channel, allow_checkpoint_deliveries
Create Date: 2026-08-03 18:30:00
"""

from __future__ import annotations

from collections.abc import Sequence


revision: str = "merge_external_http_with_main"
down_revision: tuple[str, str] = (
    "add_external_http_channel",
    "allow_checkpoint_deliveries",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Merge two independent migration branches; no schema change is required."""


def downgrade() -> None:
    """Split migration branches; no schema change is required."""
