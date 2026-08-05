"""Merge external HTTP channel and tenant backfill migration heads.

Revision ID: merge_external_http_tenant
Revises: merge_external_http_with_main, f060_tenant_id_backfill
Create Date: 2026-08-05 17:00:00
"""

from __future__ import annotations

from collections.abc import Sequence


revision: str = "merge_external_http_tenant"
down_revision: tuple[str, str] = (
    "merge_external_http_with_main",
    "f060_tenant_id_backfill",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Merge independent migration branches; no schema change is required."""


def downgrade() -> None:
    """Split migration branches; no schema change is required."""
