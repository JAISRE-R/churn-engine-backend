"""Transactional SQLite ledger for checkout finalization and offer redemption."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class CheckoutLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS finalized_checkouts (
                    order_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    request_signature TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS redeemed_offers (
                    customer_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    order_id TEXT NOT NULL REFERENCES finalized_checkouts(order_id),
                    redeemed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (customer_id, product_id),
                    UNIQUE (order_id, product_id)
                )
                """
            )
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_finalization(
        self, connection: sqlite3.Connection, order_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT customer_id, request_signature, response_json
            FROM finalized_checkouts
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    def get_redeemed_product_ids(
        self,
        customer_id: str,
        connection: sqlite3.Connection | None = None,
    ) -> set[str]:
        if connection is not None:
            rows = connection.execute(
                "SELECT product_id FROM redeemed_offers WHERE customer_id = ?",
                (customer_id,),
            ).fetchall()
            return {row["product_id"] for row in rows}

        read_connection = self._connect()
        try:
            rows = read_connection.execute(
                "SELECT product_id FROM redeemed_offers WHERE customer_id = ?",
                (customer_id,),
            ).fetchall()
        finally:
            read_connection.close()
        return {row["product_id"] for row in rows}

    def save_finalization(
        self,
        connection: sqlite3.Connection,
        *,
        order_id: str,
        customer_id: str,
        request_signature: str,
        response_json: str,
        redeemed_product_ids: set[str],
    ) -> None:
        connection.execute(
            """
            INSERT INTO finalized_checkouts
                (order_id, customer_id, request_signature, response_json)
            VALUES (?, ?, ?, ?)
            """,
            (order_id, customer_id, request_signature, response_json),
        )
        connection.executemany(
            """
            INSERT INTO redeemed_offers (customer_id, product_id, order_id)
            VALUES (?, ?, ?)
            """,
            [
                (customer_id, product_id, order_id)
                for product_id in sorted(redeemed_product_ids)
            ],
        )
