"""
Databricks SQL connector wrapper.

Provides a context-manager-friendly client for executing SQL queries
against the Gold Layer Delta Lake tables.
"""

from __future__ import annotations

import contextlib
from typing import Any, Generator

import pandas as pd
from databricks import sql as databricks_sql

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabricksClient:
    """
    Thin wrapper around the Databricks SQL connector.

    Usage:
        client = DatabricksClient()
        df = client.query("SELECT * FROM gold.dim_books LIMIT 5")

    Or as a context manager:
        with DatabricksClient() as client:
            df = client.query(sql)
    """

    def __init__(self) -> None:
        self._cfg = settings.databricks
        self._connection = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        logger.info("Connecting to Databricks SQL warehouse …")
        self._connection = databricks_sql.connect(
            server_hostname=self._cfg.server_hostname,
            http_path=self._cfg.http_path,
            access_token=self._cfg.access_token,
        )
        logger.info("Databricks connection established.")

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Databricks connection closed.")

    def __enter__(self) -> "DatabricksClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """
        Execute a SQL statement and return the result as a pandas DataFrame.

        Args:
            sql:    SQL string. Use %(key)s for named parameters.
            params: Optional dict of named parameters.

        Returns:
            pd.DataFrame with query results.
        """
        if not self._connection:
            raise RuntimeError(
                "Not connected. Call .connect() or use as a context manager."
            )

        logger.debug("Executing SQL:\n%s", sql.strip())

        with self._connection.cursor() as cursor:
            cursor.execute(sql, params or {})
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=columns)
        logger.debug("Query returned %d rows × %d columns.", len(df), len(df.columns))
        return df

    def qualified_table(self, table: str) -> str:
        """Return fully qualified table name: catalog.schema.table"""
        return f"{self._cfg.catalog}.{self._cfg.schema}.{table}"
