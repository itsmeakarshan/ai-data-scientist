"""
Security and Functional Tests for AutoDS Safe SQL Tool
"""

import pytest
from backend.app.core.security import validate_sql_query
from backend.app.tools.safe_query import execute_safe_sql_query


def test_sql_query_validator_valid():
    """Verify standard read-only SQL queries pass validation."""
    valid, msg = validate_sql_query("SELECT job_category, COUNT(*) FROM dataset GROUP BY job_category;")
    assert valid is True

    valid_with, msg = validate_sql_query("WITH cte AS (SELECT * FROM dataset) SELECT AVG(income) FROM cte;")
    assert valid_with is True


def test_sql_query_validator_blocked():
    """Verify destructive SQL statements are strictly rejected."""
    valid_drop, _ = validate_sql_query("DROP TABLE dataset;")
    assert valid_drop is False

    valid_delete, _ = validate_sql_query("DELETE FROM dataset WHERE age > 50;")
    assert valid_delete is False

    valid_update, _ = validate_sql_query("UPDATE dataset SET income = 0;")
    assert valid_update is False

    valid_attach, _ = validate_sql_query("ATTACH 'some_db.db' AS malicious;")
    assert valid_attach is False


def test_duckdb_execution(synthetic_csv_file):
    """Test real DuckDB query execution on synthetic dataset."""
    query = "SELECT job_category, COUNT(*) as count, AVG(income) as avg_inc FROM dataset GROUP BY job_category ORDER BY count DESC;"
    res = execute_safe_sql_query(str(synthetic_csv_file), query)

    assert "columns" in res
    assert "rows" in res
    assert res["row_count"] > 0
    assert "job_category" in res["columns"]
    assert res["execution_time_ms"] >= 0
