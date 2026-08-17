"""
AutoDS Safe SQL Query Tool
Executes verified read-only analytical SQL queries over local CSV/Parquet datasets using DuckDB.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.security import validate_file_path, validate_sql_query


def execute_safe_sql_query(
    file_path: str,
    sql_query: str,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Safely execute a read-only analytical SQL query on a dataset file.
    The table is registered as 'dataset' and 'data'.
    """
    # 1. Security validation of file path
    valid_path = validate_file_path(file_path)
    
    # 2. Security validation of SQL query
    is_valid, error_msg = validate_sql_query(sql_query)
    if not is_valid:
        raise ValueError(f"Security Alert: Disallowed SQL query. {error_msg}")

    # Set row limit
    max_limit = min(limit or settings.SAFE_SQL_ROW_LIMIT, settings.SAFE_SQL_ROW_LIMIT)

    start_time = time.time()
    con = duckdb.connect(database=":memory:", read_only=False)
    
    try:
        # Load dataset into in-memory table
        file_ext = valid_path.suffix.lower()
        if file_ext in (".csv", ".txt"):
            con.execute(f"CREATE TABLE dataset AS SELECT * FROM read_csv_auto('{valid_path}', header=True);")
        elif file_ext in (".parquet", ".pq"):
            con.execute(f"CREATE TABLE dataset AS SELECT * FROM read_parquet('{valid_path}');")
        else:
            # Fallback via pandas
            import pandas as pd
            df = pd.read_excel(valid_path) if file_ext in (".xlsx", ".xls") else pd.read_json(valid_path)
            con.register("dataset", df)

        # Create alias 'data'
        con.execute("CREATE VIEW data AS SELECT * FROM dataset;")

        # Append LIMIT if not present
        executed_sql = sql_query.strip().rstrip(";")
        if "limit" not in executed_sql.lower():
            executed_sql = f"{executed_sql} LIMIT {max_limit}"

        result = con.execute(executed_sql)
        columns = [desc[0] for desc in result.description]
        raw_rows = result.fetchall()
        
        # Format as list of dicts
        rows = [dict(zip(columns, row)) for row in raw_rows]
        exec_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "execution_time_ms": exec_ms,
            "sql_executed": executed_sql,
        }
    finally:
        con.close()
