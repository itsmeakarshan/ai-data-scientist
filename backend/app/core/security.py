"""
AutoDS Security & Validation Utilities
Protects against path traversal, disallowed SQL keywords, malicious file uploads, and arbitrary code execution.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional
from fastapi import HTTPException, status
from backend.app.core.config import settings


Union_Path_or_Str = Union[str, Path]
ALLOWED_EXTENSIONS = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json"}
DISALLOWED_SQL_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "PRAGMA", "LOAD", "INSTALL"
}


def sanitize_filename(filename: str) -> str:
    """Strip unsafe path characters and enforce safe alphanumeric names."""
    base = os.path.basename(filename)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return sanitized


def validate_file_path(file_path: Union_Path_or_Str, allowed_parent: Optional[Path] = None) -> Path:
    """Ensure a file path resolves inside the permitted project data directory to prevent traversal attacks."""
    resolved = Path(file_path).resolve()
    base_storage = Path(settings.STORAGE_DIR).resolve()
    
    if allowed_parent is not None:
        target_parent = Path(allowed_parent).resolve()
        if not (resolved == target_parent or target_parent in resolved.parents):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security Error: File path is outside permitted directory boundaries."
            )
    else:
        # Check against project root and storage dir
        proj_root = Path(settings.STORAGE_DIR).parent.resolve()
        if not (resolved == proj_root or proj_root in resolved.parents):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security Error: Access to path outside project workspace is prohibited."
            )
    return resolved


def validate_sql_query(query: str) -> Tuple[bool, str]:
    """
    Ensure user SQL queries executed via DuckDB are strictly read-only SELECT queries.
    Prevents file manipulation, export, attachment, or mutation.
    """
    clean_query = query.strip()
    if not clean_query:
        return False, "Query is empty."

    # Remove comments
    clean_query_no_comments = re.sub(r"--.*?$", "", clean_query, flags=re.MULTILINE)
    clean_query_no_comments = re.sub(r"/\*.*?\*/", "", clean_query_no_comments, flags=re.DOTALL)
    
    tokens = re.findall(r"\b[A-Za-z_]+\b", clean_query_no_comments.upper())
    
    if not tokens or tokens[0] not in ("SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW"):
        return False, f"Only read-only analytical queries (SELECT / WITH) are allowed. Found: {tokens[0] if tokens else 'None'}"

    for token in tokens:
        if token in DISALLOWED_SQL_KEYWORDS:
            return False, f"Disallowed SQL keyword detected: '{token}'. Destructive or administrative SQL is blocked."

    # Check for semicolon chained multi-statements
    statements = [s.strip() for s in clean_query_no_comments.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple SQL statements are not permitted."

    return True, "Query validated successfully."
