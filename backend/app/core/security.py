"""
AutoDS Security & Validation Utilities
Protects against path traversal, disallowed SQL keywords, malicious file uploads, and arbitrary code execution.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

from fastapi import HTTPException, status

from backend.app.core.config import settings
from backend.app.core.logging import logger

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
    """
    Ensure a file path resolves safely inside the permitted project data or workspace directory.
    Prevents path traversal attacks while supporting portable relative and absolute paths.
    """
    raw_str = str(file_path).strip()
    if not raw_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security Error: File path cannot be empty."
        )

    base_dir = Path(settings.BASE_DIR).resolve()
    storage_dir = Path(settings.STORAGE_DIR).resolve()
    raw_dir = Path(settings.data_raw_dir).resolve()

    # Determine permitted workspace roots
    allowed_roots: List[Path] = [storage_dir, base_dir]

    # In test environments, permit the temporary test directory
    if settings.ENVIRONMENT == "test" or os.environ.get("PYTEST_CURRENT_TEST"):
        allowed_roots.append(Path(tempfile.gettempdir()).resolve())
        allowed_roots.append(storage_dir)

    if allowed_parent is not None:
        target_parent = Path(allowed_parent).resolve()
        allowed_roots = [target_parent]

    # Resolve candidate path by searching canonical locations
    path_obj = Path(raw_str)
    candidate: Optional[Path] = None

    if (base_dir / path_obj).exists():
        candidate = (base_dir / path_obj).resolve()
    elif (storage_dir / path_obj).exists():
        candidate = (storage_dir / path_obj).resolve()
    elif (raw_dir / path_obj).exists():
        candidate = (raw_dir / path_obj).resolve()
    elif (raw_dir / path_obj.name).exists():
        candidate = (raw_dir / path_obj.name).resolve()
    elif path_obj.is_absolute() and path_obj.exists():
        candidate = path_obj.resolve()
    else:
        # Fallback candidate for validation
        candidate = (base_dir / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()

    # Security check: must reside within permitted roots
    is_safe = any(candidate == root or root in candidate.parents for root in allowed_roots)

    logger.info(
        f"[AutoDS Path Security] Stored input: '{raw_str}' | Resolved path: '{candidate}' | "
        f"Storage root: '{storage_dir}' | Workspace safe: {is_safe}"
    )

    if not is_safe:
        logger.warning(
            f"[AutoDS Path Security] Blocked path traversal attempt outside workspace roots: "
            f"Input: '{raw_str}' -> Resolved: '{candidate}'"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Security Error: Access to path outside project workspace is prohibited."
        )

    if not candidate.exists():
        logger.warning(f"[AutoDS Path Security] Target dataset file does not exist: '{candidate}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file not found: '{path_obj.name}'. Please verify dataset upload."
        )

    return candidate


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
