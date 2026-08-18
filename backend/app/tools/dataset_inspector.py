"""
AutoDS Dataset Inspector Tool
Inspects, loads, and infers file types, delimiters, sizes, and schema metadata safely.
"""

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from backend.app.core.logging import logger
from backend.app.core.security import validate_file_path

Union_Path_or_Str = Union[str, Path]


def compute_file_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of a file for reproducibility and caching."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_csv_delimiter(file_path: Path, num_bytes: int = 16384) -> str:
    """Sniff CSV delimiter (comma, semicolon, tab, pipe). Defaults to comma if ambiguous."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(num_bytes)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
    except Exception as e:
        logger.debug(f"Delimiter sniffing failed ({e}), defaulting to comma.")
        # Try checking first line manually
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
                if first_line.count(";") > first_line.count(","):
                    return ";"
                elif first_line.count("\t") > first_line.count(","):
                    return "\t"
        except Exception:
            pass
        return ","


def load_dataset_as_dataframe(
    file_path: Union_Path_or_Str,
    sample_rows: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely load a dataset into a pandas DataFrame from CSV, Parquet, or Excel.
    Returns the DataFrame and a metadata dictionary.
    """
    valid_path = validate_file_path(file_path)
    file_ext = valid_path.suffix.lower()
    meta: Dict[str, Any] = {
        "file_path": str(valid_path),
        "file_name": valid_path.name,
        "file_size_bytes": os.path.getsize(valid_path),
        "checksum": compute_file_sha256(valid_path),
        "file_type": file_ext.replace(".", ""),
    }

    if file_ext in (".csv", ".txt"):
        delimiter = detect_csv_delimiter(valid_path)
        meta["delimiter"] = delimiter
        df = pd.read_csv(
            valid_path,
            sep=delimiter,
            nrows=sample_rows,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False
        )
    elif file_ext in (".parquet", ".pq"):
        df = pd.read_parquet(valid_path)
        if sample_rows:
            df = df.head(sample_rows)
    elif file_ext in (".xlsx", ".xls"):
        df = pd.read_excel(valid_path, nrows=sample_rows)
    elif file_ext == ".json":
        df = pd.read_json(valid_path)
        if sample_rows:
            df = df.head(sample_rows)
    else:
        raise ValueError(f"Unsupported file format: '{file_ext}'. Supported: CSV, Parquet, Excel, JSON.")

    meta["row_count"] = len(df)
    meta["col_count"] = len(df.columns)
    meta["memory_usage_mb"] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)

    return df, meta
