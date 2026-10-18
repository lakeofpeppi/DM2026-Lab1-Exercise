import json
from typing import Any, Dict, List, Literal, Optional
from langchain_core.tools import tool
import numpy as np


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools bound to the provided session instance."""

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        mode: Literal["head", "tail"] = "head",
        max_text_len: int = 200,
    ) -> Dict[str, Any]:
        """Peeks at a sample of rows from session.dataframe.

        Returns dataset shape, column names, column data types, and a list of
        sample row records with long text values truncated to avoid overwhelming
        the context window.

        Args:
            n_rows: Number of sample rows to return (default 5).
            mode: Whether to inspect "head" or "tail" rows (default "head").
            max_text_len: Maximum character length for text fields (default 200).

        Returns:
            Dict containing result_id, n_documents, n_columns, columns, dtypes,
            and sample_rows.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        if n_rows <= 0:
            raise ValueError("n_rows must be a positive integer.")

        if max_text_len <= 0:
            raise ValueError("max_text_len must be a positive integer.")

        if mode not in ("head", "tail"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'head' or 'tail'.")

        df = session.dataframe

        # Select sample slice
        if mode == "head":
            sample_df = df.head(n_rows).copy()
        else:
            sample_df = df.tail(n_rows).copy()

        # Truncate string columns (especially text) if longer than max_text_len
        for col in sample_df.columns:
            if sample_df[col].dtype == object or str(sample_df[col].dtype) == "string":
                sample_df[col] = sample_df[col].apply(
                    lambda val: (
                        val[:max_text_len] + "..."
                        if isinstance(val, str) and len(val) > max_text_len
                        else val
                    )
                )

        # Convert data types and records to Python-native types for JSON safety
        dtypes_dict = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

        # Replace NaN/NaT values with None across all types, then sanitize through json.dumps with allow_nan=False
        clean_df = sample_df.astype(object).where(sample_df.notna(), None)
        raw_records = clean_df.to_dict(orient="records")
        sample_rows = json.loads(json.dumps(raw_records, default=str, allow_nan=False))

        result_id = session.next_result_id("inspect")
        args = {"n_rows": n_rows, "mode": mode, "max_text_len": max_text_len}

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "n_columns": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "dtypes": dtypes_dict,
            "mode": mode,
            "sample_rows": sample_rows,
        }

        session.store_result("inspect_data_tool", args, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text documents in session.dataframe.

        Identifies rows where text is null, empty string, or whitespace only.

        Returns:
            Dict containing result_id, n_documents, missing_count, missing_indices,
            and missing_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        df = session.dataframe
        n_docs = len(df)

        if "text" not in df.columns:
            raise ValueError("DataFrame does not contain a 'text' column.")

        # Identify missing or empty text (null, empty string, or whitespace only)
        text_series = df["text"]
        missing_mask = (
            text_series.isna()
            | (text_series.astype(str).str.strip() == "")
        )

        # Use np.flatnonzero to ensure zero-based integer row positions regardless of custom indexing
        missing_indices = np.flatnonzero(missing_mask.to_numpy()).tolist()
        missing_count = len(missing_indices)
        missing_pct = float(100.0 * missing_count / n_docs) if n_docs > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        args = {}

        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "missing_pct": round(missing_pct, 4),
        }

        session.store_result("check_missing_tool", args, summary)
        return summary

    return [inspect_data_tool, check_missing_tool]
