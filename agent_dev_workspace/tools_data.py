import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from langchain_core.tools import tool
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import numpy as np
import seaborn as sns


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

    @tool
    def check_duplicates_tool(drop: bool = False, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops them.

        When checking, counts duplicate occurrences using keep="first" (1 duplicate for 1 extra copy).
        When drop=True, removes all copies of duplicated rows using keep=False.

        Args:
            drop: Whether to drop duplicate rows from session.dataframe (default False).
            subset: List of column names to consider for identifying duplicates (default ['text']).

        Returns:
            Dict containing result_id, n_documents, duplicate_count, duplicate_indices,
            and remaining_documents.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        df = session.dataframe
        n_docs = len(df)

        cols_to_check = subset if subset is not None else ["text"]
        for col in cols_to_check:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Identify duplicate occurrences (keep="first" counts extra duplicate copies)
        dup_mask = df.duplicated(subset=cols_to_check, keep="first")
        dup_indices = np.flatnonzero(dup_mask.to_numpy()).tolist()
        dup_count = len(dup_indices)

        if drop and dup_count > 0:
            session.dataframe = df.drop_duplicates(subset=cols_to_check, keep=False).reset_index(drop=True)

        remaining_docs = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")
        args = {"drop": drop, "subset": subset}

        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "duplicate_count": dup_count,
            "duplicate_indices": dup_indices,
            "remaining_documents": remaining_docs,
        }

        session.store_result("check_duplicates_tool", args, summary)
        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Subsamples session.dataframe using pandas DataFrame.sample.

        Args:
            n: Number of items to sample (default 5).
            frac: Fraction of items to sample. If set, n is ignored.
            random_state: Seed for random number generator for reproducibility.
            replace: Allow or disallow sampling of the same row more than once.

        Returns:
            Dict containing result_id, n_documents, sample_size, sampled_indices,
            and sample_rows.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        df = session.dataframe
        n_docs = len(df)

        if frac is not None:
            if frac <= 0:
                raise ValueError("frac must be greater than 0.")
            sample_df = df.sample(frac=frac, random_state=random_state, replace=replace)
        else:
            if n <= 0:
                raise ValueError("n must be a positive integer.")
            if n > n_docs and not replace:
                raise ValueError(f"Cannot sample n={n} documents from dataset of size {n_docs} without replace=True.")
            sample_df = df.sample(n=n, random_state=random_state, replace=replace)

        sampled_indices = sample_df.index.tolist()
        sample_size = len(sample_df)

        clean_sample = sample_df.astype(object).where(sample_df.notna(), None)
        raw_records = clean_sample.to_dict(orient="records")
        sample_rows = json.loads(json.dumps(raw_records, default=str, allow_nan=False))

        result_id = session.next_result_id("sample_data")
        args = {
            "n": n,
            "frac": frac,
            "random_state": random_state,
            "replace": replace,
        }

        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "sample_size": sample_size,
            "sampled_indices": sampled_indices,
            "sample_rows": sample_rows,
        }

        session.store_result("sample_data_tool", args, summary, full_report=sample_df)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text length statistics and creates a boxplot by category.

        Computes character length (text.apply(len)) overall and grouped by
        category_name, then sets a Seaborn boxplot on session.pending_figure.

        Returns:
            Dict containing result_id, n_documents, overall_stats, and
            per_category_stats.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        df = session.dataframe
        if "text" not in df.columns:
            raise ValueError("DataFrame does not contain a 'text' column.")

        # Compute character length per document
        text_length = df["text"].apply(lambda s: len(s) if isinstance(s, str) else 0)

        # Overall describe stats
        desc_overall = text_length.describe()
        overall_stats = {
            "count": int(desc_overall["count"]),
            "mean": round(float(desc_overall["mean"]), 4),
            "std": round(float(desc_overall["std"]), 4) if not np.isnan(desc_overall["std"]) else 0.0,
            "min": round(float(desc_overall["min"]), 4),
            "25%": round(float(desc_overall["25%"]), 4),
            "50%": round(float(desc_overall["50%"]), 4),
            "75%": round(float(desc_overall["75%"]), 4),
            "max": round(float(desc_overall["max"]), 4),
        }

        # Per-category describe stats
        per_category_stats = {}
        category_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else "category")

        if category_col in df.columns:
            for cat, group_indices in df.groupby(category_col).groups.items():
                cat_len = text_length.loc[group_indices]
                cat_desc = cat_len.describe()
                per_category_stats[str(cat)] = {
                    "count": int(cat_desc["count"]),
                    "mean": round(float(cat_desc["mean"]), 4),
                    "std": round(float(cat_desc["std"]), 4) if not np.isnan(cat_desc["std"]) else 0.0,
                    "min": round(float(cat_desc["min"]), 4),
                    "25%": round(float(cat_desc["25%"]), 4),
                    "50%": round(float(cat_desc["50%"]), 4),
                    "75%": round(float(cat_desc["75%"]), 4),
                    "max": round(float(cat_desc["max"]), 4),
                }

        # Build boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = df.copy()
        plot_df["text_length"] = text_length
        sns.boxplot(data=plot_df, x=category_col, y="text_length", ax=ax)
        ax.set_title("Text Length Distribution by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        args = {}

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result("describe_data_tool", args, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document text into unigrams using NLTK word_tokenize on lowercased text.

        Stores tokens in session.dataframe['unigrams'] and session.artifacts['tokens'].

        Returns:
            Dict containing result_id, n_documents, and sample_tokens.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        df = session.dataframe
        if "text" not in df.columns:
            raise ValueError("DataFrame does not contain a 'text' column.")

        tokenized_docs = []
        for val in df["text"]:
            if val is not None and isinstance(val, str) and val.strip():
                tokens = word_tokenize(val.lower(), preserve_line=True)
            else:
                tokens = []
            tokenized_docs.append(tokens)

        # Store in session
        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["tokens"] = tokenized_docs

        sample_tokens = tokenized_docs[:5]

        result_id = session.next_result_id("tokenize")
        args = {}

        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "sample_tokens": sample_tokens,
        }

        session.store_result("tokenize_tool", args, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """Lists files and directories in the repository or a specified subdirectory.

        Args:
            subdirectory: Subdirectory path relative to the repository root (default "").

        Returns:
            Dict containing result_id, files, and directories.
        """
        repo_root = Path.cwd().resolve()

        if Path(subdirectory).is_absolute():
            raise ValueError("Subdirectory path must be relative to the repository root.")

        target_path = (repo_root / subdirectory).resolve()

        try:
            target_path.relative_to(repo_root)
        except ValueError:
            raise ValueError("Path traversal outside the repository root is not allowed.")

        if not target_path.exists():
            raise ValueError(f"Directory '{subdirectory}' does not exist.")

        if not target_path.is_dir():
            raise ValueError(f"Path '{subdirectory}' is not a directory.")

        files = []
        directories = []

        for entry in os.listdir(target_path):
            if entry.startswith("."):
                continue
            full_entry_path = target_path / entry
            if full_entry_path.is_dir():
                directories.append(entry)
            else:
                files.append(entry)

        files.sort()
        directories.sort()

        result_id = session.next_result_id("list_files")
        args = {"subdirectory": subdirectory}

        summary = {
            "result_id": result_id,
            "files": files,
            "directories": directories,
        }

        session.store_result("list_files_tool", args, summary)
        return summary

    return [
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        tokenize_tool,
        list_files_tool,
    ]
