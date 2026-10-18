from typing import Any, Dict, List, Optional, Union
from langchain_core.tools import tool
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools bound to the provided session instance."""

    @tool
    def build_dtm_tool(
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from text in session.dataframe.

        Fits CountVectorizer on session.dataframe['text'], populates
        session.feature_matrix and session.feature_names, stores the fitted
        vectorizer in session.artifacts['count_vectorizer'], and returns matrix
        dimensions and sparsity statistics.

        Args:
            min_df: Minimum document frequency for terms (int absolute count, e.g. 1, or float ratio, e.g. 0.05).
            max_df: Maximum document frequency for terms (int absolute count or float ratio, default 1.0).
            ngram_range: Range of n-grams as [min_n, max_n], default [1, 1].

        Returns:
            Dict with result_id, n_documents, vocabulary_size, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataframe loaded in session. Call load_dataset_tool first.")

        # Ensure text column exists and convert missing/NaN values to empty strings
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Convert ngram_range list [min_n, max_n] to tuple (min_n, max_n) for scikit-learn
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        # Initialize and fit CountVectorizer with preserved min_df/max_df types
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state fields
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Sync document labels and category list if category_name column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Calculate sparsity metrics
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        non_zero = int(feature_matrix.nnz)

        if total_elements > 0:
            sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements))
        else:
            sparsity_pct = 0.0

        result_id = session.next_result_id("dtm")
        args = {"min_df": min_df, "max_df": max_df, "ngram_range": ngram_range}

        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates total term frequencies across all documents in session.feature_matrix.

        Sums columns of the document-term matrix to compute overall term counts,
        optionally returning the top_n most frequent terms.

        Args:
            top_n: Optional integer specifying how many top terms to return. If None,
                returns frequencies for all vocabulary terms.

        Returns:
            Dict containing result_id, total_terms, and term_frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No DTM found in session state. Call build_dtm_tool first.")

        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be a positive integer or None.")

        # Compute column sums across DTM sparse matrix
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        # Create list of (term, count) tuples
        term_counts = [
            (term, int(count))
            for term, count in zip(session.feature_names, col_sums)
        ]

        # Sort terms by frequency descending; for ties, sort alphabetically ascending for deterministic output
        sorted_terms = sorted(term_counts, key=lambda x: (-x[1], x[0]))

        # Slice top_n if provided
        if top_n is not None:
            sorted_terms = sorted_terms[:top_n]

        term_frequencies = {term: count for term, count in sorted_terms}

        result_id = session.next_result_id("term_freq")
        args = {"top_n": top_n}

        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": term_frequencies,
        }

        session.store_result("term_frequency_tool", args, summary)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Generates a heatmap of a slice of the global document-term matrix.

        Slices the top n_documents rows and top n_terms columns of session.feature_matrix
        in their existing positional order, renders a Seaborn heatmap, sets
        session.pending_figure, and returns the matrix slice and labels.

        Args:
            n_terms: Number of terms (columns) to slice from the start (default 20).
            n_documents: Number of documents (rows) to slice from the start (default 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, terms, documents,
            and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No DTM found in session state. Call build_dtm_tool first.")

        if n_terms <= 0:
            raise ValueError("n_terms must be a positive integer.")

        if n_documents <= 0:
            raise ValueError("n_documents must be a positive integer.")

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)
        total_docs, total_terms = matrix.shape

        # Bound slices to actual matrix dimensions ("fewer than requested isn't an error")
        actual_n_terms = min(n_terms, total_terms)
        actual_n_docs = min(n_documents, total_docs)

        sliced_terms = feature_names[:actual_n_terms]
        doc_indices = list(range(actual_n_docs))

        # Extract submatrix slice [0:actual_n_docs, 0:actual_n_terms] as dense numpy array
        matrix_slice_arr = matrix[:actual_n_docs, :actual_n_terms].toarray()

        # Build Seaborn heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.8), max(4, actual_n_docs * 0.5)))
        sns.heatmap(
            matrix_slice_arr,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({actual_n_docs} docs x {actual_n_terms} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        fig.tight_layout()

        # Set matplotlib figure on session state for UI rendering
        session.pending_figure = fig

        # Convert numpy array to JSON-serializable nested list of integers
        matrix_slice_list = matrix_slice_arr.astype(int).tolist()

        result_id = session.next_result_id("dtm_heatmap")
        args = {"n_terms": n_terms, "n_documents": n_documents}

        summary = {
            "result_id": result_id,
            "n_documents": actual_n_docs,
            "n_terms": actual_n_terms,
            "terms": sliced_terms,
            "documents": [f"Doc {i}" for i in doc_indices],
            "matrix_slice": matrix_slice_list,
        }

        session.store_result("dtm_heatmap_tool", args, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
