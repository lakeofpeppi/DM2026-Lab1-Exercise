from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools bound to the provided session instance."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
        apply_in_place: bool = False,
    ) -> Dict[str, Any]:
        """Filters terms in the document-term matrix based on feature variance.

        Calculates variance across all documents for each term in session.feature_matrix.
        Terms with variance < threshold are flagged for removal.

        Args:
            threshold: Minimum variance threshold for keeping a term (default 0.0).
            apply_in_place: If True, updates session.feature_matrix and session.feature_names
                in place with the surviving columns (default False).

        Returns:
            Dict containing result_id, threshold, original_terms, kept_terms_count,
            removed_terms_count, and apply_in_place.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No DTM found in session state. Call build_dtm_tool first.")

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float.")

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)

        # Efficient variance calculation on sparse matrix: E[X^2] - (E[X])^2
        mean = np.asarray(matrix.mean(axis=0)).flatten()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).flatten()
        variances = mean_sq - (mean ** 2)

        # Create full report DataFrame sorted by variance descending, then term ascending for tie-breaking
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances,
        })
        report_df.sort_values(by=["variance", "term"], ascending=[False, True], inplace=True)
        report_df.reset_index(drop=True, inplace=True)

        # Identify kept vs removed terms based on threshold
        kept_mask = variances >= threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        # Safeguard: raise ValueError if apply_in_place is True and no terms would remain
        if apply_in_place and len(kept_terms) == 0:
            raise ValueError(f"Threshold {threshold} removes all terms. Cannot apply in place with 0 surviving features.")

        # Update session state if apply_in_place is True
        if apply_in_place:
            session.feature_matrix = matrix[:, kept_mask]
            session.feature_names = kept_terms

            # If CountVectorizer is stored in session.artifacts, update vocabulary mapping
            if "count_vectorizer" in session.artifacts and session.artifacts["count_vectorizer"] is not None:
                vec = session.artifacts["count_vectorizer"]
                if hasattr(vec, "vocabulary_"):
                    vec.vocabulary_ = {term: idx for idx, term in enumerate(kept_terms)}

        result_id = session.next_result_id("variance_filter")
        args = {"threshold": threshold, "apply_in_place": apply_in_place}

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "original_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "apply_in_place": apply_in_place,
        }

        # Store result with summary and full_report DataFrame for visualize_result_tool
        session.store_result("variance_filter_tool", args, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        apply_in_place: bool = False,
    ) -> Dict[str, Any]:
        """Filters terms based on Pearson correlation with a binary target category.

        Calculates Pearson correlation between each term's count vector and a one-vs-rest
        binary target vector for target_class. Terms with |pearson_r| < threshold are removed.

        Args:
            target_class: Category name for one-vs-rest correlation (e.g. "catB").
            threshold: Minimum absolute correlation threshold for keeping a term (default 0.0).
            apply_in_place: If True, updates session.feature_matrix and session.feature_names
                in place with surviving columns (default False).

        Returns:
            Dict containing result_id, target_class, threshold, original_terms,
            kept_terms_count, removed_terms_count, and apply_in_place.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No DTM found in session state. Call build_dtm_tool first.")

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("No category_name labels found in session.dataframe. Load dataset first.")

        valid_categories = session.dataframe["category_name"].unique().tolist()
        if target_class not in valid_categories:
            raise ValueError(f"Invalid target_class '{target_class}'. Valid categories: {valid_categories}")

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float.")

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)
        n_samples = matrix.shape[0]

        # Build one-vs-rest binary target vector y (shape: n_samples,)
        y = (session.dataframe["category_name"].values == target_class).astype(float)
        y_mean = y.mean()
        y_std = y.std()

        if y_std == 0:
            raise ValueError(f"Target category '{target_class}' has zero variance across documents.")

        # Vectorized sparse matrix calculations:
        # E[X] and E[X^2] per column
        X_mean = np.asarray(matrix.mean(axis=0)).flatten()
        X_mean_sq = np.asarray(matrix.power(2).mean(axis=0)).flatten()
        X_std = np.sqrt(np.maximum(0.0, X_mean_sq - (X_mean ** 2)))

        # Matrix-vector multiplication for E[X * y]: matrix.T @ y gives sum(X_ij * y_i) for each term j
        X_y_sum = np.asarray(matrix.T.dot(y)).flatten()
        E_X_y = X_y_sum / n_samples

        # Covariance per column: Cov(X_j, y) = E[X_j * y] - E[X_j] * E[y]
        cov = E_X_y - (X_mean * y_mean)

        # Vectorized correlation calculation with zero-variance protection
        pearson_rs = np.zeros(matrix.shape[1], dtype=float)
        valid_mask = X_std > 0
        pearson_rs[valid_mask] = cov[valid_mask] / (X_std[valid_mask] * y_std)

        # Create full report DataFrame sorted by absolute correlation magnitude descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_rs,
            "abs_r": np.abs(pearson_rs),
        })
        report_df.sort_values(by=["abs_r", "term"], ascending=[False, True], inplace=True)
        report_df.drop(columns=["abs_r"], inplace=True)
        report_df.reset_index(drop=True, inplace=True)

        # Filter terms based on absolute correlation magnitude threshold
        kept_mask = np.abs(pearson_rs) >= threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        # Safeguard: raise ValueError if apply_in_place is True and no terms would remain
        if apply_in_place and len(kept_terms) == 0:
            raise ValueError(f"Threshold {threshold} removes all terms. Cannot apply in place with 0 surviving features.")

        # Update session state if apply_in_place is True
        if apply_in_place:
            session.feature_matrix = matrix[:, kept_mask]
            session.feature_names = kept_terms

            if "count_vectorizer" in session.artifacts and session.artifacts["count_vectorizer"] is not None:
                vec = session.artifacts["count_vectorizer"]
                if hasattr(vec, "vocabulary_"):
                    vec.vocabulary_ = {term: idx for idx, term in enumerate(kept_terms)}

        result_id = session.next_result_id("pearson_filter")
        args = {"target_class": target_class, "threshold": threshold, "apply_in_place": apply_in_place}

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "original_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "apply_in_place": apply_in_place,
        }

        # Store result with summary and full_report DataFrame for visualize_result_tool
        session.store_result("pearson_filter_tool", args, summary, full_report=report_df)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        apply_in_place: bool = False,
    ) -> Dict[str, Any]:
        """Filters terms based on Spearman rank correlation with a binary target category.

        Calculates Spearman rank correlation between each term's count vector and a one-vs-rest
        binary target vector for target_class. Terms with |spearman_r| < threshold are removed.

        Args:
            target_class: Category name for one-vs-rest correlation (e.g. "catB").
            threshold: Minimum absolute rank correlation threshold for keeping a term (default 0.0).
            apply_in_place: If True, updates session.feature_matrix and session.feature_names
                in place with surviving columns (default False).

        Returns:
            Dict containing result_id, target_class, threshold, original_terms,
            kept_terms_count, removed_terms_count, and apply_in_place.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No DTM found in session state. Call build_dtm_tool first.")

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError("No category_name labels found in session.dataframe. Load dataset first.")

        valid_categories = session.dataframe["category_name"].unique().tolist()
        if target_class not in valid_categories:
            raise ValueError(f"Invalid target_class '{target_class}'. Valid categories: {valid_categories}")

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float.")

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)

        # Build one-vs-rest binary target vector y
        y = (session.dataframe["category_name"].values == target_class).astype(float)
        if y.std() == 0:
            raise ValueError(f"Target category '{target_class}' has zero variance across documents.")

        # Convert to CSC format once for fast column slicing without converting to dense
        csc_matrix = matrix.tocsc()

        # Compute Spearman rank correlation column by column
        spearman_rs = []
        for j in range(csc_matrix.shape[1]):
            x = csc_matrix.getcol(j).toarray().flatten()
            if np.std(x) == 0:
                spearman_rs.append(0.0)
            else:
                r, _ = spearmanr(x, y)
                spearman_rs.append(0.0 if np.isnan(r) else float(r))

        spearman_rs = np.array(spearman_rs)

        # Create full report DataFrame sorted by absolute correlation magnitude descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_rs,
            "abs_r": np.abs(spearman_rs),
        })
        report_df.sort_values(by=["abs_r", "term"], ascending=[False, True], inplace=True)
        report_df.drop(columns=["abs_r"], inplace=True)
        report_df.reset_index(drop=True, inplace=True)

        # Filter terms based on absolute correlation magnitude threshold
        kept_mask = np.abs(spearman_rs) >= threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        # Safeguard: raise ValueError if apply_in_place is True and no terms would remain
        if apply_in_place and len(kept_terms) == 0:
            raise ValueError(f"Threshold {threshold} removes all terms. Cannot apply in place with 0 surviving features.")

        # Update session state if apply_in_place is True
        if apply_in_place:
            session.feature_matrix = matrix[:, kept_mask]
            session.feature_names = kept_terms

            if "count_vectorizer" in session.artifacts and session.artifacts["count_vectorizer"] is not None:
                vec = session.artifacts["count_vectorizer"]
                if hasattr(vec, "vocabulary_"):
                    vec.vocabulary_ = {term: idx for idx, term in enumerate(kept_terms)}

        result_id = session.next_result_id("spearman_filter")
        args = {"target_class": target_class, "threshold": threshold, "apply_in_place": apply_in_place}

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "original_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "apply_in_place": apply_in_place,
        }

        # Store result with summary and full_report DataFrame for visualize_result_tool
        session.store_result("spearman_filter_tool", args, summary, full_report=report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
