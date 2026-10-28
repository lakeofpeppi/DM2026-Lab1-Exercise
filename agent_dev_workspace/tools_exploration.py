from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session: Any) -> List[Any]:
    """
    Factory function producing exploration tools initialized with session context.
    """

    @tool
    def cosine_similarity_tool(doc_id_1: int, doc_id_2: int) -> Dict[str, Any]:
        """
        Computes the cosine similarity between two documents in session.feature_matrix.

        Args:
            doc_id_1: Row index of the first document.
            doc_id_2: Row index of the second document.

        Returns:
            Dict containing status, result_id, document indices, and calculated cosine similarity score.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]

        if doc_id_1 < 0 or doc_id_1 >= n_docs:
            return {"status": "error", "message": f"doc_id_1 ({doc_id_1}) is out of bounds [0, {n_docs - 1}]."}

        if doc_id_2 < 0 or doc_id_2 >= n_docs:
            return {"status": "error", "message": f"doc_id_2 ({doc_id_2}) is out of bounds [0, {n_docs - 1}]."}

        # Extract vectors using row slicing [doc_id:doc_id+1] for 2D shape compatibility with both sparse and dense matrices
        vec1 = session.feature_matrix[doc_id_1:doc_id_1 + 1]
        vec2 = session.feature_matrix[doc_id_2:doc_id_2 + 1]

        # Compute cosine similarity
        sim_matrix = cosine_similarity(vec1, vec2)
        score = float(sim_matrix[0, 0])

        # Safely handle NaN results (e.g. zero vectors)
        if np.isnan(score):
            score = 0.0
        else:
            score = round(score, 4)

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "status": "success",
            "doc_id_1": doc_id_1,
            "doc_id_2": doc_id_2,
            "similarity": score
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_id_1": doc_id_1, "doc_id_2": doc_id_2},
            summary
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(top_k: int = 20) -> Dict[str, Any]:
        """
        Computes Pearson correlation matrix over top variance terms and generates a heatmap.

        Args:
            top_k: Number of highest variance features to include (default: 20).

        Returns:
            Dict containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix or feature_names found. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_rows, n_cols = X.shape

        if n_rows == 0 or n_cols == 0:
            return {"status": "error", "message": f"feature_matrix has empty dimensions ({n_rows} rows, {n_cols} columns)."}

        if top_k <= 0:
            return {"status": "error", "message": "top_k must be a positive integer greater than 0."}

        feature_names = np.array(session.feature_names)

        if len(feature_names) != n_cols:
            return {"status": "error", "message": f"feature_names length ({len(feature_names)}) does not match feature_matrix columns ({n_cols})."}

        # Compute variance efficiently without converting entire matrix to dense
        if hasattr(X, "toarray"):
            # Sparse matrix: Var(X) = E[X^2] - (E[X])^2
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
        else:
            variances = np.var(X, axis=0)

        # Select top-k variance features (sorted descending by variance)
        top_indices = np.argsort(variances)[::-1][:top_k]
        top_terms = list(feature_names[top_indices])

        # Convert ONLY selected top_k columns to dense for correlation computation
        if hasattr(X, "toarray"):
            X_top = X[:, top_indices].toarray()
        else:
            X_top = X[:, top_indices]

        # Compute Pearson correlation matrix
        corr_matrix = np.corrcoef(X_top, rowvar=False)

        # Handle scalar output if top_k == 1 or 1 term available
        if corr_matrix.ndim == 0 or corr_matrix.shape == ():
            corr_matrix = np.array([[1.0 if not np.isnan(corr_matrix) else 0.0]])

        # Handle NaNs resulting from constant features (std=0)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # Create Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1,
            vmax=1
        )
        ax.set_title("Feature Correlation Matrix")
        plt.tight_layout()

        # Store pending figure
        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        corr_matrix_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        summary = {
            "result_id": result_id,
            "status": "success",
            "top_k": len(top_terms),
            "terms": top_terms,
            "correlation_matrix": corr_matrix_list
        }

        session.store_result("feature_correlation_matrix_tool", {"top_k": top_k}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
