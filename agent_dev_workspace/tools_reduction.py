from typing import Any, Dict, List, Literal, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session: Any) -> List[Any]:
    """
    Factory function producing reduction and binarization tools initialized with session context.
    """

    @tool
    def binarize_labels_tool(column_name: str = "category_name") -> Dict[str, Any]:
        """
        One-hot encodes categorical labels from session.dataframe.

        Args:
            column_name: Name of the column in session.dataframe to one-hot encode.

        Returns:
            Dict containing status, result_id, categories, shape, and sample count.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        if column_name not in session.dataframe.columns:
            return {"status": "error", "message": f"Column '{column_name}' not found in dataframe."}

        # Perform one-hot encoding
        one_hot_df = pd.get_dummies(session.dataframe[column_name], dtype=int)
        categories = list(one_hot_df.columns)
        encoded_array = one_hot_df.to_numpy()

        # Store in session artifacts
        session.artifacts["one_hot_labels"] = encoded_array
        session.artifacts["binarized_categories"] = categories

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column_name,
            "categories": [str(c) for c in categories],
            "num_samples": int(len(encoded_array)),
            "shape": list(encoded_array.shape)
        }

        session.store_result("binarize_labels_tool", {"column_name": column_name}, summary)
        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """
        Reduces feature_matrix dimensionality using PCA, t-SNE, or UMAP and plots a 2D scatter plot.

        Args:
            method: Dimensionality reduction algorithm ("pca", "tsne", or "umap").
            n_components: Target dimensions (must be 2).
            random_state: Seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.

        Returns:
            Dict containing reduced coordinates, method stats, and result_id.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found. Run build_dtm_tool first."}

        if n_components != 2:
            return {"status": "error", "message": "n_components must be exactly 2 for 2D visualization."}

        X = session.feature_matrix
        n_samples = X.shape[0]
        method_lower = method.lower()

        # Parameter validation
        if method_lower == "tsne":
            if perplexity <= 0 or perplexity >= n_samples:
                return {
                    "status": "error",
                    "message": f"t-SNE perplexity must be strictly between 0 and n_samples ({n_samples}), got {perplexity}."
                }
        elif method_lower == "umap":
            if n_neighbors <= 1 or n_neighbors >= n_samples:
                return {
                    "status": "error",
                    "message": f"UMAP n_neighbors must be > 1 and < n_samples ({n_samples}), got {n_neighbors}."
                }

        # Compute 2D coordinates
        explained_variance_ratio = None
        if method_lower == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method_lower == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init="random", learning_rate=200.0)
            X_dense = X.toarray() if hasattr(X, "toarray") else X
            coords = reducer.fit_transform(X_dense)
        elif method_lower == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
        else:
            return {"status": "error", "message": f"Unsupported method '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Store reduced coordinates in artifacts
        session.artifacts[f"reduced_{method_lower}"] = coords

        # Create 2D Scatter Plot
        categories = session.dataframe["category_name"] if session.dataframe is not None and "category_name" in session.dataframe.columns else session.labels
        
        fig, ax = plt.subplots(figsize=(8, 6))
        if categories is not None:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=categories, palette="Set1", s=100, ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color="blue", s=100)

        ax.set_title(f"{method_lower.upper()} Dimensionality Reduction")
        ax.set_xlabel("Component 1" if method_lower == "pca" else "Dimension 1")
        ax.set_ylabel("Component 2" if method_lower == "pca" else "Dimension 2")
        plt.tight_layout()

        # Set pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id(f"reduce_{method_lower}")
        coords_list = [[round(float(x), 4), round(float(y), 4)] for x, y in coords]

        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method_lower,
            "n_components": n_components,
            "num_samples": len(coords_list),
            "coordinates": coords_list
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method_lower,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary
        )
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
