from typing import Any, Dict, List, Literal, Optional
import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI mining modules and converters
import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fp_growth
import PAMI.frequentPattern.topk.FAE as topk_fae
import PAMI.frequentPattern.maximal.MaxFPGrowth as max_fp_growth

def make_tools(session: Any) -> List[Any]:
    """
    Factory function producing pattern mining tools initialized with session context.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = 1,
        k: Optional[int] = 10
    ) -> Dict[str, Any]:
        """
        Mines frequent patterns for a specific category using category-local vocabulary and PAMI algorithms.

        Args:
            category_name: Target category in session.dataframe to mine.
            filtering_method: Vocabulary filter ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required for fpgrowth and maxfpgrowth).
            k: Top-k pattern limit (required for topk algorithm).

        Returns:
            Dict containing mined patterns, support scores, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if "category_name" not in df.columns:
            return {"status": "error", "message": "DataFrame missing 'category_name' column."}

        cat_df = df[df["category_name"] == category_name]
        if cat_df.empty:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        # Algorithm argument validation
        alg_lower = algorithm.lower()
        filter_lower = filtering_method.lower()
        artifact_key = f"patterns_{category_name}_{filter_lower}_{alg_lower}"

        if alg_lower in ["fpgrowth", "maxfpgrowth"] and (min_sup is None or min_sup <= 0):
            return {"status": "error", "message": f"Algorithm '{algorithm}' requires min_sup > 0."}
        if alg_lower == "topk" and (k is None or k <= 0):
            return {"status": "error", "message": "Algorithm 'topk' requires k > 0."}

        # Step 1: Category-local CountVectorizer with English stopwords removal
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(cat_df["text"])
        except ValueError:
            dtm = None

        if dtm is None or dtm.shape[1] == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filter_lower,
                "algorithm": alg_lower,
                "num_patterns": 0,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())

        # Step 2: Apply chosen vocabulary filtering method on sparse matrix directly
        if filter_lower == "variance":
            mean = np.array(dtm.mean(axis=0)).ravel()
            mean_sq = np.array(dtm.power(2).mean(axis=0)).ravel()
            scores = mean_sq - (mean ** 2)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores >= p5) & (scores <= p95)
        elif filter_lower == "term_frequency":
            scores = np.array(dtm.sum(axis=0)).ravel()
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores >= p5) & (scores <= p95)
        elif filter_lower == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm)
            scores = np.array(tfidf_matrix.mean(axis=0)).ravel()
            p20 = np.percentile(scores, 20)
            kept_mask = scores >= p20
        else:
            return {"status": "error", "message": f"Unsupported filtering_method '{filtering_method}'."}

        kept_indices = np.where(kept_mask)[0]
        if len(kept_indices) == 0:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filter_lower,
                "algorithm": alg_lower,
                "num_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_terms = feature_names[kept_indices]
        filtered_dtm_dense = dtm[:, kept_indices].toarray()

        # Convert filtered matrix to DataFrame for PAMI DB conversion
        binary_df = pd.DataFrame(filtered_dtm_dense > 0, columns=kept_terms)

        # Step 3: Transaction DB conversion & PAMI pattern mining
        patterns_list = []
        with tempfile.TemporaryDirectory() as temp_dir:
            db_file = os.path.join(temp_dir, "transactional.db")
            
            # Convert DF to DB
            conv = db_converter.DF2DB(binary_df)
            conv.convert2TransactionalDatabase(db_file, condition=">=", thresholdValue=0.5)

            # Mine patterns with requested algorithm
            if alg_lower == "fpgrowth":
                miner = fp_growth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            elif alg_lower == "topk":
                miner = topk_fae.FAE(iFile=db_file, k=k, sep="\t")
            elif alg_lower == "maxfpgrowth":
                miner = max_fp_growth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            else:
                return {"status": "error", "message": f"Unsupported algorithm '{algorithm}'."}

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pattern_str = str(row["Patterns"])
                    supp = int(row["Support"])
                    items = [item.strip() for item in pattern_str.split("\t") if item.strip()]
                    patterns_list.append({"pattern": items, "support": supp})

        # Store in session artifacts
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filter_lower,
            "algorithm": alg_lower,
            "num_patterns": len(patterns_list),
            "patterns": patterns_list
        }

        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
        return summary

    return [mine_patterns_tool]
