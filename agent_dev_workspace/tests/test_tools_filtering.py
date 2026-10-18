import pandas as pd
import pytest
from scipy.sparse import csr_matrix, hstack
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_fixture():
    # 1. Create a single shared session instance
    session = SessionState()

    # 2. Load dataset via premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 3. Instantiate DTM tool and build initial DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # --- Test 1: Full Report & Exact Variances (Known Answer #5) ---
    res_default = variance_filter_tool.invoke({"threshold": 0.0, "apply_in_place": False})
    assert res_default["original_terms"] == 5
    assert res_default["kept_terms_count"] == 5

    # Check full_report stored in results_store
    result_entry = session.results_store[res_default["result_id"]]
    full_report = result_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Verify exact known answer variances from TEST_FIXTURE.md #5
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-4)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-4)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-4)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-4)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-4)

    # Verify full_report sorting (descending by variance)
    assert list(full_report["term"]) == ["alpha", "gamma", "delta", "beta", "always"]

    # --- Test 2: Thresholding at 0.15 (apply_in_place=False) ---
    res_thresh = variance_filter_tool.invoke({"threshold": 0.15, "apply_in_place": False})
    assert res_thresh["kept_terms_count"] == 4
    assert res_thresh["removed_terms_count"] == 1

    # Verify session matrix is UNCHANGED
    assert session.feature_matrix.shape == (8, 5)
    assert len(session.feature_names) == 5

    # --- Test 3: Safeguard against empty features with apply_in_place=True ---
    with pytest.raises(ValueError, match="Cannot apply in place with 0 surviving features"):
        variance_filter_tool.invoke({"threshold": 100.0, "apply_in_place": True})

    # Verify session matrix is still intact after failed attempt
    assert session.feature_matrix.shape == (8, 5)

    # --- Test 4: In-Place Filtering (apply_in_place=True, threshold=0.15) ---
    res_inplace = variance_filter_tool.invoke({"threshold": 0.15, "apply_in_place": True})
    assert res_inplace["kept_terms_count"] == 4

    # Verify session state IS UPDATED
    assert session.feature_matrix.shape == (8, 4)
    assert session.feature_names == ["alpha", "beta", "delta", "gamma"]  # 'always' removed

    # Verify CountVectorizer vocabulary and .transform(["alpha gamma"]) behavior
    vec = session.artifacts["count_vectorizer"]
    assert set(vec.vocabulary_.keys()) == {"alpha", "beta", "delta", "gamma"}

    # Test transforming new text using updated vectorizer
    transformed_matrix = vec.transform(["alpha gamma"])
    assert transformed_matrix.shape == (1, 4)

    # Transform output array should match column indices of kept_terms:
    # 'alpha' is col 0 (count 1), 'beta' is col 1 (0), 'delta' is col 2 (0), 'gamma' is col 3 (1)
    transformed_counts = transformed_matrix.toarray().flatten().tolist()
    assert transformed_counts == [1, 0, 0, 1]


def test_pearson_filter_tool_fixture():
    # 1. Create a single shared session instance
    session = SessionState()

    # 2. Load dataset via premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 3. Instantiate DTM tool and build initial DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # --- Test 1: Known Answer #6 Correlations for catB ---
    res_catB = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0, "apply_in_place": False})
    assert res_catB["kept_terms_count"] == 5

    # Check full_report stored in results_store
    full_report = session.results_store[res_catB["result_id"]]["full_report"]
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Verify exact Pearson correlations from TEST_FIXTURE.md #6
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)

    # Verify sorting by absolute correlation magnitude descending
    # |1.0| > |0.8307| > |-0.7746| > |-0.6882| > |0.3780|
    assert list(full_report["term"]) == ["delta", "gamma", "beta", "alpha", "always"]

    # --- Test 2: Thresholding at |r| >= 0.7 ---
    res_thresh = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.7, "apply_in_place": False})
    assert res_thresh["kept_terms_count"] == 3  # delta, gamma, beta
    assert res_thresh["removed_terms_count"] == 2  # alpha, always

    # Verify session matrix is UNCHANGED
    assert session.feature_matrix.shape == (8, 5)

    # --- Test 3: Invalid target class validation ---
    with pytest.raises(ValueError, match="Invalid target_class"):
        pearson_filter_tool.invoke({"target_class": "nonexistent_category"})

    # --- Test 4: In-Place Filtering (threshold=0.7, apply_in_place=True) ---
    res_inplace = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.7, "apply_in_place": True})
    assert res_inplace["kept_terms_count"] == 3

    # Verify session matrix updated
    assert session.feature_matrix.shape == (8, 3)
    assert set(session.feature_names) == {"beta", "delta", "gamma"}

    # Verify vectorizer alignment & vocabulary exact index match
    vec = session.artifacts["count_vectorizer"]
    expected_vocab = {term: idx for idx, term in enumerate(session.feature_names)}
    assert vec.vocabulary_ == expected_vocab

    transformed = vec.transform(["delta beta"])
    assert transformed.shape == (1, 3)

    # Check transformed counts match vocabulary indices
    vocab = vec.vocabulary_
    transformed_arr = transformed.toarray().flatten()
    assert transformed_arr[vocab["delta"]] == 1
    assert transformed_arr[vocab["beta"]] == 1
    assert transformed_arr[vocab["gamma"]] == 0


def test_spearman_filter_tool_fixture():
    # 1. Create a single shared session instance
    session = SessionState()

    # 2. Load dataset via premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 3. Instantiate DTM tool and build initial DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # --- Test 1: Known Answer #6 Spearman Correlations for catB ---
    res_catB = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0, "apply_in_place": False})
    assert res_catB["kept_terms_count"] == 5

    # Check full_report stored in results_store
    full_report = session.results_store[res_catB["result_id"]]["full_report"]
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Verify exact Spearman rank correlations from TEST_FIXTURE.md #6
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert r_dict["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert r_dict["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert r_dict["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert r_dict["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert r_dict["always"] == pytest.approx(0.3780, abs=1e-3)

    # Verify sorting by absolute rank correlation magnitude descending
    # |1.0| > |0.9363| > |-0.7746| > |-0.7500| > |0.3780|
    assert list(full_report["term"]) == ["delta", "gamma", "beta", "alpha", "always"]

    # --- Test 2: Thresholding at |spearman_r| >= 0.8 ---
    res_thresh = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.8, "apply_in_place": False})
    assert res_thresh["kept_terms_count"] == 2  # delta, gamma
    assert res_thresh["removed_terms_count"] == 3  # beta, alpha, always

    # Verify session matrix is UNCHANGED
    assert session.feature_matrix.shape == (8, 5)

    # --- Test 3: Safeguard when threshold removes all terms in place ---
    with pytest.raises(ValueError, match="Cannot apply in place with 0 surviving features"):
        spearman_filter_tool.invoke({"target_class": "catB", "threshold": 2.0, "apply_in_place": True})

    # Verify session matrix and feature names remain untouched
    assert session.feature_matrix.shape == (8, 5)
    assert len(session.feature_names) == 5

    # --- Test 4: Invalid target class validation ---
    with pytest.raises(ValueError, match="Invalid target_class"):
        spearman_filter_tool.invoke({"target_class": "nonexistent_category"})

    # --- Test 5: Zero-variance column handling ---
    # Append constant column (all zeros) to feature_matrix
    zero_col = csr_matrix((8, 1))
    session.feature_matrix = hstack([session.feature_matrix, zero_col]).tocsr()
    session.feature_names.append("zero_var_term")

    res_zero = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0, "apply_in_place": False})
    zero_report = session.results_store[res_zero["result_id"]]["full_report"]
    zero_r_dict = dict(zip(zero_report["term"], zero_report["spearman_r"]))
    assert zero_r_dict["zero_var_term"] == 0.0

    # Clean up zero_var_term from session state
    session.feature_matrix = session.feature_matrix[:, :5]
    session.feature_names = session.feature_names[:5]

    # --- Test 6: In-Place Filtering (threshold=0.8, apply_in_place=True) ---
    res_inplace = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.8, "apply_in_place": True})
    assert res_inplace["kept_terms_count"] == 2

    # Verify session matrix updated
    assert session.feature_matrix.shape == (8, 2)
    assert set(session.feature_names) == {"delta", "gamma"}

    # Verify vectorizer alignment & transform
    vec = session.artifacts["count_vectorizer"]
    expected_vocab = {term: idx for idx, term in enumerate(session.feature_names)}
    assert vec.vocabulary_ == expected_vocab

    transformed = vec.transform(["gamma delta"])
    assert transformed.shape == (1, 2)

    # Check transformed counts match vocabulary indices
    vocab = vec.vocabulary_
    transformed_arr = transformed.toarray().flatten()
    assert transformed_arr[vocab["delta"]] == 1
    assert transformed_arr[vocab["gamma"]] == 1
