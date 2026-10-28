import pytest
import numpy as np
import scipy.sparse as sp
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools

def setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load 8-row sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session

def test_cosine_similarity_known_answers():
    session = setup_session_with_dtm()
    tools = make_tools(session)
    cosine_sim_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Known Answer #16 pairs:
    # 0 vs 1: 0.9847
    # 0 vs 3: 0.0909
    # 3 vs 4: 0.9847
    # 0 vs 7: 0.0000

    test_cases = [
        (0, 1, 0.9847),
        (0, 3, 0.0909),
        (3, 4, 0.9847),
        (0, 7, 0.0000)
    ]

    for doc1, doc2, expected in test_cases:
        res = cosine_sim_tool.invoke({"doc_id_1": doc1, "doc_id_2": doc2})
        assert res["status"] == "success"
        assert isinstance(res["result_id"], str) and len(res["result_id"]) > 0
        assert res["doc_id_1"] == doc1
        assert res["doc_id_2"] == doc2
        assert res["similarity"] == pytest.approx(expected, abs=1e-3)

def test_cosine_similarity_invalid_indices():
    session = setup_session_with_dtm()
    tools = make_tools(session)
    cosine_sim_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Negative index
    res = cosine_sim_tool.invoke({"doc_id_1": -1, "doc_id_2": 0})
    assert res["status"] == "error"
    assert "out of bounds" in res["message"]

    # Index out of bounds (n_docs = 8)
    res = cosine_sim_tool.invoke({"doc_id_1": 0, "doc_id_2": 10})
    assert res["status"] == "error"
    assert "out of bounds" in res["message"]

def test_cosine_similarity_missing_dtm():
    session = SessionState()
    tools = make_tools(session)
    cosine_sim_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    res = cosine_sim_tool.invoke({"doc_id_1": 0, "doc_id_2": 1})
    assert res["status"] == "error"
    assert "No feature_matrix found" in res["message"]

def test_cosine_similarity_dense_array_and_zero_vectors():
    session = SessionState()
    # Mock dense numpy feature_matrix with a zero row
    session.feature_matrix = np.array([
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0]  # Zero vector
    ])

    tools = make_tools(session)
    cosine_sim_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Identical non-zero vectors
    res1 = cosine_sim_tool.invoke({"doc_id_1": 0, "doc_id_2": 1})
    assert res1["status"] == "success"
    assert res1["similarity"] == pytest.approx(1.0, abs=1e-4)

    # Vector vs Zero vector -> Should safely return 0.0
    res2 = cosine_sim_tool.invoke({"doc_id_1": 0, "doc_id_2": 2})
    assert res2["status"] == "success"
    assert res2["similarity"] == 0.0

def test_feature_correlation_matrix_known_answer_19():
    session = setup_session_with_dtm()
    tools = make_tools(session)
    corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({"top_k": 20})
    assert res["status"] == "success"
    assert isinstance(res["result_id"], str) and len(res["result_id"]) > 0

    # Known Answer #19:
    # Feature ordering by variance descending: ["alpha", "gamma", "delta", "beta", "always"]
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Full correlation matrix (5x5) from Known Answer #19
    expected_corr = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000]
    ]

    for expected_row, actual_row in zip(expected_corr, res["correlation_matrix"]):
        assert actual_row == pytest.approx(expected_row, abs=1e-3)

    # Assert pending_figure creation
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

def test_feature_correlation_matrix_edge_cases():
    session = setup_session_with_dtm()
    tools = make_tools(session)
    corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    # Invalid top_k
    res1 = corr_tool.invoke({"top_k": 0})
    assert res1["status"] == "error"
    assert "top_k must be a positive integer" in res1["message"]

    res2 = corr_tool.invoke({"top_k": -5})
    assert res2["status"] == "error"

    # Missing feature_matrix or feature_names
    session_no_dtm = SessionState()
    tools_no_dtm = make_tools(session_no_dtm)
    corr_tool_no_dtm = next(t for t in tools_no_dtm if t.name == "feature_correlation_matrix_tool")
    res_no_dtm = corr_tool_no_dtm.invoke({"top_k": 5})
    assert res_no_dtm["status"] == "error"

    # Mismatched feature names length
    session_mismatch = setup_session_with_dtm()
    session_mismatch.feature_names = ["alpha", "beta"] # 2 names for 5 columns
    corr_tool_mismatch = next(t for t in make_tools(session_mismatch) if t.name == "feature_correlation_matrix_tool")
    res_mismatch = corr_tool_mismatch.invoke({"top_k": 5})
    assert res_mismatch["status"] == "error"
    assert "does not match" in res_mismatch["message"]

    # Empty matrix
    session_empty = SessionState()
    session_empty.feature_matrix = np.empty((0, 0))
    session_empty.feature_names = []
    corr_tool_empty = next(t for t in make_tools(session_empty) if t.name == "feature_correlation_matrix_tool")
    res_empty = corr_tool_empty.invoke({"top_k": 5})
    assert res_empty["status"] == "error"
    assert "empty dimensions" in res_empty["message"]

    # Non-constant feature with top_k = 1 -> Should produce [[1.0]]
    session_nonconstant = SessionState()
    session_nonconstant.feature_matrix = np.array([[1.0], [2.0], [3.0]])
    session_nonconstant.feature_names = ["f1"]
    corr_tool_nonconstant = next(t for t in make_tools(session_nonconstant) if t.name == "feature_correlation_matrix_tool")
    res_k1_nonconstant = corr_tool_nonconstant.invoke({"top_k": 1})
    assert res_k1_nonconstant["status"] == "success"
    assert res_k1_nonconstant["correlation_matrix"] == [[1.0]]

    # Constant features (std = 0) with top_k = 1 and top_k = 2 -> NaN converted to 0.0
    session_constant = SessionState()
    session_constant.feature_matrix = np.array([
        [1.0, 5.0],
        [1.0, 5.0],
        [1.0, 5.0]
    ])
    session_constant.feature_names = ["c1", "c2"]
    corr_tool_constant = next(t for t in make_tools(session_constant) if t.name == "feature_correlation_matrix_tool")

    # Single constant feature top_k = 1 -> [[0.0]]
    res_k1_constant = corr_tool_constant.invoke({"top_k": 1})
    assert res_k1_constant["status"] == "success"
    assert res_k1_constant["correlation_matrix"] == [[0.0]]

    # Two constant features top_k = 2 -> 2x2 matrix of zeros [[0.0, 0.0], [0.0, 0.0]]
    res_k2_constant = corr_tool_constant.invoke({"top_k": 2})
    assert res_k2_constant["status"] == "success"
    assert res_k2_constant["correlation_matrix"] == [[0.0, 0.0], [0.0, 0.0]]
