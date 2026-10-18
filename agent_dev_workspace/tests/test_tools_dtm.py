import matplotlib.figure
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from sklearn.feature_extraction.text import CountVectorizer
from tools_dtm import make_tools


def test_build_dtm_tool_fixture():
    # 1. Create a single shared session instance
    session = SessionState()

    # 2. Load premade load_dataset_tool to load sample_fixture.csv into session
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 3. Instantiate student tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 4. Invoke build_dtm_tool with defaults
    result = build_dtm_tool.invoke({})

    # 5. Assertions against Known Answer #4 from TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["vocabulary_size"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Verify session state fields populated directly
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Verify count_vectorizer stored in session.artifacts
    assert "count_vectorizer" in session.artifacts
    assert isinstance(session.artifacts["count_vectorizer"], CountVectorizer)


def test_term_frequency_tool_fixture():
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

    # 3. Instantiate student tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Prerequisite: build DTM first
    build_dtm_tool.invoke({})

    # --- Test 1: Full term frequencies (Known Answer #14 in TEST_FIXTURE.md) ---
    result_full = term_frequency_tool.invoke({})
    assert result_full["total_terms"] == 5

    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert result_full["term_frequencies"] == expected_freqs

    # --- Test 2: top_n=3 with deterministic tie-breaking ---
    # Both "always" (7) and "gamma" (7) have count 7; tie-breaking orders "always" before "gamma".
    # Top 3 terms: "always": 7, "gamma": 7, "alpha": 6.
    result_top3 = term_frequency_tool.invoke({"top_n": 3})
    expected_top3 = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
    }
    assert result_top3["term_frequencies"] == expected_top3
    assert list(result_top3["term_frequencies"].keys()) == ["always", "gamma", "alpha"]

    # --- Test 3: Invalid top_n validation ---
    with pytest.raises(ValueError):
        term_frequency_tool.invoke({"top_n": 0})

    with pytest.raises(ValueError):
        term_frequency_tool.invoke({"top_n": -2})


def test_dtm_heatmap_tool_fixture():
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

    # 3. Instantiate student tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Prerequisite: build DTM first
    build_dtm_tool.invoke({})

    # --- Test 1: Default 20x20 slice (Bounded to 8x5 full matrix, Known Answer #21) ---
    res_default = dtm_heatmap_tool.invoke({})
    assert res_default["n_documents"] == 8
    assert res_default["n_terms"] == 5
    assert res_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_default["documents"] == ["Doc 0", "Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5", "Doc 6", "Doc 7"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7
    ]
    assert res_default["matrix_slice"] == expected_full_matrix

    # Verify pending_figure is set to a Matplotlib Figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # --- Test 2: Smaller slice (3x2 corner) ---
    res_small = dtm_heatmap_tool.invoke({"n_documents": 3, "n_terms": 2})
    assert res_small["n_documents"] == 3
    assert res_small["n_terms"] == 2
    assert res_small["terms"] == ["alpha", "always"]
    assert res_small["documents"] == ["Doc 0", "Doc 1", "Doc 2"]
    assert res_small["matrix_slice"] == [
        [3, 1],
        [2, 1],
        [1, 1],
    ]

    # --- Test 3: Invalid dimensions validation ---
    with pytest.raises(ValueError):
        dtm_heatmap_tool.invoke({"n_terms": 0})

    with pytest.raises(ValueError):
        dtm_heatmap_tool.invoke({"n_documents": -5})
