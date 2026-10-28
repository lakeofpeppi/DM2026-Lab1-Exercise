import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def test_binarize_labels_tool():
    # Setup session and load fixture
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load 8-row sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student tool
    tools = make_tools(session)
    binarize_labels_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    # Execute tool
    result = binarize_labels_tool.invoke({"column_name": "category_name"})

    # Assert return summary
    assert result["status"] == "success"
    assert result["categories"] == ["catA", "catB"]
    assert result["shape"] == [8, 2]
    assert result["num_samples"] == 8

    # Assert artifact storage
    assert "one_hot_labels" in session.artifacts
    assert session.artifacts["binarized_categories"] == ["catA", "catB"]

    encoded = session.artifacts["one_hot_labels"]
    assert encoded.shape == (8, 2)

    # Expected exact matrix mapping
    expected_catA = np.array([1, 0])
    expected_catB = np.array([0, 1])

    # Documents 0, 1, 2, 7 -> catA
    for doc_idx in [0, 1, 2, 7]:
        np.testing.assert_array_equal(encoded[doc_idx], expected_catA)

    # Documents 3, 4, 5, 6 -> catB
    for doc_idx in [3, 4, 5, 6]:
        np.testing.assert_array_equal(encoded[doc_idx], expected_catB)

def test_reduce_dimensions_pca():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Test PCA (Known Answer #7)
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    assert res["status"] == "success"
    assert res["method"] == "pca"
    assert len(res["coordinates"]) == 8
    assert res["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)
    
    # Check pending_figure
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # All 8 documents coordinates from Known Answer #7
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    for idx, (expected, actual) in enumerate(zip(expected_coords, res["coordinates"])):
        assert actual == pytest.approx(expected, abs=1e-3)

def test_reduce_dimensions_tsne():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known Answer #8 reference coordinates (scikit-learn historical reference):
    # Doc 0: [1330.9991, 32.1039], Doc 1: [951.2459, 393.5588], Doc 2: [530.8209, 682.9791],
    # Doc 3: [-1558.6448, -314.1625], Doc 4: [-915.5153, -346.3907], Doc 5: [-280.0313, -1261.2115],
    # Doc 6: [-280.0313, -1261.2115], Doc 7: [-701.3674, 1465.3973]

    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    assert res["status"] == "success"
    assert res["method"] == "tsne"
    assert len(res["coordinates"]) == 8

    # Assert 8x2 shape and finite float values
    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    assert np.all(np.isfinite(coords))

    # Assert pending_figure creation
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Assert deterministic output given random_state=42
    res_second = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    np.testing.assert_allclose(coords, np.array(res_second["coordinates"]), rtol=1e-5)

def test_reduce_dimensions_umap():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Test UMAP (Known Answer #9)
    res = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert res["status"] == "success"
    assert res["method"] == "umap"
    assert len(res["coordinates"]) == 8
    
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # All 8 documents coordinates from Known Answer #9
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]
    for expected, actual in zip(expected_coords, res["coordinates"]):
        assert actual == pytest.approx(expected, abs=1e-1)
