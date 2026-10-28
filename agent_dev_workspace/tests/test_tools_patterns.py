import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def setup_session_with_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load 8-row sample fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_mine_patterns_fpgrowth_known_answer_17():
    session = setup_session_with_data()
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. TF-IDF filtering + FPGrowth on catA (min_sup = 1)
    res_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["status"] == "success"
    assert isinstance(res_catA["result_id"], str) and len(res_catA["result_id"]) > 0
    assert res_catA["category_name"] == "catA"
    assert res_catA["filtering_method"] == "tfidf"
    assert res_catA["algorithm"] == "fpgrowth"
    assert res_catA["num_patterns"] == 1
    assert res_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == [{"pattern": ["alpha"], "support": 3}]

    # 2. TF-IDF filtering + FPGrowth on catB (min_sup = 1)
    res_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB["status"] == "success"
    assert isinstance(res_catB["result_id"], str) and len(res_catB["result_id"]) > 0
    assert res_catB["category_name"] == "catB"
    assert res_catB["filtering_method"] == "tfidf"
    assert res_catB["algorithm"] == "fpgrowth"
    assert res_catB["num_patterns"] == 1
    assert res_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    assert session.artifacts["patterns_catB_tfidf_fpgrowth"] == [{"pattern": ["gamma"], "support": 4}]

    # 3. Variance & Term Frequency filtering on catA and catB -> 0 terms survive
    for cat in ["catA", "catB"]:
        for filter_method in ["variance", "term_frequency"]:
            res_empty = mine_tool.invoke({
                "category_name": cat,
                "filtering_method": filter_method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res_empty["status"] == "success"
            assert isinstance(res_empty["result_id"], str) and len(res_empty["result_id"]) > 0
            assert res_empty["category_name"] == cat
            assert res_empty["filtering_method"] == filter_method
            assert res_empty["algorithm"] == "fpgrowth"
            assert res_empty["num_patterns"] == 0
            assert res_empty["patterns"] == []
            assert "no terms survived filtering" in res_empty["note"]
            assert session.artifacts[f"patterns_{cat}_{filter_method}_fpgrowth"] == []

def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    session = setup_session_with_data()
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. topk algorithm (k = 1) on catA -> returns {alpha} support 3
    res_topk_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA["status"] == "success"
    assert isinstance(res_topk_catA["result_id"], str) and len(res_topk_catA["result_id"]) > 0
    assert res_topk_catA["num_patterns"] == 1
    assert res_topk_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    assert session.artifacts["patterns_catA_tfidf_topk"] == [{"pattern": ["alpha"], "support": 3}]

    # 2. topk algorithm (k = 1) on catB -> returns {gamma} support 4
    res_topk_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catB["status"] == "success"
    assert isinstance(res_topk_catB["result_id"], str) and len(res_topk_catB["result_id"]) > 0
    assert res_topk_catB["num_patterns"] == 1
    assert res_topk_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    assert session.artifacts["patterns_catB_tfidf_topk"] == [{"pattern": ["gamma"], "support": 4}]

    # 3. maxfpgrowth algorithm on catA and catB -> returns 0 patterns (Known Answer #20)
    for cat in ["catA", "catB"]:
        res_max = mine_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1
        })
        assert res_max["status"] == "success"
        assert isinstance(res_max["result_id"], str) and len(res_max["result_id"]) > 0
        assert res_max["num_patterns"] == 0
        assert res_max["patterns"] == []
        assert session.artifacts[f"patterns_{cat}_tfidf_maxfpgrowth"] == []

def test_mine_patterns_edge_cases_and_invalid_inputs():
    session = setup_session_with_data()
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Invalid category name
    res_invalid_cat = mine_tool.invoke({"category_name": "non_existent_cat"})
    assert res_invalid_cat["status"] == "error"
    assert "No documents found" in res_invalid_cat["message"]

    # Invalid min_sup <= 0
    res_invalid_sup = mine_tool.invoke({"category_name": "catA", "algorithm": "fpgrowth", "min_sup": 0})
    assert res_invalid_sup["status"] == "error"
    assert "requires min_sup > 0" in res_invalid_sup["message"]

    # Invalid k <= 0 for topk
    res_invalid_k = mine_tool.invoke({"category_name": "catA", "algorithm": "topk", "k": 0})
    assert res_invalid_k["status"] == "error"
    assert "requires k > 0" in res_invalid_k["message"]

    # Missing dataframe
    session_no_data = SessionState()
    tools_no_data = make_tools(session_no_data)
    mine_no_data = next(t for t in tools_no_data if t.name == "mine_patterns_tool")
    res_no_data = mine_no_data.invoke({"category_name": "catA"})
    assert res_no_data["status"] == "error"
    assert "No dataset loaded" in res_no_data["message"]

    # Text that contains only stopwords -> empty vocabulary after stopword removal
    session_stop_only = SessionState()
    import pandas as pd
    session_stop_only.dataframe = pd.DataFrame({
        "text": ["always always", "always"],
        "category_name": ["stop_cat", "stop_cat"]
    })
    tools_stop = make_tools(session_stop_only)
    mine_stop = next(t for t in tools_stop if t.name == "mine_patterns_tool")
    res_stop = mine_stop.invoke({"category_name": "stop_cat", "filtering_method": "tfidf"})
    assert res_stop["status"] == "success"
    assert res_stop["num_patterns"] == 0
    assert "no terms survived stopword removal" in res_stop["note"]
    assert session_stop_only.artifacts["patterns_stop_cat_tfidf_fpgrowth"] == []
