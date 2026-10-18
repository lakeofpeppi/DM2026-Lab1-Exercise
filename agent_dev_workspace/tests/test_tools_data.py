import json
import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_inspect_data_tool_custom():
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

    # 3. Instantiate inspect_data_tool bound to this session
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Check actual session.dataframe column names after load_dataset_tool runs
    assert "text" in session.dataframe.columns
    assert "category_name" in session.dataframe.columns

    # --- Test 1: Default 5-row preview ---
    result_default = inspect_data_tool.invoke({})
    assert result_default["n_documents"] == 8
    assert result_default["n_columns"] == len(session.dataframe.columns)
    assert len(result_default["sample_rows"]) == 5
    assert result_default["mode"] == "head"
    assert "text" in result_default["columns"]

    # Verify JSON serializability of summary
    json_str = json.dumps(result_default, allow_nan=False)
    assert json_str is not None

    # --- Test 2: Tail mode (verify rows match actual tail of session.dataframe) ---
    result_tail = inspect_data_tool.invoke({"n_rows": 3, "mode": "tail"})
    assert len(result_tail["sample_rows"]) == 3
    assert result_tail["mode"] == "tail"

    expected_tail_texts = session.dataframe["text"].tail(3).tolist()
    actual_tail_texts = [row["text"] for row in result_tail["sample_rows"]]
    assert actual_tail_texts == expected_tail_texts

    # --- Test 3: Text truncation without modifying original session.dataframe ---
    original_text_0 = session.dataframe["text"].iloc[0]
    assert len(original_text_0) > 10

    result_trunc = inspect_data_tool.invoke({"max_text_len": 10})
    assert result_trunc["sample_rows"][0]["text"].endswith("...")
    assert len(result_trunc["sample_rows"][0]["text"]) == 13  # 10 chars + '...'

    # Verify original dataframe is untouched
    assert session.dataframe["text"].iloc[0] == original_text_0

    # --- Test 4: Numeric column with NaN handling for JSON safety ---
    session.dataframe["numeric_with_nan"] = [1.5, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]
    result_nan = inspect_data_tool.invoke({"n_rows": 5})

    # Verify second row's numeric_with_nan is serialized as None/null without raising ValueError in json.dumps
    assert result_nan["sample_rows"][1]["numeric_with_nan"] is None
    json_nan_str = json.dumps(result_nan, allow_nan=False)
    assert "null" in json_nan_str

    # Clean up added column
    session.dataframe.drop(columns=["numeric_with_nan"], inplace=True)

    # --- Test 5: Parameter validation / Invalid inputs ---
    with pytest.raises(ValueError):
        inspect_data_tool.invoke({"n_rows": 0})

    with pytest.raises(ValueError):
        inspect_data_tool.invoke({"max_text_len": -5})

    with pytest.raises(ValueError):
        inspect_data_tool.invoke({"mode": "invalid_mode"})


def test_check_missing_tool_fixture():
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

    # 3. Instantiate check_missing_tool bound to this session
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # --- Test 1: Known Answer #2 from TEST_FIXTURE.md ---
    res_fixture = check_missing_tool.invoke({})
    assert res_fixture["n_documents"] == 8
    assert res_fixture["missing_count"] == 1
    assert res_fixture["missing_indices"] == [7]
    assert res_fixture["missing_pct"] == pytest.approx(12.5, abs=1e-3)

    # --- Test 2: Custom Non-Default DataFrame Indexing ---
    # Temporarily set custom non-zero-based index
    orig_index = session.dataframe.index
    session.dataframe.index = [100, 200, 300, 400, 500, 600, 700, 800]

    res_custom_idx = check_missing_tool.invoke({})
    # Verify missing_indices returns zero-based row position [7], NOT index label [800]
    assert res_custom_idx["missing_indices"] == [7]

    # Restore original index
    session.dataframe.index = orig_index

    # --- Test 3: Null values & Whitespace-only text ---
    # Create isolated copy of dataframe with null and whitespace values
    orig_df_backup = session.dataframe.copy()
    session.dataframe.loc[session.dataframe.index[1], "text"] = "   \t\n "  # whitespace row 1
    session.dataframe.loc[session.dataframe.index[2], "text"] = np.nan       # null row 2

    res_whitespace_null = check_missing_tool.invoke({})
    assert res_whitespace_null["missing_count"] == 3
    assert res_whitespace_null["missing_indices"] == [1, 2, 7]

    # Restore original dataframe
    session.dataframe = orig_df_backup.copy()

    # --- Test 4: Completely Clean Dataset (No missing or empty text) ---
    # Filter out empty/null/whitespace rows
    clean_mask = (
        session.dataframe["text"].notna()
        & (session.dataframe["text"].astype(str).str.strip() != "")
    )
    session.dataframe = session.dataframe[clean_mask].reset_index(drop=True)

    res_clean = check_missing_tool.invoke({})
    assert res_clean["n_documents"] == 7
    assert res_clean["missing_count"] == 0
    assert res_clean["missing_indices"] == []
    assert res_clean["missing_pct"] == 0.0
