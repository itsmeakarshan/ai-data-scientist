"""
Unit & Integration Tests for Gemini AI Client
Tests planning, tool calling, API failure fallback, quota backoff, and missing key handling.
"""

from unittest.mock import MagicMock, patch
import pytest
from backend.app.agents.gemini_client import GeminiAgentClient


def test_gemini_client_missing_key_fallback():
    """Verify Gemini client cleanly initializes in deterministic mode when no key is set."""
    with patch("backend.app.core.config.settings.GEMINI_API_KEY", ""):
        client = GeminiAgentClient()
        assert not client.is_active
        
        # Test deterministic plan generation
        plan = client.generate_plan(
            user_goal="Predict customer churn",
            problem_info={"problem_type": "classification", "sub_type": "binary"},
            profile_summary={"row_count": 1000, "col_count": 10}
        )
        assert plan["planner_source"] == "deterministic_heuristic_engine"
        assert "candidate_models" in plan
        assert "LightGBM" in plan["candidate_models"]
        
        # Test deterministic business insights
        insights = client.generate_business_insights(
            dataset_name="test_data",
            problem_type="classification",
            best_model_name="LightGBM",
            test_metrics={"roc_auc": 0.88, "f1_macro": 0.85},
            top_features=[{"feature": "age", "importance_pct": 25.0}]
        )
        assert len(insights) >= 3
        assert any(i["category"] == "model_derived" for i in insights)

        # Test deterministic chat response
        chat_res = client.run_agent_chat(
            user_message="Which model performed best?",
            context_data={"best_model": {"model_name": "RandomForest", "metrics": {"test": {"roc_auc": 0.91}}}}
        )
        assert chat_res["source"] == "deterministic_grounded_engine"
        assert "RandomForest" in chat_res["reply"]


def test_gemini_client_planning_mocked_success():
    """Verify Gemini client correctly consumes and parses JSON response when active."""
    client = GeminiAgentClient()
    mock_genai_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"validation_strategy": "stratified_kfold", "candidate_models": ["LightGBM", "XGBoost"], "steps": []}'
    mock_genai_client.models.generate_content.return_value = mock_response
    
    client.client = mock_genai_client
    
    plan = client.generate_plan(
        user_goal="Maximize accuracy",
        problem_info={"problem_type": "classification"},
        profile_summary={"row_count": 500, "col_count": 8}
    )
    assert plan["validation_strategy"] == "stratified_kfold"
    assert "XGBoost" in plan["candidate_models"]
    assert plan["planner_source"] == f"gemini:{client.model_name}"


def test_gemini_client_tool_calling_chat():
    """Verify Gemini client executes chat with tools using Chat.send_message."""
    client = GeminiAgentClient()
    mock_genai_client = MagicMock()
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Based on the SQL query, there are 4,118 positive subscribers in the dataset."
    mock_chat.send_message.return_value = mock_response
    mock_chat.get_history.return_value = []
    mock_genai_client.chats.create.return_value = mock_chat
    
    client.client = mock_genai_client
    
    def sample_tool(query: str) -> str:
        return "4118 rows"

    res = client.run_agent_chat(
        user_message="How many subscribers are there?",
        tools=[sample_tool],
        context_data={"dataset_name": "bank_marketing"}
    )
    assert res["source"] == f"gemini:{client.model_name}"
    assert "4,118 positive subscribers" in res["reply"]
    mock_genai_client.chats.create.assert_called_once()
    mock_chat.send_message.assert_called_once_with("How many subscribers are there?")


def test_gemini_client_api_failure_and_quota_fallback():
    """Verify Gemini client falls back gracefully on API errors or 429 quota exhaustion."""
    client = GeminiAgentClient()
    mock_genai_client = MagicMock()
    mock_genai_client.models.generate_content.side_effect = Exception("429 Resource Exhausted: Quota limit reached")
    mock_genai_client.chats.create.side_effect = Exception("503 Service Unavailable")
    
    client.client = mock_genai_client
    
    plan = client.generate_plan(
        user_goal="Predict sales",
        problem_info={"problem_type": "regression"},
        profile_summary={"row_count": 200, "col_count": 5}
    )
    assert plan["planner_source"] == "deterministic_heuristic_engine"
    assert "Ridge" in plan["candidate_models"]
    
    chat_res = client.run_agent_chat(
        user_message="Explain feature importance",
        context_data={"top_features": [{"feature": "income", "importance_pct": 30.5}]}
    )
    assert chat_res["source"] == "deterministic_grounded_engine"
    assert "income" in chat_res["reply"]
