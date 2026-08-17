"""
AutoDS Grounded Chat Agent
Provides evidence-backed conversational responses, data explanations, and automatic safe SQL tool invocation.
"""

import re
from typing import Any, Dict, List, Optional
from backend.app.agents.gemini_client import gemini_client
from backend.app.core.logging import logger
from backend.app.models.entities import AnalysisRun, Dataset, ModelRecord, Report
from backend.app.tools.safe_query import execute_safe_sql_query


def answer_chat_query(
    user_message: str,
    dataset: Optional[Dataset],
    latest_run: Optional[AnalysisRun],
    session_history: List[Dict[str, str]],
    sync_db_session: Any
) -> Dict[str, Any]:
    """
    Produce a grounded response with automatic SQL/tool execution if an empirical question is asked.
    """
    tool_calls = None
    tool_results = None
    context_data: Dict[str, Any] = {}

    if dataset:
        context_data["dataset_name"] = dataset.name
        context_data["row_count"] = dataset.row_count
        context_data["col_count"] = dataset.col_count
        if dataset.profile:
            context_data["dataset_profile"] = {
                "summary_stats": dataset.profile.summary_stats,
                "missingness_report": dataset.profile.missingness_report,
                "column_types": dataset.profile.column_types,
                "candidate_targets": dataset.profile.candidate_targets,
            }

    if latest_run:
        context_data["problem_type"] = latest_run.problem_type
        context_data["target_column"] = latest_run.target_column
        context_data["user_goal"] = latest_run.user_goal
        context_data["critic_findings"] = latest_run.critic_findings_json

        # Fetch champion model if available
        if latest_run.final_model_id:
            champion = sync_db_session.query(ModelRecord).filter(ModelRecord.id == latest_run.final_model_id).first()
            if champion:
                context_data["best_model"] = {
                    "model_name": champion.name,
                    "task_type": champion.task_type,
                    "metrics": champion.metrics_json,
                }
                context_data["top_features"] = champion.feature_importance_json.get("rankings", [])[:8]

    # Check if user passed direct SQL or asked a direct statistical/aggregation query on the data
    msg_lower = user_message.lower()
    
    # 1. Reject destructive SQL attempts immediately
    destructive_keywords = ("drop table", "delete from", "update ", "insert into", "truncate ", "alter table", "attach ", "copy ")
    if any(k in msg_lower for k in destructive_keywords):
        return {
            "reply": "⚠️ **Security Notice:** Disallowed SQL operation detected. AutoDS strictly blocks destructive or modifying statements (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ATTACH`, `TRUNCATE`). Only read-only analytical `SELECT` queries are permitted in the sandboxed environment.",
            "tool_calls": {"tool_name": "security_guard", "blocked": True},
            "tool_results": None,
        }

    # 2. Direct Custom SQL Execution
    if (msg_lower.startswith("select ") or msg_lower.startswith("with ")) and dataset and dataset.file_path:
        try:
            sql_res = execute_safe_sql_query(dataset.file_path, user_message)
            tool_calls = {"tool_name": "execute_safe_sql_query", "query": user_message}
            tool_results = sql_res
            row_summary = f"Executed in {sql_res['execution_time_ms']}ms. Returned {sql_res['row_count']} rows:\n\n"
            if sql_res["rows"]:
                import json
                row_summary += f"```json\n{json.dumps(sql_res['rows'][:10], indent=2, default=str)}\n```"
            return {
                "reply": f"**Analytical SQL Execution Result:**\n\n{row_summary}",
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            }
        except Exception as e:
            return {
                "reply": f"**SQL Query Error:** {str(e)}",
                "tool_calls": {"tool_name": "execute_safe_sql_query", "error": str(e)},
                "tool_results": None,
            }

    # 3. Natural language query intent -> Auto-SQL generation
    is_data_query = any(k in msg_lower for k in ("how many", "what percentage", "average", "mean", "count of", "distribution of", "highest", "lowest", "total rows"))
    if is_data_query and dataset and dataset.file_path:
        sql_candidate = None
        if "how many" in msg_lower or "count" in msg_lower or "total rows" in msg_lower:
            matched_col = None
            for col in (dataset.profile.column_types.keys() if dataset.profile else []):
                if col.lower() in msg_lower:
                    matched_col = col
                    break
            if matched_col:
                sql_candidate = f"SELECT {matched_col}, COUNT(*) as count FROM dataset GROUP BY {matched_col} ORDER BY count DESC LIMIT 10;"
            else:
                sql_candidate = "SELECT COUNT(*) as total_rows FROM dataset;"

        elif "average" in msg_lower or "mean" in msg_lower:
            num_cols = [c for c, t in (dataset.profile.column_types.items() if dataset.profile else []) if t == "numeric"]
            matched_num = next((c for c in num_cols if c.lower() in msg_lower), None)
            if matched_num:
                sql_candidate = f"SELECT AVG({matched_num}) as avg_{matched_num}, MIN({matched_num}) as min_{matched_num}, MAX({matched_num}) as max_{matched_num} FROM dataset;"

        if sql_candidate:
            try:
                sql_res = execute_safe_sql_query(dataset.file_path, sql_candidate)
                tool_calls = {"tool_name": "execute_safe_sql_query", "query": sql_candidate}
                tool_results = sql_res
                context_data["sql_query_result"] = sql_res
            except Exception as e:
                logger.debug(f"Auto-SQL query failed: {e}")

    # Generate grounded reply
    reply_text = gemini_client.chat_response(
        user_message=user_message,
        conversation_history=session_history,
        context_data=context_data
    )

    # If SQL was executed, append evidence badge
    if tool_results and "rows" in tool_results:
        reply_text += f"\n\n> **Verified SQL Evidence:**\n```sql\n{tool_results.get('sql_executed')}\n```\nResult: `{tool_results.get('rows')}`"

    return {
        "reply": reply_text,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
    }
