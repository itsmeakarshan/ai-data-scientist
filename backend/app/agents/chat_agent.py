"""
AutoDS Grounded Chat Agent
Provides evidence-backed conversational responses, data explanations, and automatic safe SQL tool invocation.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from backend.app.agents.gemini_client import gemini_client
from backend.app.core.logging import logger
from backend.app.models.entities import AnalysisRun, Dataset
from backend.app.services.analysis_context_builder import AnalysisContextBuilder
from backend.app.tools.safe_query import execute_safe_sql_query


def answer_chat_query(
    user_message: str,
    dataset: Optional[Dataset] = None,
    latest_run: Optional[AnalysisRun] = None,
    session_history: Optional[List[Dict[str, str]]] = None,
    sync_db_session: Any = None,
    analysis_id: Optional[str] = None,
    report_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    comparison_analysis_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Produce a grounded response with automatic SQL/tool execution if an empirical question is asked.
    Builds full, structured analysis context via AnalysisContextBuilder.
    Uses official Gemini Chat.send_message with automatic function calling when active, and
    deterministic heuristics when running offline.
    """
    tool_calls = None
    tool_results = None

    # 1. Build structured analysis context using AnalysisContextBuilder
    context_data: Dict[str, Any] = {}
    if sync_db_session:
        target_analysis_id = analysis_id or (latest_run.id if latest_run else None)
        target_dataset_id = dataset_id or (dataset.id if dataset else None)

        # Check if user message explicitly requests comparison with another dataset/analysis
        msg_lower_check = user_message.lower()
        comp_id = comparison_analysis_id
        if not comp_id and any(k in msg_lower_check for k in ("compare", "comparison", "versus", "vs")):
            # Look for another completed analysis to compare with
            other_run = sync_db_session.query(AnalysisRun).filter(
                AnalysisRun.status == "COMPLETED",
                AnalysisRun.id != target_analysis_id
            ).order_by(AnalysisRun.created_at.desc()).first()
            if other_run:
                comp_id = other_run.id

        context_data = AnalysisContextBuilder.build_context(
            sync_db=sync_db_session,
            analysis_id=target_analysis_id,
            report_id=report_id,
            dataset_id=target_dataset_id,
            comparison_analysis_id=comp_id
        )

    # 2. Reject destructive SQL attempts immediately
    msg_lower = user_message.lower()
    destructive_keywords = ("drop table", "delete from", "update ", "insert into", "truncate ", "alter table", "attach ", "copy ")
    if any(k in msg_lower for k in destructive_keywords):
        return {
            "reply": "⚠️ **Security Notice:** Disallowed SQL operation detected. AutoDS strictly blocks destructive or modifying statements (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ATTACH`, `TRUNCATE`). Only read-only analytical `SELECT` queries are permitted in the sandboxed environment.",
            "tool_calls": {"tool_name": "security_guard", "blocked": True},
            "tool_results": None,
        }

    # Resolve active dataset file path if available
    ds_file_path = context_data.get("dataset", {}).get("file_path") or (dataset.file_path if dataset else None)

    # 3. Direct Custom SQL Execution
    if (msg_lower.startswith("select ") or msg_lower.startswith("with ")) and ds_file_path:
        try:
            sql_res = execute_safe_sql_query(ds_file_path, user_message)
            tool_calls = {"tool_name": "execute_safe_sql_query", "query": user_message}
            tool_results = sql_res
            row_summary = f"Executed in {sql_res['execution_time_ms']}ms. Returned {sql_res['row_count']} rows:\n\n"
            if sql_res["rows"]:
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

    # 4. Define Tools for Gemini Agent Calling
    agent_tools: List[Callable] = []
    if ds_file_path:
        def execute_analytical_sql(sql_query: str) -> str:
            """Executes a read-only analytical SQL query against the dataset table."""
            try:
                res = execute_safe_sql_query(ds_file_path, sql_query)
                return f"Query returned {res['row_count']} rows: {json.dumps(res['rows'][:10], default=str)}"
            except Exception as ex:
                return f"SQL Error: {ex}"

        agent_tools.append(execute_analytical_sql)

    # 5. Natural language query intent -> Auto-SQL generation fallback
    is_data_query = any(k in msg_lower for k in ("how many", "what percentage", "average", "mean", "count of", "distribution of", "highest", "lowest", "total rows"))
    if is_data_query and ds_file_path:
        sql_candidate = None
        col_types = context_data.get("dataset", {}).get("column_types", {})
        if "how many" in msg_lower or "count" in msg_lower or "total rows" in msg_lower:
            matched_col = None
            for col in col_types.keys():
                if col.lower() in msg_lower:
                    matched_col = col
                    break
            if matched_col:
                sql_candidate = f"SELECT {matched_col}, COUNT(*) as count FROM dataset GROUP BY {matched_col} ORDER BY count DESC LIMIT 10;"
            else:
                sql_candidate = "SELECT COUNT(*) as total_rows FROM dataset;"

        elif "average" in msg_lower or "mean" in msg_lower:
            num_cols = [c for c, t in col_types.items() if t == "numeric"]
            matched_num = next((c for c in num_cols if c.lower() in msg_lower), None)
            if matched_num:
                sql_candidate = f"SELECT AVG({matched_num}) as avg_{matched_num}, MIN({matched_num}) as min_{matched_num}, MAX({matched_num}) as max_{matched_num} FROM dataset;"

        if sql_candidate:
            try:
                sql_res = execute_safe_sql_query(ds_file_path, sql_candidate)
                tool_calls = {"tool_name": "execute_safe_sql_query", "query": sql_candidate}
                tool_results = sql_res
                context_data["sql_query_result"] = sql_res
            except Exception as e:
                logger.debug(f"Auto-SQL query failed: {e}")

    # Generate response via Chat.send_message
    agent_output = gemini_client.run_agent_chat(
        user_message=user_message,
        tools=agent_tools if agent_tools else None,
        context_data=context_data
    )
    reply_text = agent_output.get("reply", "")

    if agent_output.get("tool_calls"):
        tool_calls = agent_output["tool_calls"]

    # If SQL was executed, append evidence badge
    if tool_results and "rows" in tool_results:
        reply_text += f"\n\n> **Verified SQL Evidence:**\n```sql\n{tool_results.get('sql_executed')}\n```\nResult: `{tool_results.get('rows')}`"

    return {
        "reply": reply_text,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
    }

