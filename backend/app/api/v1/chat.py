"""
AutoDS Conversational Agent API Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from backend.app.agents.chat_agent import answer_chat_query
from backend.app.core.database import SyncSessionLocal, get_db
from backend.app.models.entities import AnalysisRun, ChatMessage, ChatSession, Dataset
from backend.app.schemas.domain import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionResponse,
)


router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post("/chat", response_model=ChatMessageResponse)
async def chat_with_agent(
    req: ChatMessageCreate,
    db: AsyncSession = Depends(get_db)
):
    """Interact with the AutoDS Agent grounded in dataset and experiment evidence."""
    # Find or create session
    session_id = req.session_id
    if session_id:
        res = await db.execute(select(ChatSession).filter(ChatSession.id == session_id))
        session = res.scalar_one_or_none()
        if not session:
            session = ChatSession(id=session_id, dataset_id=req.dataset_id, title=req.content[:30])
            db.add(session)
            await db.flush()
    else:
        session = ChatSession(dataset_id=req.dataset_id, title=req.content[:30])
        db.add(session)
        await db.flush()
        session_id = session.id

    # Record User Message
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=req.content
    )
    db.add(user_msg)
    await db.commit()

    # Gather Dataset & Analysis Context using synchronous DB session for tool compatibility
    sync_db = SyncSessionLocal()
    try:
        ds_id = req.dataset_id or session.dataset_id
        dataset_obj = sync_db.query(Dataset).filter(Dataset.id == ds_id).first() if ds_id else None
        latest_run = sync_db.query(AnalysisRun).filter(AnalysisRun.dataset_id == ds_id).order_by(AnalysisRun.created_at.desc()).first() if ds_id else None
        
        # Recent history
        hist_msgs = sync_db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.desc()).limit(6).all()
        formatted_history = [{"role": m.role, "content": m.content} for m in reversed(hist_msgs)]

        agent_result = answer_chat_query(
            user_message=req.content,
            dataset=dataset_obj,
            latest_run=latest_run,
            session_history=formatted_history,
            sync_db_session=sync_db
        )
    finally:
        sync_db.close()

    # Record Assistant Message
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=agent_result["reply"],
        tool_calls_json=agent_result.get("tool_calls"),
        tool_results_json=agent_result.get("tool_results"),
    )
    db.add(assistant_msg)
    await db.commit()
    await db.refresh(assistant_msg)

    return assistant_msg


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """List all active chat sessions."""
    res = await db.execute(
        select(ChatSession)
        .options(selectinload(ChatSession.messages))
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(res.scalars().all())


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get chat session messages and tool invocation trace."""
    res = await db.execute(
        select(ChatSession)
        .options(selectinload(ChatSession.messages))
        .filter(ChatSession.id == session_id)
    )
    session = res.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return session
