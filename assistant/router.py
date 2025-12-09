"""
FastAPI router for the Flight Delay Prediction Agent.
"""

from fastapi import APIRouter

from assistant.schemas import ChatRequest, ChatResponse
from assistant.agent import run_agent_turn

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(request: ChatRequest):
    """
    Chat with the Flight Delay Prediction Agent.
    
    Send a message and receive a response. The agent will ask clarifying
    questions until it has all the information needed to make a prediction.
    
    Args:
        request: Chat request with message and optional session_id.
    
    Returns:
        Agent response with reply text and current slot state.
    """
    return await run_agent_turn(request)
