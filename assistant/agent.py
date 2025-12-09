"""
Flight Delay Prediction Agent.

Orchestrates conversation with user to collect flight information
and provide delay predictions.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from assistant.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from assistant.tools import AGENT_TOOLS
from assistant.schemas import FlightSlots, PredictionResult, ChatRequest, ChatResponse
from assistant.rag import get_airport_by_code, get_airline_by_code

logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system_default.md").read_text(encoding="utf-8")


class SessionState:
    """Holds conversation state for a session."""
    def __init__(self):
        self.messages: List = []
        self.slots: FlightSlots = FlightSlots()
        self.last_prediction: Optional[PredictionResult] = None


_sessions: Dict[str, SessionState] = {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@lru_cache(maxsize=1)
def _get_llm_with_tools():
    """Get the tool-bound LLM instance (built once and cached)."""
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    return llm.bind_tools(AGENT_TOOLS)


def _build_slot_status(state: SessionState) -> str:
    """Build JSON string of current slot values."""
    return json.dumps(state.slots.model_dump(), default=str)


def _build_prediction_context(state: SessionState) -> str:
    """Build enriched context when prediction is ready."""
    pred = state.last_prediction
    slots = state.slots
    
    origin_info = get_airport_by_code(slots.origin_airport_code) if slots.origin_airport_code else None
    dest_info = get_airport_by_code(slots.destination_airport_code) if slots.destination_airport_code else None
    airline_info = get_airline_by_code(slots.airline_code) if slots.airline_code else None
    
    origin_name = f"{origin_info['name']} ({slots.origin_airport_code})" if origin_info else slots.origin_airport_code
    dest_name = f"{dest_info['name']} ({slots.destination_airport_code})" if dest_info else slots.destination_airport_code
    airline_name = airline_info['name'] if airline_info else slots.airline_code
    
    prob_pct = round(pred.delay_probability * 100, 1)
    delay_mins = round(pred.delay_minutes)
    
    return f"""PREDICTION COMPLETE.

Communicate the prediction result to the user based ONLY on these facts:

Origin airport: {origin_name}
Destination airport: {dest_name}
Airline: {airline_name}
Departure time: {slots.scheduled_departure_hour}:00
Delay probability: {prob_pct}%
Model prediction: {'delayed' if pred.delayed else 'on time'}
{f'Estimated delay (if delayed): {delay_mins} minutes' if delay_mins > 0 else ''}

Write 1-3 natural sentences in the user's language. Be friendly and professional. Adapt your tone based on the prediction - REASSURING if on time, HELPFUL if delayed. End with a brief note that this is a statistical prediction based on historical data. No Markdown or bullet points."""


def _update_slots_from_tool_calls(state: SessionState, messages: List) -> None:
    """Extract slot values from tool calls and responses."""
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = tool_call.get('name', '')
                args = tool_call.get('args', {})
                logger.debug(f"Tool call: {name} with args: {args}")
                
                if name in ('call_predict_api', 'update_flight_slots'):
                    for field_name in FlightSlots.model_fields:
                        if args.get(field_name) is not None:
                            setattr(state.slots, field_name, args[field_name])
                    logger.info(f"Slots updated via {name}: {state.slots}")

                elif name == 'calculate_distance':
                    if args.get('origin_airport_code'):
                        state.slots.origin_airport_code = args.get('origin_airport_code')
                    if args.get('destination_airport_code'):
                        state.slots.destination_airport_code = args.get('destination_airport_code')
        
        if isinstance(msg, ToolMessage):
            try:
                content = msg.content
                logger.debug(f"ToolMessage content: {content}")
                if isinstance(content, str):
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if 'delay_probability' in data and data['delay_probability'] is not None:
                            state.last_prediction = PredictionResult(
                                delay_probability=data['delay_probability'],
                                delayed=data.get('delayed', False),
                                delay_minutes=data.get('delay_minutes') or 0.0
                            )
                            logger.info(f"Prediction extracted: {state.last_prediction}")
                        if 'distance_miles' in data:
                            state.slots.distance = data['distance_miles']
                        if 'updated_slots' in data:
                            for k, v in data['updated_slots'].items():
                                if hasattr(state.slots, k) and v is not None:
                                    setattr(state.slots, k, v)
                            logger.info(f"Slots updated from ToolMessage: {state.slots}")
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"Failed to parse ToolMessage: {e}")



def extract_text(response):
    content = response.content
    
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return " ".join(parts).strip()

    return str(content or "").strip()

# =============================================================================
#  MAIN AGENT ENTRYPOINT
# =============================================================================

async def run_agent_turn(request: ChatRequest) -> ChatResponse:
    """
    Run a single turn of the agent conversation.
    
    Uses a ReAct-style loop with max n iterations. The LLM can call multiple
    tools per iteration and will continue until it produces a natural language
    response or reaches the iteration limit.
    
    Args:
        request: Chat request with message and optional session_id.
    
    Returns:
        Chat response with reply and current slot state.
    """
    session_id = request.session_id or "default"
    
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    
    state = _sessions[session_id]
    
    try:
        # Add user message to conversation history
        state.messages.append(HumanMessage(content=request.message))
        
        llm_with_tools = _get_llm_with_tools()

        # Build messages for LLM
        all_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=f"CURRENT SLOT STATE:\n{_build_slot_status(state)}"),
        ] + state.messages
        
        max_iterations = 8
        iteration = 0
        final_reply: Optional[str] = None
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[Session {session_id}] Iteration {iteration}/{max_iterations}")
            
            # Call LLM
            response = await llm_with_tools.ainvoke(all_messages)
            
            # If no tool calls, this is the final response
            if not response.tool_calls:
                final_reply = extract_text(response)
                if final_reply:
                    state.messages.append(AIMessage(content=final_reply))
                logger.info(f"[Session {session_id}] LLM produced final response")
                break
            
            # Process tool calls
            logger.info(f"[Session {session_id}] LLM produced {len(response.tool_calls)} tool call(s)")
            all_messages.append(response)

            tool_messages: List[ToolMessage] = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                logger.info(f"[Session {session_id}] Executing: {tool_name}")
                
                # Find tool function
                tool_fn = None
                for t in AGENT_TOOLS:
                    if t.name == tool_name:
                        tool_fn = t
                        break
                
                if tool_fn:
                    # Gate: block call_predict_api if slots incomplete
                    if tool_name == 'call_predict_api' and not state.slots.is_complete():
                        missing = state.slots.missing_fields()
                        result_str = json.dumps({
                            "success": False,
                            "error": f"Cannot predict yet. Missing fields: {missing}",
                            "missing_fields": missing,
                        })
                        logger.warning(f"[Session {session_id}] Blocked call_predict_api - missing: {missing}")
                    else:
                        try:
                            result = tool_fn.invoke(tool_args)
                            result_str = json.dumps(result) if not isinstance(result, str) else result
                        except Exception as e:
                            logger.error(f"[Session {session_id}] Tool {tool_name} failed: {e}")
                            result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
                
                tool_msg = ToolMessage(content=result_str, tool_call_id=tool_id)
                all_messages.append(tool_msg)
                tool_messages.append(tool_msg)

            # Update slots once from this iteration's tool calls + results
            _update_slots_from_tool_calls(state, [response, *tool_messages])

            # If prediction is ready, add context for final response
            if state.last_prediction is not None:
                logger.info(f"[Session {session_id}] Prediction ready, adding context for final response")
                prediction_context = _build_prediction_context(state)
                all_messages.append(SystemMessage(content=prediction_context))
        
        # Fallback if no final reply or empty/whitespace
        if not final_reply or (isinstance(final_reply, str) and not final_reply.strip()):
            missing = state.slots.missing_fields()
            if state.last_prediction:
                # We have prediction but LLM didn't respond - generate basic response
                pred = state.last_prediction
                final_reply = f"Prediction complete: delay probability {round(pred.delay_probability * 100, 1)}%."
            elif missing:
                final_reply = f"To complete the prediction, I still need: {', '.join(missing)}."
            else:
                final_reply = "How can I help you predict your flight delay?"
            state.messages.append(AIMessage(content=final_reply))
        
        # Trim history if too long
        if len(state.messages) > 30:
            state.messages = state.messages[-30:]
        
        logger.info(f"[Session {session_id}] Turn complete. Missing: {state.slots.missing_fields() or 'none'}")
        
        return ChatResponse(
            reply=final_reply,
            slots=state.slots,
            prediction=state.last_prediction,
        )
        
    except Exception as e:
        logger.exception("Agent error")
        return ChatResponse(
            reply=f"An error occurred: {str(e)}",
            slots=state.slots if session_id in _sessions else FlightSlots(),
            prediction=None,
        )


