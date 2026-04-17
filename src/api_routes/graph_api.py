from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from src.api_routes.apis_schemas.graph_schemas import (
    UserMessage,
    ChatResponse,
    GraphStateResponse,
)
from src.Services.graph_service import get_compiled_graph
from src.Services.retrieval_service import get_tree_retrieval
from loguru import logger
import uuid

router = APIRouter(prefix="/graph", tags=["Graph Agent"])

retrieval_service = get_tree_retrieval()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(message: UserMessage):
    """
    Send a message to the LangGraph agent and get a response.
    """
    thread_id = message.thread_id or str(uuid.uuid4())
    logger.info(f"Received message for thread {thread_id}: {message.content[:50]}...")

    try:
        agent = get_compiled_graph()

        # Get document titles as required by the graph state
        docs_titles = retrieval_service.get_docs_titles()

        # Initial state - Include user_query and docs_titles as expected by the graph nodes
        inputs = {
            "messages": [HumanMessage(content=message.content)],
            "user_query": message.content,
            "docs_titles": docs_titles,
        }
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke agent
        result = await agent.ainvoke(inputs, config=config)

        # Extract last message or relevant output
        # Based on GraphState, the messages list will contain the conversation
        last_message = result["messages"][-1]

        return {
            "thread_id": thread_id,
            "response": last_message.content
            if hasattr(last_message, "content")
            else str(last_message),
            "full_state": {
                k: v for k, v in result.items() if k != "messages"
            },  # Return other state bits but exclude bulky messages
        }

    except Exception as e:
        logger.error(f"Error in chat_with_agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{thread_id}", response_model=GraphStateResponse)
async def get_agent_state(thread_id: str):
    """
    Retrieve the current state of a conversation by thread_id.
    """
    logger.info(f"Fetching state for thread {thread_id}")
    try:
        agent = get_compiled_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = await agent.aget_state(config)
        return {"thread_id": thread_id, "state": state.values if state else None}
    except Exception as e:
        logger.error(f"Error fetching agent state: {e}")
        raise HTTPException(status_code=500, detail=str(e))
