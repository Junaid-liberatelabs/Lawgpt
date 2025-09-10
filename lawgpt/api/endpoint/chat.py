from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import logging

from lawgpt.api.schema.chat import ChatRequest, ChatResponse
from lawgpt.llm.workflow.graph import create_chat_workflow

logger = logging.getLogger(__name__)
router = APIRouter()

logger.info("Chat endpoint router initialized")

# Single workflow instance for stateless operation
workflow = create_chat_workflow()
logger.info("Global stateless workflow created")


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    """
    Chat endpoint that processes user messages through LangGraph workflow - Stateless
    """
    logger.info(f"Chat endpoint received request - model: {chat_request.llm_model_id}, case_rag: {chat_request.is_case_rag}, law_rag: {chat_request.is_law_rag}, message_length: {len(chat_request.message)}")
    try:
        # Prepare input with proper LangChain message format
        from langchain_core.messages import HumanMessage
        
        # Stateless input - no thread_id or session management
        input_data = {
            "messages": [HumanMessage(content=chat_request.message)],
            "is_case_rag": chat_request.is_case_rag,
            "is_law_rag": chat_request.is_law_rag,
            "llm_model_id": chat_request.llm_model_id,
            "rag_context": []
        }
        logger.info(f"Prepared stateless workflow input data")
        
        # Run the workflow - no config needed for stateless operation
        logger.info(f"Invoking stateless workflow...")
        result = await workflow.ainvoke(input_data)
        logger.info(f"Workflow completed")
        
        # Extract the final response
        final_message = result["messages"][-1].content
        logger.info(f"Chat endpoint response generated - response_length: {len(final_message)}")
        
        return ChatResponse(response=final_message)
        
    except Exception as e:
        logger.error(f"Chat endpoint error - model: {chat_request.llm_model_id}, error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @router.post("/chat/reset")
# async def reset_chat(request: Request, thread_id: str) -> Dict[str, Any]:
#     """
#     Reset chat endpoint - No-op for stateless operation
#     """
#     logger.info(f"Reset chat endpoint called for thread: {thread_id} - No-op for stateless operation")
#     return {"message": f"Chat is stateless - no reset needed for thread: {thread_id}"}


# @router.post("/chat/reset-all")
# async def reset_all_chats(request: Request) -> Dict[str, Any]:
#     """
#     Reset all chat sessions - No-op for stateless operation
#     """
#     logger.info("Reset all chats endpoint called - No-op for stateless operation")
#     return {"message": "Chat is stateless - no reset needed"}