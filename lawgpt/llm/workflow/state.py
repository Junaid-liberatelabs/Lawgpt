from typing import Annotated, TypedDict, List, Dict, Any
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class ChatState(TypedDict):
    """State for the chat workflow - Stateless, no thread_id"""
    is_case_rag: bool
    is_law_rag: bool
    llm_model_id: str
    messages: Annotated[List[AnyMessage], add_messages]
    rag_context: List[Dict[str, Any]]
