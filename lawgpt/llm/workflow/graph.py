
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from lawgpt.llm.workflow.state import ChatState
from lawgpt.llm.workflow.agent import ChatAgent
from lawgpt.data_pipeline.rag_case_pipeline import CaseRAGPipeline
from lawgpt.data_pipeline.rag_law_pipeline import LawRAGPipeline

logger = logging.getLogger(__name__)


def create_chat_workflow():
    """Create and return the chat workflow graph - Stateless, no memory"""
    logger.info("Creating stateless chat workflow graph...")
    
    # Define workflow nodes
    async def rag_node(state: ChatState) -> ChatState:
        """Node to handle RAG retrieval based on flags"""
        logger.info(f"RAG node starting - case_rag: {state['is_case_rag']}, law_rag: {state['is_law_rag']}")
        
        rag_context = []
        user_message = state["messages"][-1].content
        logger.info(f"RAG node processing message - length: {len(user_message)}, user_message_preview: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        
        # Case RAG
        if state["is_case_rag"]:
            try:
                logger.info("Case RAG: Initializing pipeline and searching...")
                case_pipeline = CaseRAGPipeline()
                case_results = case_pipeline.search_by_text(user_message, limit=3)
                logger.info(f"Case RAG: Retrieved {len(case_results)} results")
                
                for i, result in enumerate(case_results):
                    payload = result["payload"]
                    content = f"""
                    Case Title: {payload.get('case_title', '')}
                    Division: {payload.get('division', '')}
                    Law Category: {payload.get('law_category', '')}
                    Law Act: {payload.get('law_act', '')}
                    Reference: {payload.get('reference', '')}
                    Case Details: {payload.get('case_details', '')}
                    """
                    
                    rag_context.append({
                        "type": "case",
                        "content": content.strip(),
                        "score": result.get('score', 0)
                    })
                    
                    # Log truncated context preview
                    preview = content.strip()[:100] + "..." if len(content.strip()) > 100 else content.strip()
                    logger.info(f"Case RAG Result {i+1}: {preview}")
                
                logger.info(f"Case RAG: Successfully processed {len(case_results)} results into context")
            except Exception as e:
                logger.error(f"Case RAG error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}")
        
        # Law RAG
        if state["is_law_rag"]:
            try:
                logger.info("Law RAG: Initializing pipeline and searching...")
                law_pipeline = LawRAGPipeline()
                law_results = law_pipeline.search_by_text(user_message, limit=3)
                logger.info(f"Law RAG: Retrieved {len(law_results)} results")
                
                for i, result in enumerate(law_results):
                    payload = result["payload"]
                    content = f"""
                    Part Section: {payload.get('part_section', '')}
                    Law Text: {payload.get('law_text', '')}
                    Is Chunked: {payload.get('is_chunked', False)}
                    Chunk Index: {payload.get('chunk_index', 0)} of {payload.get('total_chunks', 1)}
                    """
                    
                    rag_context.append({
                        "type": "law",
                        "content": content.strip(),
                        "score": result.get('score', 0)
                    })
                    
                    # Log truncated context preview
                    preview = content.strip()[:100] + "..." if len(content.strip()) > 100 else content.strip()
                    logger.info(f"Law RAG Result {i+1}: {preview}")
                
                logger.info(f"Law RAG: Successfully processed {len(law_results)} results into context")
            except Exception as e:
                logger.error(f"Law RAG error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}")
        
        # Update state with RAG context
        state["rag_context"] = rag_context
        logger.info(f"RAG node completed - total_context_items: {len(rag_context)}")
        return state
    
    async def llm_node(state: ChatState) -> ChatState:
        """Node to generate LLM response"""
        logger.info(f"LLM node starting - model: {state['llm_model_id']}, rag_items: {len(state.get('rag_context', []))}")
        
        try:
            # Initialize the LLM directly
            logger.info(f"LLM node: Initializing ChatAgent with model: {state['llm_model_id']}")
            chat_agent = ChatAgent(model_id=state["llm_model_id"])
            
            # Get the latest user message
            user_message = ""
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_message = msg.content
                    break
            
            # Generate response using chat agent (which handles custom_llm internally)
            logger.info(f"LLM node: Generating response for user_message_length: {len(user_message)}")
            response_text = await chat_agent.generate_response(
                user_input=user_message,
                rag_context=state["rag_context"]
            )
            logger.info(f"LLM node: Response generated successfully - length: {len(response_text)}")
            
            # Add AI response to messages
            from langchain_core.messages import AIMessage
            logger.info(f"LLM node completed successfully")
            return {"messages": [AIMessage(content=response_text)]}
            
        except Exception as e:
            logger.error(f"LLM node error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=error_response)]}
    
    def end_node(state: ChatState) -> ChatState:
        """End node - just returns the state"""
        logger.info(f"Workflow completed - final_message_count: {len(state.get('messages', []))}")
        return state
    
    # Create the workflow graph
    workflow = StateGraph(ChatState)
    logger.info("StateGraph created with ChatState schema")
    
    # Add nodes
    workflow.add_node("rag", rag_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("end", end_node)
    logger.info("Added workflow nodes: rag, llm, end")
    
    # Define edges
    workflow.add_edge(START, "rag")
    workflow.add_edge("rag", "llm")
    workflow.add_edge("llm", "end")
    workflow.add_edge("end", END)
    logger.info("Defined workflow edges: START->rag->llm->end->END")
    
    # No memory saver needed for stateless workflow
    logger.info("Workflow configured as stateless - no memory checkpointer")
    
    # Compile the workflow without checkpointer for stateless operation
    app = workflow.compile()
    logger.info("Workflow compiled successfully without checkpointer (stateless)")
    
    return app