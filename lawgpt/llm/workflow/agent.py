
import yaml
import os
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from lawgpt.core.config import settings
from lawgpt.llm.workflow.custom_llm import CustomLLMChatAgent

logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(self, model_id: str = "gemini"):
        """Initialize ChatAgent with specified model"""
        logger.info(f"ChatAgent initializing with model_id: {model_id}")
        self.model_id = model_id
        if model_id == "custom_llm":
            # For custom LLM, we use the specialized agent
            logger.info("Initializing CustomLLMChatAgent for custom_llm model")
            self.custom_agent = CustomLLMChatAgent()
            self.llm = self.custom_agent.llm
        else:
            logger.info(f"Initializing standard LLM for model: {model_id}")
            self.llm = self._initialize_llm(model_id)
            
        # Only load prompt template for non-custom models
        if model_id != "custom_llm":
            self.prompt_template = self._load_prompt_template()
            logger.info(f"Loaded prompt template for model: {model_id}")
        else:
            self.prompt_template = None
            logger.info("No prompt template needed for custom_llm model")
    
    def _load_prompt_template(self) -> ChatPromptTemplate:
        """Load prompt template from YAML file"""
        try:
            # Adjust path to be relative to the project root
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "chat_prompt.yml")
            logger.info(f"Loading prompt template from: {prompt_path}")
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt_data = yaml.safe_load(file)
            
            system_prompt = prompt_data.get("SYSTEM_PROMPT", "You are a helpful legal assistant.")
            logger.info(f"Loaded system prompt with length: {len(system_prompt)}")
            
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{user_input}")
            ])
        except Exception as e:
            logger.warning(f"Failed to load prompt template from YAML: {e}, using fallback")
            # Fallback prompt if file loading fails
            return ChatPromptTemplate.from_messages([
                ("human", "{user_input}")
            ])
    
    def _initialize_llm(self, model_id: str):
        """Initialize the appropriate LLM based on model_id"""
        if model_id == "gemini":
            logger.info("Initializing ChatGoogleGenerativeAI with gemini-2.0-flash-exp")
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite", 
                temperature=0,
                google_api_key=settings.GOOGLE_API_KEY
            )
        elif model_id == "openai":
            logger.info("Initializing ChatOpenAI with gpt-5-mini")
            return ChatOpenAI(
                model="gpt-5-nano", 
                temperature=0,
                api_key=settings.OPENAI_API_KEY
            )
        else:
            logger.error(f"Unsupported model_id: {model_id}")
            raise ValueError(f"Unsupported model_id: {model_id}")
    
    async def generate_response(self, user_input: str, rag_context: list = None) -> str:
        """Generate response using the configured LLM - Stateless"""
        logger.info(f"ChatAgent generating response - model: {self.model_id}, user_input_length: {len(user_input)}, rag_items: {len(rag_context) if rag_context else 0}")
        try:
            # Handle custom LLM differently 
            if self.model_id == "custom_llm":
                logger.info("Delegating to CustomLLMChatAgent for response generation")
                return await self.custom_agent.generate_response(
                    user_input=user_input,
                    rag_context=rag_context
                )
            
            # Standard LLM handling (gemini, openai)
            logger.info(f"Standard LLM processing for model: {self.model_id}")
            # Prepare context if RAG results are available
            context_text = ""
            if rag_context:
                context_parts = []
                for i, item in enumerate(rag_context):
                    if item.get("type") == "case":
                        content = item.get('content', '')
                        context_parts.append(f"Legal Case: {content}")
                        # Log truncated context preview
                        preview = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"🔍 RAG Context Item {i+1} (Case): {preview}")
                    elif item.get("type") == "law":
                        content = item.get('content', '')
                        context_parts.append(f"Law Reference: {content}")
                        # Log truncated context preview
                        preview = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"🔍 RAG Context Item {i+1} (Law): {preview}")
                
                if context_parts:
                    context_text = f"\n\nRelevant Context:\n{chr(10).join(context_parts)}"
                    logger.info(f"Prepared RAG context with {len(context_parts)} items, total_length: {len(context_text)}")
            
            # Combine user input with context
            full_input = user_input + context_text
            logger.info(f"Generated full input with length: {len(full_input)}")
            
            # Generate response
            prompt = self.prompt_template.format_messages(user_input=full_input)
            logger.info(f"Formatted prompt for {self.model_id}, invoking LLM...")
            response = await self.llm.ainvoke(prompt)
            
            logger.info(f"ChatAgent response generated successfully - model: {self.model_id}, response_length: {len(response.content)}")
            return response.content
            
        except Exception as e:
            logger.error(f"ChatAgent error - model: {self.model_id}, error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
