import requests
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from lawgpt.core.config import settings

logger = logging.getLogger(__name__)


class CustomLLMAPI(BaseChatModel):
    """
    Custom LLM implementation that interfaces with Modal API endpoint
    following the pattern from test_api.py - Stateless, no chat history
    """
    
    api_url: str = Field(default="")
    api_key: str = Field(default="")
    model_name: str = Field(default="custom-modal-llm")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_url = settings.CUSTOM_MODEL_URL or ""
        self.api_key = settings.CUSTOM_MODEL_API_KEY or ""
        self.model_name = "custom-modal-llm"
        
        logger.info(f"CustomLLMAPI initialized with URL: {self.api_url[:50]}{'...' if len(self.api_url) > 50 else ''}")
    
    def _llm_type(self) -> str:
        return "custom_modal_llm"
    
    def _format_messages_for_api(self, messages: List[BaseMessage], rag_context: str = "") -> Dict[str, Any]:
        """
        Convert LangChain messages to the format expected by the Modal API
        following the test_api.py pattern - No chat history
        """
        system_prompt = None
        user_prompt = ""
        
        # Extract system prompt and latest user message
        for message in messages:
            if isinstance(message, SystemMessage):
                system_prompt = message.content
            elif isinstance(message, HumanMessage):
                user_prompt = message.content
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = "You are a helpful legal assistant specializing in Bangladeshi law. Provide accurate, detailed responses."
        
        # Prepare API payload following test_api.py format
        payload = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "chat_history": [],  # Always empty - no chat history
            "rag_context": rag_context,
            "max_new_tokens": 2048,
        }
        
        logger.info(f"Prepared API payload - has_rag_context: {len(rag_context) > 0}, user_prompt_length: {len(user_prompt)}")
        
        return payload
    
    async def _agenerate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> ChatResult:
        """
        Async method to generate response from custom Modal API - Stateless
        """
        # Extract rag_context from kwargs if available
        rag_context = kwargs.get('rag_context', "")
        
        try:
            # Format messages for API
            payload = self._format_messages_for_api(messages, rag_context)
            
            logger.info(f"Sending request to custom LLM API: {self.api_url}")
            logger.info(f"Request details - user_prompt_length: {len(payload['user_prompt'])}, system_prompt_length: {len(payload['system_prompt'])}, rag_context_length: {len(payload['rag_context'])}")
            
            # Make API request
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get('response', '')
                
                logger.info(f"Custom LLM API response received. Model: {result.get('model_name', 'unknown')}, response_length: {len(assistant_response)}, inference_time: {result.get('inference_time', 0):.2f}s")
                
                # Return in LangChain format
                generation = ChatGeneration(message=AIMessage(content=assistant_response))
                return ChatResult(generations=[generation])
            
            else:
                error_msg = f"Custom LLM API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                generation = ChatGeneration(message=AIMessage(content=f"I apologize, but I encountered an API error: {error_msg}"))
                return ChatResult(generations=[generation])
                
        except requests.exceptions.Timeout:
            error_msg = "Custom LLM API request timed out (>5 minutes)"
            logger.error(error_msg)
            generation = ChatGeneration(message=AIMessage(content=f"I apologize, but the request timed out: {error_msg}"))
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Custom LLM API network error: {e}"
            logger.error(error_msg)
            generation = ChatGeneration(message=AIMessage(content=f"I apologize, but I encountered a network error: {error_msg}"))
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_msg = f"Custom LLM API unexpected error: {e}"
            logger.error(error_msg)
            generation = ChatGeneration(message=AIMessage(content=f"I apologize, but I encountered an unexpected error: {error_msg}"))
            return ChatResult(generations=[generation])
    
    def _generate(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> ChatResult:
        """
        Sync method (calls async version)
        """
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._agenerate(messages, stop, **kwargs))


class CustomLLMChatAgent:
    """
    Custom Chat Agent that uses CustomLLMAPI - Stateless, no conversation management
    """
    
    def __init__(self):
        self.llm = CustomLLMAPI()
        self.system_prompt = self._load_system_prompt()
        logger.info(f"CustomLLMChatAgent initialized with system_prompt_length: {len(self.system_prompt)}")
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from YAML file or use default"""
        try:
            import yaml
            import os
            
            prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "chat_prompt.yml")
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt_data = yaml.safe_load(file)
            
            return prompt_data.get("SYSTEM_PROMPT", "You are a helpful legal assistant specializing in Bangladeshi law.")
        except Exception as e:
            logger.warning(f"Could not load system prompt from YAML: {e}")
            return "You are a helpful legal assistant specializing in Bangladeshi law."
    
    async def generate_response(
        self, 
        user_input: str, 
        rag_context: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate response using custom LLM - Stateless, no chat history
        """
        try:
            logger.info(f"CustomLLMChatAgent generating response - user_input_length: {len(user_input)}, rag_items: {len(rag_context) if rag_context else 0}")
            
            # Prepare RAG context string if available
            rag_context_text = ""
            if rag_context:
                context_parts = []
                for i, item in enumerate(rag_context):
                    if item.get("type") == "case":
                        content = item.get('content', '')
                        context_parts.append(f"Legal Case: {content}")
                        # Log truncated context preview
                        preview = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"RAG Context Item {i+1} (Case): {preview}")
                    elif item.get("type") == "law":
                        content = item.get('content', '')
                        context_parts.append(f"Law Reference: {content}")
                        # Log truncated context preview
                        preview = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"RAG Context Item {i+1} (Law): {preview}")
                
                if context_parts:
                    rag_context_text = f"\n\nRelevant Context:\n{chr(10).join(context_parts)}"
                    logger.info(f"Prepared RAG context with {len(context_parts)} items, total_length: {len(rag_context_text)}")
            
            # Prepare messages for the LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input + rag_context_text)
            ]
            
            # Generate response with rag_context only (no thread_id)
            result = await self.llm._agenerate(
                messages, 
                rag_context=rag_context_text
            )
            
            response_content = result.generations[0].message.content
            logger.info(f"CustomLLMChatAgent response generated successfully - response_length: {len(response_content)}")
            return response_content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"