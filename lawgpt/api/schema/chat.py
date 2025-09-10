from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    llm_model_id: str
    is_case_rag: bool
    is_law_rag: bool

class ChatResponse(BaseModel):
    response: str
