from pydantic import BaseModel, Field

class CaseSummarizerSchema(BaseModel):
    case_summary: str = Field(description="short and detailed summary of the case")