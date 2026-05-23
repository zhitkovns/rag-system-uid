from typing import List, Optional

from pydantic import BaseModel, Field


class UserIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    use_llm: Optional[bool] = None


class UserOut(BaseModel):
    answer: Optional[str] = None
    answer_short: Optional[str] = None
    answer_long: Optional[str] = None
    short: List[str]
    long: List[str]
    llm_used: bool = False