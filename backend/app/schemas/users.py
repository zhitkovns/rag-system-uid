from pydantic import BaseModel, Field
from typing import List


class UserIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class UserOut(BaseModel):
    short: List[str]
    long: List[str]