from pydantic import BaseModel

class TextSubmission(BaseModel):
    text: str
    mode: str
