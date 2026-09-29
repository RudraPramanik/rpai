"""Voice transcription request/response schemas."""

from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str = Field(description="Transcribed utterance text")
