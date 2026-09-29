"""POST /api/voice/transcribe — speech-to-text only (answers go through /api/chat)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.providers.factory import get_transcription_provider
from app.providers.litellm_provider import ProviderConfigError
from app.schemas.voice import TranscribeResponse

router = APIRouter(tags=["voice"])


@router.post("/api/voice/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio blob from MediaRecorder"),
) -> TranscribeResponse:
    """Accept an audio upload and return transcribed text. Does not call chat/RAG."""
    settings = get_settings()
    filename = file.filename or "audio.webm"
    content_type = file.content_type

    try:
        audio = await file.read()
    except Exception as exc:  # noqa: BLE001 — surface as client error
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}") from exc

    if not audio:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    max_bytes = settings.stt_max_upload_bytes
    if len(audio) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio upload exceeds the {max_bytes} byte limit. "
                "Record a shorter clip and try again."
            ),
        )

    try:
        provider = get_transcription_provider(settings)
        text = provider.transcribe(
            audio,
            filename=filename,
            content_type=content_type,
        )
    except ProviderConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — map provider failures
        raise HTTPException(
            status_code=503,
            detail=f"Transcription unavailable: {exc}",
        ) from exc

    return TranscribeResponse(text=text)
