"""OpenAI-compatible hosted transcription (NVIDIA NIM / NVCF ASR HTTP)."""

from __future__ import annotations

import httpx

from app.providers.litellm_provider import ProviderConfigError


class OpenAICompatibleTranscriptionProvider:
    """POST multipart audio to `{api_base}/audio/transcriptions`.

    Used for NVIDIA NVCF ASR (Parakeet) free endpoints that speak the OpenAI
    transcription shape. Callers depend on TranscriptionProvider, not this class.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        api_base: str,
        language: str | None = "en-US",
        timeout_seconds: float = 90.0,
        send_model_field: bool = False,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ProviderConfigError(
                "NVIDIA_NIM_API_KEY is not set. Add it to backend/.env for voice "
                "transcription (see .env.example)."
            )
        base = (api_base or "").strip().rstrip("/")
        if not base:
            raise ProviderConfigError(
                "STT_API_BASE is not set. Point it at the NVIDIA NVCF ASR base "
                "(e.g. Parakeet invocation URL .../v1). See backend/.env.example."
            )
        self._model = model
        self._api_key = str(api_key).strip()
        self._api_base = base
        self._language = language
        self._timeout = timeout_seconds
        self._send_model_field = send_model_field

    @property
    def model_id(self) -> str:
        return self._model

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str | None = None,
    ) -> str:
        if not audio:
            raise ValueError("audio payload is empty")

        mime = content_type or "application/octet-stream"
        data: dict[str, str] = {}
        if self._language:
            data["language"] = self._language
        if self._send_model_field and self._model:
            data["model"] = self._model
        data["response_format"] = "json"

        files = {"file": (filename or "audio.webm", audio, mime)}
        url = f"{self._api_base}/audio/transcriptions"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files=files,
                    data=data,
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"STT request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text.strip()[:400] or response.reason_phrase
            raise RuntimeError(
                f"STT provider returned {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("STT provider returned non-JSON body") from exc

        if isinstance(payload, dict):
            text = payload.get("text")
            if text is None:
                raise RuntimeError("STT provider JSON missing 'text' field")
            return str(text).strip()
        if isinstance(payload, str):
            return payload.strip()
        raise RuntimeError("Unexpected STT response shape")
