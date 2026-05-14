from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import HttpUrl

from src.transcripts.models import Transcript, TranscriptSegment
from src.transcripts.youtube import extract_video_id


class TranscriptFetchError(RuntimeError):
    pass


class SuperdataTranscriptFetcher:
    """Fetch transcripts from Supadata while preserving the spec's env naming."""

    endpoint = "https://api.supadata.ai/v1/transcript"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
        max_poll_seconds: float = 90.0,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_seconds = max_poll_seconds

    def fetch(self, url: str) -> Transcript:
        video_id = extract_video_id(url)
        data = self._request_transcript(url)
        return self._normalize_response(url=url, video_id=video_id, data=data)

    def _request_transcript(self, url: str) -> dict[str, Any]:
        headers = {"x-api-key": self.api_key}
        params: dict[str, Any] = {"url": url, "text": "false", "mode": "auto"}
        try:
            response = httpx.get(
                self.endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise TranscriptFetchError(f"Supadata transcript request failed: {exc}") from exc

        if response.status_code == 202:
            job_id = response.json().get("jobId")
            if not job_id:
                raise TranscriptFetchError("Supadata returned 202 without jobId")
            return self._poll_job(job_id)

        if response.status_code >= 400:
            raise TranscriptFetchError(
                f"Supadata transcript request failed with HTTP {response.status_code}: "
                f"{response.text}"
            )
        return response.json()

    def _poll_job(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.max_poll_seconds
        url = f"{self.endpoint}/{job_id}"
        headers = {"x-api-key": self.api_key}
        while time.monotonic() < deadline:
            response = httpx.get(url, headers=headers, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                raise TranscriptFetchError(
                    f"Supadata job status failed with HTTP {response.status_code}: "
                    f"{response.text}"
                )
            data = response.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "failed":
                raise TranscriptFetchError(f"Supadata transcript job failed: {data}")
            time.sleep(self.poll_interval_seconds)
        raise TranscriptFetchError(f"Supadata transcript job timed out: {job_id}")

    def _normalize_response(
        self, url: str, video_id: str, data: dict[str, Any]
    ) -> Transcript:
        content = data.get("content") or data.get("result")
        language = data.get("lang") or data.get("language")
        segments: list[TranscriptSegment] = []

        if isinstance(content, str):
            raw_text = content.strip()
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                offset_ms = item.get("offset")
                duration_ms = item.get("duration")
                start_seconds = (
                    float(offset_ms) / 1000 if isinstance(offset_ms, int | float) else None
                )
                end_seconds = None
                if start_seconds is not None and isinstance(duration_ms, int | float):
                    end_seconds = start_seconds + (float(duration_ms) / 1000)
                segments.append(
                    TranscriptSegment(
                        text=text,
                        offset_ms=int(offset_ms) if isinstance(offset_ms, int | float) else None,
                        duration_ms=(
                            int(duration_ms) if isinstance(duration_ms, int | float) else None
                        ),
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        language=item.get("lang"),
                    )
                )
            raw_text = " ".join(segment.text for segment in segments).strip()
            if not language and content:
                first = next((item for item in content if isinstance(item, dict)), {})
                language = first.get("lang")
        else:
            raise TranscriptFetchError("Supadata response did not include transcript content")

        if not raw_text:
            raise TranscriptFetchError("Supadata returned an empty transcript")

        return Transcript(
            video_id=video_id,
            url=HttpUrl(url),
            title=data.get("title"),
            language=language,
            provider="supadata",
            raw_text=raw_text,
            segments=segments,
            fetched_at=datetime.now(timezone.utc),
        )
