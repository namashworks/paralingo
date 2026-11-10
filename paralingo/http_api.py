# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""HTTP API for Paralingo, protected by API keys.

    pip install "paralingo[api]"
    python -m paralingo.api_keys --keys keys.json create --name my-bot --scopes profiles:read profiles:write audio:analyze
    python -m paralingo.http_api --root ./profiles --keys keys.json --port 8000

Send the key as ``Authorization: Bearer spk_...`` or ``X-API-Key: spk_...``.
Interactive docs are served at ``/docs``.

Every key belongs to one tenant, and each tenant's profiles live in their own folder, so customers
never see each other's data. Remote clients cannot pass server file paths; audio is uploaded instead.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import __url__, __version__
from .agent_tools import TOOL_SPECS, SpeechProfileTools
from .api_keys import ApiKey, KeyStore
from .engine import SpeechProfileEngine
from .store import (ConcurrencyConflict, ConsentError, LockTimeout, ProfileError, ProfileExists, ProfileNotFound,
                    SpeechProfileStore)

try:
    from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise ImportError('The HTTP API needs extra packages: pip install "paralingo[api]"') from e

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
READ_TOOLS = {"paralingo_get_summary", "paralingo_get_guidance", "paralingo_read_section",
              "paralingo_get_trend", "paralingo_query_observations"}
TOOL_SCOPE = {s["name"]: ("profiles:read" if s["name"] in READ_TOOLS else
                          "audio:analyze" if s["name"] == "paralingo_analyze_utterance" else
                          "profiles:write") for s in TOOL_SPECS}
STATUS_FOR = {"ProfileNotFound": 404, "ConsentError": 403, "ConcurrencyConflict": 409, "ProfileExists": 409,
              "LockTimeout": 503}
ATTRIBUTION = f"paralingo/{__version__} (+{__url__})"


class SessionIn(BaseModel):
    channel: str | None = None
    context: str | None = None


class _RateLimiter:
    """In-memory token bucket per key (per process)."""

    def __init__(self) -> None:
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key_id: str, per_minute: int) -> float:
        """Return 0 if allowed, else seconds to wait."""
        now = time.monotonic()
        rate = per_minute / 60.0
        with self._lock:
            tokens, last = self._b.get(key_id, (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - last) * rate)
            if tokens < 1.0:
                self._b[key_id] = (tokens, now)
                return (1.0 - tokens) / rate
            self._b[key_id] = (tokens - 1.0, now)
            return 0.0


def _error_status(exc: Exception) -> int:
    return STATUS_FOR.get(type(exc).__name__, 400)


def create_app(root: str | os.PathLike[str], keys_file: str | os.PathLike[str]) -> FastAPI:
    root_path = Path(root)
    keys = KeyStore(keys_file)
    limiter = _RateLimiter()
    stores: dict[str, SpeechProfileStore] = {}
    stores_lock = threading.Lock()

    app = FastAPI(
        title="Paralingo API",
        version=__version__,
        description=("Read and write per-speaker speech profiles: prosody, pauses, voice quality, fluency, "
                     f"emotion, skills, accent and agent guidance.\n\nOpen source project: {__url__}"),
        license_info={"name": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
        contact={"name": "Namash Aggarwal", "url": __url__},
    )

    @app.middleware("http")
    async def attribution_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Powered-By"] = ATTRIBUTION
        return response

    def store_for(tenant: str) -> SpeechProfileStore:
        with stores_lock:
            if tenant not in stores:
                stores[tenant] = SpeechProfileStore(root_path / tenant)
            return stores[tenant]

    def auth(request: Request) -> ApiKey:
        hdr = request.headers.get("authorization", "")
        presented = hdr[7:].strip() if hdr.lower().startswith("bearer ") else request.headers.get("x-api-key")
        key = keys.verify(presented)
        if key is None:
            raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        wait = limiter.check(key.key_id, key.rate_per_minute)
        if wait:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(int(wait) + 1)})
        return key

    def need(key: ApiKey, scope: str) -> None:
        if not key.allows(scope):
            raise HTTPException(403, f"API key lacks scope '{scope}'")

    def engine(key: ApiKey) -> SpeechProfileEngine:
        return SpeechProfileEngine(store_for(key.tenant), agent_id=f"api:{key.name}")

    def tools(key: ApiKey) -> SpeechProfileTools:
        # audio_root=None: remote callers may not reference server file paths
        return SpeechProfileTools(store_for(key.tenant), agent_id=f"api:{key.name}", audio_root=None)

    def run(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except (ProfileNotFound, ConsentError, ConcurrencyConflict, ProfileExists, LockTimeout, ProfileError,
                ValueError) as exc:
            raise HTTPException(_error_status(exc), f"{type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------ routes
    @app.get("/v1/health", tags=["meta"])
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "project": __url__}

    @app.get("/v1/tools", tags=["tools"])
    def list_tools(format: str = "anthropic", key: ApiKey = Depends(auth)) -> list[dict[str, Any]]:
        if format not in ("anthropic", "openai", "mcp"):
            raise HTTPException(400, "format must be anthropic, openai or mcp")
        allowed = {n for n, s in TOOL_SCOPE.items() if key.allows(s)}
        defs = tools(key).definitions(format)
        name_of = (lambda d: d["function"]["name"]) if format == "openai" else (lambda d: d["name"])
        return [d for d in defs if name_of(d) in allowed]

    @app.post("/v1/tools/{name}", tags=["tools"])
    def call_tool(name: str, body: dict[str, Any] | None = None, key: ApiKey = Depends(auth)) -> JSONResponse:
        if name not in TOOL_SCOPE:
            raise HTTPException(404, f"unknown tool '{name}'")
        need(key, TOOL_SCOPE[name])
        args = dict(body or {})
        if "audio_path" in args:
            raise HTTPException(400, "audio_path is not accepted over HTTP; upload audio to /v1/speakers/{id}/utterances")
        out = json.loads(tools(key).execute(name, args))
        if out["ok"]:
            return JSONResponse(out)
        etype = out["error"].split(":", 1)[0]
        return JSONResponse(out, status_code=STATUS_FOR.get(etype, 400))

    @app.get("/v1/speakers", tags=["speakers"])
    def list_speakers(key: ApiKey = Depends(auth)) -> list[str]:
        need(key, "profiles:read")
        return store_for(key.tenant).list_speakers()

    @app.get("/v1/speakers/{speaker_id}/summary", tags=["speakers"])
    def summary(speaker_id: str, key: ApiKey = Depends(auth)) -> dict[str, Any]:
        need(key, "profiles:read")
        return run(engine(key).get_summary, speaker_id)

    @app.get("/v1/speakers/{speaker_id}/guidance", tags=["speakers"])
    def guidance(speaker_id: str, key: ApiKey = Depends(auth)) -> dict[str, Any]:
        need(key, "profiles:read")
        p = run(engine(key).get_profile, speaker_id)
        return {"guidance": p["agent_guidance"], "current_emotion": p["assessments"]["emotional_state"].get("current")}

    @app.post("/v1/speakers/{speaker_id}/sessions", tags=["sessions"], status_code=201)
    def start_session(speaker_id: str, body: SessionIn | None = None, key: ApiKey = Depends(auth)) -> dict[str, str]:
        need(key, "profiles:write")
        b = body or SessionIn()
        return {"session_id": run(engine(key).start_session, speaker_id, channel=b.channel, context=b.context)}

    @app.post("/v1/speakers/{speaker_id}/sessions/{session_id}/end", tags=["sessions"])
    def end_session(speaker_id: str, session_id: str, key: ApiKey = Depends(auth)) -> dict[str, Any]:
        need(key, "profiles:write")
        return run(engine(key).end_session, speaker_id, session_id)

    @app.post("/v1/speakers/{speaker_id}/utterances", tags=["sessions"])
    def analyze(
        speaker_id: str,
        session_id: str | None = Form(None),
        transcript: str | None = Form(None),
        response_latency_s: float | None = Form(None),
        t_start: float | None = Form(None),
        t_end: float | None = Form(None),
        word_timestamps: str | None = Form(None, description="JSON list of {word, start, end}"),
        external: str | None = Form(None, description="JSON object of other models' outputs"),
        audio: UploadFile | None = File(None, description="WAV (PCM) audio of this user turn"),
        key: ApiKey = Depends(auth),
    ) -> dict[str, Any]:
        need(key, "audio:analyze")
        try:
            words = json.loads(word_timestamps) if word_timestamps else None
            ext = json.loads(external) if external else None
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"invalid JSON field: {exc}") from exc
        if audio is None and not transcript and not words:
            raise HTTPException(400, "send at least one of: audio, transcript, word_timestamps")
        tmp_path = None
        try:
            if audio is not None:
                data = audio.file.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "audio larger than 25 MB")
                if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
                    raise HTTPException(415, "audio must be a WAV file")
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            try:
                return run(engine(key).analyze_utterance, speaker_id, session_id=session_id, transcript=transcript,
                           audio_path=tmp_path, word_timestamps=words, t_start=t_start, t_end=t_end,
                           response_latency_s=response_latency_s, external=ext,
                           audio_ref=audio.filename if audio is not None else None)
            except Exception as exc:
                if isinstance(exc, HTTPException):
                    raise
                if tmp_path and type(exc).__name__ in {"Error", "EOFError"}:  # wave module errors
                    raise HTTPException(415, f"could not read WAV: {exc}") from exc
                raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)  # audio is never kept on the server

    @app.delete("/v1/speakers/{speaker_id}", tags=["speakers"])
    def erase(speaker_id: str, key: ApiKey = Depends(auth)) -> dict[str, Any]:
        need(key, "admin")
        return {"deleted": run(store_for(key.tenant).delete, speaker_id)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Paralingo HTTP API")
    ap.add_argument("--root", default="profiles", help="directory for tenant profile folders")
    ap.add_argument("--keys", default="keys.json", help="API keys file created with paralingo.api_keys")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn

    uvicorn.run(create_app(a.root, a.keys), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
