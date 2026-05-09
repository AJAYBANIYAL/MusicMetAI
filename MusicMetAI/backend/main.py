"""Music Met AI — FastAPI service for link analysis (metadata stub)."""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

app = FastAPI(
    title="Music Met AI API",
    description="Resolve pasted HTTPS music links into structured context (demo / stub).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str | list[str]]:
    """Browser-friendly root: the SPA is served by nginx on port 8080 in Docker."""
    return {
        "service": "Music Met AI API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "analyse": "POST /api/analyse with JSON body {\"url\": \"https://...\"}",
        "ui": "With docker compose, open http://localhost:8080 for the web app (this port is the API).",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


# Ephemeral YouTube MP3: in-memory catalog + temp files (no DB). Removed on DELETE or when the tab closes.
_AUDIO_LOCK = threading.Lock()
_AUDIO_FILES: dict[str, Path] = {}
_AUDIO_CREATED_AT: dict[str, float] = {}
_AUDIO_STORE = Path(tempfile.gettempdir()) / "musicmetai_youtube_mp3"
_AUDIO_STORE.mkdir(parents=True, exist_ok=True)
_AUDIO_EXTS = frozenset({".mp3"})
_MEDIA_TYPES: dict[str, str] = {
    ".mp3": "audio/mpeg",
}

MAX_MP3_UPLOAD_BYTES = 80 * 1024 * 1024
AUDIO_TTL_SECONDS = 30 * 60
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MIN_NOTE_SECONDS = 0.05
MIN_NOTE_VELOCITY = 0.18
MERGE_GAP_SECONDS = 0.06
TIMESTAMP_QUANT = 0.01


class PitchNote(BaseModel):
    start_s: float
    end_s: float
    midi: int
    note: str
    velocity: float


class InstrumentGuess(BaseModel):
    name: str
    confidence: float
    details: dict[str, float] | None = None


class ChordEvent(BaseModel):
    start_s: float
    end_s: float
    chord: str
    confidence: float


def _midi_to_note_name(midi: int) -> str:
    m = int(midi)
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


def _basic_pitch_onnx_model() -> str:
    """Use bundled ONNX weights: default *.tflite fails with tflite-runtime in this image."""
    from basic_pitch import ICASSP_2022_MODEL_PATH

    return str(Path(ICASSP_2022_MODEL_PATH).with_suffix(".onnx"))


def _run_basic_pitch(path: Path):
    from basic_pitch.inference import predict

    with contextlib.redirect_stdout(io.StringIO()):
        return predict(str(path), model_or_model_path=_basic_pitch_onnx_model())


def _note_events_to_models(note_events) -> list[PitchNote]:
    cleaned: list[tuple[float, float, int, float]] = []
    for row in note_events:
        start_s = float(row[0])
        end_s = float(row[1])
        midi = int(row[2])
        amp = float(row[3])
        if end_s <= start_s:
            continue
        if (end_s - start_s) < MIN_NOTE_SECONDS:
            continue
        if amp < MIN_NOTE_VELOCITY:
            continue
        cleaned.append((start_s, end_s, midi, amp))

    if not cleaned:
        return []

    # Merge same-MIDI notes that are extremely close (common Basic Pitch jitter).
    cleaned.sort(key=lambda n: (n[2], n[0], n[1]))
    merged: list[tuple[float, float, int, float]] = []
    for start_s, end_s, midi, amp in cleaned:
        if not merged:
            merged.append((start_s, end_s, midi, amp))
            continue
        p_start, p_end, p_midi, p_amp = merged[-1]
        if midi == p_midi and start_s <= (p_end + MERGE_GAP_SECONDS):
            merged[-1] = (p_start, max(p_end, end_s), p_midi, max(p_amp, amp))
        else:
            merged.append((start_s, end_s, midi, amp))

    # Remove near-duplicate overlaps (same MIDI, almost same interval) keeping stronger one.
    dedup: list[tuple[float, float, int, float]] = []
    for cur in merged:
        c_start, c_end, c_midi, c_amp = cur
        replaced = False
        for i, ex in enumerate(dedup):
            e_start, e_end, e_midi, e_amp = ex
            if c_midi != e_midi:
                continue
            if abs(c_start - e_start) <= 0.03 and abs(c_end - e_end) <= 0.03:
                if c_amp > e_amp:
                    dedup[i] = cur
                replaced = True
                break
        if not replaced:
            dedup.append(cur)

    out: list[PitchNote] = []
    for start_s, end_s, midi, amp in sorted(dedup, key=lambda n: (n[0], n[2])):
        q_start = round(round(start_s / TIMESTAMP_QUANT) * TIMESTAMP_QUANT, 3)
        q_end = round(round(end_s / TIMESTAMP_QUANT) * TIMESTAMP_QUANT, 3)
        if q_end <= q_start:
            q_end = round(q_start + TIMESTAMP_QUANT, 3)
        out.append(
            PitchNote(
                start_s=q_start,
                end_s=q_end,
                midi=midi,
                note=_midi_to_note_name(midi),
                velocity=round(amp, 4),
            )
        )
    return out


def _midi_var_len(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 0x7F
    while True:
        value >>= 7
        if value == 0:
            break
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def _build_midi_bytes(notes: list[PitchNote]) -> bytes:
    # Format-0 MIDI file with a single track.
    ticks_per_quarter = 480
    tempo_micro_per_quarter = 500_000  # 120 BPM
    ticks_per_second = (ticks_per_quarter * 1_000_000) / tempo_micro_per_quarter

    events: list[tuple[int, bytes]] = []
    for n in notes:
        start_tick = max(0, int(round(float(n.start_s) * ticks_per_second)))
        end_tick = max(start_tick + 1, int(round(float(n.end_s) * ticks_per_second)))
        midi = max(0, min(127, int(n.midi)))
        vel = max(1, min(127, int(round(float(n.velocity) * 127))))
        events.append((start_tick, bytes((0x90, midi, vel))))
        events.append((end_tick, bytes((0x80, midi, 0))))

    # Deterministic order for simultaneous events:
    # note-off before note-on at same tick to avoid stuck-note overlaps.
    events.sort(key=lambda e: (e[0], 0 if (e[1][0] & 0xF0) == 0x80 else 1, e[1][1]))

    track = bytearray()
    # Tempo meta event at time 0.
    track.extend(_midi_var_len(0))
    track.extend(b"\xFF\x51\x03")
    track.extend(tempo_micro_per_quarter.to_bytes(3, "big"))

    last_tick = 0
    for tick, msg in events:
        delta = tick - last_tick
        track.extend(_midi_var_len(delta))
        track.extend(msg)
        last_tick = tick

    # End of track
    track.extend(_midi_var_len(0))
    track.extend(b"\xFF\x2F\x00")

    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big")  # format 0
    header += (1).to_bytes(2, "big")  # 1 track
    header += ticks_per_quarter.to_bytes(2, "big")
    chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    return header + chunk


def _detect_chord_from_pcs(pcs: set[int]) -> tuple[str | None, float]:
    if len(pcs) < 2:
        return None, 0.0
    triads = [
        ("maj", {0, 4, 7}),
        ("min", {0, 3, 7}),
        ("dim", {0, 3, 6}),
        ("aug", {0, 4, 8}),
        ("sus2", {0, 2, 7}),
        ("sus4", {0, 5, 7}),
    ]
    best_name: str | None = None
    best_score = 0.0
    for root in range(12):
        norm = {(pc - root) % 12 for pc in pcs}
        for suffix, pattern in triads:
            overlap = len(norm & pattern)
            score = overlap / max(len(pattern), len(norm))
            if score > best_score:
                root_name = _NOTE_NAMES[root]
                best_name = root_name if suffix == "maj" else f"{root_name}{suffix}"
                best_score = score
    if best_score < 0.5:
        return None, best_score
    return best_name, round(best_score, 3)


def _notes_to_chords(notes: list[PitchNote]) -> list[ChordEvent]:
    if not notes:
        return []
    window = 0.4
    stride = 0.2
    min_t = min(n.start_s for n in notes)
    max_t = max(n.end_s for n in notes)
    out: list[ChordEvent] = []
    t = min_t
    while t < max_t:
        w_end = t + window
        active = [n for n in notes if n.start_s < w_end and n.end_s > t]
        if active:
            pcs = {n.midi % 12 for n in active}
            name, conf = _detect_chord_from_pcs(pcs)
            if name is not None:
                if out and out[-1].chord == name and (t - out[-1].end_s) <= stride:
                    out[-1].end_s = round(w_end, 3)
                    out[-1].confidence = round(max(out[-1].confidence, conf), 3)
                else:
                    out.append(
                        ChordEvent(
                            start_s=round(t, 3),
                            end_s=round(w_end, 3),
                            chord=name,
                            confidence=conf,
                        )
                    )
        t += stride
    return out


def _guess_instrument(path: Path) -> InstrumentGuess:
    """
    Heuristic instrument guess (lightweight).
    This is NOT a classifier; it's a quick signal for the UI pipeline.
    This version focuses on guitar vs piano as primary labels.
    """
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return InstrumentGuess(name="unknown", confidence=0.0, details=None)

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if y.size == 0:
        return InstrumentGuess(name="unknown", confidence=0.0, details=None)

    y = y.astype(np.float32, copy=False)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    y_h, y_p = librosa.effects.hpss(y)
    h = float(np.mean(np.abs(y_h)))
    p = float(np.mean(np.abs(y_p)))
    hp_ratio = h / (h + p + 1e-9)

    details = {
        "rms": round(rms, 6),
        "zcr": round(zcr, 6),
        "centroid_hz": round(centroid, 2),
        "harmonic_ratio": round(hp_ratio, 4),
    }

    if rms < 1e-4:
        return InstrumentGuess(name="silence", confidence=0.9, details=details)

    if hp_ratio < 0.35:
        return InstrumentGuess(name="drums_or_percussive", confidence=0.55, details=details)

    # Focused guess: guitar vs piano.
    # - Piano tends to have lower spectral centroid (more low/mid energy) and can have a stronger percussive transient.
    # - Guitar (esp. electric) often has higher centroid in the mid range with lower ZCR than noisy content.
    piano_score = 0.0
    guitar_score = 0.0

    # centroid contribution
    if centroid < 1000:
        piano_score += 1.0
    elif 1000 <= centroid <= 2400:
        guitar_score += 1.0
    else:
        guitar_score += 0.3

    # zcr: very high ZCR is often noisy/hi-hat/sizzle; moderate/low favors harmonic instruments
    if zcr < 0.08:
        guitar_score += 0.6
        piano_score += 0.4
    elif zcr < 0.14:
        guitar_score += 0.4
        piano_score += 0.3
    else:
        piano_score += 0.1

    # harmonic content
    piano_score += max(0.0, min(0.6, (hp_ratio - 0.35))) * 0.4
    guitar_score += max(0.0, min(0.6, (hp_ratio - 0.35))) * 0.6

    details = {
        **details,
        "piano_score": round(piano_score, 3),
        "guitar_score": round(guitar_score, 3),
    }

    if piano_score <= 0.2 and guitar_score <= 0.2:
        return InstrumentGuess(name="unknown", confidence=0.0, details=details)

    if abs(piano_score - guitar_score) < 0.25:
        # ambiguous / both present (e.g., backing track)
        confidence = 0.55
        return InstrumentGuess(name="guitar+piano", confidence=confidence, details=details)

    if piano_score > guitar_score:
        confidence = min(0.9, 0.55 + (piano_score - guitar_score) * 0.25)
        return InstrumentGuess(name="piano", confidence=confidence, details=details)

    confidence = min(0.9, 0.55 + (guitar_score - piano_score) * 0.25)
    return InstrumentGuess(name="guitar", confidence=confidence, details=details)


def _require_youtube_url(raw: str) -> str:
    url = _normalize_url(raw)
    if _detect_source(url) != "youtube":
        raise HTTPException(
            status_code=400,
            detail="Only YouTube URLs are supported for MP3 download.",
        )
    return url


def _register_audio(audio_id: str, path: Path) -> None:
    with _AUDIO_LOCK:
        _AUDIO_FILES[audio_id] = path
        _AUDIO_CREATED_AT[audio_id] = time.time()


def _pop_audio_path(audio_id: str) -> Path | None:
    with _AUDIO_LOCK:
        _AUDIO_CREATED_AT.pop(audio_id, None)
        return _AUDIO_FILES.pop(audio_id, None)


def _unlink_audio(audio_id: str) -> None:
    path = _pop_audio_path(audio_id)
    if path is not None and path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


def _purge_expired_audio_files(now: float | None = None) -> None:
    ts = now if now is not None else time.time()
    expired_ids: list[str] = []
    with _AUDIO_LOCK:
        for audio_id, created in _AUDIO_CREATED_AT.items():
            if ts - created > AUDIO_TTL_SECONDS:
                expired_ids.append(audio_id)
    for audio_id in expired_ids:
        _unlink_audio(audio_id)


def _cleanup_partial_files(prefix: str) -> None:
    for p in _AUDIO_STORE.glob(f"{prefix}.*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


class AnalyseRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048, description="HTTPS music or media URL")


class AnalyseResponse(BaseModel):
    url: str
    source: str
    title: str | None
    summary: str
    hints: list[str]


def _normalize_url(raw: str) -> str:
    trimmed = raw.strip()
    if not trimmed.lower().startswith(("http://", "https://")):
        trimmed = f"https://{trimmed}"
    parsed = urlparse(trimmed)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must use HTTP or HTTPS")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
        raise HTTPException(status_code=400, detail="Only HTTPS is allowed (except localhost)")
    return trimmed


def _detect_source(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "youtube.com" in host or host == "youtu.be" or host.endswith(".youtube.com"):
        return "youtube"
    if "spotify.com" in host:
        return "spotify"
    if "music.apple.com" in host or host.endswith(".apple.com"):
        return "apple_music"
    if "soundcloud.com" in host:
        return "soundcloud"
    if "bandcamp.com" in host:
        return "bandcamp"
    return "web"


def _stub_title(url: str, source: str) -> str | None:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    segments = [s for s in path.split("/") if s]
    if source == "youtube":
        if not segments or segments[-1] in ("watch", "embed", "shorts"):
            return None
        last = segments[-1]
        if len(last) == 11 and re.match(r"^[a-zA-Z0-9_-]+$", last):
            return None
        slug = last
    else:
        slug = segments[-1] if segments else ""
    if not slug:
        return None
    readable = re.sub(r"[-_]+", " ", slug)
    readable = re.sub(r"\s+", " ", readable).strip()
    if len(readable) < 2:
        return None
    return readable[:120].title()


def _build_summary(url: str, source: str) -> tuple[str, list[str]]:
    title_bit = _stub_title(url, source)
    label = {
        "youtube": "YouTube",
        "spotify": "Spotify",
        "apple_music": "Apple Music",
        "soundcloud": "SoundCloud",
        "bandcamp": "Bandcamp",
        "web": "Open web",
    }[source]
    hints = [
        f"Resolved host: {urlparse(url).hostname or 'unknown'}",
        "Full streaming metadata requires provider APIs; this response is a structured placeholder.",
    ]
    if source == "youtube":
        summary = (
            f"Link points to {label} content"
            + (f" (“{title_bit}”)." if title_bit else ".")
            + " Duration, channel, and chapter data would come from oEmbed or Data API in production."
        )
        hints.append("Consider YouTube oEmbed for title/thumbnail without an API key.")
    elif source == "spotify":
        summary = (
            f"Spotify URI or share URL detected"
            + (f" (path hint: “{title_bit}”)." if title_bit else ".")
            + " Track and album names need the Web API with OAuth for reliable reads."
        )
        hints.append("Spotify Web API returns ISRC, popularity, and audio features when authorized.")
    elif source == "apple_music":
        summary = (
            "Apple Music share link detected. Catalog resolution typically uses the MusicKit API."
        )
    elif source in ("soundcloud", "bandcamp"):
        summary = f"{label} link detected. Many pages expose oEmbed or Open Graph tags for titles."
    else:
        summary = (
            "Generic HTTPS link. In production, fetch Open Graph / oEmbed and normalize into one schema."
        )
    return summary, hints


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyse", response_model=AnalyseResponse)
def analyse(body: AnalyseRequest) -> AnalyseResponse:
    url = _normalize_url(body.url)
    source = _detect_source(url)
    title = _stub_title(url, source)
    summary, hints = _build_summary(url, source)
    return AnalyseResponse(url=url, source=source, title=title, summary=summary, hints=hints)


class YoutubeAudioResponse(BaseModel):
    id: str
    stream_path: str


@app.post("/api/audio/youtube", response_model=YoutubeAudioResponse)
def download_youtube_mp3(body: AnalyseRequest) -> YoutubeAudioResponse:
    _purge_expired_audio_files()
    url = _require_youtube_url(body.url)
    audio_id = str(uuid.uuid4())
    outtmpl = str(_AUDIO_STORE / f"{audio_id}.%(ext)s")
    opts: dict = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 2,
        "socket_timeout": 120,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        _cleanup_partial_files(audio_id)
        raise HTTPException(status_code=502, detail=f"Download failed: {e!s}") from e

    chosen: Path | None = None
    mp3 = _AUDIO_STORE / f"{audio_id}.mp3"
    if mp3.is_file():
        chosen = mp3
    else:
        for p in sorted(_AUDIO_STORE.glob(f"{audio_id}.*")):
            if p.suffix.lower() in _AUDIO_EXTS and p.is_file():
                chosen = p
                break
    if chosen is None:
        _cleanup_partial_files(audio_id)
        raise HTTPException(status_code=502, detail="No MP3 file was produced.")

    _register_audio(audio_id, chosen)
    return YoutubeAudioResponse(id=audio_id, stream_path=f"/api/audio/stream/{audio_id}")


@app.get("/api/audio/stream/{audio_id}")
def stream_audio(audio_id: str):
    with _AUDIO_LOCK:
        path = _AUDIO_FILES.get(audio_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found or already removed.")
    media = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, filename="preview.mp3")


@app.delete("/api/audio/{audio_id}")
def delete_audio(audio_id: str) -> dict[str, bool]:
    _unlink_audio(audio_id)
    return {"ok": True}


class UploadMp3Response(BaseModel):
    id: str
    stream_path: str
    instrument: InstrumentGuess
    notes: list[PitchNote]
    note_count: int


class AnalyseNotesResponse(BaseModel):
    instrument: InstrumentGuess
    notes: list[PitchNote]
    note_count: int


class AnalyseChordsResponse(BaseModel):
    chords: list[ChordEvent]
    chord_count: int


@app.post("/api/audio/upload", response_model=UploadMp3Response)
async def upload_mp3_extract_notes(file: UploadFile = File(...)) -> UploadMp3Response:
    _purge_expired_audio_files()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if Path(file.filename).suffix.lower() != ".mp3":
        raise HTTPException(status_code=400, detail="Only .mp3 files are allowed.")
    audio_id = str(uuid.uuid4())
    dest = _AUDIO_STORE / f"{audio_id}.mp3"
    total = 0
    try:
        with dest.open("wb") as buf:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MP3_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_MP3_UPLOAD_BYTES // (1024 * 1024)} MB).",
                    )
                buf.write(chunk)
    except HTTPException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    if total == 0:
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file.")
    _register_audio(audio_id, dest)
    instrument = _guess_instrument(dest)
    try:
        _, _, note_events = _run_basic_pitch(dest)
    except Exception as e:
        _unlink_audio(audio_id)
        raise HTTPException(status_code=502, detail=f"Note extraction failed: {e!s}") from e
    notes = _note_events_to_models(note_events)
    return UploadMp3Response(
        id=audio_id,
        stream_path=f"/api/audio/stream/{audio_id}",
        instrument=instrument,
        notes=notes,
        note_count=len(notes),
    )


@app.post("/api/audio/{audio_id}/analyse-notes", response_model=AnalyseNotesResponse)
def extract_notes_from_session(audio_id: str) -> AnalyseNotesResponse:
    _purge_expired_audio_files()
    with _AUDIO_LOCK:
        path = _AUDIO_FILES.get(audio_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found or already removed.")
    if path.suffix.lower() != ".mp3":
        raise HTTPException(status_code=400, detail="Note extraction requires an MP3 file.")
    instrument = _guess_instrument(path)
    try:
        _, _, note_events = _run_basic_pitch(path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Note extraction failed: {e!s}") from e
    notes = _note_events_to_models(note_events)
    return AnalyseNotesResponse(instrument=instrument, notes=notes, note_count=len(notes))


@app.get("/api/audio/{audio_id}/midi")
def download_session_midi(audio_id: str) -> Response:
    _purge_expired_audio_files()
    with _AUDIO_LOCK:
        path = _AUDIO_FILES.get(audio_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found or already removed.")
    if path.suffix.lower() != ".mp3":
        raise HTTPException(status_code=400, detail="MIDI export requires an MP3 file.")
    try:
        _, _, note_events = _run_basic_pitch(path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MIDI export failed: {e!s}") from e
    notes = _note_events_to_models(note_events)
    midi_bytes = _build_midi_bytes(notes)
    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="musicmetai-{audio_id}.mid"'},
    )


@app.post("/api/audio/{audio_id}/analyse-chords", response_model=AnalyseChordsResponse)
def extract_chords_from_session(audio_id: str) -> AnalyseChordsResponse:
    _purge_expired_audio_files()
    with _AUDIO_LOCK:
        path = _AUDIO_FILES.get(audio_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found or already removed.")
    if path.suffix.lower() != ".mp3":
        raise HTTPException(status_code=400, detail="Chord extraction requires an MP3 file.")
    try:
        _, _, note_events = _run_basic_pitch(path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chord extraction failed: {e!s}") from e
    notes = _note_events_to_models(note_events)
    chords = _notes_to_chords(notes)
    return AnalyseChordsResponse(chords=chords, chord_count=len(chords))
