import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Disc3, Download, Link2, Music2, Play, Upload, Volume2 } from "lucide-react";

type PitchNote = {
  start_s: number;
  end_s: number;
  midi: number;
  note: string;
  velocity: number;
};

type ChordEvent = {
  start_s: number;
  end_s: number;
  chord: string;
  confidence: number;
};

type InstrumentGuess = {
  name: string;
  confidence: number;
  details?: Record<string, number> | null;
};

function isYoutubeUrl(raw: string): boolean {
  const s = raw.trim();
  if (!s) return false;
  try {
    const withProto = s.includes("://") ? s : `https://${s}`;
    const u = new URL(withProto);
    const h = u.hostname.toLowerCase();
    return h === "youtu.be" || h.includes("youtube.com");
  } catch {
    return false;
  }
}

function formatTimestamp(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00.00";
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  const pad = s < 10 ? "0" : "";
  return `${m}:${pad}${s.toFixed(2)}`;
}

const midiToPc = (midi: number) => ((midi % 12) + 12) % 12;

function midiToName(midi: number): string {
  const pcs = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const pc = pcs[midiToPc(midi)];
  const oct = Math.floor(midi / 12) - 1;
  return `${pc}${oct}`;
}

const midiToFreq = (midi: number) => 440 * Math.pow(2, (midi - 69) / 12);
const VOCAL_PHRASES = [
  "Sing: la la la on a comfortable note",
  "Sing: do re mi fa sol fa mi re do",
  "Sing: hold the vowel 'aaah' for 3 seconds",
  "Sing: hum from low to high slowly",
];

function detectPitchAutocorr(buffer: Float32Array, sampleRate: number): number | null {
  let rms = 0;
  for (let i = 0; i < buffer.length; i += 1) rms += buffer[i] * buffer[i];
  rms = Math.sqrt(rms / buffer.length);
  if (rms < 0.01) return null;

  let bestOffset = -1;
  let bestCorrelation = 0;
  const minHz = 70;
  const maxHz = 1200;
  const minOffset = Math.floor(sampleRate / maxHz);
  const maxOffset = Math.floor(sampleRate / minHz);

  for (let offset = minOffset; offset <= maxOffset; offset += 1) {
    let correlation = 0;
    for (let i = 0; i < buffer.length - offset; i += 1) {
      correlation += buffer[i] * buffer[i + offset];
    }
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
    }
  }

  if (bestOffset <= 0) return null;
  return sampleRate / bestOffset;
}

function freqToMidi(freq: number): number {
  return Math.round(69 + 12 * Math.log2(freq / 440));
}

function classifyVoiceType(minMidi: number, maxMidi: number): string {
  if (maxMidi <= 55) return "Bass";
  if (maxMidi <= 62) return "Baritone";
  if (maxMidi <= 69) return "Tenor";
  if (minMidi >= 55 && maxMidi >= 74) return "Soprano";
  if (minMidi >= 52 && maxMidi >= 70) return "Mezzo-soprano";
  return "Alto";
}

type ParsedMidiNote = { midi: number; velocity: number; startSec: number; endSec: number };

function parseMidiFile(arrayBuffer: ArrayBuffer): ParsedMidiNote[] {
  const view = new DataView(arrayBuffer);
  let pos = 0;
  const readU8 = () => view.getUint8(pos++);
  const readU16 = () => { const v = view.getUint16(pos); pos += 2; return v; };
  const readU32 = () => { const v = view.getUint32(pos); pos += 4; return v; };
  const readStr = (n: number) => { let s = ""; for (let i = 0; i < n; i += 1) s += String.fromCharCode(readU8()); return s; };
  const readVar = () => {
    let value = 0;
    while (true) {
      const b = readU8();
      value = (value << 7) | (b & 0x7f);
      if ((b & 0x80) === 0) break;
    }
    return value;
  };

  if (readStr(4) !== "MThd") throw new Error("Invalid MIDI header");
  const headerLen = readU32();
  const format = readU16();
  const trackCount = readU16();
  const division = readU16();
  pos = 8 + headerLen;
  if (division <= 0) throw new Error("Unsupported MIDI timing");
  if (format > 1) throw new Error("Unsupported MIDI format");

  let tempo = 500000; // default 120 BPM
  const notes: ParsedMidiNote[] = [];
  const open = new Map<string, { tick: number; velocity: number }>();

  for (let t = 0; t < trackCount; t += 1) {
    if (readStr(4) !== "MTrk") throw new Error("Invalid MIDI track chunk");
    const trackLen = readU32();
    const trackEnd = pos + trackLen;
    let tick = 0;
    let runningStatus = 0;

    while (pos < trackEnd) {
      tick += readVar();
      let status = readU8();
      if (status < 0x80) {
        pos -= 1;
        status = runningStatus;
      } else {
        runningStatus = status;
      }

      if (status === 0xff) {
        const metaType = readU8();
        const len = readVar();
        if (metaType === 0x51 && len === 3) {
          tempo = (readU8() << 16) | (readU8() << 8) | readU8();
        } else {
          pos += len;
        }
        continue;
      }

      if (status === 0xf0 || status === 0xf7) {
        pos += readVar();
        continue;
      }

      const event = status & 0xf0;
      const ch = status & 0x0f;
      const note = readU8();
      const v = readU8();
      const key = `${ch}:${note}`;

      if (event === 0x90 && v > 0) {
        open.set(key, { tick, velocity: v });
      } else if (event === 0x80 || (event === 0x90 && v === 0)) {
        const start = open.get(key);
        if (start) {
          notes.push({
            midi: note,
            velocity: start.velocity / 127,
            startSec: (start.tick * tempo) / (division * 1_000_000),
            endSec: (tick * tempo) / (division * 1_000_000),
          });
          open.delete(key);
        }
      }
    }
    pos = trackEnd;
  }

  return notes
    .filter((n) => n.endSec > n.startSec)
    .sort((a, b) => a.startSec - b.startSec);
}

function playMidiNote(midi: number, seconds = 0.6): void {
  try {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const now = ctx.currentTime;
    osc.type = "triangle";
    osc.frequency.value = midiToFreq(midi);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + seconds);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + seconds + 0.02);
    osc.onended = () => void ctx.close();
  } catch {
    /* noop */
  }
}

const guitarTuning = [
  { name: "E2", midi: 40 },
  { name: "A2", midi: 45 },
  { name: "D3", midi: 50 },
  { name: "G3", midi: 55 },
  { name: "B3", midi: 59 },
  { name: "E4", midi: 64 },
] as const;
const MAX_GUITAR_FRET = 24;

function guitarPositionsForMidi(midi: number): string {
  const positions: string[] = [];
  for (let i = guitarTuning.length - 1; i >= 0; i -= 1) {
    const fret = midi - guitarTuning[i].midi;
    if (fret >= 0 && fret <= MAX_GUITAR_FRET) positions.push(`${guitarTuning[i].name}:${fret}`);
  }
  return positions.length ? positions.join(" · ") : "—";
}

type InstrumentViewProps = {
  selectedMidi: number | null;
  playingMidi: number | null;
  activeMidis: Set<number>;
  activePitchClasses: Set<number>;
  onSelectMidi: (midi: number) => void;
  onPlayMidi: (midi: number) => void;
};

function GuitarFretboard({
  selectedMidi,
  playingMidi,
  activeMidis,
  activePitchClasses,
  onSelectMidi,
  onPlayMidi,
}: InstrumentViewProps) {
  const frets = Array.from({ length: MAX_GUITAR_FRET + 1 }, (_, i) => i);
  return (
    <div className="border-2 border-ink bg-paper p-5">
      <div className="mb-4 flex items-baseline justify-between">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink">Fretboard / 0–24</p>
        <p className="font-mono text-[10px] text-muted-foreground">EADGBE</p>
      </div>
      <div className="max-h-[70vh] overflow-y-auto overflow-x-auto">
        <table className="min-w-[1220px] border-separate border-spacing-1">
          <thead>
            <tr>
              <th className="w-12" />
              {frets.map((f) => (
                <th key={f} className="px-2 py-1 text-center font-mono text-[10px] text-muted-foreground">
                  {f === 0 ? "○" : f}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {guitarTuning
              .slice()
              .reverse()
              .map((s) => (
                <tr key={s.name}>
                  <td className="pr-2 text-right font-mono text-[10px] font-bold text-ink">{s.name}</td>
                  {frets.map((f) => {
                    const midi = s.midi + f;
                    const pc = midiToPc(midi);
                    const exact = activeMidis.has(midi);
                    const pitchClassOnly = activePitchClasses.has(pc) && !exact;
                    const isSelected = selectedMidi === midi;
                    const isPlaying = playingMidi === midi;
                    return (
                      <td key={f} className="text-center">
                        <button
                          type="button"
                          className={`h-9 w-9 border-2 text-[11px] font-bold transition ${
                            isPlaying
                              ? "border-ink bg-accent text-accent-foreground"
                              : isSelected
                                ? "border-ink bg-highlight text-ink"
                                : exact
                                  ? "border-ink bg-ink text-paper"
                                  : pitchClassOnly
                                    ? "border-ink/30 bg-paper text-ink"
                                    : "border-transparent bg-paper text-muted-foreground hover:border-ink/30"
                          }`}
                          onClick={() => {
                            onSelectMidi(midi);
                            onPlayMidi(midi);
                          }}
                          title={`${midiToName(midi)} (MIDI ${midi})`}
                        >
                          {f}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PianoKeyboard({
  selectedMidi,
  playingMidi,
  activeMidis,
  activePitchClasses,
  onSelectMidi,
  onPlayMidi,
}: InstrumentViewProps) {
  const ranges = [
    { label: "Low / A0-B2", startMidi: 21, endMidi: 47 },
    { label: "Mid / C3-B5", startMidi: 48, endMidi: 83 },
    { label: "High / C6-C8", startMidi: 84, endMidi: 108 },
  ] as const;
  const isBlack = (pc: number) => pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;

  return (
    <div className="border-2 border-ink bg-paper p-5">
      <div className="mb-4 flex items-baseline justify-between">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink">Keyboard / A0–C8</p>
        <p className="font-mono text-[10px] text-muted-foreground">88 keys</p>
      </div>
      <div className="max-h-[70vh] space-y-4 overflow-y-auto">
        {ranges.map((range) => {
          const keys = Array.from({ length: range.endMidi - range.startMidi + 1 }, (_, i) => range.startMidi + i);
          const whites = keys.filter((m) => !isBlack(midiToPc(m)));
          return (
            <div key={range.label}>
              <p className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{range.label}</p>
              <div className="overflow-x-auto">
                <div className="relative h-36 min-w-[820px] border-2 border-ink">
                  <div className="absolute inset-0 flex">
                    {whites.map((m) => {
                      const exact = activeMidis.has(m);
                      const pitchClassOnly = activePitchClasses.has(midiToPc(m)) && !exact;
                      const isSelected = selectedMidi === m;
                      const isPlaying = playingMidi === m;
                      return (
                        <button
                          key={m}
                          type="button"
                          className={`relative h-full w-10 border-r border-ink/40 last:border-r-0 ${
                            isPlaying
                              ? "bg-accent"
                              : isSelected
                                ? "bg-highlight"
                                : exact
                                  ? "bg-ink/10"
                                  : "bg-paper"
                          }`}
                          onClick={() => {
                            onSelectMidi(m);
                            onPlayMidi(m);
                          }}
                          title={`${midiToName(m)} (MIDI ${m})`}
                        >
                          {exact ? (
                            <span className="absolute left-1/2 top-2 h-2 w-2 -translate-x-1/2 rounded-full bg-accent" />
                          ) : pitchClassOnly ? (
                            <span className="absolute left-1/2 top-2 h-1.5 w-1.5 -translate-x-1/2 rounded-full bg-ink/40" />
                          ) : null}
                          <span className="absolute bottom-1.5 left-1/2 -translate-x-1/2 font-mono text-[9px] text-ink/60">
                            {midiToName(m)}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  <div className="pointer-events-none absolute inset-0">
                    {keys
                      .filter((m) => isBlack(midiToPc(m)))
                      .map((m) => {
                        const exact = activeMidis.has(m);
                        const pitchClassOnly = activePitchClasses.has(midiToPc(m)) && !exact;
                        const isSelected = selectedMidi === m;
                        const isPlaying = playingMidi === m;
                        const idx = whites.findIndex((w) => w > m);
                        const left = Math.max(0, idx * 40 - 12);
                        return (
                          <button
                            key={m}
                            type="button"
                            className={`pointer-events-auto absolute top-0 w-6 border-2 border-ink ${
                              isPlaying ? "bg-accent" : isSelected ? "bg-highlight" : "bg-ink"
                            }`}
                            style={{ left, height: "60%" }}
                            onClick={() => {
                              onSelectMidi(m);
                              onPlayMidi(m);
                            }}
                            title={`${midiToName(m)} (MIDI ${m})`}
                          >
                            {exact ? (
                              <span className="absolute left-1/2 top-1.5 h-1.5 w-1.5 -translate-x-1/2 rounded-full bg-accent" />
                            ) : pitchClassOnly ? (
                              <span className="absolute left-1/2 top-1.5 h-1 w-1 -translate-x-1/2 rounded-full bg-paper/60" />
                            ) : null}
                          </button>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-4 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full bg-accent" /> playing</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-highlight" /> selected</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 bg-ink" /> mapped</span>
      </div>
    </div>
  );
}

export default function App() {
  const [trackUrl, setTrackUrl] = useState("");
  const [activeAudioId, setActiveAudioId] = useState<string | null>(null);
  const [youtubeLoading, setYoutubeLoading] = useState(false);
  const [youtubeError, setYoutubeError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [pitchNotes, setPitchNotes] = useState<PitchNote[] | null>(null);
  const [pitchLoading, setPitchLoading] = useState(false);
  const [pitchError, setPitchError] = useState<string | null>(null);
  const [instrument, setInstrument] = useState<InstrumentGuess | null>(null);
  const [selectedMidi, setSelectedMidi] = useState<number | null>(null);
  const [playingMidi, setPlayingMidi] = useState<number | null>(null);
  const [activeView, setActiveView] = useState<"notes" | "guitar" | "piano">("notes");
  const [midiExportLoading, setMidiExportLoading] = useState(false);
  const [appMode, setAppMode] = useState<"note_mapper" | "chord_finder" | "vocal_analyser" | "tuner" | "midi_player">("note_mapper");
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [vocalListening, setVocalListening] = useState(false);
  const [vocalError, setVocalError] = useState<string | null>(null);
  const [vocalDetectedMidis, setVocalDetectedMidis] = useState<number[]>([]);
  const [vocalCurrentMidi, setVocalCurrentMidi] = useState<number | null>(null);
  const [vocalTypeResult, setVocalTypeResult] = useState<{ label: string; minMidi: number; maxMidi: number } | null>(null);
  const [tunerListening, setTunerListening] = useState(false);
  const [tunerError, setTunerError] = useState<string | null>(null);
  const [tunerMidi, setTunerMidi] = useState<number | null>(null);
  const [tunerCents, setTunerCents] = useState(0);
  const [chords, setChords] = useState<ChordEvent[] | null>(null);
  const [chordLoading, setChordLoading] = useState(false);
  const [chordError, setChordError] = useState<string | null>(null);
  const [midiPlayerNotes, setMidiPlayerNotes] = useState<ParsedMidiNote[]>([]);
  const [midiPlayerFileName, setMidiPlayerFileName] = useState<string | null>(null);
  const [midiPlayerError, setMidiPlayerError] = useState<string | null>(null);
  const [midiPlayerPlaying, setMidiPlayerPlaying] = useState(false);

  const audioIdRef = useRef<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playTokenRef = useRef(0);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micAudioCtxRef = useRef<AudioContext | null>(null);
  const micRafRef = useRef<number | null>(null);
  const midiPlayerCtxRef = useRef<AudioContext | null>(null);
  const midiPlayerTimeoutsRef = useRef<number[]>([]);

  useEffect(() => {
    audioIdRef.current = activeAudioId;
  }, [activeAudioId]);

  useEffect(() => {
    const release = () => {
      const id = audioIdRef.current;
      if (!id) return;
      fetch(`/api/audio/${id}`, { method: "DELETE", keepalive: true });
    };
    window.addEventListener("pagehide", release);
    window.addEventListener("beforeunload", release);
    return () => {
      window.removeEventListener("pagehide", release);
      window.removeEventListener("beforeunload", release);
    };
  }, []);

  const stopMicSession = () => {
    if (micRafRef.current != null) {
      cancelAnimationFrame(micRafRef.current);
      micRafRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (micAudioCtxRef.current) {
      void micAudioCtxRef.current.close();
      micAudioCtxRef.current = null;
    }
    setVocalListening(false);
    setTunerListening(false);
  };

  const stopMidiPlayback = () => {
    midiPlayerTimeoutsRef.current.forEach((id) => window.clearTimeout(id));
    midiPlayerTimeoutsRef.current = [];
    if (midiPlayerCtxRef.current) {
      void midiPlayerCtxRef.current.close();
      midiPlayerCtxRef.current = null;
    }
    setMidiPlayerPlaying(false);
  };

  useEffect(
    () => () => {
      stopMicSession();
      stopMidiPlayback();
    },
    [],
  );

  const releaseServerAudio = async (id: string | null) => {
    if (!id) return;
    await fetch(`/api/audio/${id}`, { method: "DELETE" });
  };

  const playAndHighlight = (midi: number, seconds = 0.6) => {
    setSelectedMidi(midi);
    setPlayingMidi(midi);
    playMidiNote(midi, seconds);
    const token = ++playTokenRef.current;
    window.setTimeout(() => {
      if (token !== playTokenRef.current) return;
      setPlayingMidi((cur) => (cur === midi ? null : cur));
    }, Math.max(120, Math.round(seconds * 1000) + 60));
  };

  const handleYoutubeMp3 = async () => {
    const trimmed = trackUrl.trim();
    if (!trimmed || !isYoutubeUrl(trimmed)) return;
    setYoutubeError(null);
    setPitchError(null);
    setPitchNotes(null);
    setInstrument(null);
    setSelectedMidi(null);
    setYoutubeLoading(true);
    try {
      if (activeAudioId) await releaseServerAudio(activeAudioId);
      const res = await fetch("/api/audio/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
      });
      if (!res.ok) throw new Error("Download failed");
      const data = (await res.json()) as { id: string };
      setActiveAudioId(data.id);
    } catch (e) {
      setActiveAudioId(null);
      setYoutubeError(e instanceof Error ? e.message : "Download failed");
    } finally {
      setYoutubeLoading(false);
    }
  };

  const handleMp3Upload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".mp3")) {
      setUploadError("Only .mp3 files are accepted.");
      return;
    }
    setUploadError(null);
    setPitchError(null);
    setPitchLoading(true);
    try {
      if (activeAudioId) await releaseServerAudio(activeAudioId);
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/audio/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error("Upload failed");
      const data = (await res.json()) as { id: string; notes: PitchNote[]; instrument?: InstrumentGuess };
      setActiveAudioId(data.id);
      setPitchNotes(data.notes);
      setInstrument(data.instrument ?? null);
    } catch (e) {
      setActiveAudioId(null);
      setPitchError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setPitchLoading(false);
    }
  };

  const handleExtractNotes = async () => {
    if (!activeAudioId) return;
    setPitchLoading(true);
    setPitchError(null);
    try {
      const res = await fetch(`/api/audio/${activeAudioId}/analyse-notes`, { method: "POST" });
      if (!res.ok) throw new Error("Note extraction failed");
      const data = (await res.json()) as { notes: PitchNote[]; instrument?: InstrumentGuess };
      setPitchNotes(data.notes);
      setInstrument(data.instrument ?? null);
    } catch (e) {
      setPitchError(e instanceof Error ? e.message : "Note extraction failed");
    } finally {
      setPitchLoading(false);
    }
  };

  const seekAudioTo = (seconds: number) => {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = seconds;
    void el.play();
  };

  const handleMidiDownload = async () => {
    if (!activeAudioId || midiExportLoading) return;
    setMidiExportLoading(true);
    try {
      const res = await fetch(`/api/audio/${activeAudioId}/midi`);
      if (!res.ok) throw new Error("MIDI export failed");
      const blob = await res.blob();
      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = `musicmetai-${activeAudioId}.mid`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(href);
    } catch (e) {
      setPitchError(e instanceof Error ? e.message : "MIDI export failed");
    } finally {
      setMidiExportLoading(false);
    }
  };

  const handleExtractChords = async () => {
    if (!activeAudioId) return;
    setChordLoading(true);
    setChordError(null);
    try {
      const res = await fetch(`/api/audio/${activeAudioId}/analyse-chords`, { method: "POST" });
      if (!res.ok) throw new Error("Chord extraction failed");
      const data = (await res.json()) as { chords: ChordEvent[] };
      setChords(data.chords);
    } catch (e) {
      setChordError(e instanceof Error ? e.message : "Chord extraction failed");
    } finally {
      setChordLoading(false);
    }
  };

  const startVocalAnalyser = async () => {
    stopMicSession();
    setVocalError(null);
    setVocalTypeResult(null);
    setVocalDetectedMidis([]);
    setVocalCurrentMidi(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);
      const buffer = new Float32Array(analyser.fftSize);
      micStreamRef.current = stream;
      micAudioCtxRef.current = ctx;
      setVocalListening(true);

      const loop = () => {
        analyser.getFloatTimeDomainData(buffer);
        const freq = detectPitchAutocorr(buffer, ctx.sampleRate);
        if (freq && freq > 70 && freq < 1200) {
          const midi = freqToMidi(freq);
          setVocalCurrentMidi(midi);
          setVocalDetectedMidis((prev) => (prev.length > 1200 ? [...prev.slice(1), midi] : [...prev, midi]));
        }
        micRafRef.current = requestAnimationFrame(loop);
      };
      micRafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      setVocalError(e instanceof Error ? e.message : "Microphone access failed.");
      stopMicSession();
    }
  };

  const stopVocalAnalyser = () => {
    const collected = vocalDetectedMidis;
    stopMicSession();
    if (collected.length < 8) {
      setVocalError("Not enough sung notes captured. Try again and sing longer.");
      return;
    }
    const sorted = [...collected].sort((a, b) => a - b);
    const minMidi = sorted[Math.floor(sorted.length * 0.1)];
    const maxMidi = sorted[Math.floor(sorted.length * 0.9)];
    const label = classifyVoiceType(minMidi, maxMidi);
    setVocalTypeResult({ label, minMidi, maxMidi });
  };

  const startTuner = async () => {
    stopMicSession();
    setTunerError(null);
    setTunerMidi(null);
    setTunerCents(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);
      const buffer = new Float32Array(analyser.fftSize);
      micStreamRef.current = stream;
      micAudioCtxRef.current = ctx;
      setTunerListening(true);

      const loop = () => {
        analyser.getFloatTimeDomainData(buffer);
        const freq = detectPitchAutocorr(buffer, ctx.sampleRate);
        if (freq && freq > 40 && freq < 2000) {
          const midi = freqToMidi(freq);
          const target = midiToFreq(midi);
          const cents = Math.max(-50, Math.min(50, Math.round(1200 * Math.log2(freq / target))));
          setTunerMidi(midi);
          setTunerCents(cents);
        }
        micRafRef.current = requestAnimationFrame(loop);
      };
      micRafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      setTunerError(e instanceof Error ? e.message : "Microphone access failed.");
      stopMicSession();
    }
  };

  const handleMidiPlayerUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setMidiPlayerError(null);
    stopMidiPlayback();
    try {
      const buf = await file.arrayBuffer();
      const notes = parseMidiFile(buf);
      if (notes.length === 0) throw new Error("No note events found in MIDI file.");
      setMidiPlayerNotes(notes);
      setMidiPlayerFileName(file.name);
    } catch (err) {
      setMidiPlayerNotes([]);
      setMidiPlayerFileName(null);
      setMidiPlayerError(err instanceof Error ? err.message : "Failed to parse MIDI.");
    }
  };

  const startMidiPlayback = async () => {
    if (midiPlayerNotes.length === 0 || midiPlayerPlaying) return;
    stopMidiPlayback();
    const ctx = new AudioContext();
    midiPlayerCtxRef.current = ctx;
    setMidiPlayerPlaying(true);
    const startAt = ctx.currentTime + 0.03;
    const maxEnd = Math.max(...midiPlayerNotes.map((n) => n.endSec));

    midiPlayerNotes.forEach((n) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.value = midiToFreq(n.midi);
      const vel = Math.max(0.02, Math.min(0.28, n.velocity * 0.32));
      gain.gain.setValueAtTime(0.0001, startAt + n.startSec);
      gain.gain.exponentialRampToValueAtTime(vel, startAt + n.startSec + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, startAt + n.endSec);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(startAt + n.startSec);
      osc.stop(startAt + n.endSec + 0.03);
    });

    const tid = window.setTimeout(() => {
      stopMidiPlayback();
    }, Math.ceil((maxEnd + 0.2) * 1000));
    midiPlayerTimeoutsRef.current.push(tid);
  };

  const pitchMidis = useMemo(() => new Set<number>((pitchNotes ?? []).map((n) => n.midi)), [pitchNotes]);
  const pitchClasses = useMemo(() => new Set<number>((pitchNotes ?? []).map((n) => midiToPc(n.midi))), [pitchNotes]);
  const uniqueMidis = useMemo(() => Array.from(pitchMidis).sort((a, b) => a - b), [pitchMidis]);

  const views: Array<{ id: "notes" | "guitar" | "piano"; label: string }> = [
    { id: "notes", label: "Notes" },
    { id: "guitar", label: "Guitar" },
    { id: "piano", label: "Piano" },
  ];

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="bubble-layer" aria-hidden="true">
        <span className="bubble b1" />
        <span className="bubble b2" />
        <span className="bubble b3" />
        <span className="bubble b4" />
        <span className="bubble b5" />
        <span className="bubble b6" />
      </div>
      <header className="relative z-10 border-b-2 border-ink">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center border-2 border-ink bg-ink">
              <Disc3 size={18} className="text-paper" strokeWidth={2.5} />
            </div>
            <div   className="leading-tight">
              <p className="font-mono text-[13px] tracking-[0.25em] text-ink">MusicMetAI</p>
              <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground">Beta Version 0.1</p>
            </div>
          </div>
          <div className="hidden items-center gap-6 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground sm:flex">
            <span>Find Notes & chords for Guitar and Piano</span>
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
          </div>
        </div>
      </header>

      <nav className="relative z-10 border-b-2 border-ink bg-card">
        <div className="mx-auto flex max-w-[1400px] items-center gap-2 px-4 py-2">
          {[
            { id: "note_mapper", label: "Note Mapper" },
            { id: "chord_finder", label: "Chord Finder" },
            { id: "vocal_analyser", label: "Vocal Type Analyser" },
            { id: "tuner", label: "Tuner" },
            { id: "midi_player", label: "MIDI Player" },
          ].map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => {
                stopMicSession();
                stopMidiPlayback();
                setAppMode(m.id as typeof appMode);
              }}
              className={`border-2 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.2em] ${
                appMode === m.id ? "border-ink bg-ink text-paper" : "border-ink/40 bg-paper text-ink hover:border-ink"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </nav>

      <section className={`relative z-10 border-b-2 border-ink ${appMode !== "note_mapper" ? "hidden" : ""}`}>
        <div className="mx-auto grid max-w-[1400px] items-start gap-4 px-4 py-6 lg:grid-cols-12 lg:py-7">
          <div className="lg:col-span-8">
            <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-accent">Beta ending soon</p>
            <h1 className="mt-2 text-[clamp(1.35rem,2.8vw,2.25rem)] font-bold leading-[1.05] tracking-tight">
              Identify Notes. <span className="italic text-accent">See</span> the notes. Learn Them.
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-muted-foreground sm:text-[15px]">
              Drop an MP3 or paste a ytlink. We extract the melody, map every note onto guitar and piano, and let you audition them - one click at a time.
            </p>
          </div>
          <div className="lg:col-span-4">
            <div className="border-2 border-ink bg-card p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Workflow</p>
              <ol className="mt-2 grid gap-y-1 text-sm">
                {[
                  "Upload MP3 or fetch from Ytlink",
                  "Extract refined timestamped notes",
                  "Open guitar / piano mapping",
                  "Play and validate placements",
                  "Download MIDI files",
                ].map((step, i) => (
                  <li key={step} className="flex gap-2">
                    <span className="font-mono text-xs font-bold text-accent">0{i + 1}</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      </section>

      <main className={`relative z-10 mx-auto max-w-[1400px] px-6 py-10 ${appMode !== "note_mapper" ? "hidden" : ""}`}>
        <div className="border-2 border-ink bg-card">
          <div className="flex items-center justify-between border-b-2 border-ink px-5 py-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.25em]">Input</p>
            <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
              {activeAudioId ? "session active" : "no source"}
            </p>
          </div>
          <div className="grid gap-3 p-5 md:grid-cols-[1fr_auto_auto]">
            <div className="relative">
              <Link2 size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink/50" />
              <input
                type="url"
                value={trackUrl}
                onChange={(e) => setTrackUrl(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && isYoutubeUrl(trackUrl)) void handleYoutubeMp3();
                }}
                placeholder="Paste a Ytlink URL..."
                className="h-12 w-full border-2 border-ink bg-paper pl-10 pr-3 font-mono text-sm placeholder:text-ink/40 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-card"
              />
            </div>
            <button
              type="button"
              onClick={() => void handleYoutubeMp3()}
              disabled={!trackUrl.trim() || !isYoutubeUrl(trackUrl) || youtubeLoading || pitchLoading}
              className="inline-flex h-12 items-center justify-center gap-2 border-2 border-ink bg-accent px-5 font-mono text-xs font-bold uppercase tracking-[0.18em] text-accent-foreground transition hover:bg-ink hover:text-paper disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Play size={14} strokeWidth={3} />
              {youtubeLoading ? "Fetching..." : "Fetch"}
            </button>
            <label className="inline-flex h-12 cursor-pointer items-center justify-center gap-2 border-2 border-ink bg-paper px-5 font-mono text-xs font-bold uppercase tracking-[0.18em] text-ink transition hover:bg-ink hover:text-paper">
              <Upload size={14} strokeWidth={3} />
              Upload MP3
              <input
                type="file"
                accept=".mp3,audio/mpeg"
                className="sr-only"
                disabled={pitchLoading || youtubeLoading}
                onChange={(e) => void handleMp3Upload(e)}
              />
            </label>
          </div>

          {(uploadError || youtubeError || pitchError) && (
            <div className="border-t-2 border-ink bg-accent/10 px-5 py-3 font-mono text-xs text-ink">
              {uploadError || youtubeError || pitchError}
            </div>
          )}
        </div>

        {activeAudioId ? (
          <div className="mt-6 grid gap-6 lg:grid-cols-12">
            <div className="lg:col-span-12">
              <div className="border-2 border-ink bg-card">
                <div className="flex flex-wrap items-center justify-between gap-4 border-b-2 border-ink px-5 py-3">
                  <div className="flex items-center gap-3">
                    <Music2 size={16} />
                    <p className="font-mono text-[11px] uppercase tracking-[0.25em]">Session</p>
                    {instrument ? (
                      <span className="border border-ink/30 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider">
                        {instrument.name} - {Math.round(instrument.confidence * 100)}%
                      </span>
                    ) : null}
                  </div>
                  <div className="flex items-center gap-2">
                    {activeAudioId && pitchNotes === null ? (
                      <button
                        type="button"
                        onClick={() => void handleExtractNotes()}
                        className="border-2 border-ink bg-ink px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-paper hover:bg-accent hover:text-accent-foreground"
                      >
                        {pitchLoading ? "Analysing..." : "Extract Notes"}
                      </button>
                    ) : null}
                    {pitchNotes ? (
                      <>
                        <button
                          type="button"
                          onClick={() => void handleMidiDownload()}
                          disabled={midiExportLoading}
                          className="inline-flex items-center gap-1.5 border-2 border-ink bg-accent px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-accent-foreground hover:bg-ink hover:text-paper disabled:opacity-40"
                        >
                          <Download size={12} />
                          {midiExportLoading ? "Exporting..." : "Download MIDI"}
                        </button>
                        <div className="flex border-2 border-ink">
                          {views.map((v) => (
                            <button
                              key={v.id}
                              type="button"
                              onClick={() => setActiveView(v.id)}
                              className={`px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.2em] ${
                                activeView === v.id ? "bg-ink text-paper" : "bg-paper text-ink hover:bg-ink/10"
                              }`}
                            >
                              {v.label}
                            </button>
                          ))}
                        </div>
                      </>
                    ) : null}
                  </div>
                </div>
                <div className="p-5">
                  <audio ref={audioRef} key={activeAudioId} src={`/api/audio/stream/${activeAudioId}`} controls className="w-full" />
                </div>
              </div>
            </div>

            {pitchNotes ? (
              <>
                {activeView === "notes" ? (
                  <div className="lg:col-span-12">
                    <div className="border-2 border-ink bg-card">
                      <div className="flex items-center justify-between border-b-2 border-ink px-5 py-2">
                        <p className="font-mono text-[10px] uppercase tracking-[0.25em]">Transcription</p>
                        <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                          {pitchNotes.length} events
                        </p>
                      </div>
                      <div className="max-h-[480px] overflow-y-auto">
                        <table className="w-full min-w-[600px] border-collapse text-left text-sm">
                          <thead className="sticky top-0 bg-card">
                            <tr className="border-b-2 border-ink font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                              <th className="px-4 py-2.5">Time</th>
                              <th className="px-4 py-2.5">Note</th>
                              <th className="px-4 py-2.5">Guitar</th>
                              <th className="px-4 py-2.5">Piano</th>
                              <th className="px-4 py-2.5 text-right">Audition</th>
                            </tr>
                          </thead>
                          <tbody>
                            {pitchNotes.map((n, i) => (
                              <tr
                                key={`${n.start_s}-${n.end_s}-${n.midi}-${i}`}
                                onClick={() => seekAudioTo(n.start_s)}
                                className={`cursor-pointer border-b border-ink/10 transition hover:bg-highlight/40 ${
                                  selectedMidi === n.midi ? "bg-highlight/30" : ""
                                }`}
                              >
                                <td className="px-4 py-2.5 font-mono text-xs text-accent">
                                  {formatTimestamp(n.start_s)}
                                  <span className="text-ink/40">{" -> "}</span>
                                  {formatTimestamp(n.end_s)}
                                </td>
                                <td className="px-4 py-2.5">
                                  <span className="font-bold">{n.note}</span>
                                  <span className="ml-2 font-mono text-[10px] text-muted-foreground">MIDI {n.midi}</span>
                                </td>
                                <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{guitarPositionsForMidi(n.midi)}</td>
                                <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{midiToName(n.midi)}</td>
                                <td className="px-4 py-2.5 text-right">
                                  <button
                                    type="button"
                                    className="inline-flex items-center gap-1.5 border-2 border-ink bg-paper px-2.5 py-1 font-mono text-[10px] font-bold uppercase tracking-wider hover:bg-ink hover:text-paper"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      playAndHighlight(n.midi, 0.6);
                                    }}
                                  >
                                    <Volume2 size={12} /> Play
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="lg:col-span-3">
                      <div className="border-2 border-ink bg-card p-4">
                        <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Notes detected</p>
                        <p className="mt-2 font-mono text-[10px] uppercase tracking-wider text-ink">{uniqueMidis.length} unique</p>
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {uniqueMidis.map((m) => (
                            <button
                              key={m}
                              type="button"
                              onClick={() => playAndHighlight(m, 0.6)}
                              className={`border-2 px-2 py-1 font-mono text-[10px] font-bold ${
                                selectedMidi === m ? "border-ink bg-accent text-accent-foreground" : "border-ink/30 bg-paper hover:border-ink"
                              }`}
                            >
                              {midiToName(m)}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="lg:col-span-9">
                      {activeView === "guitar" ? (
                        <GuitarFretboard
                          activeMidis={pitchMidis}
                          activePitchClasses={pitchClasses}
                          selectedMidi={selectedMidi}
                          playingMidi={playingMidi}
                          onSelectMidi={setSelectedMidi}
                          onPlayMidi={(m) => playAndHighlight(m, 0.6)}
                        />
                      ) : (
                        <PianoKeyboard
                          activeMidis={pitchMidis}
                          activePitchClasses={pitchClasses}
                          selectedMidi={selectedMidi}
                          playingMidi={playingMidi}
                          onSelectMidi={setSelectedMidi}
                          onPlayMidi={(m) => playAndHighlight(m, 0.6)}
                        />
                      )}
                    </div>
                  </>
                )}
              </>
            ) : null}
          </div>
        ) : (
          <div className="mt-6 border-2 border-dashed border-ink/40 bg-card p-12 text-center">
            <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-muted-foreground">No source loaded</p>
            <p className="mt-3 text-xl font-bold">Upload an MP3 or paste a Ytlink to begin.</p>
          </div>
        )}
      </main>

      {appMode === "chord_finder" ? (
        <section className="relative z-10 mx-auto max-w-[1400px] px-6 py-8">
          <div className="border-2 border-ink bg-card p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Chord Finder</p>
            {!activeAudioId ? (
              <p className="mt-3 text-sm text-muted-foreground">Load audio in `Note Mapper` first, then come here to detect chords.</p>
            ) : (
              <>
                <div className="mt-3 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void handleExtractChords()}
                    disabled={chordLoading}
                    className="border-2 border-ink bg-accent px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-accent-foreground disabled:opacity-40"
                  >
                    {chordLoading ? "Analysing..." : "Analyse Chords"}
                  </button>
                  {chords ? <span className="font-mono text-xs text-muted-foreground">{chords.length} chord events</span> : null}
                </div>
                {chordError ? <p className="mt-3 font-mono text-xs text-accent">{chordError}</p> : null}
                {chords ? (
                  <div className="mt-4 max-h-[520px] overflow-y-auto border-2 border-ink bg-paper">
                    <table className="w-full min-w-[500px] border-collapse text-left text-sm">
                      <thead className="sticky top-0 bg-card">
                        <tr className="border-b-2 border-ink font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                          <th className="px-4 py-2.5">Time</th>
                          <th className="px-4 py-2.5">Chord</th>
                          <th className="px-4 py-2.5">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {chords.map((c, i) => (
                          <tr key={`${c.start_s}-${c.end_s}-${c.chord}-${i}`} className="border-b border-ink/10">
                            <td className="px-4 py-2.5 font-mono text-xs text-accent">
                              {formatTimestamp(c.start_s)} <span className="text-ink/40">{" -> "}</span> {formatTimestamp(c.end_s)}
                            </td>
                            <td className="px-4 py-2.5 font-mono text-sm font-bold">{c.chord}</td>
                            <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{Math.round(c.confidence * 100)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
              </>
            )}
          </div>
        </section>
      ) : null}

      {appMode === "vocal_analyser" ? (
        <section className="relative z-10 mx-auto max-w-[1400px] px-6 py-8">
          <div className="border-2 border-ink bg-card p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Vocal Type Analyser</p>
            <p className="mt-2 text-lg font-semibold">Phrase: {VOCAL_PHRASES[phraseIndex]}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              <button type="button" onClick={() => setPhraseIndex((v) => (v + 1) % VOCAL_PHRASES.length)} className="border-2 border-ink bg-paper px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em] hover:bg-ink hover:text-paper">Next phrase</button>
              {!vocalListening ? (
                <button type="button" onClick={() => void startVocalAnalyser()} className="border-2 border-ink bg-accent px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-accent-foreground">Start listening</button>
              ) : (
                <button type="button" onClick={stopVocalAnalyser} className="border-2 border-ink bg-ink px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-paper">Stop & analyse</button>
              )}
            </div>
            <p className="mt-3 font-mono text-xs text-muted-foreground">
              {vocalCurrentMidi != null ? `Current note: ${midiToName(vocalCurrentMidi)} (MIDI ${vocalCurrentMidi})` : "Current note: -"}
            </p>
            {vocalTypeResult ? (
              <div className="mt-4 border-2 border-ink bg-paper p-3">
                <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink">Detected voice type: {vocalTypeResult.label}</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Range: {midiToName(vocalTypeResult.minMidi)} to {midiToName(vocalTypeResult.maxMidi)}
                </p>
              </div>
            ) : null}
            {vocalError ? <p className="mt-3 font-mono text-xs text-accent">{vocalError}</p> : null}
          </div>
        </section>
      ) : null}

      {appMode === "tuner" ? (
        <section className="relative z-10 mx-auto max-w-[1400px] px-6 py-8">
          <div className="border-2 border-ink bg-card p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">Tuner</p>
            <div className="mt-3 flex items-center gap-2">
              {!tunerListening ? (
                <button type="button" onClick={() => void startTuner()} className="border-2 border-ink bg-accent px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-accent-foreground">Start tuner</button>
              ) : (
                <button type="button" onClick={stopMicSession} className="border-2 border-ink bg-ink px-3 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-paper">Stop tuner</button>
              )}
            </div>
            <p className="mt-4 text-3xl font-bold">{tunerMidi != null ? midiToName(tunerMidi) : "--"}</p>
            <p className="font-mono text-sm text-muted-foreground">{tunerMidi != null ? `${tunerCents > 0 ? "+" : ""}${tunerCents} cents` : "Sing a steady note into your mic."}</p>
            <div className="mt-4 h-3 w-full border-2 border-ink bg-paper">
              <div className="h-full bg-accent transition-all" style={{ width: `${Math.min(100, Math.max(0, ((tunerCents + 50) / 100) * 100))}%` }} />
            </div>
            {tunerError ? <p className="mt-3 font-mono text-xs text-accent">{tunerError}</p> : null}
          </div>
        </section>
      ) : null}

      {appMode === "midi_player" ? (
        <section className="relative z-10 mx-auto max-w-[1400px] px-6 py-8">
          <div className="border-2 border-ink bg-card p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">MIDI Player</p>
            <p className="mt-2 text-sm text-muted-foreground">Upload a `.mid`/`.midi` file and play it with the built-in synth.</p>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <label className="inline-flex cursor-pointer items-center gap-2 border-2 border-ink bg-paper px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-ink hover:bg-ink hover:text-paper">
                <Upload size={12} />
                Upload MIDI
                <input type="file" accept=".mid,.midi,audio/midi" className="sr-only" onChange={(e) => void handleMidiPlayerUpload(e)} />
              </label>
              <button
                type="button"
                onClick={() => void startMidiPlayback()}
                disabled={midiPlayerNotes.length === 0 || midiPlayerPlaying}
                className="inline-flex items-center gap-1.5 border-2 border-ink bg-accent px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-accent-foreground disabled:opacity-40"
              >
                <Play size={12} />
                {midiPlayerPlaying ? "Playing..." : "Play MIDI"}
              </button>
              <button
                type="button"
                onClick={stopMidiPlayback}
                disabled={!midiPlayerPlaying}
                className="border-2 border-ink bg-ink px-3 py-1.5 font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-paper disabled:opacity-40"
              >
                Stop
              </button>
            </div>
            <div className="mt-4 border-2 border-ink bg-paper p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink">Loaded file: {midiPlayerFileName ?? "-"}</p>
              <p className="mt-1 text-sm text-muted-foreground">Notes parsed: {midiPlayerNotes.length}</p>
              {midiPlayerNotes.length > 0 ? (
                <p className="text-sm text-muted-foreground">
                  Duration: {Math.max(...midiPlayerNotes.map((n) => n.endSec)).toFixed(2)}s
                </p>
              ) : null}
            </div>
            {midiPlayerError ? <p className="mt-3 font-mono text-xs text-accent">{midiPlayerError}</p> : null}
          </div>
        </section>
      ) : null}

      <footer className="relative z-10 border-t-2 border-ink">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-6 py-5 font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
  <span>MusicMetAI</span>

  <div className="flex flex-wrap items-center gap-4">
    <span>Support & bugs: whenmusicmetai@gmail.com</span>

    <a
      href="https://instagram.com/yourusername"
      target="_blank"
      rel="noopener noreferrer"
      className="hover:text-white transition-colors"
    >
      Instagram
    </a>

    <a
      href="https://facebook.com/yourusername"
      target="_blank"
      rel="noopener noreferrer"
      className="hover:text-white transition-colors"
    >
      Facebook
    </a>

    <a
      href="https://youtube.com/@yourchannel"
      target="_blank"
      rel="noopener noreferrer"
      className="hover:text-white transition-colors"
    >
      YouTube
    </a>
    <a
      href="https://youtube.com/@yourchannel"
      target="_blank"
      rel="noopener noreferrer"
      className="hover:text-white transition-colors"
    >
      Linkdln
    </a>
  </div>
</div>
      </footer>
    </div>
  );
}
