#!/usr/bin/env python3
"""CutLogic: auto-cut a raw video to match a text script.

Pipeline: extract audio (ffmpeg) -> transcribe with word timestamps (Deepgram)
-> align script sentences to transcript words -> cut & concat (ffmpeg).

Stdlib-only; requires ffmpeg/ffprobe on PATH and DEEPGRAM_API_KEY in the
environment or a .env file next to this script.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true"


def die(msg: str) -> "NoReturn":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("DEEPGRAM_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not key:
        die(
            "DEEPGRAM_API_KEY not set.\n"
            "  1. Sign up at https://console.deepgram.com (free credit included)\n"
            "  2. Create an API key\n"
            "  3. export DEEPGRAM_API_KEY=your_key  (or put it in a .env file here)"
        )
    return key


def run(cmd: list, desc: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"{desc} failed:\n{proc.stderr.strip()[-2000:]}")


# ---------------------------------------------------------------- transcription

def extract_audio(video: Path, work: Path) -> Path:
    audio = work / f"{video.stem}.audio.ogg"
    if audio.exists() and audio.stat().st_mtime >= video.stat().st_mtime:
        print(f"[1/4] audio already extracted: {audio}")
        return audio
    print(f"[1/4] extracting audio -> {audio}")
    run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
         "-c:a", "libopus", "-b:a", "32k", str(audio)],
        "audio extraction",
    )
    return audio


def transcribe(video: Path, audio: Path, work: Path, api_key: str) -> list:
    """Return flat word list [{word, start, end}, ...], caching the raw response."""
    cache = work / f"{video.stem}.transcript.json"
    if cache.exists() and cache.stat().st_mtime >= video.stat().st_mtime:
        print(f"[2/4] using cached transcript: {cache}")
        data = json.loads(cache.read_text())
    else:
        print("[2/4] transcribing with Deepgram (nova-3)...")
        req = urllib.request.Request(
            DEEPGRAM_URL,
            data=audio.read_bytes(),
            headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/ogg"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            die(f"Deepgram HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        except urllib.error.URLError as e:
            die(f"could not reach Deepgram: {e.reason}")
        cache.write_text(json.dumps(data, indent=2))
        print(f"      transcript saved to {cache}")

    try:
        words = data["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError):
        die("transcript has no words — is there speech in the video?")
    if not words:
        die("transcript is empty — is there speech in the video?")
    return words


# ------------------------------------------------------------------- alignment

_norm_re = re.compile(r"[^a-z0-9' ]+")


def norm_tokens(text: str) -> list:
    return _norm_re.sub(" ", text.lower()).split()


def split_sentences(script: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", script.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Match:
    sentence: str
    score: float
    wstart: int  # index into word list
    wend: int    # inclusive
    t0: float
    t1: float


def align(sentences: list, words: list, threshold: float) -> tuple:
    """Monotonically match each sentence to the best transcript window.

    Among near-equal matches, prefers the later one — the last take of a
    flubbed line is usually the keeper.
    """
    wtokens = [norm_tokens(w["word"]) for w in words]
    wtokens = [(t[0] if t else "") for t in wtokens]

    matches, warnings = [], []
    search_start = 0
    for sent in sentences:
        stoks = norm_tokens(sent)
        if not stoks:
            continue
        n = len(stoks)
        lengths = sorted({max(1, round(n * f)) for f in (0.8, 1.0, 1.2)})
        best = None  # (score, start, end)
        sm = difflib.SequenceMatcher(autojunk=False)
        sm.set_seq2(stoks)
        for start in range(search_start, len(words)):
            for length in lengths:
                end = start + length
                if end > len(words):
                    continue
                sm.set_seq1(wtokens[start:end])
                if sm.real_quick_ratio() < threshold:
                    continue
                score = sm.ratio()
                if best is None or score > best[0] + 0.02:
                    best = (score, start, end - 1)
                elif score >= threshold and score >= best[0] - 0.02:
                    # near-tie at/above threshold: prefer the later take
                    best = (score, start, end - 1)
        if best and best[0] >= threshold:
            score, ws, we = best
            matches.append(Match(sent, score, ws, we, words[ws]["start"], words[we]["end"]))
            search_start = we + 1
        else:
            got = f"best score {best[0]:.2f}" if best else "no candidate"
            warnings.append(f"unmatched (skipped): \"{sent[:60]}\" ({got})")
    return matches, warnings


def build_segments(matches: list, words: list, pad_pre: float, pad_post: float,
                   merge_gap: float, max_pause: float, duration: float) -> list:
    # Split each matched sentence at internal silences longer than max_pause
    # (reading pauses, breaths) so dead air inside a take gets cut too.
    spans = []
    for m in matches:
        run_start = m.wstart
        for i in range(m.wstart, m.wend):
            if words[i + 1]["start"] - words[i]["end"] > max_pause:
                spans.append((words[run_start]["start"], words[i]["end"]))
                run_start = i + 1
        spans.append((words[run_start]["start"], words[m.wend]["end"]))

    segs = []
    for t0, t1 in spans:
        t0 = max(0.0, t0 - pad_pre)
        t1 = min(duration, t1 + pad_post)
        if segs and t0 - segs[-1][1] <= merge_gap:
            segs[-1][1] = max(segs[-1][1], t1)
        else:
            segs.append([t0, t1])
    return segs


# --------------------------------------------------------------------- cutting

def probe_duration(video: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        die(f"ffprobe could not read {video}:\n{proc.stderr.strip()}")


def render(video: Path, segs: list, work: Path, output: Path) -> None:
    print(f"[4/4] cutting {len(segs)} segment(s) and concatenating...")
    seg_files = []
    for i, (t0, t1) in enumerate(segs):
        seg = work / f"seg_{i:03d}.mp4"
        run(
            ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(video),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", str(seg)],
            f"cutting segment {i} ({t0:.2f}-{t1:.2f}s)",
        )
        seg_files.append(seg)
        print(f"      seg {i:03d}: {t0:8.2f}s -> {t1:8.2f}s")

    concat_list = work / "segments.txt"
    concat_list.write_text("".join(f"file '{s.resolve()}'\n" for s in seg_files))
    run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(output)],
        "concatenation",
    )


# ------------------------------------------------------------------------ main

def fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-cut a raw video to match a script.")
    ap.add_argument("video", type=Path, help="raw video file")
    ap.add_argument("script", type=Path, help="text script of the final cut")
    ap.add_argument("-o", "--output", type=Path, default=Path("output.mp4"))
    ap.add_argument("--dry-run", action="store_true", help="print cut list, don't render")
    ap.add_argument("--threshold", type=float, default=0.8, help="match score cutoff (0-1)")
    ap.add_argument("--pad-pre", type=float, default=0.15, help="seconds kept before each match")
    ap.add_argument("--pad-post", type=float, default=0.25, help="seconds kept after each match")
    ap.add_argument("--merge-gap", type=float, default=0.3,
                    help="merge segments closer than this many seconds")
    ap.add_argument("--max-pause", type=float, default=0.6,
                    help="cut silences inside a sentence longer than this many seconds")
    ap.add_argument("--work-dir", type=Path, default=Path("work"))
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found on PATH (brew install ffmpeg)")
    if not args.video.exists():
        die(f"video not found: {args.video}")
    if not args.script.exists():
        die(f"script not found: {args.script}")

    api_key = load_api_key()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    audio = extract_audio(args.video, args.work_dir)
    words = transcribe(args.video, audio, args.work_dir, api_key)
    duration = probe_duration(args.video)
    sentences = split_sentences(args.script.read_text())
    print(f"[3/4] aligning {len(sentences)} script sentence(s) "
          f"against {len(words)} transcript words...")

    matches, warnings = align(sentences, words, args.threshold)
    for w in warnings:
        print(f"      warning: {w}")
    if not matches:
        die("no script sentences matched the transcript — check the script, "
            "or lower --threshold")

    print(f"\n{'score':>6}  {'start':>9}  {'end':>9}  sentence")
    for m in matches:
        print(f"{m.score:6.2f}  {fmt_t(m.t0):>9}  {fmt_t(m.t1):>9}  {m.sentence[:70]}")

    segs = build_segments(matches, words, args.pad_pre, args.pad_post,
                          args.merge_gap, args.max_pause, duration)
    kept = sum(t1 - t0 for t0, t1 in segs)
    print(f"\n{len(matches)}/{len(sentences)} sentences matched -> {len(segs)} segment(s), "
          f"keeping {fmt_t(kept)} of {fmt_t(duration)}")

    cuts_file = args.work_dir / "cuts.json"
    cuts_file.write_text(json.dumps({
        "video": str(args.video),
        "segments": [{"start": t0, "end": t1} for t0, t1 in segs],
        "matches": [{"sentence": m.sentence, "score": round(m.score, 3),
                     "start": m.t0, "end": m.t1} for m in matches],
        "warnings": warnings,
    }, indent=2))
    print(f"cut list written to {cuts_file}")

    if args.dry_run:
        print("dry run — not rendering.")
        return

    render(args.video, segs, args.work_dir, args.output)
    print(f"\ndone: {args.output}")


if __name__ == "__main__":
    main()
