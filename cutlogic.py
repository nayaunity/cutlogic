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


def deepgram_post(payload: bytes, api_key: str) -> dict:
    req = urllib.request.Request(
        DEEPGRAM_URL,
        data=payload,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/ogg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        die(f"Deepgram HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as e:
        die(f"could not reach Deepgram: {e.reason}")


def resp_words(data: dict) -> list:
    try:
        return data["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError):
        return []


def recover_gaps(video: Path, words: list, work: Path, api_key: str,
                 min_gap: float = 1.5) -> list:
    """Re-transcribe long inter-word silences in isolation.

    The full-file pass sometimes skips quiet false starts inside pauses;
    those hidden words matter because cut padding must not bleed into them.
    """
    recovered = []
    ctx = 1.0  # transcribe with surrounding context; tiny clips transcribe poorly
    for i in range(len(words) - 1):
        g0, g1 = words[i]["end"], words[i + 1]["start"]
        if g1 - g0 < min_gap:
            continue
        c0 = max(0.0, g0 - ctx)
        clip = work / "gap.ogg"
        run(["ffmpeg", "-y", "-ss", f"{c0:.2f}", "-to", f"{g1 + ctx:.2f}", "-i", str(video),
             "-vn", "-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(clip)],
            f"extracting gap {g0:.1f}-{g1:.1f}s")
        for w in resp_words(deepgram_post(clip.read_bytes(), api_key)):
            s, e = w["start"] + c0, w["end"] + c0
            if not (g0 <= (s + e) / 2 <= g1):
                continue  # context region; those words are already in the list
            recovered.append({"word": w["word"], "start": round(s, 3), "end": round(e, 3)})
    clip = work / "gap.ogg"
    if clip.exists():
        clip.unlink()
    if recovered:
        print(f"      recovered {len(recovered)} hidden word(s) inside pauses")
    return sorted(words + recovered, key=lambda w: w["start"])


def transcribe(video: Path, audio: Path, work: Path, api_key: str) -> list:
    """Return flat word list [{word, start, end}, ...], caching the result."""
    cache = work / f"{video.stem}.transcript.json"
    if cache.exists() and cache.stat().st_mtime >= video.stat().st_mtime:
        print(f"[2/4] using cached transcript: {cache}")
        data = json.loads(cache.read_text())
    else:
        print("[2/4] transcribing with Deepgram (nova-3)...")
        data = deepgram_post(audio.read_bytes(), api_key)
        cache.write_text(json.dumps(data, indent=2))
        print(f"      transcript saved to {cache}")

    words = resp_words(data)
    if not words:
        die("transcript has no words — is there speech in the video?")

    if data.get("cutlogic_gap_recovered") != 2:
        words = recover_gaps(video, words, work, api_key)
        data["results"]["channels"][0]["alternatives"][0]["words"] = words
        data["cutlogic_gap_recovered"] = 2
        cache.write_text(json.dumps(data, indent=2))
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
    wjoined = ["".join(t) for t in wtokens]  # full word, spaces stripped
    wtokens = [(t[0] if t else "") for t in wtokens]
    MIN_TAIL = 0.15  # seconds of clean air needed after a take's last word

    def tail_air(we: int) -> float:
        if we + 1 >= len(words):
            return float("inf")
        return words[we + 1]["start"] - words[we]["end"]

    matches, warnings = [], []
    search_start = 0
    for sent in sentences:
        stoks = norm_tokens(sent)
        if not stoks:
            continue
        n = len(stoks)
        lengths = sorted({max(1, round(n * f)) for f in (0.8, 1.0, 1.2)})
        sm = difflib.SequenceMatcher(autojunk=False)
        sm.set_seq2(stoks)
        smc = difflib.SequenceMatcher(autojunk=False)
        smc.set_seq2("".join(stoks))

        def char_score(s: int, e: int) -> float:
            smc.set_seq1("".join(wjoined[s:e + 1]))
            return smc.ratio()

        def search(from_idx: int):
            """Best window at/after from_idx: coarse token scan, then char-level
            boundary refinement (compound-word splits like "SkillsBuild" vs
            "skills build" would skew token-level edges)."""
            best = None  # (score, start, end)
            for start in range(from_idx, len(words)):
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
            if best is None or best[0] < threshold:
                return best
            best = (char_score(best[1], best[2]), best[1], best[2])
            improved = True
            while improved:
                improved = False
                _, ws, we = best
                for ds in range(-3, 4):
                    for de in range(-3, 4):
                        s2, e2 = ws + ds, we + de
                        if s2 < from_idx or e2 >= len(words) or e2 < s2:
                            continue
                        cs = char_score(s2, e2)
                        if cs > best[0] + 1e-9:
                            best = (cs, s2, e2)
                            improved = True
            return best

        best = search(search_start)
        if best and best[0] >= threshold:
            score, ws, we = best
            if tail_air(we) < MIN_TAIL:
                # The speaker restarted right on top of this take's last word,
                # so no cut point can keep the word intact. A later take that
                # ends into clean air beats a slightly better-worded one.
                alt = search(we + 1)
                if alt and alt[0] >= max(threshold, score - 0.2) \
                        and tail_air(alt[2]) >= MIN_TAIL:
                    print(f"      note: best take of \"{sent[:50]}\" has no clean "
                          f"ending; using a later take (score {alt[0]:.2f})")
                    score, ws, we = alt
            matches.append(Match(sent, score, ws, we, words[ws]["start"], words[we]["end"]))
            search_start = we + 1
        else:
            got = f"best score {best[0]:.2f}" if best else "no candidate"
            warnings.append(f"unmatched (skipped): \"{sent[:60]}\" ({got})")
    return matches, warnings


def build_segments(matches: list, words: list, video: Path,
                   pad_pre: float, pad_post: float,
                   merge_gap: float, max_pause: float, duration: float) -> list:
    # Split each matched sentence at internal silences longer than max_pause
    # (reading pauses, breaths) so dead air inside a take gets cut too.
    wnorm = ["".join(norm_tokens(w["word"])) for w in words]
    spans = []  # [first_word_index, last_word_index]
    for m in matches:
        mspans = []
        run_start = m.wstart
        for i in range(m.wstart, m.wend):
            if words[i + 1]["start"] - words[i]["end"] > max_pause:
                mspans.append([run_start, i])
                run_start = i + 1
        mspans.append([run_start, m.wend])
        # Stutter removal: if the last words of a sub-span are re-spoken at the
        # start of the next one ("you can even | can even learn..."), the first
        # occurrence is a false start — trim it so the line isn't repeated.
        for a, b in zip(mspans, mspans[1:]):
            maxk = min(a[1] - a[0] + 1, b[1] - b[0] + 1)
            for k in range(maxk, 1, -1):
                if wnorm[a[1] - k + 1:a[1] + 1] == wnorm[b[0]:b[0] + k]:
                    a[1] -= k
                    break
        # ASR sometimes silently drops a stuttered restart, leaving speech the
        # word list doesn't know about inside a "pause". Check the pause audio:
        # if speech energy runs continuously into the next sub-span, snap that
        # span's start back to the true silence boundary (so no utterance is
        # entered mid-word), and drop a short earlier sub-span as a false start.
        mspans = [s + [None] for s in mspans if s[1] >= s[0]]  # [i0, i1, t0_override]
        dropped = set()
        for idx, (a, b) in enumerate(zip(mspans, mspans[1:])):
            g0, g1 = words[a[1]]["end"] + 0.05, words[b[0]]["start"]
            if g1 - g0 < 0.8:
                continue
            iv = speech_intervals(video, g0, g1)
            if iv and iv[-1][1] >= g1 - 0.25:  # hidden speech leads into b
                b[2] = max(g0, iv[-1][0] - 0.05)
                if a[1] - a[0] + 1 <= 4:
                    dropped.add(idx)
                    print(f"      note: dropping false start before "
                          f"{words[b[0]]['start']:.2f}s (hidden retake in pause)")
        spans.extend(s for i, s in enumerate(mspans) if i not in dropped)

    segs = []
    for i0, i1, t0_override in spans:
        # Pad, but never into a neighboring word — that's how a false start
        # ("next we ha-") bleeds into the end of the previous cut.
        if t0_override is not None:
            t0 = t0_override
        else:
            t0 = words[i0]["start"] - pad_pre
            if i0 > 0:
                t0 = max(t0, words[i0 - 1]["end"] + 0.05)
            t0 = max(0.0, min(t0, words[i0]["start"]))
        t1 = words[i1]["end"] + pad_post
        if i1 + 1 < len(words):
            t1 = min(t1, words[i1 + 1]["start"] - 0.05)
        t1 = min(duration, max(t1, words[i1]["end"]))
        # ASR word timestamps absorb breaths and voice decay, leaving hidden
        # air inside the cut. The two edges need different treatment:
        # - Heads: the inhale before speech can be as loud as quiet speech, so
        #   no energy floor separates them — but duration does. Breaths are
        #   sub-0.15s bursts; voice comes in sustained runs. Snap the head to
        #   the last substantial voiced run (voice-level threshold, chained
        #   across stop-consonant closures), margin for soft onset consonants.
        # - Tails: soft word endings (trailing sibilants, decay) sit far below
        #   voice level, so use the sensitive threshold there; post-speech
        #   breath is separated from the word by registering silence.
        iv = speech_intervals(video, t0, min(t1, words[i0]["end"] + 0.1),
                              noise="-16dB", min_silence=0.05)
        runs = []
        for s, e in iv:
            if runs and s - runs[-1][1] <= 0.09:
                runs[-1][1] = e
            else:
                runs.append([s, e])
        if runs:
            t0 = max(t0, runs[-1][0] - pad_pre - 0.10)
        iv = speech_intervals(video, max(t0, words[i1]["start"] - 0.1), t1,
                              noise="-27dB", min_silence=0.12)
        if iv:
            t1 = max(min(t1, iv[-1][1] + pad_post), t0 + 0.1)
        if segs and t0 - segs[-1][1] <= merge_gap:
            segs[-1][1] = max(segs[-1][1], t1)
        else:
            segs.append([t0, t1])
    return segs


def speech_intervals(video: Path, t0: float, t1: float,
                     noise: str = "-35dB", min_silence: float = 0.25) -> list:
    """Actual speech spans (by audio energy) inside [t0, t1] of the video.

    The default -35dB floor is paranoid (quiet mumbles count as speech) —
    right for detecting hidden retakes. Boundary tightening passes -27dB so
    breaths and room noise count as silence and get cut through.
    """
    proc = subprocess.run(
        ["ffmpeg", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(video),
         "-vn", "-af", f"silencedetect=noise={noise}:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    intervals, cur = [], t0
    for kind, val in re.findall(r"silence_(start|end): ([0-9.]+)", proc.stderr):
        t = t0 + float(val)
        if kind == "start":
            if cur is not None and t - cur >= 0.15:
                intervals.append((cur, t))
            cur = None
        else:
            cur = t
    if cur is not None and t1 - cur >= 0.15:
        intervals.append((cur, t1))
    return intervals


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
    ap.add_argument("--pad-pre", type=float, default=0.05, help="seconds kept before each match")
    ap.add_argument("--pad-post", type=float, default=0.12, help="seconds kept after each match")
    ap.add_argument("--merge-gap", type=float, default=0.15,
                    help="merge segments closer than this many seconds")
    ap.add_argument("--max-pause", type=float, default=0.35,
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

    segs = build_segments(matches, words, args.video, args.pad_pre, args.pad_post,
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
