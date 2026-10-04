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
import math
import os
import re
import shutil
import struct
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
        print(f"[1/5] audio already extracted: {audio}")
        return audio
    print(f"[1/5] extracting audio -> {audio}")
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
    # Never merge a recovered word that overlaps a word we already have —
    # duplicates skew the neighbor-based padding clamps.
    recovered = [r for r in recovered
                 if not any(r["start"] < w["end"] + 0.05 and w["start"] < r["end"] + 0.05
                            for w in words)]
    if recovered:
        print(f"      recovered {len(recovered)} hidden word(s) inside pauses")
    return sorted(words + recovered, key=lambda w: w["start"])


def transcribe(video: Path, audio: Path, work: Path, api_key: str) -> list:
    """Return flat word list [{word, start, end}, ...], caching the result."""
    cache = work / f"{video.stem}.transcript.json"
    if cache.exists() and cache.stat().st_mtime >= video.stat().st_mtime:
        print(f"[2/5] using cached transcript: {cache}")
        data = json.loads(cache.read_text())
    else:
        print("[2/5] transcribing with Deepgram (nova-3)...")
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

# Contractions expand on both sides so "here's the plan" matches "here is the
# plan" at the token level, not only after char refinement.
_CONTRACTIONS = {
    "i'm": "i am", "i'll": "i will", "i've": "i have", "i'd": "i would",
    "you're": "you are", "you'll": "you will", "you've": "you have",
    "we're": "we are", "we'll": "we will", "we've": "we have",
    "they're": "they are", "they'll": "they will", "they've": "they have",
    "it's": "it is", "that's": "that is", "here's": "here is",
    "there's": "there is", "what's": "what is", "who's": "who is",
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "didn't": "did not",
}


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> list:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + (_num_words(n % 10) if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (_num_words(n % 100) if n % 100 else [])
    if n < 1_000_000:
        return _num_words(n // 1000) + ["thousand"] + (_num_words(n % 1000) if n % 1000 else [])
    return [str(n)]


def norm_tokens(text: str) -> list:
    # "$400,000" -> "400000"; scripts write digits, speakers say words —
    # normalize both sides to words so "90 days" matches "ninety days".
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    out = []
    for t in _norm_re.sub(" ", text).split():
        if t.isdigit():
            out.extend(_num_words(int(t)))
        elif re.fullmatch(r"\d+k", t):
            out.extend(_num_words(int(t[:-1]) * 1000))
        elif t == "k":
            out.append("thousand")
        else:
            out.extend(_CONTRACTIONS.get(t, t).split())
    return out


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
    low: bool = False  # accepted below threshold (delivery deviates); review


def align(sentences: list, words: list, threshold: float,
          min_score: float = 0.45) -> tuple:
    """Monotonically match each sentence to the best transcript window.

    Among near-equal matches, prefers the later one — the last take of a
    flubbed line is usually the keeper.
    """
    wexp = [norm_tokens(w["word"]) for w in words]  # every token a word expands to
    wjoined = ["".join(t) for t in wexp]  # full word, spaces stripped
    MIN_TAIL = 0.15  # seconds of clean air needed after a take's last word

    def tail_air(we: int) -> float:
        if we + 1 >= len(words):
            return float("inf")
        return words[we + 1]["start"] - words[we]["end"]

    def search(stoks: list, from_idx: int, prefer_later: bool = True,
               until: int = None):
        """Best window at/after from_idx: coarse token scan, then char-level
        boundary refinement (compound-word splits like "SkillsBuild" vs
        "skills build" would skew token-level edges)."""
        n = len(stoks)
        # Window lengths are in ASR words, but n counts script tokens, and one
        # spoken word can expand to many tokens ("$33,330" -> thirty three
        # thousand three hundred thirty). Always try one- and two-word
        # windows too so such lines can still match.
        lengths = sorted({1, 2} | {max(1, round(n * f)) for f in (0.6, 0.8, 1.0, 1.2)})
        sm = difflib.SequenceMatcher(autojunk=False)
        sm.set_seq2(stoks)
        smc = difflib.SequenceMatcher(autojunk=False)
        smc.set_seq2("".join(stoks))

        def char_score(s: int, e: int) -> float:
            smc.set_seq1("".join(wjoined[s:e + 1]))
            return smc.ratio()

        best = None  # (score, start, end)
        for start in range(from_idx, min(len(words), until or len(words))):
            for length in lengths:
                end = start + length
                if end > len(words):
                    continue
                sm.set_seq1([tok for k in range(start, end) for tok in wexp[k]])
                if sm.real_quick_ratio() < threshold:
                    continue
                score = sm.ratio()
                if best is None or score > best[0] + 0.02:
                    best = (score, start, end - 1)
                elif prefer_later and score >= threshold and score >= best[0] - 0.02:
                    # Near-tie at/above threshold. The token scan only sees
                    # each word's first token, so "$2,500" and "$0" look the
                    # same here; break the tie on full characters and only
                    # then prefer the later take.
                    if char_score(start, end - 1) >= char_score(best[1], best[2]) - 0.02:
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

    def search_clean_tail(stoks: list, from_idx: int, label: str,
                          prefer_later: bool = True, until: int = None):
        """search(), then swap to a later take if the best one's last word has
        a restart on top of it — no cut point could keep that word intact."""
        best = search(stoks, from_idx, prefer_later, until)
        if best and best[0] >= threshold and tail_air(best[2]) < MIN_TAIL:
            alt = search(stoks, best[2] + 1, prefer_later, until)
            if alt and alt[0] >= max(threshold, best[0] - 0.2) \
                    and tail_air(alt[2]) >= MIN_TAIL:
                print(f"      note: best take of \"{label[:50]}\" has no clean "
                      f"ending; using a later take (score {alt[0]:.2f})")
                return alt
        return best

    matches, warnings = [], []
    search_start = 0
    for sent in sentences:
        stoks = norm_tokens(sent)
        if not stoks:
            continue
        best = search_clean_tail(stoks, search_start, sent)

        # A weak whole-sentence match usually means no single take contains
        # the full line. The script's commas mark where splicing is legal:
        # retry clause by clause, each clause matched to its own best take.
        if best is None or best[0] < 0.97:
            parts = [p.strip() for p in sent.split(",") if p.strip()]
            if len(parts) < 2:
                # No commas — split before conjunctions and clause markers;
                # retakes restart there ("...like a startup / and I just
                # realized...", "followed by Wisprflow / where he talks out...").
                parts = [p.strip() for p in
                         re.split(r"\s+(?=(?:and|but|or|so|because|where|when|while|then)\s)",
                                  sent) if p.strip()]
            clauses, cur = [], ""
            for part in parts:
                cur = f"{cur} {part}".strip() if cur else part
                if len(norm_tokens(cur)) >= 3:
                    clauses.append(cur)
                    cur = ""
            if cur and clauses:
                clauses[-1] = f"{clauses[-1]} {cur}".strip()
            # Pieces of one sentence live near the whole-sentence match —
            # bound their search to that neighborhood, not the full video.
            bound = (best[2] + 60) if best else None

            def try_pieces(prefer_later: bool):
                # A greedy later-take chain can trap later clauses on flubbed
                # takes; the earlier-take chain sometimes completes instead.
                out, pos = [], search_start
                for i, clause in enumerate(clauses):
                    # Mid-sentence seams are expected to abut the next word,
                    # so the clean-tail take swap only applies to the last
                    # clause; phonetic tail rules handle interior boundaries.
                    if i == len(clauses) - 1:
                        b = search_clean_tail(norm_tokens(clause), pos, clause,
                                              prefer_later, until=bound)
                    else:
                        b = search(norm_tokens(clause), pos, prefer_later,
                                   until=bound)
                    if not b or b[0] < threshold:
                        return None
                    out.append((clause, b))
                    pos = b[2] + 1
                return out

            pieces = None
            if len(clauses) >= 2:
                cands = [p for p in (try_pieces(True), try_pieces(False)) if p]
                if cands:
                    pieces = max(cands, key=lambda p: min(b[0] for _, b in p))
            if pieces and (best is None or
                           min(b[0] for _, b in pieces) > best[0]):
                print(f"      note: no single take covers \"{sent[:50]}\"; "
                      f"splicing {len(pieces)} clauses from separate takes")
                for clause, (score, ws, we) in pieces:
                    matches.append(Match(clause, score, ws, we,
                                         words[ws]["start"], words[we]["end"]))
                search_start = pieces[-1][1][2] + 1
                continue

        if best and best[0] >= threshold:
            score, ws, we = best
            matches.append(Match(sent, score, ws, we, words[ws]["start"], words[we]["end"]))
            search_start = we + 1
        else:
            # Scripts repeat lines (taglines, hooks) the speaker only recorded
            # once. If no take exists after the cursor, reuse an earlier one —
            # cutting the same footage twice is what a human editor would do.
            reuse = search(stoks, 0)
            if reuse and reuse[0] >= threshold:
                score, ws, we = reuse
                print(f"      note: reusing earlier take at {words[ws]['start']:.2f}s "
                      f"for repeated line \"{sent[:50]}\"")
                matches.append(Match(sent, score, ws, we,
                                     words[ws]["start"], words[we]["end"]))
                # search_start intentionally not moved: the cursor tracks the
                # forward pass; a reused take is out-of-order by design.
            elif best and best[0] >= min_score:
                # Imperfect scripts are normal in production: a mid-score match
                # is usually the right take delivered in different words.
                # Keep it, marked for review, rather than silently dropping
                # the line from the video.
                score, ws, we = best
                matches.append(Match(sent, score, ws, we,
                                     words[ws]["start"], words[we]["end"],
                                     low=True))
                search_start = we + 1
            else:
                got = f"best score {best[0]:.2f}" if best else "no candidate"
                warnings.append(f"unmatched (skipped): \"{sent[:60]}\" ({got})")
    return matches, warnings


def build_segments(matches: list, words: list, video: Path,
                   pad_pre: float, pad_post: float,
                   merge_gap: float, max_pause: float, duration: float) -> list:
    global LAST_FOLLOW
    # Split each matched sentence at internal silences longer than max_pause
    # (reading pauses, breaths) so dead air inside a take gets cut too.
    wnorm = ["".join(norm_tokens(w["word"])) for w in words]
    spans = []  # [first_word_index, last_word_index]
    for m in matches:
        # ASR sometimes splits one spoken word into two identical contiguous
        # tokens ("i" 507.43-507.67 + "i" 507.67-508.23); the aligner's
        # later-take preference then opens the cut on the second token, after
        # the voice has already ended. Back up onto the first one.
        while m.wstart > 0 and wnorm[m.wstart - 1] == wnorm[m.wstart] and \
                words[m.wstart]["start"] - words[m.wstart - 1]["end"] <= 0.05:
            m.wstart -= 1
        mspans = []
        run_start = m.wstart
        for i in range(m.wstart, m.wend):
            gap = words[i + 1]["start"] - words[i]["end"]
            # Short ASR gaps aren't always pauses: a drawn-out word or an
            # untranscribed syllable can fill one, and cutting it removes
            # speech. Check the audio. Long gaps (>= 0.8s) are always split
            # so the hidden-retake logic below can inspect them.
            if gap > max_pause and (gap >= 0.8 or gap_is_quiet(
                    video, words[i]["end"], words[i + 1]["start"])):
                mspans.append([run_start, i])
                run_start = i + 1
        mspans.append([run_start, m.wend])
        # Stutter removal: if the last words of a sub-span are re-spoken at the
        # start of the next one ("you can even | can even learn..."), the first
        # occurrence is a false start — trim it so the line isn't repeated.
        for a, b in zip(mspans, mspans[1:]):
            # k=1 included: a single word repeated across a pause-split
            # ("I- ... I had") is a restart, not deliberate repetition —
            # deliberate doubles have no pause and never get split.
            maxk = min(a[1] - a[0] + 1, b[1] - b[0] + 1)
            for k in range(maxk, 0, -1):
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
            if g1 - g0 >= 0.8:
                runs = voiced_runs(video, g0, g1)
                if runs and runs[-1][1] >= g1 - 0.25:  # hidden speech leads into b
                    b[2] = max(g0, runs[-1][0] - 0.05)
                    if a[1] - a[0] + 1 <= 4:
                        dropped.add(idx)
                        print(f"      note: dropping false start before "
                              f"{words[b[0]]['start']:.2f}s (hidden retake in pause)")
            elif len(set(wnorm[a[0]:a[1] + 1])) == 1 and g1 - g0 >= 0.3:
                # A lone word stalled between pauses ("launch plan | for | a
                # membership") is a false start when the ASR absorbed its
                # restart into the next word: voice leads into b well before
                # b's first marked word. Drop the stall, open b at the voice.
                iv = speech_intervals(video, g0, g1 + 0.2, noise="-27dB",
                                      min_silence=0.05)
                if iv and iv[-1][1] >= g1 and iv[-1][0] <= g1 - 0.04:
                    b[2] = max(g0, iv[-1][0] - 0.05)
                    dropped.add(idx)
                    print(f"      note: dropping stalled word "
                          f"'{words[a[0]]['word']}' at {words[a[0]]['start']:.2f}s "
                          f"(restart absorbed into next word)")
        spans.extend(s for i, s in enumerate(mspans) if i not in dropped)

    segs = []
    for i0, i1, t0_override in spans:
        # Pad, but never into a neighboring word — that's how a false start
        # ("next we ha-") bleeds into the end of the previous cut.
        if t0_override is not None:
            t0 = t0_override
        else:
            # ASR word starts can be late as well as early: look back
            # further than the pad and let head_onset() find the onset.
            t0 = words[i0]["start"] - max(pad_pre, 0.12)
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
        # ASR word starts can absorb a breath and a whole inhale pause, so the
        # first word's marked end may still be before the voice begins — scan
        # well past it and snap to the first sustained voiced run.
        win_end = min(t1, max(words[i0]["end"] + 0.1, t0 + 2.5))
        runs = voiced_runs(video, t0, win_end)
        if runs:
            # head_onset keeps a quiet voiced first word ("a", "the") by
            # ZCR, so no separate volume safeguard is needed here.
            t0 = head_onset(video, t0, runs[0][0], words[i0]["start"],
                            words[i0]["word"], margin=pad_pre)
        else:
            # No sustained voice found (a short quiet word): don't keep
            # the whole look-back as dead air.
            t0 = max(t0, words[i0]["start"] - pad_pre)
        tail_from = max(t0, words[i1]["start"] - 0.1)
        # ASR word ends can also be EARLY (a drawn-out "monetizing", a
        # trailing "L"): let the tail follow voice past the marked end, but
        # only voice contiguous with the word — never the next word or a
        # breath after a pause.
        t1_ext = words[i1]["end"] + 0.45
        if i1 + 1 < len(words):
            t1_ext = min(t1_ext, words[i1 + 1]["start"] - 0.05)
        t1_ext = min(duration, max(t1_ext, t1))
        v_end = last_voice(video, tail_from, t1_ext)
        if v_end > t1 - pad_post + 0.02:
            lv = rms_windows(video, max(tail_from, words[i1]["end"] - 0.06), v_end, 0.02)
            run, contiguous = 0, True
            for x in lv:
                run = run + 1 if x < ROOM_DB + 3.0 else 0
                if run >= 5:
                    contiguous = False
                    break
            if contiguous:
                t1 = max(t1, min(t1_ext, v_end + pad_post))
            else:
                v_end = last_voice(video, tail_from, t1)
        last = "".join(norm_tokens(words[i1]["word"]))
        if v_end >= tail_from and words[i1]["start"] >= v_end - 0.05:
            # The whole last word sits under the tail threshold (a
            # trailed-off "now"): it is a word, not a fading ending, so
            # follow it to its marked end while it stays above room tone.
            lim = min(t1_ext, words[i1]["end"] + 0.1)
            if lim > v_end:
                lw = level_windows(video, v_end, lim, 0.02)
                k, dips = 0, 0
                while k < len(lw) and dips <= 1:
                    if lw[k][0] > ROOM_DB + 4.0:
                        dips = 0
                    else:
                        dips += 1
                    k += 1
                v_end = v_end + max(0, k - dips) * 0.02
                t1 = max(t1, min(t1_ext, v_end + pad_post))
        sib_final = last[-1:] in "sz" or words[i1]["word"].startswith("$")
        if sib_final and v_end >= tail_from:
            # A final "s" is often detached: a short gap, then a soft
            # high-frequency burst near room tone ("dollar...s"). Look
            # ahead to the word's marked end for it and keep it.
            lim = min(t1_ext, words[i1]["end"] + 0.1)
            if lim > v_end + 0.04:
                lw = level_windows(video, v_end, lim, 0.02)
                run_s, burst_end, missed = 0, None, False
                for k, (r, z) in enumerate(lw):
                    if k * 0.02 > 0.4 and burst_end is None:
                        break
                    if z >= 0.25 and r > ROOM_DB - 1.0:
                        run_s += 1
                        missed = False
                        if run_s >= 3:
                            burst_end = k + 1
                    elif run_s and not missed:
                        missed = True  # tolerate one weak window inside the burst
                    else:
                        if burst_end is not None:
                            break
                        run_s, missed = 0, False
                if burst_end is not None:
                    v_end = v_end + burst_end * 0.02
                    LAST_FOLLOW = 0.0
                    t1 = max(t1, min(t1_ext, v_end + 0.03))
        if v_end >= tail_from:
            # The trailing-sound follow already acts as padding.
            snapped = min(t1, v_end + max(0.02, pad_post - LAST_FOLLOW))
            # A final stop consonant ("need", "build") has a near-silent
            # closure + release just past the energy end. Keep a release
            # allowance, capped near the ASR word end — ASR ends overrun
            # into silence, so they can't be trusted on their own either.
            if last[-1:] in "bdgkpt":
                snapped = max(snapped,
                              min(v_end + 0.08, words[i1]["end"] + 0.02, t1))
            elif last[-1:] in "sz":
                # Trailing sibilants drag on: pad only a little after the
                # measured end instead of the full pad_post. Never cut into
                # the "s" itself — on a quiet recording that removes it.
                snapped = min(snapped, max(v_end + 0.03, t0 + 0.1))
            t1 = max(snapped, t0 + 0.1)
        if last[-1:] in "sz":
            t1 = max(min(t1, words[i1]["end"] + 0.2), t0 + 0.1)
        # Merge only forward-adjacent spans; a reused earlier take jumps
        # backward in source time and must stay its own segment.
        if segs and t0 >= segs[-1][0] and t0 - segs[-1][1] <= merge_gap:
            segs[-1][1] = max(segs[-1][1], t1)
        else:
            segs.append([t0, t1])

    # Anti-jitter: halting delivery produces runs of sub-second segments —
    # four cuts in two seconds reads as flicker. Merge shots shorter than
    # MIN_SHOT into a neighbor when the pause between them is small enough
    # to keep; a beat of natural pause beats machine-gun cuts.
    MIN_SHOT, KEEPABLE_GAP = 0.8, 0.7
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(segs):
            if seg[1] - seg[0] >= MIN_SHOT:
                continue
            cands = []
            if i > 0 and 0 <= seg[0] - segs[i - 1][1] <= KEEPABLE_GAP:
                cands.append((seg[0] - segs[i - 1][1], i - 1))
            if i + 1 < len(segs) and 0 <= segs[i + 1][0] - seg[1] <= KEEPABLE_GAP:
                cands.append((segs[i + 1][0] - seg[1], i + 1))
            if not cands:
                continue
            _, j = min(cands)
            a, b = (j, i) if j < i else (i, j)
            segs[a][1] = segs[b][1]
            del segs[b]
            changed = True
            break
    return segs


# Every energy threshold below was tuned on footage whose spoken sentences
# average REF_SPEECH_LEVEL dB. A quieter recording (phone across the room, a
# presenter who drops her voice reading figures) pushes real speech under
# those fixed floors and the cutter trims it as silence. calibrate_level()
# measures the actual speech level and LEVEL_OFFSET shifts every floor by
# the difference, clamped so a very quiet file never sinks into room noise.
REF_SPEECH_LEVEL = -15.0
LEVEL_OFFSET = 0.0
NOISE_FLOOR = -50.0
TAIL_DB = -25.0  # RMS level below which a word's tail counts as over
ROOM_DB = -60.0  # measured room tone (RMS)
WHISPER_DB = -54.0  # floor for following a trailed-off word ending
SPEECH_DB = -15.0  # measured median speech level (RMS)
LAST_FOLLOW = 0.0  # set by last_voice(): seconds of trailing sound followed


def calibrate_level(video: Path, matches: list) -> float:
    global LEVEL_OFFSET, NOISE_FLOOR, TAIL_DB, ROOM_DB, WHISPER_DB, SPEECH_DB
    spans = [(m.t0, m.t1) for m in matches if m.t1 - m.t0 >= 2.0][:8]
    if not spans:
        spans = [(m.t0, m.t1) for m in matches][:8]
    levels = sorted(mean_volume(video, a, b) for a, b in spans)
    level = levels[len(levels) // 2]
    SPEECH_DB = level
    LEVEL_OFFSET = max(-30.0, min(5.0, level - REF_SPEECH_LEVEL))
    # Room tone: the pauses between matched takes. silencedetect compares
    # sample peaks, which sit well above the RMS mean_volume reports, so no
    # floor may go within ~12 dB of the room RMS or tails never snap and
    # every cut drags its ASR word-end overrun along.
    gaps = [(a.t1 + 0.3, b.t0 - 0.3) for a, b in zip(matches, matches[1:])
            if b.t0 - a.t1 >= 1.2][:8]
    TAIL_DB = level - 18.0
    # The gaps also hold unscripted retakes, so take a low percentile of
    # short windows rather than a mean — the quiet tenth is room tone.
    win = []
    for a, b in gaps:
        win.extend(rms_windows(video, a, b, 0.1))
    if win:
        win.sort()
        noise = win[len(win) // 10]
        ROOM_DB = noise
        # Trailed-off endings are followed down to just above room tone on
        # a quiet recording, but on loud footage that floor is 35 dB under
        # speech and reverb decay lives there: cap it relative to speech so
        # tails don't drag.
        WHISPER_DB = max(noise + 8.0, level - 25.0)
        NOISE_FLOOR = noise + 12.0
        # last_voice() compares RMS, so it can sit closer to the room RMS
        # than the peak-based silencedetect floor above. On a quiet phone
        # recording (speech ~-41 dB, room ~-61 dB) a +12 floor lands only
        # 8 dB under speech and trims soft final syllables ("dollars").
        TAIL_DB = max(TAIL_DB, noise + 8.0)
        print(f"      speech level {level:.1f} dB, room tone {noise:.1f} dB "
              f"-> thresholds shifted {LEVEL_OFFSET:+.1f} dB, "
              f"floor {NOISE_FLOOR:.1f} dB, tail {TAIL_DB:.1f} dB")
    else:
        print(f"      speech level {level:.1f} dB -> thresholds shifted "
              f"{LEVEL_OFFSET:+.1f} dB, tail {TAIL_DB:.1f} dB")
    return LEVEL_OFFSET


def rms_windows(video: Path, t0: float, t1: float, step: float = 0.02) -> list:
    """RMS level (dB) of consecutive `step`-second windows in [t0, t1]."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
    )
    raw = proc.stdout
    n = len(raw) // 2
    if n == 0:
        return []
    samples = struct.unpack(f"<{n}h", raw)
    w = max(1, int(16000 * step))
    out = []
    for i in range(0, n - w + 1, w):
        chunk = samples[i:i + w]
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
        out.append(20 * math.log10(rms / 32768 + 1e-9))
    return out


def level_windows(video: Path, t0: float, t1: float, step: float = 0.02) -> list:
    """(RMS dB, zero-crossing rate) per `step`-second window in [t0, t1].

    ZCR separates what energy can't on a close-miked phone recording:
    an inhale is broadband noise (high ZCR), a vowel is periodic (low)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
    )
    raw = proc.stdout
    n = len(raw) // 2
    if n == 0:
        return []
    samples = struct.unpack(f"<{n}h", raw)
    w = max(1, int(16000 * step))
    out = []
    for i in range(0, n - w + 1, w):
        chunk = samples[i:i + w]
        rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
        zc = sum(1 for a, b in zip(chunk, chunk[1:]) if (a < 0) != (b < 0))
        out.append((20 * math.log10(rms / 32768 + 1e-9), zc / len(chunk)))
    return out


ZCR_NOISY = 0.25  # above this a 20ms window is a fricative/plosive/aspiration


def head_onset(video: Path, t0: float, run_start: float, w0s: float,
               first_word: str = "", margin: float = 0.02) -> float:
    """Where a cut should open ahead of the first word.

    On a close-miked phone recording the inhale before a line is nearly
    as loud as speech, so energy alone can't find the word. Measured on
    such footage: the inhale is low-frequency (ZCR ~0.05) and 12-20 dB
    under the vowel that follows; consonant onsets ("s", "st", "t") are
    high-frequency (ZCR 0.4-0.8) and run straight into the vowel.

    So: find the first window at speech level that is periodic (the
    vowel), then walk back over high-ZCR windows for up to 0.25s (the
    consonant onset), and over low-ZCR windows only while they are at
    speech level (a quiet real word) or within 40ms of the vowel (a nasal
    or approximant onset: "m", "w"). Stop at room tone. 30ms margin.
    """
    step = 0.02
    # A nasal onset ("m", "n") is a quiet low-ZCR murmur that looks just
    # like the inhale; a voiced "th" is similar. Let those words keep a
    # longer soft onset; everything else gets one window.
    fw = first_word.lower()
    ramped = fw[:1] in "aeiouwlry"  # vowels and glides ramp up monotonically
    soft = 0.16 if fw[:1] in "mn" else 0.02
    a = max(t0, min(run_start, w0s) - 0.2)
    b = max(run_start, w0s) + 0.6
    win = level_windows(video, a, b, step)
    if not win:
        return max(t0, run_start - 0.03)
    i = next((k for k, (r, z) in enumerate(win)
              if r >= SPEECH_DB - 2.0 and z < 0.2), None)
    if i is None:
        i = min(len(win) - 1, max(0, int(round((run_start - a) / step))))
    j = i
    noisy = rising = quiet = 0.0
    floor = win[i][0]  # strictly falling as we walk back = a real attack
    while j > 0:
        r, z = win[j - 1]
        if z >= ZCR_NOISY:
            if r <= ROOM_DB + 2.0 or noisy + step > 0.25:
                break
            noisy += step
        else:
            if r <= ROOM_DB + 4.0:
                break
            if r >= SPEECH_DB - 2.0:
                pass  # speech level: a quiet real word, keep
            elif ramped and r < floor - 1.0 and rising + step <= 0.12:
                rising += step  # the vowel's own attack ramp (monotonic)
                floor = r
            elif quiet + step <= soft:
                quiet += step  # nasal murmur / voiced onset allowance
            else:
                break
        j -= 1
    return max(t0, a + j * step - margin)


def gap_is_quiet(video: Path, g0: float, g1: float) -> bool:
    """True if the interior of an ASR word gap really drops to tail level
    for most of its length (i.e. it is a pause, not mistimed speech)."""
    if g1 - g0 < 0.2:
        return True
    lv = rms_windows(video, g0 + 0.05, g1 - 0.05, 0.02)
    if not lv:
        return True
    return sum(1 for x in lv if x <= TAIL_DB) >= 0.5 * len(lv)


def last_voice(video: Path, t0: float, t1: float) -> float:
    """End time of the last window in [t0, t1] whose RMS is above TAIL_DB.

    Peak-based silencedetect keeps reverb decay and breath alive long after
    a word is over; RMS against a level calibrated between speech and room
    tone finds where the word actually stops. Returns t0 if nothing is
    voiced. Sets LAST_FOLLOW to how far past that point a trailing sound
    was followed, so the caller can shrink its pad by that much.
    """
    global LAST_FOLLOW
    step = 0.02
    win = level_windows(video, t0, t1, step)
    LAST_FOLLOW = 0.0
    for i in range(len(win) - 1, -1, -1):
        if win[i][0] > TAIL_DB:
            # A speaker trailing off ends a line under TAIL_DB but above
            # room tone, and a final sibilant sits even lower but is
            # high-frequency (ZCR). Follow either for up to 0.2s; reverb
            # decay and breath are low-ZCR and near room tone, so they
            # are still cut.
            j = i
            while j + 1 < len(win) and j - i < int(0.06 / step):
                r, z = win[j + 1]
                if r > WHISPER_DB or (r > ROOM_DB + 5.0 and z >= 0.35):
                    j += 1
                else:
                    break
            LAST_FOLLOW = (j - i) * step
            return t0 + (j + 1) * step
    return t0


def _shift_noise(noise: str) -> str:
    base = float(noise.rstrip("dB"))
    return f"{max(NOISE_FLOOR, base + LEVEL_OFFSET):.1f}dB"


def speech_intervals(video: Path, t0: float, t1: float,
                     noise: str = "-35dB", min_silence: float = 0.25) -> list:
    """Actual speech spans (by audio energy) inside [t0, t1] of the video.

    The default -35dB floor is paranoid (quiet mumbles count as speech) —
    right for detecting hidden retakes. Boundary tightening passes -27dB so
    breaths and room noise count as silence and get cut through. Floors are
    relative to REF_SPEECH_LEVEL and shifted by LEVEL_OFFSET for the file.
    """
    noise = _shift_noise(noise)
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


def mean_volume(video: Path, t0: float, t1: float) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(video),
         "-vn", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"mean_volume: (-?[0-9.]+) dB", proc.stderr)
    return float(m.group(1)) if m else -99.0


def voiced_runs(video: Path, t0: float, t1: float) -> list:
    """Sustained voice runs in [t0, t1] — breath-proof speech detection.

    Breaths can be as loud as quiet speech, so no energy floor separates
    them; duration does. Detect at voice level (-16dB), chain across brief
    stop-consonant closures, and keep only sustained runs.
    """
    iv = speech_intervals(video, t0, t1, noise="-16dB", min_silence=0.05)
    runs = []
    for s, e in iv:
        if runs and s - runs[-1][1] <= 0.09:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    return [r for r in runs if r[1] - r[0] >= 0.18]


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
    print(f"[4/5] cutting {len(segs)} segment(s) and concatenating...")
    # Segments carry PCM audio: AAC frames are 23ms blocks with a priming
    # delay, so stream-copying AAC segments leaves a partial frame and a
    # timestamp overlap at every join. Players resolve those differently,
    # and lip sync drifts. PCM joins are sample-exact; AAC is encoded once
    # at the concat step.
    seg_files = []
    for i, (t0, t1) in enumerate(segs):
        seg = work / f"seg_{i:03d}.mov"
        run(
            ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(video),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-c:a", "pcm_s16le", str(seg)],
            f"cutting segment {i} ({t0:.2f}-{t1:.2f}s)",
        )
        seg_files.append(seg)
        print(f"      seg {i:03d}: {t0:8.2f}s -> {t1:8.2f}s")

    concat_list = work / "segments.txt"
    # Pin each file's duration to the intended cut length so the concat
    # offsets follow the audio exactly instead of the video's overhanging
    # last frame.
    concat_list.write_text("".join(
        f"file '{s.resolve()}'\nduration {t1 - t0:.6f}\n"
        for s, (t0, t1) in zip(seg_files, segs)))
    # Each segment's video overhangs its audio by up to one frame (the last
    # frame's display time runs past the cut), so the concat offset leaves a
    # few-ms hole in the audio at every join. Without async resampling the
    # AAC encoder closes those holes and the audio runs ahead of the video,
    # drifting further with every cut. aresample=async pads them instead.
    run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c:v", "copy",
         "-af", "aresample=async=1000:min_hard_comp=0.002:first_pts=0",
         "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", str(output)],
        "concatenation",
    )
    for seg in seg_files:
        seg.unlink()


# ---------------------------------------------------------------------- verify

def check_edges(video: Path, segs: list, words: list, api_key: str,
                probe: bool = True) -> list:
    """Inspect every cut edge in the SOURCE audio.

    The transcript diff in verify() hears the output, but it is blind to a
    breath kept ahead of a line, a soft ending trimmed as silence, or speech
    inside a "pause" the ASR mis-timed. These checks look at the audio on
    either side of each edge instead. Returns (source_time, message) pairs.
    """
    issues = []
    quiet = ROOM_DB + 8.0  # TAIL_DB floor: real voice, not room hum
    for k, (t0, t1) in enumerate(segs):
        inside = [w for w in words if w["end"] > t0 + 0.02 and w["start"] < t1]
        if not inside:
            continue
        first, last = inside[0]["word"], inside[-1]["word"]
        # Tail: voice still going right after the cut.
        after = rms_windows(video, t1, t1 + 0.15, 0.05)
        if after and max(after) > quiet:
            issues.append((t1, f"voice continues after the cut ending \"{last}\" "
                               f"({max(after):.0f} dB) — clipped ending?"))
        # Removed gap between close segments: must be silence, not speech.
        if k + 1 < len(segs) and 0 < segs[k + 1][0] - t1 < 1.0:
            gap = rms_windows(video, t1, segs[k + 1][0], 0.05)
            if gap and max(gap) > ROOM_DB + 8.0:
                issues.append((t1, f"speech inside the removed gap after \"{last}\" "
                                   f"({max(gap):.0f} dB)"))
        # Head: breath kept ahead of the first word, or an attack cut into.
        pre = level_windows(video, t0 - 0.06, t0, 0.02)
        if len(pre) >= 2 and pre[-1][0] > ROOM_DB + 12.0 and pre[-1][1] < 0.2 \
                and pre[-1][0] >= pre[0][0] + 3.0:
            issues.append((t0, f"cut opens on rising voice before \"{first}\" — "
                               f"clipped opener?"))
        head = level_windows(video, t0, t0 + 0.4, 0.02)
        vi = next((i for i, (r, z) in enumerate(head)
                   if r >= SPEECH_DB - 2.0 and z < 0.2), len(head))
        breath = [r for r, z in head[:vi]
                  if ROOM_DB + 5.0 < r < SPEECH_DB - 2.0 and z < 0.25]
        allow = 10 if first[:1].lower() in "mn" else 4  # nasal murmur is kept by design
        if len(breath) > allow:
            issues.append((t0, f"{len(breath) * 20}ms of breath-level audio before "
                               f"\"{first}\""))
    if probe:
        # Transcribe the first 2.4s of each cut from the source, with 0.5s of
        # silence prepended (bare short clips come back empty), and check
        # the first word is heard. ASR confusions on function words are
        # common, so a loose prefix match is used.
        tmp = video.parent / ".cutlogic-probe.ogg"
        for t0, t1 in segs:
            inside = [w for w in words if w["end"] > t0 + 0.02 and w["start"] < t1]
            if not inside:
                continue
            exp = "".join(norm_tokens(inside[0]["word"]))
            heard = []
            for _ in range(2):
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", "2.4",
                     "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                     "-af", "adelay=500", "-c:a", "libopus", "-b:a", "48k", str(tmp)],
                    capture_output=True)
                try:
                    heard = ["".join(norm_tokens(w["word"]))
                             for w in resp_words(deepgram_post(tmp.read_bytes(), api_key))]
                except Exception:
                    heard = []
                if heard:
                    break
            ok = bool(heard) and (heard[0] == exp or heard[0][:3] == exp[:3]
                                  or heard[0].startswith(exp)
                                  or (len(heard) > 1 and heard[1] == exp))
            if not ok:
                issues.append((t0, f"opener \"{inside[0]['word']}\" heard as "
                                   f"\"{' '.join(heard[:3])}\" — clipped, or an ASR confusion"))
        if tmp.exists():
            tmp.unlink()
    return sorted(issues)


def verify(output: Path, script_text: str, work: Path, api_key: str,
           video: Path = None, segs: list = None, words_src: list = None,
           probe: bool = True) -> None:
    """QC pass: transcribe the rendered cut and diff it against the script.

    Catches what input-side analysis can't — clipped words at cut boundaries,
    leaked false starts — because it hears exactly what a viewer will hear.
    """
    print("[5/5] verifying: transcribing the render and diffing against the script...")
    audio = work / "verify.ogg"
    run(["ffmpeg", "-y", "-i", str(output), "-vn", "-ac", "1",
         "-c:a", "libopus", "-b:a", "32k", str(audio)],
        "verify audio extraction")
    words = resp_words(deepgram_post(audio.read_bytes(), api_key))
    audio.unlink()
    if not words:
        print("      warning: verify transcription heard no words; skipping QC")
        return

    want = norm_tokens(script_text)
    got, times = [], []
    for w in words:
        for tok in norm_tokens(w["word"]):
            got.append(tok)
            times.append(w["start"])

    sm = difflib.SequenceMatcher(None, want, got, autojunk=False)
    issues = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        wtxt, gtxt = " ".join(want[i1:i2]), " ".join(got[j1:j2])
        if "".join(want[i1:i2]) == "".join(got[j1:j2]):
            continue  # tokenization only ("boot camp" vs "bootcamp") — cosmetic
        at = times[min(j1, len(times) - 1)]
        if tag == "delete":
            issues.append((at, f"missing from video: \"{wtxt}\""))
        elif tag == "insert":
            issues.append((at, f"extra in video: \"{gtxt}\""))
        else:
            cr = difflib.SequenceMatcher(
                None, "".join(want[i1:i2]), "".join(got[j1:j2])).ratio()
            kind = "variant" if cr >= 0.8 else "mismatch"
            issues.append((at, f"{kind}: script \"{wtxt}\" -> heard \"{gtxt}\""))

    fidelity = sm.ratio()
    print(f"      script fidelity: {fidelity:.1%}")
    for at, msg in issues:
        print(f"      [{fmt_t(at)}] {msg}")
    if issues:
        print("      (flagged spots deserve a listen — transcription itself is "
              "imperfect, so not every flag is a real defect)")
    else:
        print("      no differences beyond spelling — cut matches the script")
    edge_issues = []
    if video is not None and segs and words_src:
        print(f"      checking {len(segs)} cut edges in the source audio"
              + (" and probing each opener..." if probe else "..."))
        edge_issues = check_edges(video, segs, words_src, api_key, probe)
        for at, msg in edge_issues:
            print(f"      [source {fmt_t(at)}] {msg}")
        if not edge_issues:
            print("      cut edges clean: no breath kept, no clipped word, no speech in removed gaps")
    (work / f"{output.stem}.verify.json").write_text(json.dumps({
        "fidelity": round(fidelity, 4),
        "issues": [{"at": at, "issue": msg} for at, msg in issues],
        "edge_issues": [{"source_at": at, "issue": msg} for at, msg in edge_issues],
        "transcript": " ".join(w["word"] for w in words),
    }, indent=2))


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
    ap.add_argument("--min-score", type=float, default=0.45,
                    help="floor for low-confidence matches; below this a line is skipped")
    ap.add_argument("--pad-pre", type=float, default=0.05, help="seconds kept before each match")
    ap.add_argument("--pad-post", type=float, default=0.12, help="seconds kept after each match")
    ap.add_argument("--merge-gap", type=float, default=0.15,
                    help="merge segments closer than this many seconds")
    ap.add_argument("--max-pause", type=float, default=0.35,
                    help="cut silences inside a sentence longer than this many seconds")
    ap.add_argument("--work-dir", type=Path, default=Path("work"))
    ap.add_argument("--no-probe", action="store_true",
                    help="verify: skip transcribing each cut's opener (faster, fewer API calls)")
    ap.add_argument("--no-capcut", action="store_true",
                    help="don't hand the cut to CapCut as an editable draft after rendering")
    ap.add_argument("--capcut-name", help="name for the CapCut draft (default: '<output> cutlogic <date time>')")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the QC pass (transcribe the render, diff against script)")
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
    print(f"[3/5] aligning {len(sentences)} script sentence(s) "
          f"against {len(words)} transcript words...")

    matches, warnings = align(sentences, words, args.threshold, args.min_score)
    for w in warnings:
        print(f"      warning: {w}")
    if not matches:
        die("no script sentences matched the transcript — check the script, "
            "or lower --threshold")

    print(f"\n{'score':>6}  {'start':>9}  {'end':>9}  sentence")
    for m in matches:
        flag = "LOW" if m.low else "   "
        print(f"{flag} {m.score:5.2f}  {fmt_t(m.t0):>9}  {fmt_t(m.t1):>9}  {m.sentence[:70]}")

    calibrate_level(args.video, matches)
    segs = build_segments(matches, words, args.video, args.pad_pre, args.pad_post,
                          args.merge_gap, args.max_pause, duration)
    kept = sum(t1 - t0 for t0, t1 in segs)
    lows = [m for m in matches if m.low]
    if lows:
        print(f"\n{len(lows)} low-confidence line(s) — delivery likely deviates "
              f"from the script there; review those timestamps")
    print(f"\n{len(matches)} take(s) matched for {len(sentences)} script sentence(s) "
          f"-> {len(segs)} segment(s), "
          f"keeping {fmt_t(kept)} of {fmt_t(duration)}")

    cuts_file = args.work_dir / f"{args.video.stem}.cuts.json"
    cuts_file.write_text(json.dumps({
        "video": str(args.video),
        "segments": [{"start": t0, "end": t1} for t0, t1 in segs],
        "matches": [{"sentence": m.sentence, "score": round(m.score, 3), "low": m.low,
                     "start": m.t0, "end": m.t1} for m in matches],
        "warnings": warnings,
    }, indent=2))
    print(f"cut list written to {cuts_file}")

    if args.dry_run:
        print("dry run — not rendering.")
        return

    render(args.video, segs, args.work_dir, args.output)
    if not args.no_verify:
        verify(args.output, args.script.read_text(), args.work_dir, api_key,
               video=args.video, segs=segs, words_src=words, probe=not args.no_probe)
    if not args.no_capcut:
        # Hand the same cut list to CapCut as trimmed clips of the source so
        # any edge can be nudged there. Never fails the cut: the MP4 exists.
        try:
            import capcut_handoff
            capcut_handoff.handoff(args.video, [tuple(s) for s in segs], args.output,
                                   args.work_dir, name=args.capcut_name)
        except Exception as e:  # noqa: BLE001
            print(f"      CapCut hand-off skipped: {e}")
    print(f"\ndone: {args.output}")


if __name__ == "__main__":
    main()
