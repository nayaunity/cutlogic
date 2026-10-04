"""Word timestamps for a cutlogic render, on the OUTPUT timeline.

cutlogic knows exactly which source words made the cut and where each
segment landed, so the rendered video's transcript can be derived instead
of re-transcribed: every kept source word is shifted by its segment's
offset. Names keep the script's spelling (the ASR's "chatgeektewerk" becomes
"ChatGPT Work") by aligning each segment's words to the script line that
produced it.

    python3 output_words.py work/<video>.cuts.json <script.txt> \
        --words-out transcript.json [--joins-out joins.json] [--srt-out captions.srt]

transcript.json is the flat [{text,start,end}] array HyperFrames tools and
the videoAnimations editor read; joins.json lists the output times of every
cut join so overlays can avoid straddling one.
"""
import argparse
import difflib
import json
import re
from pathlib import Path

import cutlogic as c


def load_words(video: Path, work: Path) -> list:
    tp = work / f"{video.stem}.transcript.json"
    data = json.loads(tp.read_text())
    return data if isinstance(data, list) else c.resp_words(data)


def script_spelling(seg_words: list, sentences: list) -> list:
    """Return display text per word, taking spelling from the best-matching
    script sentence when the normalised tokens line up; else the ASR word."""
    out = [w.get("punctuated_word") or w["word"] for w in seg_words]
    toks = ["".join(c.norm_tokens(w["word"])) for w in seg_words]
    best, best_r = None, 0.0
    for s in sentences:
        stoks = s.split()
        snorm = ["".join(c.norm_tokens(t)) for t in stoks]
        r = difflib.SequenceMatcher(None, toks, snorm, autojunk=False).ratio()
        if r > best_r:
            best, best_r = (stoks, snorm), r
    if best is None or best_r < 0.6:
        return out
    stoks, snorm = best
    sm = difflib.SequenceMatcher(None, toks, snorm, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                out[i1 + k] = stoks[j1 + k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cuts", type=Path, help="work/<video>.cuts.json")
    ap.add_argument("script", type=Path, help="the script the cut was made from")
    ap.add_argument("--words-out", type=Path, required=True)
    ap.add_argument("--joins-out", type=Path)
    ap.add_argument("--srt-out", type=Path)
    args = ap.parse_args()

    cuts = json.loads(args.cuts.read_text())
    video, work = Path(cuts["video"]), args.cuts.resolve().parent
    words = load_words(video, work)
    sentences = [re.sub(r"\s+", " ", s) for s in c.split_sentences(args.script.read_text())]

    out, joins, cursor = [], [], 0.0
    for seg in cuts["segments"]:
        t0, t1 = seg["start"], seg["end"]
        inside = [w for w in words if w["start"] < t1 - 0.02 and w["end"] > t0 + 0.02]
        texts = script_spelling(inside, sentences)
        for w, text in zip(inside, texts):
            ws = max(w["start"], t0) - t0 + cursor
            we = min(w["end"], t1) - t0 + cursor
            if we - ws < 0.02:
                continue
            out.append({"text": text, "start": round(ws, 3), "end": round(we, 3)})
        cursor += t1 - t0
        joins.append(round(cursor, 3))
    joins = joins[:-1]  # the last value is the end of the video, not a join

    args.words_out.parent.mkdir(parents=True, exist_ok=True)
    args.words_out.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    if args.joins_out:
        args.joins_out.write_text(json.dumps(joins))
    if args.srt_out:
        def ts(t):
            h, r = divmod(t, 3600); m, s = divmod(r, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"
        lines = []
        for i, w in enumerate(out, 1):
            lines += [str(i), f"{ts(w['start'])} --> {ts(w['end'])}", w["text"], ""]
        args.srt_out.write_text("\n".join(lines))
    print(f"{len(out)} words over {cursor:.2f}s, {len(joins)} joins -> {args.words_out}")


if __name__ == "__main__":
    main()
