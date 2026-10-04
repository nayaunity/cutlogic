"""Cut a voiceover-over-b-roll vlog reel from a project folder.

    project/
      script.txt      the voiceover script (what the final VO says)
      vo/             voiceover takes: any number of video or audio files
      broll/          clips to cut to

    python3 vlogcut.py <project> --name "oxford vlog" [--no-capcut] [--threshold 0.65]

Steps
  1. voiceover: transcribe every take (cached), align the script across all
     of them (later takes win, as in cutlogic), cut each line tight with the
     cutlogic boundary rules, and concatenate the audio into work/vo.wav with
     word timestamps on the output timeline.
  2. catalog: for every b-roll clip, probe it, sample frames, measure motion,
     and have Claude describe what is on screen (cached in work/catalog.json).
  3. match: split the VO into 1-3 s phrases and ask Claude for one clip per
     phrase; enforce no repeat within 20 s, at most two phrases on one clip,
     the most active window of the clip, and a face clip on the last phrase.
  4. draft: b-roll on CapCut's main track, VO on its own audio track, brand
     captions with keyword accents, then a preview MP4 and a contact sheet
     in work/ so the assembly is checked before CapCut opens.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cutlogic as c  # noqa: E402
import capcut_handoff as ch  # noqa: E402
import capcut_reel as reel  # noqa: E402
import output_words as ow  # noqa: E402

CLAUDE_MODEL = os.environ.get("VLOGCUT_MODEL", "claude-sonnet-5")
CANVAS = (1080, 1920)
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mts", ".avi", ".mkv", ".webm"}
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".aiff", ".flac"}
PHRASE_MAX, PHRASE_MIN, PHRASE_PAUSE = 2.6, 1.0, 0.35
REUSE_GAP, MAX_RUN = 20.0, 2


# ------------------------------------------------------------ helpers

def log(msg):
    print(msg, flush=True)


def claude_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    for env in (HERE / ".env", HERE.parent / "videoAnimations/.env"):
        if env.exists():
            m = re.search(r'ANTHROPIC_API_KEY\s*=\s*"?([^"\n]+)', env.read_text())
            if m:
                return m.group(1).strip()
    c.die("ANTHROPIC_API_KEY not set (export it or put it in .env)")


def claude(messages, system=None, max_tokens=2000, model=None) -> str:
    import urllib.request
    body = {"model": model or CLAUDE_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": claude_key(), "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    text = "".join(b.get("text", "") for b in out["content"])
    if not text.strip():
        raise RuntimeError(f"Claude returned no text (stop_reason={out.get('stop_reason')}, "
                           f"blocks={[b['type'] for b in out['content']]})")
    return text


def claude_json(messages, system=None, max_tokens=2000, model=None):
    """Parse the JSON object in a reply; retries once asking for JSON only."""
    text = claude(messages, system, max_tokens, model)
    for attempt in (0, 1):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
        m = re.search(r"\{.*\}|\[.*\]", body, re.S)
        try:
            return json.loads(m.group(0) if m else body)
        except json.JSONDecodeError:
            if attempt:
                raise RuntimeError("Claude did not return JSON:\n" + text[:800])
            text = claude(messages + [{"role": "assistant", "content": text},
                                      {"role": "user", "content": "Reply again with the JSON object only, no prose, no code fence."}],
                          system, max_tokens, model)


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:stream_side_data=rotation:stream_tags=rotate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    st = (j.get("streams") or [{}])[0]
    w, h = int(st.get("width", 0) or 0), int(st.get("height", 0) or 0)
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        rot = int(float(sd.get("rotation", 0) or 0))
    rot = rot or int(float((st.get("tags") or {}).get("rotate", 0) or 0))
    if rot % 180:
        w, h = h, w
    return {"w": w, "h": h, "dur": float(j.get("format", {}).get("duration", 0) or 0), "has_video": bool(st)}


# ------------------------------------------------------------ 1. voiceover

def voiceover(project: Path, work: Path, api_key: str, threshold: float, pad_pre: float, pad_post: float,
              merge_gap: float, max_pause: float) -> dict:
    takes = sorted(p for p in (project / "vo").iterdir() if p.suffix.lower() in VIDEO_EXT | AUDIO_EXT)
    if not takes:
        c.die("no voiceover files in vo/")
    script = (project / "script.txt").read_text()
    sentences = c.split_sentences(script)
    per, cat, offset = [], [], 0.0
    for i, take in enumerate(takes):
        audio = c.extract_audio(take, work)
        words = c.transcribe(take, audio, work, api_key)
        dur = c.probe_duration(take)
        per.append({"path": take, "words": words, "dur": dur, "offset": offset, "first": len(cat)})
        for w in words:
            ww = dict(w); ww["start"] += offset; ww["end"] += offset; ww["take"] = i
            cat.append(ww)
        offset += dur + 2.0
    log(f"[vo] {len(takes)} take(s), {len(cat)} words; aligning {len(sentences)} script lines")
    matches, warnings = c.align(sentences, cat, threshold, 0.45)
    for w in warnings:
        log(f"      warning: {w}")
    if not matches:
        c.die("no script line matched any take")
    # group by take, keep script order across takes
    order, groups = [], {}
    for m in matches:
        ti = cat[m.wstart]["take"]
        groups.setdefault(ti, []).append(m)
        if ti not in order:
            order.append(ti)
    pieces = []  # (take path, t0, t1, take index)
    for ti in order:
        t = per[ti]
        local = [replace(m, wstart=m.wstart - t["first"], wend=m.wend - t["first"],
                         t0=m.t0 - t["offset"], t1=m.t1 - t["offset"]) for m in groups[ti]]
        c.calibrate_level(t["path"], local)
        segs = c.build_segments(local, t["words"], t["path"], pad_pre, pad_post, merge_gap, max_pause, t["dur"])
        pieces += [(t["path"], s, e, ti) for s, e in segs]
    # concatenate PCM pieces into one VO bed
    parts = []
    for k, (path, s, e, ti) in enumerate(pieces):
        part = work / f"vo_part{k:03d}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(path),
                        "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(part)], check=True)
        parts.append(part)
    lst = work / "vo_parts.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    vo = work / "vo.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(vo)], check=True)
    for p in parts:
        p.unlink()
    # words on the output timeline, script spelling
    out_words, cursor, joins = [], 0.0, []
    for path, s, e, ti in pieces:
        inside = [w for w in per[ti]["words"] if w["start"] < e - 0.02 and w["end"] > s + 0.02]
        texts = ow.script_spelling(inside, [re.sub(r"\s+", " ", x) for x in sentences])
        for w, text in zip(inside, texts):
            ws, we = max(w["start"], s) - s + cursor, min(w["end"], e) - s + cursor
            if we - ws >= 0.02:
                out_words.append({"text": text, "start": round(ws, 3), "end": round(we, 3)})
        cursor += e - s
        joins.append(round(cursor, 3))
    D = round(cursor, 3)
    face_take = next((t["path"] for t in per if probe(t["path"])["has_video"]), None)
    res = {"vo": vo, "D": D, "words": out_words, "joins": joins[:-1], "pieces": pieces, "face_take": face_take}
    (work / "vo.json").write_text(json.dumps({"D": D, "words": out_words, "joins": joins[:-1],
                                              "pieces": [(str(p), s, e) for p, s, e, _ in pieces]}, indent=1))
    log(f"[vo] {D:.1f}s of voiceover from {len(pieces)} piece(s) -> {vo}")
    return res


# ------------------------------------------------------------ 2. catalog

def motion_profile(path: Path, step: float = 0.5) -> list:
    """Mean frame-to-frame change per `step` seconds (0-1), tiny grey frames."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps=1/{step},scale=48:27,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    n = 48 * 27
    frames = [out[i:i + n] for i in range(0, len(out) - n + 1, n)]
    prof = [0.0]
    for a, b in zip(frames, frames[1:]):
        prof.append(sum(abs(x - y) for x, y in zip(a, b)) / (n * 255.0))
    return [round(p, 4) for p in prof]


def sample_frames(path: Path, dur: float, work: Path, n: int) -> list:
    outs = []
    for k in range(n):
        t = dur * (k + 0.5) / n
        f = work / "frames" / f"{path.stem}_{k}.jpg"
        f.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                        "-vf", "scale='if(gt(iw,ih),512,-2)':'if(gt(iw,ih),-2,512)'", "-q:v", "5", str(f)], check=False)
        if f.exists():
            outs.append(f)
    return outs


CATALOG_SYSTEM = ("You catalog b-roll clips for a creator's vertical vlog reels. Frames are shown in time order. "
                  "Answer with one JSON object only.")


def describe_clip(frames: list) -> dict:
    content = []
    for f in frames:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.b64encode(f.read_bytes()).decode()}})
    content.append({"type": "text", "text": (
        "Describe this clip for matching to voiceover phrases. JSON keys: "
        "description (one vivid sentence, what happens), subject, setting, action, "
        "naya_on_camera (true if the creator herself is the subject), face_visible (true if a face is clearly visible), "
        "tags (6-10 lowercase keywords: objects, places, moods, activities), mood, quality (1-5 for sharpness and light), "
        f"best_frame (0-{len(frames) - 1}: the frame index that best shows the clip's subject, flattering and clear).")})
    return claude_json([{"role": "user", "content": content}], CATALOG_SYSTEM, 1500)


def catalog(project: Path, work: Path) -> list:
    clips = sorted(p for p in (project / "broll").iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        c.die("no clips in broll/")
    cache_p = work / "catalog.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    out = []
    for p in clips:
        key = f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
        if key in cache and "best_t" in cache[key]:
            out.append(cache[key]); continue
        info = probe(p)
        if info["dur"] < 0.8:
            continue
        log(f"[catalog] {p.name} ({info['dur']:.1f}s, {info['w']}x{info['h']})")
        n_frames = min(8, max(3, int(info["dur"] // 2) + 1))
        frames = sample_frames(p, info["dur"], work, n_frames)
        desc = describe_clip(frames)
        try:
            bf = int(desc.get("best_frame", n_frames // 2))
        except (TypeError, ValueError):
            bf = n_frames // 2
        desc["best_t"] = round(info["dur"] * (max(0, min(n_frames - 1, bf)) + 0.5) / n_frames, 2)
        entry = {"id": p.name, "path": str(p.resolve()), **info, "motion": motion_profile(p), **desc}
        cache[key] = entry; out.append(entry)
        cache_p.write_text(json.dumps(cache, indent=1))
    log(f"[catalog] {len(out)} clip(s)")
    return out


# ------------------------------------------------------------ 3. match

def phrases(words: list, D: float) -> list:
    out, cur = [], []
    def flush():
        if cur:
            out.append({"i": len(out), "text": " ".join(w["text"] for w in cur), "start": cur[0]["start"], "end": cur[-1]["end"]})
            cur.clear()
    soft = None  # index in cur of the last comma or small pause, a better break than a hard cut
    for k, w in enumerate(words):
        cur.append(w)
        nxt = words[k + 1] if k + 1 < len(words) else None
        span = w["end"] - cur[0]["start"]
        ends_sentence = w["text"].rstrip()[-1:] in ".?!"
        pause = (nxt["start"] - w["end"]) if nxt else 9
        if w["text"].rstrip()[-1:] in ",;:" or pause > 0.15:
            if w["end"] - cur[0]["start"] >= PHRASE_MIN:
                soft = len(cur)
        if nxt is None or ends_sentence or pause > PHRASE_PAUSE:
            flush(); soft = None
        elif span >= PHRASE_MAX:
            if soft and soft < len(cur):
                rest = cur[soft:]; del cur[soft:]
                flush(); cur.extend(rest)
            else:
                flush()
            soft = None
    # merge too-short phrases into the previous one; phrases abut (no gaps on the picture)
    merged = []
    for p in out:
        if merged and p["end"] - p["start"] < PHRASE_MIN:
            merged[-1]["end"] = p["end"]; merged[-1]["text"] += " " + p["text"]
        else:
            merged.append(p)
    for i, p in enumerate(merged):
        p["i"] = i
        p["start"] = 0.0 if i == 0 else merged[i - 1]["end"]
    merged[-1]["end"] = D
    return merged


MATCH_SYSTEM = (
    "You are the editor of a creator's vertical vlog reel: a voiceover over fast b-roll, one clip per phrase, "
    "a new shot every 1-3 s. Pick the clip whose content best illustrates or rhymes with each phrase: literal "
    "matches first (she says 'studying' -> a clip of her at a desk), then mood or setting matches. Rules: never "
    "use the same clip for two phrases less than 20 s apart unless there is no alternative; use every clip at "
    "least once before repeating any when phrases outnumber clips; prefer higher quality clips; the LAST phrase "
    "should be a clip where her face is visible. Answer with JSON only: "
    '{"assignments": [{"phrase": <i>, "clip": "<id>", "why": "<6 words>"}]} covering every phrase.')


def match(phr: list, cat: list) -> list:
    clips = [{"id": e["id"], "duration": round(e["dur"], 1), "description": e["description"], "tags": e.get("tags", []),
              "naya_on_camera": e.get("naya_on_camera"), "face_visible": e.get("face_visible"), "quality": e.get("quality")}
             for e in cat]
    user = ("PHRASES (in order, with seconds):\n" + "\n".join(f'{p["i"]}: [{p["start"]:.1f}-{p["end"]:.1f}] {p["text"]}' for p in phr)
            + "\n\nCLIPS:\n" + json.dumps(clips, ensure_ascii=False))
    res = claude_json([{"role": "user", "content": user}], MATCH_SYSTEM, 16000)
    by_id = {e["id"]: e for e in cat}
    want = {a["phrase"]: a for a in res.get("assignments", []) if a.get("clip") in by_id}
    # enforce the rules the model may have bent
    last_use, uses, run = {}, {e["id"]: 0 for e in cat}, 0
    out, prev = [], None
    for p in phr:
        cid = want.get(p["i"], {}).get("clip")
        why = want.get(p["i"], {}).get("why", "")
        bad = cid is None or (cid != prev and p["start"] - last_use.get(cid, -99) < REUSE_GAP) or (cid == prev and run >= MAX_RUN)
        if bad:
            fresh = sorted(cat, key=lambda e: (uses[e["id"]], last_use.get(e["id"], -99)))
            alt = next((e["id"] for e in fresh if e["id"] != prev and p["start"] - last_use.get(e["id"], -99) >= REUSE_GAP), fresh[0]["id"])
            why = (why + " | swapped: repeat rule").strip(" |"); cid = alt
        run = run + 1 if cid == prev else 1
        uses[cid] += 1; last_use[cid] = p["start"]; prev = cid
        out.append({**p, "clip": cid, "why": why})
    # last phrase: a face clip if one exists and the pick has none
    faces = [e["id"] for e in cat if e.get("face_visible")]
    if faces and out and not by_id[out[-1]["clip"]].get("face_visible"):
        out[-1]["clip"] = max(faces, key=lambda i: by_id[i].get("quality", 0)); out[-1]["why"] += " | face for the close"
    return out


def choose_windows(assign: list, cat: list) -> list:
    """Source window per phrase: the most active stretch of the clip not yet
    used; a phrase continuing the previous phrase's clip keeps rolling."""
    by_id = {e["id"]: e for e in cat}
    used = {e["id"]: [] for e in cat}
    prev_clip, prev_end = None, 0.0
    for a in assign:
        e = by_id[a["clip"]]; L = a["end"] - a["start"]
        if a["clip"] == prev_clip and prev_end + L <= e["dur"] - 0.05:
            s = prev_end
        else:
            prof, step = e["motion"], 0.5
            # phone clips start with a settle and end with the hand reaching for
            # the stop button: keep 0.5 s off the head and 1.0 s off the tail
            best, best_v = 0.5, -1.0
            top = max(0.5, e["dur"] - L - 1.0)
            s_cands = [x * 0.25 for x in range(int(top / 0.25) + 1)]
            bt = e.get("best_t")
            for s0 in s_cands:
                if s0 > top: break
                if any(s0 < u1 and s0 + L > u0 for u0, u1 in used[a["clip"]]):
                    continue
                i0, i1 = int(s0 / step), max(int(s0 / step) + 1, int((s0 + L) / step))
                v = sum(prof[i0:i1]) / max(1, i1 - i0) if prof else 0
                if bt is not None:  # the representative moment outweighs raw motion
                    v += 1.0 if s0 <= bt <= s0 + L else (0.5 if abs(bt - (s0 + L / 2)) < 3.0 else 0.0)
                if v > best_v:
                    best, best_v = s0, v
            s = best if best_v >= 0 else 0.5
            if s + L > e["dur"] - 0.05:
                s = max(0.0, e["dur"] - L - 0.05)
        a["src_start"], a["src_end"] = round(s, 3), round(min(s + L, e["dur"]), 3)
        used[a["clip"]].append((s, s + L)); prev_clip, prev_end = a["clip"], s + L
    return assign


# ------------------------------------------------------------ 4. draft + preview

def cover_scale(e: dict) -> tuple:
    """CapCut fits a clip inside the canvas at scale 1.0; scale up to fill for
    vertical-ish clips, keep landscape clips whole over a blurred fill."""
    if not e["w"] or not e["h"]:
        return 1.0, None
    fit = min(CANVAS[0] / e["w"], CANVAS[1] / e["h"])
    cover = max(CANVAS[0] / e["w"], CANVAS[1] / e["h"])
    if e["w"] / e["h"] > 0.8:
        return 1.0, 3
    return round(cover / fit, 4), None


def build_draft(assign: list, cat: list, vo: dict, work: Path, name: str, open_app: bool) -> Path:
    by_id = {e["id"]: e for e in cat}
    proc = ch.ensure_server(work)
    try:
        draft = ch._post("/create_draft", {"width": CANVAS[0], "height": CANVAS[1]})["draft_id"]
        def post(ep, body):
            if ep == "/add_text" and "transform_y" in body:
                body["transform_y"] = reel.safe_text_y(body["transform_y"], body.get("font_size", 15), body.get("text", ""))
            return ch._post(ep, {"draft_id": draft, **body})
        n = {"clips": 0, "captions": 0, "keywords": 0}
        for a in assign:
            e = by_id[a["clip"]]; sc, blur = cover_scale(e)
            body = {"video_url": e["path"], "start": a["src_start"], "end": a["src_end"], "target_start": round(a["start"], 3),
                    "track_name": "video_main", "volume": 0.0, "scale_x": sc, "scale_y": sc}
            if blur:
                body["background_blur"] = blur
            post("/add_video", body); n["clips"] += 1
        post("/add_audio", {"audio_url": str(vo["vo"].resolve()), "target_start": 0, "volume": 1.0, "track_name": "voiceover"})
        # captions: the reel chunker, clamped to phrase ends so text changes with the shot
        D, words = vo["D"], vo["words"]
        ends = [a["end"] for a in assign]
        def clamp_end(st, en):
            nxt = min((j for j in ends if j > st + 0.05), default=D)
            return min(en, nxt - 0.02)
        for sent in reel.sentences(words):
            kw = reel.pick_keyword(sent); cur = []
            def flush(nxt_start):
                if not cur: return
                st = cur[0]["start"]; en = clamp_end(st, min(nxt_start, cur[-1]["end"] + 0.5))
                if en - st >= 0.08:
                    post("/add_text", {"text": " ".join(w["text"] for w in cur), "start": st, "end": en, "font": reel.FONT,
                                       "font_color": reel.WHITE, "font_size": 11, "letter_spacing": reel.TRACK,
                                       "transform_y": reel.Y_CAPTION, "track_name": "text_captions"}); n["captions"] += 1
                cur.clear()
            for i, w in enumerate(sent):
                nxt = sent[i + 1]["start"] if i + 1 < len(sent) else w["end"] + 0.5
                if kw is not None and w is kw:
                    flush(w["start"])
                    en = clamp_end(w["start"], min(nxt, w["end"] + 0.5))
                    if en - w["start"] >= 0.08:
                        post("/add_text", {"text": w["text"], "start": w["start"], "end": en, "font": reel.FONT, "font_color": reel.BUTTER,
                                           "font_size": 14.3, "letter_spacing": reel.TRACK, "transform_y": reel.Y_CAPTION,
                                           "track_name": "text_keyword", "intro_animation": "Pop_Up", "intro_duration": 0.25}); n["keywords"] += 1
                    continue
                cur.append(w)
                if len(cur) == 3 or nxt - w["end"] > 0.3 or (i + 1 < len(sent) and sent[i + 1] is kw):
                    flush(nxt)
            flush(sent[-1]["end"] + 0.5)
        out_base = (work / "capcut").resolve(); out_base.mkdir(parents=True, exist_ok=True)
        ch._post("/save_draft", {"draft_id": draft, "draft_folder": str(out_base), "auto_deploy": False})
        draft_dir = out_base / draft
    finally:
        if proc is not None:
            proc.terminate()
    dest = ch.install_into_capcut(draft_dir, name, source=None)
    log(f"[draft] {json.dumps(n)} -> {dest}")
    if open_app:
        ch.open_capcut(dest)
    return dest


def preview(assign: list, cat: list, vo: dict, work: Path) -> Path:
    """Low-res MP4 of the assembly (b-roll + VO, no captions) and a 1 fps
    contact sheet, so the cut is looked at before CapCut opens."""
    by_id = {e["id"]: e for e in cat}
    inputs, fc, k = [], [], 0
    for a in assign:
        e = by_id[a["clip"]]
        inputs += ["-ss", f"{a['src_start']:.3f}", "-t", f"{a['src_end'] - a['src_start']:.3f}", "-i", e["path"]]
        if e["w"] / max(1, e["h"]) > 0.8:
            fc.append(f"[{k}:v]scale=540:-2,pad=540:960:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=24[v{k}]")
        else:
            fc.append(f"[{k}:v]scale=540:960:force_original_aspect_ratio=increase,crop=540:960,setsar=1,fps=24[v{k}]")
        k += 1
    inputs += ["-i", str(vo["vo"])]
    fc.append("".join(f"[v{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0[v]")
    out = work / "preview.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", f"{k}:a",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-c:a", "aac", "-shortest", str(out)], check=True)
    sheet = work / "preview-sheet.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-vf", "fps=1,scale=180:320,tile=10x6", "-frames:v", "1", str(sheet)], check=False)
    log(f"[preview] {out}\n          {sheet}")
    return out


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("project", type=Path)
    ap.add_argument("--name", help="CapCut draft name (default: '<project> vlog')")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--pad-pre", type=float, default=0.02)
    ap.add_argument("--pad-post", type=float, default=0.04)
    ap.add_argument("--merge-gap", type=float, default=0.1)
    ap.add_argument("--max-pause", type=float, default=0.2)
    ap.add_argument("--no-capcut", action="store_true")
    ap.add_argument("--stop-after", choices=["vo", "catalog", "match"], help="run part of the pipeline")
    a = ap.parse_args()
    project = a.project.resolve()
    work = project / "work"; work.mkdir(exist_ok=True)
    api_key = c.load_api_key()

    vo = voiceover(project, work, api_key, a.threshold, a.pad_pre, a.pad_post, a.merge_gap, a.max_pause)
    if a.stop_after == "vo": return
    cat = catalog(project, work)
    if a.stop_after == "catalog": return
    phr = phrases(vo["words"], vo["D"])
    log(f"[match] {len(phr)} phrases over {vo['D']:.1f}s, {len(cat)} clips")
    assign = choose_windows(match(phr, cat), cat)
    (work / "assembly.json").write_text(json.dumps(assign, indent=1))
    for x in assign:
        log(f"  {x['start']:6.2f}-{x['end']:6.2f}  {x['clip']:<22} @{x['src_start']:5.1f}  {x['text'][:48]:<48}  {x['why']}")
    preview(assign, cat, vo, work)
    if a.stop_after == "match": return
    build_draft(assign, cat, vo, work, a.name or f"{project.name} vlog", open_app=not a.no_capcut)


if __name__ == "__main__":
    main()
