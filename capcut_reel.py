"""Build a styled, fully editable CapCut draft of a cutlogic render.

Everything lands as native CapCut layers through VectCutAPI, so each caption,
keyword, callout, logo, badge, zoom keyframe and sound cue can be moved,
restyled or deleted in CapCut. The edit follows Naya's 2026-10-03 reel style
guide: Poppins Bold at -1 tracking, white captions of 1-3 words at lower
chest, one keyword per sentence in butter yellow and 30% bigger with a pop,
a caps hook title in the top third from 0.0s, number callouts for every
spoken figure, literal logo pop-ups that slide in for ~2.5s, brand-oxblood
badge pills for chapter marks and the held CTA, slow + punch zooms, and
light whoosh/pop cues on their own tracks. No text layer crosses a cut join.

    python3 capcut_reel.py work/<video>.cuts.json <output words.json> \
        --logos <dir> --sfx <dir> --name "<draft name>" [--cta GAMMA]

Keywords sit on the track `text_keyword` so they can all be switched to
Parslay in CapCut with one multi-select (the API only knows CapCut's
built-in fonts).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capcut_handoff as ch  # noqa: E402

FONT = "Poppins_Bold"
WHITE = "#FFFFFF"
BUTTER = "#FFE9A3"
OXBLOOD = "#4D1B27"
TRACK = -0.05  # -> -1 in CapCut's letter-spacing units
CANVAS = (1080, 1920)

# vertical positions in half-canvas-height units (negative = down)
Y_HOOK_1, Y_HOOK_2 = 0.66, 0.50
Y_CAPTION = -0.30     # lower chest
Y_CALLOUT = -0.52     # over the desk edge
Y_BADGE, X_BADGE = 0.74, 0.42   # top-right, above the shoulder
Y_POPUP, X_POPUP = -0.50, 0.34

STOP = set("""a an the and or but so to of in on at for with from by as is are was were be been being it its this
that these those i me my we our you your he she they them their his her him then than there here when where which
who what how why if not no yes do did does done have has had having get got go going went just also very really
really more most much many some any all every each own same other such into out up down off over under again
still ever never now about through because while until after before like one two three four five six seven eight
nine ten first last next week month year day days months years thing things way ways lot lots kind sort etcetera
make made making am im ive id ill youre youve thats whats lets dont didnt wasnt isnt its gonna wanna""".split())

LOGOS = {  # spoken word -> asset file; the first two mentions get a pop-up, at least 20s apart
    "gamma": "gamma.png", "linkedin": "linkedin.png", "linkedin's": "linkedin.png",
    "chatgpt": "chatgpt.png", "manychat": "manychat.png", "adweek": "adweek.png", "oxford": "oxford.png",
}


def norm(t):
    return re.sub(r"[^a-z0-9$']", "", t.lower())


def is_figure(t):
    return bool(re.search(r"\d", t))


def sentences(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.?!]$", w["text"]):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def pick_keyword(sent):
    cands = [w for w in sent if not is_figure(w["text"]) and norm(w["text"]).strip("'") not in STOP
             and len(norm(w["text"])) >= 4]
    if not cands:
        return None
    # the longest content word, ties to the later one (the payoff tends to come last)
    return max(cands, key=lambda w: (len(norm(w["text"])), w["start"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cuts", type=Path)
    ap.add_argument("words", type=Path, help="output-timeline words from output_words.py")
    ap.add_argument("--logos", type=Path, required=True)
    ap.add_argument("--sfx", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cta", default="GAMMA")
    ap.add_argument("--work", type=Path, default=Path("work"))
    a = ap.parse_args()

    cuts = json.loads(a.cuts.read_text())
    video = Path(cuts["video"])
    segs = [(round(s["start"], 3), round(s["end"], 3)) for s in cuts["segments"]]
    words = json.loads(a.words.read_text())
    D = round(sum(e - s for s, e in segs), 3)
    joins = []
    t = 0.0
    for s, e in segs[:-1]:
        t += e - s; joins.append(round(t, 3))

    def clamp_end(start, end):
        """A text layer never crosses a cut join (no clip bleed)."""
        for j in joins:
            if start < j < end:
                return round(j - 0.02, 3)
        return end

    def off_joins(start, end, m=0.3):
        s, e = start, end
        for j in joins:
            if abs(s - j) < m: s = round(j + m, 3)
        for j in joins:
            if abs(e - j) < m: e = round(j - m, 3)
        return s, max(e, s + 0.8)

    proc = ch.ensure_server(a.work)
    try:
        draft = ch._post("/create_draft", {"width": CANVAS[0], "height": CANVAS[1]})["draft_id"]
        post = lambda ep, body: ch._post(ep, {"draft_id": draft, **body})  # noqa: E731
        n = {"clips": 0, "captions": 0, "keywords": 0, "callouts": 0, "popups": 0, "badges": 0, "keyframes": 0, "sfx": 0}

        # 1. clips, in order
        cursor = 0.0
        for s, e in segs:
            s, e = round(float(s), 3), round(float(e), 3)  # microsecond rounding of raw floats made clips overlap by 1 us
            post("/add_video", {"video_url": str(video.resolve()), "start": s, "end": e, "target_start": round(cursor, 3), "track_name": "video_main"})
            cursor = round(cursor + (e - s), 3); n["clips"] += 1

        # 2. per sentence: keyword, figures, captions
        sents = sentences(words)
        keywords, figures, punch_times, pop_times = [], [], [], []
        chunks = []  # (text, start, end, style)
        for sent in sents:
            kw = pick_keyword(sent)
            cur = []
            def flush(nxt_start):
                if not cur: return
                st = cur[0]["start"]; en = clamp_end(st, min(nxt_start, cur[-1]["end"] + 0.5))
                if en - st >= 0.08:
                    chunks.append((" ".join(w["text"] for w in cur), st, en, "white"))
                cur.clear()
            for i, w in enumerate(sent):
                nxt = sent[i + 1]["start"] if i + 1 < len(sent) else w["end"] + 0.5
                if is_figure(w["text"]):
                    flush(w["start"]); figures.append(w); continue  # the callout carries the figure
                if kw is not None and w is kw:
                    flush(w["start"])
                    en = clamp_end(w["start"], min(nxt, w["end"] + 0.5))
                    chunks.append((w["text"], w["start"], en, "keyword")); keywords.append(w); continue
                cur.append(w)
                gap = nxt - w["end"]
                if len(cur) == 3 or gap > 0.3 or (i + 1 < len(sent) and (sent[i + 1] is kw or is_figure(sent[i + 1]["text"]))):
                    flush(nxt)
            flush(sent[-1]["end"] + 0.5)

        # no two layers on one track may touch: clamp each chunk to the next one's start on its track
        for style in ("white", "keyword"):
            same = sorted((i for i, c in enumerate(chunks) if c[3] == style), key=lambda i: chunks[i][1])
            for k, i in enumerate(same[:-1]):
                nxt = chunks[same[k + 1]][1]
                if chunks[i][2] > nxt - 0.01:
                    chunks[i] = (chunks[i][0], chunks[i][1], round(nxt - 0.01, 3), style)
        chunks = [(c[0], c[1], min(c[2], round(D - 0.02, 3)), c[3]) for c in chunks]  # nothing past the last frame
        chunks = [c for c in chunks if c[2] - c[1] >= 0.06]
        for text, st, en, style in chunks:
            if style == "keyword":
                post("/add_text", {"text": text, "start": st, "end": en, "font": FONT, "font_color": BUTTER, "font_size": 14.3,
                                   "letter_spacing": TRACK, "transform_y": Y_CAPTION, "track_name": "text_keyword",
                                   "intro_animation": "Pop_Up", "intro_duration": 0.25})
                n["keywords"] += 1; punch_times.append(st); pop_times.append(st)
            else:
                post("/add_text", {"text": text, "start": st, "end": en, "font": FONT, "font_color": WHITE, "font_size": 11,
                                   "letter_spacing": TRACK, "transform_y": Y_CAPTION, "track_name": "text_captions"})
                n["captions"] += 1

        # 3. figure callouts: every spoken figure, the month total is the hit
        figures.sort(key=lambda w: w["start"])
        for fi, w in enumerate(figures):
            txt = w["text"].rstrip(".,!?")
            big = txt.replace(",", "") in ("$33330",)
            st = w["start"]; en = clamp_end(st, min(st + (3.2 if big else 2.2), D))
            if fi + 1 < len(figures): en = min(en, round(figures[fi + 1]["start"] - 0.05, 3))
            if en - st < 0.3: continue
            post("/add_text", {"text": txt, "start": st, "end": en, "font": FONT, "font_color": BUTTER, "font_size": 28 if big else 20,
                               "letter_spacing": TRACK, "transform_y": Y_CALLOUT, "track_name": "text_callouts",
                               "intro_animation": "Bounce_In" if big else "Zoom_In", "intro_duration": 0.35, "outro_animation": "Fade_Out", "outro_duration": 0.3})
            n["callouts"] += 1; punch_times.append(st)

        # 4. logo pop-ups on named tools (literal, ~2.5s, alternate sides, 20s between repeats)
        last_logo, side = {}, "left"
        for w in words:
            key = norm(w["text"]).strip("'")
            key = key[:-2] if key.endswith("'s") else key
            f = LOGOS.get(key)
            if not f or not (a.logos / f).exists(): continue
            if w["start"] - last_logo.get(f, -99) < 20: continue
            if any(abs(w["start"] - t0) < 6 for t0 in last_logo.values()): continue
            st, en = off_joins(w["start"], w["start"] + 2.5)
            post("/add_image", {"image_url": str((a.logos / f).resolve()), "start": st, "end": en, "transform_y": Y_POPUP,
                                "transform_x": -X_POPUP if side == "left" else X_POPUP, "scale_x": 0.9, "scale_y": 0.9,
                                "track_name": "image_popups", "intro_animation": "Slide_Right" if side == "left" else "Slide_Left",
                                "intro_animation_duration": 0.4, "outro_animation": "Fade_Out", "outro_animation_duration": 0.3})
            last_logo[f] = w["start"]; side = "right" if side == "left" else "left"; n["popups"] += 1; pop_times.append(st)

        # 5. badges: chapter marks, stamps, CTA held to the end
        def badge(text, st, en, hold=False):
            st, en = (st, en) if hold else off_joins(st, en)
            post("/add_text", {"text": text, "start": st, "end": en, "font": FONT, "font_color": WHITE, "font_size": 9,
                               "letter_spacing": 0.02, "transform_x": X_BADGE, "transform_y": Y_BADGE, "track_name": "text_badges",
                               "background_color": OXBLOOD, "background_alpha": 1.0, "background_round_radius": 1.0,
                               "background_width": 1.0, "background_height": 1.0,
                               "intro_animation": "Bounce_In", "intro_duration": 0.3, **({} if hold else {"outro_animation": "Fade_Out", "outro_duration": 0.25})})
            n["badges"] += 1; pop_times.append(st)
        text_all = [(norm(w["text"]), w) for w in words]
        for i, (tok, w) in enumerate(text_all):
            if tok == "week" and i + 1 < len(text_all):
                num = {"one": 1, "two": 2, "three": 3, "four": 4}.get(text_all[i + 1][0])
                if num and (i == 0 or text_all[i - 1][0] not in ("my", "the", "this", "that")) and re.search(r"[.?!]$", words[i - 1]["text"] if i else "."):
                    badge(f"WEEK {num}", w["start"] + 0.6, w["start"] + 3.2)
            if tok == "disclaimer":
                badge("DISCLAIMER", w["start"] + 0.6, w["start"] + 3.0)
            if tok == "biggest" and i + 1 < len(text_all) and text_all[i + 1][0] in ("win", "l"):
                badge("BIGGEST WIN" if text_all[i + 1][0] == "win" else "BIGGEST L", w["start"] + 0.6, w["start"] + 3.2)
        cta_i = next((i for i, (tok, w) in enumerate(text_all) if tok == "comment"), None)
        if cta_i is not None:
            # start at the beginning of the sentence before the CTA line so it holds >= 3s
            sent_start = cta_i
            seen = 0
            while sent_start > 0:
                sent_start -= 1
                if re.search(r"[.?!]$", words[sent_start]["text"]):
                    seen += 1
                    if seen == 2: sent_start += 1; break
            badge(f"COMMENT {a.cta.upper()}", words[sent_start]["start"], D, hold=True)

        # 6. hook title, top third, from frame 0
        post("/add_text", {"text": "HOW MUCH I MADE IN", "start": 0, "end": 4.8, "font": FONT, "font_color": WHITE, "font_size": 15,
                           "letter_spacing": TRACK, "transform_y": Y_HOOK_1, "track_name": "text_title", "outro_animation": "Fade_Out", "outro_duration": 0.4})
        post("/add_text", {"text": "SEPTEMBER", "start": 0, "end": 4.8, "font": FONT, "font_color": BUTTER, "font_size": 19,
                           "letter_spacing": TRACK, "transform_y": Y_HOOK_2, "track_name": "text_title_accent",
                           "intro_animation": "Bounce_In", "intro_duration": 0.35, "outro_animation": "Fade_Out", "outro_duration": 0.4})

        # 7. zooms: slow drift on the hook clip and the month-total clip; quick punches on keywords and figures
        kf = lambda times, vals: post("/add_video_keyframe", {"track_name": "video_main", "property_types": ["uniform_scale"] * len(times),  # noqa: E731
                                                              "times": times, "values": [str(v) for v in vals]})
        kf([0.0, round(segs[0][1] - segs[0][0], 3)], [1.0, 1.06]); n["keyframes"] += 2
        punch_times = [t0 for t0 in sorted(set(round(t, 2) for t in punch_times))]
        kept = []
        for t0 in punch_times:
            if (not kept or t0 - kept[-1] >= 0.7) and 0.3 <= t0 <= D - 0.8: kept.append(t0)
        punch_times = kept
        for t0 in punch_times:
            kf([round(t0 - 0.03, 3), round(t0 + 0.1, 3), round(t0 + 0.6, 3)], [1.0, 1.12, 1.0]); n["keyframes"] += 3

        # 8. sound cues on their own tracks, low, so they can be muted or deleted
        whoosh, pop = a.sfx / "whoosh.wav", a.sfx / "pop.wav"
        import subprocess
        dur = lambda f: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout or 0)  # noqa: E731
        # a cue must end before the last frame: a sound past the picture leaves a black tail in CapCut
        def cue(f, track, t0, vol):
            d = min(dur(f), round(D - t0 - 0.02, 3))
            if d < 0.05: return
            post("/add_audio", {"audio_url": str(f.resolve()), "target_start": t0, "duration": d, "end": d, "volume": vol, "track_name": track}); n["sfx"] += 1
        def spaced(times, gap):
            out = []
            for t0 in sorted(set(round(t, 2) for t in times)):
                if not out or t0 - out[-1] >= gap: out.append(t0)
            return out
        for t0 in spaced(punch_times, 0.5):
            if whoosh.exists() and 0.3 <= t0 <= D - 0.8: cue(whoosh, "sfx_whoosh", t0, 0.25)
        for t0 in spaced(pop_times, 0.12):
            if pop.exists() and t0 <= D - 0.5: cue(pop, "sfx_pop", t0, 0.2)

        out_base = (a.work / "capcut").resolve(); out_base.mkdir(parents=True, exist_ok=True)
        ch._post("/save_draft", {"draft_id": draft, "draft_folder": str(out_base), "auto_deploy": False})
        draft_dir = out_base / draft
    finally:
        if proc is not None:
            proc.terminate()

    dest = ch.install_into_capcut(draft_dir, a.name, source=video)
    ch.open_capcut(dest)
    print(json.dumps(n), f"\ndraft \"{a.name}\" -> {dest}")


if __name__ == "__main__":
    main()
