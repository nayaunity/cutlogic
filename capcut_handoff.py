"""Hand a cutlogic cut list to CapCut desktop as an editable draft.

Each cut segment becomes its own trimmed clip of the original footage on
CapCut's main track, laid end to end, so any cut edge can be nudged in the
editor. Drafts are built through a local VectCutAPI server
(https://github.com/sun-guannan/VectCutAPI, profile ``capcut_legacy``), then
moved into CapCut's drafts folder, patched so CapCut recognises them, and
CapCut is launched.

Standalone:  python3 capcut_handoff.py work/<video>.cuts.json [--name NAME]
From cutlogic: handoff(video, segs, output, work) after render/verify.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
VECTCUT_DIR = Path(os.environ.get("VECTCUT_DIR", HERE.parent / "VectCutAPI")).resolve()
API = os.environ.get("VECTCUT_API", "http://127.0.0.1:9001")
CAPCUT_APP = Path("/Applications/CapCut.app")
DRAFTS_DIR = Path.home() / "Movies/CapCut/User Data/Projects/com.lveditor.draft"
US = 1_000_000  # CapCut stores times in microseconds


# ----------------------------------------------------------------- server

def _post(endpoint: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(API + endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    if not out.get("success"):
        raise RuntimeError(f"{endpoint}: {out.get('error') or out}")
    return out.get("output") or {}


def _server_up() -> bool:
    try:
        urllib.request.urlopen(API + "/get_transition_types", timeout=2)
        return True
    except urllib.error.HTTPError:
        return True  # any HTTP answer means the server is listening
    except Exception:
        return False


def ensure_server(work: Path):
    """Return a Popen handle if we started the server, else None."""
    if _server_up():
        return None
    py = VECTCUT_DIR / "venv-capcut/bin/python"
    server = VECTCUT_DIR / "capcut_server.py"
    if not py.exists() or not server.exists():
        raise RuntimeError(
            f"VectCutAPI not installed at {VECTCUT_DIR}; run scripts/install-vectcut.sh")
    log = open(work / "vectcut.log", "ab")
    proc = subprocess.Popen([str(py), str(server)], cwd=str(VECTCUT_DIR),
                            stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        if _server_up():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"VectCutAPI server exited; see {work / 'vectcut.log'}")
        time.sleep(0.25)
    proc.terminate()
    raise RuntimeError("VectCutAPI server did not come up on " + API)


# ------------------------------------------------------------------ draft

def probe_dims(video: Path) -> tuple:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return int(st["width"]), int(st["height"]), fps


def build_draft(video: Path, segs: list, work: Path) -> Path:
    """Create the draft on the server; returns the saved draft folder."""
    width, height, _ = probe_dims(video)
    draft_id = _post("/create_draft", {"width": width, "height": height})["draft_id"]
    cursor = 0.0
    for t0, t1 in segs:
        _post("/add_video", {
            "draft_id": draft_id, "video_url": str(video.resolve()),
            "start": round(float(t0), 3), "end": round(float(t1), 3),
            "target_start": round(cursor, 3), "track_name": "video_main",
        })
        cursor += float(t1) - float(t0)
    out_base = (work / "capcut").resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    # auto_deploy targets the JianYing container on macOS, not CapCut: do it ourselves.
    _post("/save_draft", {"draft_id": draft_id, "draft_folder": str(out_base),
                          "auto_deploy": False})
    return out_base / draft_id


# ------------------------------------------------------------- install

def _now_us() -> int:
    return int(time.time() * US)


def install_into_capcut(draft_dir: Path, name: str, source: Path = None) -> Path:
    """Move the saved draft into CapCut's drafts folder and fix its metadata.

    VectCutAPI writes draft_meta_info.json straight from its template
    (placeholder user, name and id, no materials), so CapCut would show a
    broken card. Fill every path/name/id/time field the way CapCut's own
    drafts have them.
    """
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = DRAFTS_DIR / name
    if dest.exists():
        name = f"{name} {datetime.now():%H%M%S}"
        dest = DRAFTS_DIR / name
    shutil.move(str(draft_dir), str(dest))

    info_p = dest / "draft_info.json"
    info = json.loads(info_p.read_text())
    draft_uuid = str(uuid.uuid4()).upper()
    info["id"] = draft_uuid
    info["name"] = name
    info["path"] = str(dest)
    # The server copied the source into assets/video (3+ GB for a 4K phone
    # file). CapCut's own imports just reference the original file, so do the
    # same and drop the copy; fall back to the moved copy if the original is
    # gone.
    for m in info.get("materials", {}).get("videos", []):
        p = m.get("path") or ""
        if str(draft_dir) in p:
            moved = Path(p.replace(str(draft_dir), str(dest)))
            if source is not None and source.exists():
                m["path"] = str(source.resolve())
                m["material_name"] = source.name
                if moved.exists():
                    moved.unlink()
            else:
                m["path"] = str(moved)
    info_p.write_text(json.dumps(info, ensure_ascii=False))
    (dest / "draft_info.json.bak").write_text(json.dumps(info, ensure_ascii=False))

    meta_p = dest / "draft_meta_info.json"
    meta = json.loads(meta_p.read_text())
    now = _now_us()
    vids = info.get("materials", {}).get("videos", [])
    size = sum(Path(m["path"]).stat().st_size for m in vids if Path(m.get("path", "")).exists())
    meta.update({
        "draft_fold_path": str(dest),
        "draft_root_path": str(DRAFTS_DIR),
        "draft_json_file": str(info_p),
        "draft_name": name,
        "draft_id": draft_uuid,
        "draft_cover": "draft_cover.jpg",
        "tm_draft_create": now,
        "tm_draft_modified": now,
        "tm_duration": int(info.get("duration", 0)),
        "draft_timeline_materials_size_": size,
    })
    mats = []
    for m in vids:
        p = Path(m["path"])
        mats.append({
            "ai_group_type": "", "create_time": int(time.time()),
            "duration": int(m.get("duration", 0)), "enter_from": 0,
            "extra_info": p.name, "file_Path": str(p),
            "height": int(m.get("height", 0)), "id": str(uuid.uuid4()),
            "import_time": int(time.time()), "import_time_ms": now,
            "item_source": 1, "md5": "", "metetype": "video",
            "roughcut_time_range": {"duration": int(m.get("duration", 0)), "start": 0},
            "sub_time_range": {"duration": -1, "start": -1},
            "type": 0, "width": int(m.get("width", 0)),
        })
    for entry in meta.get("draft_materials", []):
        if entry.get("type") == 0:
            entry["value"] = mats
    meta_p.write_text(json.dumps(meta, ensure_ascii=False))
    return dest


def open_capcut(dest: Path) -> None:
    subprocess.run(["open", "-a", "CapCut"], check=False)


# ----------------------------------------------------------------- entry

def handoff(video: Path, segs: list, output: Path, work: Path, name: str = None) -> Path:
    if not CAPCUT_APP.exists():
        raise RuntimeError("CapCut.app not found in /Applications")
    name = name or f"{output.stem} cutlogic {datetime.now():%Y-%m-%d %H%M}"
    print(f"[6/6] handing off to CapCut: building draft \"{name}\" "
          f"({len(segs)} clips of {video.name})...")
    proc = ensure_server(work)
    try:
        draft_dir = build_draft(video, segs, work)
    finally:
        if proc is not None:
            proc.terminate()
    dest = install_into_capcut(draft_dir, name, source=video)
    # CapCut rescans its drafts folder on launch and while running, so no
    # registry edit is needed: just place the folder and bring CapCut up.
    open_capcut(dest)
    print(f"      draft \"{name}\" is on CapCut's home screen\n      {dest}")
    return dest


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Send a cutlogic cut list to CapCut as an editable draft.")
    ap.add_argument("cuts", type=Path, help="work/<video>.cuts.json written by cutlogic")
    ap.add_argument("--name", help="draft name shown in CapCut")
    ap.add_argument("--no-open", action="store_true", help="build and install the draft but don't launch CapCut")
    args = ap.parse_args()
    cuts = json.loads(args.cuts.read_text())
    video = Path(cuts["video"])
    segs = [(s["start"], s["end"]) for s in cuts["segments"]]
    work = args.cuts.resolve().parent
    if args.no_open:
        global open_capcut
        open_capcut = lambda dest: None  # noqa: E731
    handoff(video, segs, Path(video.stem), work, name=args.name)


if __name__ == "__main__":
    main()
