"""Sticker renderers for CapCut drafts: iMessage threads, a search bar, phone
notifications, a paper card. Each returns a transparent RGBA PIL image that
is exactly CANVAS_W px wide, so CapCut shows it 1:1 at scale 1.0 and
transform_y alone places it (the strip's centre line). Drawn 2x and
downsampled for clean edges.

Kinds (plan JSON "kind" + fields):
  imessage      bubbles: [{"from": "them"|"me", "text": "..."}], align: "left"|"right"|"center"
  search        text, align
  notification  app, title, body, logo (png path, optional), align
  card          title, lines: [...], tag (optional), align
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

CANVAS_W = 1080
SS = 2  # supersample

FONTS = Path.home() / "Documents/coding2026/videoAnimations/fonts"
POPPINS_BOLD = str(FONTS / "Poppins-Bold.ttf")
POPPINS_SEMI = str(FONTS / "Poppins-SemiBold.ttf")
SF = "/System/Library/Fonts/SFCompact.ttf"          # closest to iOS Messages
HELV = "/System/Library/Fonts/Helvetica.ttc"

OXBLOOD = (0x4D, 0x1B, 0x27, 255)
PAPER = (0xF3, 0xF3, 0xF1, 255)
BUTTER = (0xFF, 0xE9, 0xA3, 255)
INK = (0x2A, 0x28, 0x28, 255)
GREY = (0x80, 0x80, 0x80, 255)
IOS_BLUE = (0x0B, 0x93, 0xF6, 255)
IOS_GREY = (0xE9, 0xE9, 0xEB, 255)


def font(path, size, weight="Regular"):
    try:
        f = ImageFont.truetype(path, int(size * SS))
    except OSError:
        return ImageFont.truetype(HELV, int(size * SS))
    try:  # SF Compact is a variable font whose default instance is Black
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def _shadow(img: Image.Image, radius=18, offset=(0, 10), alpha=70) -> Image.Image:
    """Soft drop shadow under every opaque pixel."""
    a = img.getchannel("A")
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh.putalpha(a.point(lambda v: v * alpha // 255))
    sh = sh.filter(ImageFilter.GaussianBlur(radius * SS))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.alpha_composite(sh, (offset[0] * SS, offset[1] * SS))
    out.alpha_composite(img)
    return out


def _wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def _finish(img: Image.Image) -> Image.Image:
    img = _shadow(img)
    return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)


def _place(content: Image.Image, align: str, margin=48) -> Image.Image:
    """Put a content image onto a CANVAS_W-wide strip (already at SS scale)."""
    W = CANVAS_W * SS
    strip = Image.new("RGBA", (W, content.height + 40 * SS), (0, 0, 0, 0))
    m = margin * SS
    x = {"left": m, "right": W - m - content.width}.get(align, (W - content.width) // 2)
    strip.alpha_composite(content, (x, 20 * SS))
    return strip


# ------------------------------------------------------------ iMessage

def imessage(bubbles, align="center", text_size=34, max_w=680, only=None, **_):
    """One strip with the whole thread laid out; with only=i, every bubble but
    the i-th is left transparent so the per-bubble layers stack in place."""
    fnt = font(SF, text_size)
    pad_x, pad_y, gap, r = 22 * SS, 14 * SS, 8 * SS, 22 * SS
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    laid = []
    for b in bubbles:
        lines = _wrap(tmp, b["text"], fnt, (max_w - 2 * pad_x / SS) * SS)
        lh = int(text_size * SS * 1.25)
        w = max(tmp.textlength(l, font=fnt) for l in lines) + 2 * pad_x
        h = lh * len(lines) + 2 * pad_y
        laid.append((b, lines, int(w), int(h), lh))
    width = int(max(w for _, _, w, _, _ in laid) + 8 * SS)
    width = max(width, max_w * SS // 2)
    height = sum(h for _, _, _, h, _ in laid) + gap * (len(laid) - 1) + 8 * SS
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = 4 * SS
    for i, (b, lines, w, h, lh) in enumerate(laid):
        if only is not None and i != only:
            y += h + gap; continue
        mine = b.get("from") == "me"
        x = width - w - 4 * SS if mine else 4 * SS
        fill = IOS_BLUE if mine else IOS_GREY
        d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill)
        # tail
        if mine:
            d.polygon([(x + w - 10 * SS, y + h - 16 * SS), (x + w + 6 * SS, y + h), (x + w - 18 * SS, y + h)], fill=fill)
        else:
            d.polygon([(x + 10 * SS, y + h - 16 * SS), (x - 6 * SS, y + h), (x + 18 * SS, y + h)], fill=fill)
        ty = y + pad_y
        for l in lines:
            d.text((x + pad_x, ty), l, font=fnt, fill=(255, 255, 255, 255) if mine else INK)
            ty += lh
        y += h + gap
    return _finish(_place(img, align))


# ------------------------------------------------------------ search bar

def search(text, align="center", width=880, **_):
    fnt = font(POPPINS_SEMI, 32)
    W, H = width * SS, 104 * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 2, fill=(255, 255, 255, 255), outline=(0xDD, 0xDD, 0xDA, 255), width=2 * SS)
    # magnifier
    cx, cy, rr = 52 * SS, H // 2, 16 * SS
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=GREY, width=5 * SS)
    d.line([cx + rr * 0.7, cy + rr * 0.7, cx + rr * 1.6, cy + rr * 1.6], fill=GREY, width=5 * SS)
    d.text((92 * SS, cy - 32 * SS * 0.72), text, font=fnt, fill=INK)
    tw = d.textlength(text, font=fnt)
    d.rectangle([92 * SS + tw + 6 * SS, cy - 20 * SS, 92 * SS + tw + 9 * SS, cy + 20 * SS], fill=IOS_BLUE)  # caret
    return _finish(_place(img, align))


# ------------------------------------------------------------ notification

def notification(app, title, body, logo=None, align="center", width=900, **_):
    f_app, f_title, f_body = font(POPPINS_SEMI, 20), font(POPPINS_BOLD, 30), font(POPPINS_SEMI, 27)
    W = width * SS
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    body_lines = _wrap(tmp, body, f_body, W - 150 * SS)[:2]
    H = (34 + 36 + 34 * len(body_lines) + 28) * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=34 * SS, fill=(255, 255, 255, 242))
    # app icon
    ix, iy, isz = 22 * SS, (H - 76 * SS) // 2, 76 * SS
    if logo and Path(logo).exists():
        lg = Image.open(logo).convert("RGBA")
        lg.thumbnail((isz - 8 * SS, isz - 8 * SS))
        tile = Image.new("RGBA", (isz, isz), (0, 0, 0, 0))
        ImageDraw.Draw(tile).rounded_rectangle([0, 0, isz - 1, isz - 1], radius=18 * SS, fill=PAPER)
        tile.alpha_composite(lg, ((isz - lg.width) // 2, (isz - lg.height) // 2))
        img.alpha_composite(tile, (ix, iy))
    else:
        d.rounded_rectangle([ix, iy, ix + isz, iy + isz], radius=18 * SS, fill=OXBLOOD)
        d.text((ix + isz / 2, iy + isz / 2), app[:1].upper(), font=font(POPPINS_BOLD, 36), fill=PAPER, anchor="mm")
    tx = 122 * SS
    d.text((tx, 22 * SS), app.upper(), font=f_app, fill=GREY)
    d.text((W - 24 * SS, 22 * SS), "now", font=f_app, fill=GREY, anchor="ra")
    d.text((tx, 50 * SS), title, font=f_title, fill=INK)
    y = 88 * SS
    for l in body_lines:
        d.text((tx, y), l, font=f_body, fill=(0x44, 0x44, 0x44, 255)); y += 34 * SS
    return _finish(_place(img, align))


# ------------------------------------------------------------ paper card

def card(title, lines=(), tag=None, align="center", width=720, **_):
    f_title, f_line, f_tag = font(POPPINS_BOLD, 34), font(POPPINS_SEMI, 24), font(POPPINS_BOLD, 18)
    W = width * SS
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tl = _wrap(tmp, title, f_title, W - 80 * SS)[:2]
    H = (40 + 44 * len(tl) + 16 + 36 * len(lines) + 36) * SS
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=28 * SS, fill=PAPER)
    d.rectangle([0, 0, 14 * SS, H - 1], fill=OXBLOOD)
    d.rounded_rectangle([0, 0, 28 * SS, H - 1], radius=28 * SS, fill=OXBLOOD)
    d.rectangle([14 * SS, 0, 40 * SS, H - 1], fill=PAPER)
    y = 36 * SS
    for l in tl:
        d.text((56 * SS, y), l, font=f_title, fill=OXBLOOD); y += 44 * SS
    y += 12 * SS
    for l in lines:
        d.ellipse([58 * SS, y + 12 * SS, 66 * SS, y + 20 * SS], fill=OXBLOOD)
        d.text((80 * SS, y), l, font=f_line, fill=INK); y += 36 * SS
    if tag:
        tw = d.textlength(tag, font=f_tag) + 28 * SS
        d.rounded_rectangle([W - tw - 28 * SS, 28 * SS, W - 28 * SS, 62 * SS], radius=17 * SS, fill=BUTTER)
        d.text((W - tw / 2 - 28 * SS, 45 * SS), tag, font=f_tag, fill=OXBLOOD, anchor="mm")
    return _finish(_place(img, align))


RENDERERS = {"imessage": imessage, "search": search, "notification": notification, "card": card}


def imessage_layers(spec: dict, out_stem: Path) -> list:
    """Render an iMessage thread as one PNG per bubble (same canvas, same
    transform_y) so each can rise in on its own beat. Returns
    [(png_path, "me"|"them"), ...] in thread order."""
    args = {k: v for k, v in spec.items() if k not in ("kind", "at", "dur", "y", "x", "delay", "anim", "stagger")}
    out = []
    for i, b in enumerate(spec["bubbles"]):
        p = out_stem.parent / f"{out_stem.name}_{i}.png"
        imessage(only=i, **args).save(p)
        out.append((p, b.get("from", "them")))
    return out


def render(spec: dict, out: Path) -> Path:
    img = RENDERERS[spec["kind"]](**{k: v for k, v in spec.items() if k not in ("kind", "at", "dur", "y", "x", "delay", "anim")})
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


if __name__ == "__main__":
    import json, sys
    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("work/stickers")
    for i, spec in enumerate(plan):
        p = render(spec, out / f"{i:02d}_{spec['kind']}.png")
        print(p, Image.open(p).size)
