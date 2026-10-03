# CutLogic

Auto-cut a raw video to match a text script, from the command line.

Give it a raw recording (flubs, retakes, pauses and all) plus the script of
what the final video should say. It transcribes the video with Deepgram
(word-level timestamps), aligns each script sentence to the best take in the
transcript, and uses FFmpeg to cut and join only those segments.

## Requirements

- Python 3 (stdlib only — no pip installs)
- FFmpeg (`brew install ffmpeg`)
- A Deepgram API key: sign up at <https://console.deepgram.com> (free credit
  included), create a key, then either:

  ```sh
  export DEEPGRAM_API_KEY=your_key
  ```

  or put `DEEPGRAM_API_KEY=your_key` in a `.env` file in this directory
  (gitignored).

## Usage

```sh
# Preview the cut list without rendering
python3 cutlogic.py raw.mp4 script.txt --dry-run

# Render the final cut
python3 cutlogic.py raw.mp4 script.txt -o final.mp4
```

`script.txt` is plain text — just the sentences the final video should
contain, in order.

### Tuning flags

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | 0.8 | Minimum fuzzy-match score (0–1) for a sentence to be kept |
| `--pad-pre` | 0.05 | Seconds kept before each matched sentence |
| `--pad-post` | 0.12 | Seconds kept after each matched sentence |
| `--merge-gap` | 0.15 | Segments closer than this (seconds) are merged into one |
| `--max-pause` | 0.35 | Silences inside a sentence longer than this are cut out |
| `--no-verify` | off | Skip the QC pass |
| `--work-dir` | `work/` | Where audio, transcript cache, and cut artifacts go |

## How it works

1. **Extract audio** — FFmpeg pulls a small mono Opus track so the upload is tiny.
2. **Transcribe** — one HTTPS POST to Deepgram's `nova-3` model returns every
   word with start/end timestamps. The response is cached in `work/` so
   re-runs don't re-bill.
3. **Align** — the script is split into sentences; each is fuzzy-matched
   (difflib) against sliding windows of transcript words, scanning forward
   monotonically. Among near-equal matches it prefers the **later** take —
   the last read of a flubbed line is usually the keeper. Unmatched sentences
   are warned about and skipped.
4. **Cut & join** — each matched span (plus padding) is re-encoded for
   frame-accurate cuts, then concatenated into the output. Cut boundaries are
   snapped to measured speech energy: heads skip breaths (loud but brief),
   tails keep soft word endings, and pauses hiding un-transcribed retakes are
   detected and cut around. Energy thresholds are calibrated to the
   recording's own speech level first, so a quiet phone recording (or a
   presenter who drops her voice reading numbers) isn't trimmed as silence.
5. **Verify** — the rendered cut is itself transcribed and diffed against the
   script. You get a fidelity score and a timestamped list of anything that
   differs (missing phrases, delivery deviations, suspect boundaries), saved
   to `work/verify.json`. This catches what input-side analysis can't — a
   clipped word at a cut point is audible in the output, not the input.

Inspect `work/cuts.json` to see exactly what was matched and where.
