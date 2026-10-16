"""Turns recorded follow-cam frames into a preview video with captions.

/usr/bin/python3 docs/sgm-world/follow/make_video.py <frames dir> <captions.json> <out stem> [fps] [first frame] [hold seconds]
captions.json is JSON.stringify(__rec.caps) from recorder.js: [[frame, film seconds, caption], ...].
Writes <out stem>.mp4 (H.264, sized for a claude.ai artifact), <out stem>.vtt and <out stem>.jpg (poster)."""
import json, subprocess, sys
from pathlib import Path

frames, caps_file, stem = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fps = int(sys.argv[4]) if len(sys.argv) > 4 else 20
first = int(sys.argv[5]) if len(sys.argv) > 5 else 0      # skip frames recorded before the map had loaded
hold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # then hold the first good frame this long, so the opening caption can be read
FF = "/opt/homebrew/bin/ffmpeg"
caps = json.loads(caps_file.read_text())


def ts(sec):
    h, rem = divmod(sec, 3600); m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


# one cue per caption, from the frame it first shows to the frame the next one starts
cues, cur, start = [], None, 0
caps = [c for c in caps if c[0] >= first]
shift = hold - first / fps
for k, _t, text in caps:
    if text != cur:
        if cur: cues.append((max(0, start / fps + shift) if cues else 0, k / fps + shift, cur))
        cur, start = text, k
if cur: cues.append((start / fps + shift, (caps[-1][0] + 1) / fps + shift, cur))
vtt = "WEBVTT\n\n" + "\n\n".join(f"{ts(a)} --> {ts(b)}\n{t}" for a, b, t in cues if t and not t.startswith("Getting")) + "\n"
stem.with_suffix(".vtt").write_text(vtt)
# a claude.ai artifact takes files up to 15 MB: step the quality down until the film fits under 14 MB
for crf in (26, 29, 32, 35):
    subprocess.run([FF, "-loglevel", "error", "-y", "-framerate", str(fps), "-start_number", str(first), "-i", str(frames / "%05d.jpg"),
                    "-vf", f"tpad=start_duration={hold}:start_mode=clone", "-c:v", "libx264", "-preset", "slow",
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(stem.with_suffix(".mp4"))], check=True)
    if stem.with_suffix(".mp4").stat().st_size < 14e6: break
poster = caps[len(caps) // 5][0]
subprocess.run([FF, "-loglevel", "error", "-y", "-i", str(frames / f"{poster:05d}.jpg"), "-q:v", "4", str(stem.with_suffix(".jpg"))], check=True)
mb = stem.with_suffix(".mp4").stat().st_size / 1e6
print(f"{stem}.mp4 {mb:.1f} MB (crf {crf}), {len(caps)} frames, {len(cues)} captions")
