"""
Pull recent videos from Indian-finance YouTube channels (RSS, no API key needed) and
fetch their auto-captions via youtube-transcript-api when available.

Channel RSS works via: https://www.youtube.com/feeds/videos.xml?channel_id=<UCxxxxx>

Output:
  data/derived/youtube_videos.parquet  (one row per video; metadata + transcript text + tagged symbols)
"""
from __future__ import annotations
import re, time, hashlib
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import requests

OUT = Path("data/derived/youtube_videos.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Channel id list — confirmed Indian-equities/finance
CHANNELS = [
    ("CA Rachana Phadke",        "UCdghB6djO2DHrsMpf7vpvNg"),
    ("Pranjal Kamra",            "UCwAdQUuPT6laN-AQR17fe1g"),
    ("ET Markets",               "UCD3IjE9NfcgrmLEntxxC4kw"),
    ("CNBC TV18",                "UCsjJlOkdyZUaG3WseTcsRDA"),
    ("Soic - Investing",         "UCyDjyqEvfL4afL_9qX0gwVw"),
    ("Akshat Shrivastava",       "UCwVEhEzsjLym_u1he4XWFkg"),
    ("Asset Yogi",               "UC8Sj0Q9HMqeF9eJEfPbDIVA"),  # asset class focus
    ("LabourLawAdvisor",         "UCN67XGdDJTnyWoldcjFENBA"),
    ("Stock Market Today (CNBC AWAAZ)","UCmbdnpkmyJjEjwYiPqlGUFA"),
    ("Moneycontrol",             "UCSHmoyXX7cwnZF5hCNJjt6g"),
    ("Zerodha Varsity",          "UCcyq283he07B7_KUX07mmtA"),
]


UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def load_symbols() -> set[str]:
    sm = pd.read_parquet("tmp/from_scratch_7d_run/alt2/sector_index_members.parquet")
    return set(sm["symbol"].astype(str).str.upper().unique())


def tag_symbols(text: str, syms: set[str], min_len: int = 4) -> list[str]:
    if not text:
        return []
    upper = text.upper()
    return [s for s in syms if len(s) >= min_len and re.search(rf"\b{re.escape(s)}\b", upper)]


def fetch_channel_rss(channel_id: str) -> list[dict]:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as e:
        print(f"  ch={channel_id} ERR {str(e)[:120]}")
        return []
    ns = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
    rows = []
    for entry in root.findall("a:entry", ns):
        vid = entry.findtext("yt:videoId", namespaces=ns)
        title = entry.findtext("a:title", namespaces=ns)
        published = entry.findtext("a:published", namespaces=ns)
        author = entry.findtext("a:author/a:name", namespaces=ns)
        rows.append({
            "video_id": vid,
            "title": title or "",
            "published": published,
            "author": author,
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return rows


def fetch_transcript(video_id: str) -> str | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi", "en-IN"])
        except Exception:
            return None
        return " ".join(seg.get("text", "") for seg in data)[:25000]
    except ImportError:
        return None


def main() -> None:
    syms = load_symbols()
    rows = []
    for name, cid in CHANNELS:
        items = fetch_channel_rss(cid)
        for it in items:
            it["channel_name"] = name
        print(f"  {name}: {len(items)} videos")
        rows.extend(items)
        time.sleep(0.5)

    if not rows:
        print("no videos")
        return

    df = pd.DataFrame(rows)
    df["published_ts"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()

    # only fetch transcripts for newest 30 (cheap-ish, takes ~10-30s)
    df = df.sort_values("published_ts", ascending=False)
    df = df.drop_duplicates("video_id")
    new_videos = df.head(50)
    transcripts = {}
    for _, r in new_videos.iterrows():
        t = fetch_transcript(r["video_id"])
        if t:
            transcripts[r["video_id"]] = t
        time.sleep(0.4)
    df["transcript"] = df["video_id"].map(transcripts).fillna("")

    text = (df["title"].fillna("") + " " + df["transcript"].fillna(""))
    df["symbols"] = text.apply(lambda t: tag_symbols(t, syms))
    df["n_symbols"] = df["symbols"].apply(len)
    df["has_transcript"] = df["transcript"].str.len() > 100

    if OUT.exists():
        old = pd.read_parquet(OUT)
        # keep newer transcript if we have it now
        df = pd.concat([old, df], ignore_index=True)
        # dedupe keeping the row with longer transcript
        df = df.sort_values("transcript", key=lambda s: s.str.len(), ascending=False).drop_duplicates("video_id", keep="first")
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(df):,} rows, with-transcript: {df['has_transcript'].sum()})")

    recent = df[df["published_ts"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=72)]
    print(f"\ntop YouTube-mentioned symbols (last 72h, {len(recent)} videos):")
    sym_counts = recent.explode("symbols").groupby("symbols").size().sort_values(ascending=False)
    print(sym_counts.head(15).to_string())


if __name__ == "__main__":
    main()
