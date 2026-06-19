"""
YouTube auto-transcription pipeline.

Flow:
  1. Read the channel's recent uploads via the public RSS feed (no auth, no cookies).
  2. Compare against processed_ids.json to find videos we haven't transcribed yet.
  3. For each new video: download audio only (yt-dlp), transcribe it (faster-whisper),
     save the transcript as a .txt file in transcripts/, then delete the audio.
  4. Update processed_ids.json so next week's run skips these.

This script does NOT use any YouTube API, OAuth, or cookies. It downloads videos the
same way any video downloader would, then transcribes the audio locally. This avoids
the cookie/bot-detection issues that broke the old captions-API approach.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHANNEL_ID = "UCsexetBmFzWSnDdzLT4H1yQ"
RSS_URL = f"https://www.youtube.com/feeds/videos.xml?channel_id={CHANNEL_ID}"

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = REPO_ROOT / "transcripts"
PROCESSED_IDS_PATH = REPO_ROOT / "processed_ids.json"
TMP_AUDIO_DIR = REPO_ROOT / "_tmp_audio"

WHISPER_MODEL_SIZE = "medium"
WHISPER_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_processed_ids() -> set[str]:
    if not PROCESSED_IDS_PATH.exists():
        return set()
    with open(PROCESSED_IDS_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_processed_ids(ids: set[str]) -> None:
    with open(PROCESSED_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, indent=2)


def get_channel_videos() -> list[dict]:
    """
    Fetch the channel's recent uploads from the public RSS feed.
    Returns a list of dicts: {id, title, published}
    Note: YouTube's RSS feed typically only returns the ~15 most recent uploads.
    """
    feed = feedparser.parse(RSS_URL)

    if feed.bozo:
        print(f"Warning: feed parsing issue: {feed.bozo_exception}", file=sys.stderr)

    if not feed.entries:
        print("No entries found in RSS feed. Check the channel ID.", file=sys.stderr)
        return []

    videos = []
    for entry in feed.entries:
        # yt:videoId is the cleanest way to get the video ID from this feed
        video_id = entry.get("yt_videoid")
        if not video_id:
            # fall back to parsing it out of the link
            match = re.search(r"v=([\w-]{11})", entry.get("link", ""))
            video_id = match.group(1) if match else None

        if not video_id:
            continue

        videos.append(
            {
                "id": video_id,
                "title": entry.get("title", video_id),
                "published_parsed": entry.get("published_parsed"),  # time.struct_time or None
            }
        )

    return videos


def sanitize_filename(name: str) -> str:
    """Strip characters that aren't safe in filenames, keep it readable."""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[\s]+", "-", name)
    return name[:80]  # keep filenames from getting absurdly long


def download_audio(video_id: str, dest_dir: Path) -> Path:
    """
    Download audio-only for a video using yt-dlp.
    Returns the path to the downloaded audio file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(dest_dir / f"{video_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "5",  # smaller file, still plenty for speech transcription
        "-o",
        output_template,
        url,
    ]

    print(f"  Downloading audio for {video_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  yt-dlp failed for {video_id}:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"yt-dlp failed for {video_id}")

    audio_path = dest_dir / f"{video_id}.mp3"
    if not audio_path.exists():
        raise RuntimeError(f"Expected audio file not found: {audio_path}")

    return audio_path


def transcribe_audio(audio_path: Path, model: WhisperModel) -> str:
    """Run faster-whisper on the audio file and return the full transcript text."""
    segments, info = model.transcribe(
        str(audio_path),
        language=WHISPER_LANGUAGE,
        beam_size=5,
    )

    lines = [segment.text.strip() for segment in segments]
    return " ".join(lines).strip()


def save_transcript(video_id: str, title: str, published_parsed, transcript: str) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    if published_parsed:
        date_str = datetime(*published_parsed[:6]).strftime("%Y-%m-%d")
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    safe_title = sanitize_filename(title)
    filename = f"{date_str}_{video_id}_{safe_title}.md"
    filepath = TRANSCRIPTS_DIR / filename

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    link_safe_title = title.replace("[", "(").replace("]", ")")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Title - [{link_safe_title}]({video_url})\n")
        f.write(f"Date - {date_str}\n")
        f.write(f"Transcript - {transcript}\n")

    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    processed_ids = load_processed_ids()
    print(f"Already processed: {len(processed_ids)} videos")

    videos = get_channel_videos()
    print(f"Found {len(videos)} videos in RSS feed")

    new_videos = [v for v in videos if v["id"] not in processed_ids]
    print(f"New videos to process: {len(new_videos)}")

    if not new_videos:
        print("Nothing new. Exiting.")
        return

    print(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    for video in new_videos:
        video_id = video["id"]
        title = video["title"]
        print(f"\nProcessing: {title} ({video_id})")

        try:
            audio_path = download_audio(video_id, TMP_AUDIO_DIR)
            print("  Transcribing...")
            transcript = transcribe_audio(audio_path, model)

            if not transcript:
                print(f"  Warning: empty transcript for {video_id}, skipping save.")
                continue

            saved_path = save_transcript(video_id, title, video.get("published_parsed"), transcript)
            print(f"  Saved: {saved_path.relative_to(REPO_ROOT)}")

            processed_ids.add(video_id)

        except Exception as e:
            print(f"  Error processing {video_id}: {e}", file=sys.stderr)
            # Don't add to processed_ids -- we'll retry this one next run.
            continue

        finally:
            # Always clean up the audio file, even if transcription failed partway.
            audio_file = TMP_AUDIO_DIR / f"{video_id}.mp3"
            if audio_file.exists():
                audio_file.unlink()

    save_processed_ids(processed_ids)
    print(f"\nDone. Total processed: {len(processed_ids)}")

    if TMP_AUDIO_DIR.exists() and not any(TMP_AUDIO_DIR.iterdir()):
        TMP_AUDIO_DIR.rmdir()


if __name__ == "__main__":
    main()
