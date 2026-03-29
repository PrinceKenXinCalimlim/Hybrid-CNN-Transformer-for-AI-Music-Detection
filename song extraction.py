 import pandas as pd
import subprocess
import os
import sys
import re
import time
import random  # For varied cooldowns
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ==================== CONFIGURATION ====================
CSV_PATH       = r"D:\Datasets\real_songs.csv"
OUTPUT_DIR     = r"D:\RealSong"
COOKIES_PATH   = r"D:\Datasets\cookies.txt"          # fresh cookies.txt

START_ROW      = 35001       # inclusive
END_ROW        = 39999         # exclusive   ← adjust as needed

MAX_WORKERS         = 3      # Start low (2–4) with cookies to avoid account rate limits
BATCH_COOLDOWN      = 60     # seconds — increased for safety
COOLDOWN_EVERY      = 25     # trigger cooldown after this many finished

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== SCAN EXISTING FILES ====================
print("Scanning output folder for existing files (to skip duplicates)...")
existing_files = set()
for f in os.listdir(OUTPUT_DIR):
    name_clean = os.path.splitext(f)[0].lower().strip()
    existing_files.add(name_clean)

print(f"Found {len(existing_files)} existing files — will skip duplicates.\n")

# ==================== LOAD & SLICE CSV ====================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

if len(df) < START_ROW:
    print(f"Error: CSV only has {len(df)} rows. Cannot start at {START_ROW}.")
    sys.exit(1)

df_subset = df.iloc[START_ROW:END_ROW]
total_to_process = len(df_subset)

print(f"Preparing to process {total_to_process} songs (rows {START_ROW} to {END_ROW-1}).")
print(f"Concurrency: {MAX_WORKERS} parallel downloads (low to avoid rate limits)\n")

# ==================== DOWNLOAD FUNCTION ====================
def download_song(task):
    idx, row = task
    artist = str(row["artist"]).replace(';', ' ').strip()
    title  = str(row["title"]).strip()

    full_name     = f"{artist} - {title}"
    safe_filename = re.sub(r'[\\/*?:"<>|]', '', full_name).strip()

    if safe_filename.lower() in existing_files:
        print(f"[{idx}/{total_to_process}] Skipping (already exists): {full_name}")
        return idx, True, "skipped"

    search_query = f"ytsearch1:{artist} {title} official audio"
    output_path  = Path(OUTPUT_DIR) / f"{safe_filename}.%(ext)s"

    command = [
        sys.executable, "-m", "yt_dlp",
        "--remote-components", "ejs:github",
        "--cookies", COOKIES_PATH,
        # Pacing to prevent account rate limiting (critical with logged-in cookies)
        "--sleep-requests", "3",           # sleep between HTTP requests
        "--sleep-interval", "5",           # min sleep before each download
        "--max-sleep-interval", "12",      # max → random 5–12s delay
        "-f", "251/250/249/bestaudio[ext=webm]/bestaudio/best",            # reliable fallback (audio first)
        "-N", "4",                         # concurrent fragments — reduced burst
        "--audio-quality", "0",
        "--add-metadata",
        "--ignore-errors",
        "--no-overwrites",
        "--no-playlist",
        "-o", str(output_path.with_suffix('.webm')),
        search_query
    ]

    print(f"[{idx}/{total_to_process}] Starting → {full_name}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900               # 15 min per song
        )

        if result.returncode == 0:
            print(f"[{idx}/{total_to_process}] Success → {full_name}")
            return idx, True, "downloaded"
        else:
            err_msg = result.stderr.strip()[-800:] or result.stdout.strip()[-400:]
            print(f"[{idx}/{total_to_process}] Failed → {full_name}")
            print(f"    → exit code {result.returncode} | {err_msg}")
            return idx, False, err_msg

    except subprocess.TimeoutExpired:
        print(f"[{idx}/{total_to_process}] Timeout (900s) → {full_name}")
        return idx, False, "timeout"
    except Exception as e:
        print(f"[{idx}/{total_to_process}] Exception → {full_name} | {e}")
        return idx, False, str(e)


# ==================== MAIN PARALLEL EXECUTION ====================
rows_with_index = list(enumerate(df_subset.iterrows(), START_ROW + 1))

success_count = 0
failed_count  = 0
completed     = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(download_song, (idx, row)): (idx, row)
        for idx, row_tuple in rows_with_index
        for _, row in [row_tuple]  # unpack iterrows
    }

    for future in as_completed(futures):
        idx, success, msg = future.result()
        completed += 1

        if success:
            success_count += 1
        else:
            failed_count += 1

        # Stronger global pacing — helps a lot with account limits
        if completed % COOLDOWN_EVERY == 0 and completed < total_to_process:
            extra_jitter = random.uniform(0, 30)
            print(f"\n→ Safety cooldown ({BATCH_COOLDOWN + extra_jitter:.0f}s) after {completed} songs to avoid rate limits...")
            time.sleep(BATCH_COOLDOWN + extra_jitter)

print("\n" + "="*60)
print(f"FINISHED BATCH PROCESSING")
print(f"Successful: {success_count}")
print(f"Failed   : {failed_count}")
print(f"Skipped  : {total_to_process - success_count - failed_count}")
print(f"Total    : {total_to_process}")
print("="*60)
print(f"Files saved to: {OUTPUT_DIR}")
print("Tips:")
print("- If rate-limit persists → wait 30–60 min, refresh cookies (yt-dlp --cookies-from-browser chrome --cookies ...)")
print("- Then try increasing MAX_WORKERS to 4–5 slowly.")
print("- For non-restricted songs, try commenting out --cookies line to use guest mode (no account ban risk).")