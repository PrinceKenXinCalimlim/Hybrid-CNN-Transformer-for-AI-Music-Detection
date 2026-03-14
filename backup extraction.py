import pandas as pd
import subprocess
import os
import sys
import re
import time  # For gentle rate-limiting delays

# ==================== CONFIGURATION ====================
csv_path = r"D:\Datasets\real_songs.csv"
output_dir = r"D:\RealSong"
cookies_path = r"D:\Datasets\cookies.txt"  # ← Make sure this is your fresh exported cookies.txt

start_row = 35001  # Starting row (inclusive)
end_row = 39999    # Ending row (exclusive)

os.makedirs(output_dir, exist_ok=True)

# ==================== PRE-SCAN EXISTING FILES ====================
print("Scanning output folder for existing songs (duplicate check)...")
existing_files = set()
for f in os.listdir(output_dir):
    name_clean = os.path.splitext(f)[0].lower().strip()
    existing_files.add(name_clean)

print(f"Found {len(existing_files)} existing files — skipping duplicates.")

# ==================== LOAD CSV AND SLICE RANGE ====================
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Clean any extra spaces in headers

if len(df) < start_row:
    print(f"Error: CSV only has {len(df)} rows. Cannot start at {start_row}.")
    sys.exit(1)

df_subset = df.iloc[start_row:end_row]
total_to_process = len(df_subset)

print(f"Ready to process {total_to_process} songs (rows {start_row} to {end_row-1}).")

# ==================== MAIN DOWNLOAD LOOP ====================
for i, (_, row) in enumerate(df_subset.iterrows(), 1):
    artist_raw = str(row["artist"]).replace(';', ' ').strip()
    track_raw = str(row["title"]).strip()

    full_name = f"{artist_raw} - {track_raw}"
    safe_filename = re.sub(r'[\\/*?:"<>|]', '', full_name).strip()

    if safe_filename.lower() in existing_files:
        print(f"[{i}/{total_to_process}] Skipping (already exists): {full_name}")
        continue

    search_query = f"ytsearch1:{artist_raw} {track_raw} official audio"
    output_template = os.path.join(output_dir, f"{safe_filename}.%(ext)s")

    command = [
        sys.executable, "-m", "yt_dlp",
        "--remote-components", "ejs:github",     # ← Fixes EJS / n challenge warnings (recommended)
        "--cookies", cookies_path,               # Authentication for restricted content
        "-f", "251/250/249/bestaudio[ext=webm]/bestaudio/best",  # Prioritize high-quality opus audio
        "--audio-quality", "0",                  # Highest quality
        "--add-metadata",                        # Embed title/artist/year etc.
        "--ignore-errors",                       # Continue on failures
        "--no-overwrites",                       # Skip if file exists
        "--no-playlist",                         # Avoid grabbing playlists
        "-o", output_template,
        search_query
    ]

    print(f"[{i}/{total_to_process}] Downloading: {full_name}")

    # Run with captured output for better error visibility
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"   → Failed (exit code {result.returncode}). Here's the output:")
        print(result.stdout)
        print(result.stderr)
    else:
        print(f"   → Success!")

    # Gentle delay to reduce rate-limit / bot detection risk (especially with cookies + bulk)
    time.sleep(2)  # Adjust to 2-10 seconds as needed

print("\nBatch processing complete! Check the output folder and logs for any remaining issues.")