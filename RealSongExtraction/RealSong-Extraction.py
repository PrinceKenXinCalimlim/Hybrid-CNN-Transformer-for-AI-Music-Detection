import pandas as pd
import subprocess
import os
import sys

# Configuration
csv_path = r"C:\Users\Xin\Downloads\dataset.csv"
output_dir = r"C:\Users\Xin\Downloads\songs"
limit = 4000

os.makedirs(output_dir, exist_ok=True)

# Load the dataset and limit to 3,000 rows
df = pd.read_csv(csv_path)
df_subset = df.head(limit)

for index, row in df_subset.iterrows():
    artist = str(row["artists"]).replace(';', ' ')
    track_name = str(row["track_name"])
    
    # Create a safe filename
    safe_title = "".join(c for c in track_name if c not in r'\/:*?"<>|')
    
    # Search query
    search_query = f"ytsearch1:{artist} {track_name} official audio"
    
    # Use %(ext)s so yt-dlp handles the extension correctly
    output_template = os.path.join(output_dir, f"{safe_title}.%(ext)s")

    # FIX: Use sys.executable -m yt_dlp and shell=True for Windows compatibility
    command = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio[ext=webm]/bestaudio",
        "--ignore-errors",
        "--no-overwrites",
        "--no-playlist",
        "-o", output_template,
        search_query
    ]

    print(f"[{index+1}/{limit}] Processing: {artist} - {track_name}")
    
    # Added shell=True which helps Windows resolve the command path
    subprocess.run(command, shell=True)