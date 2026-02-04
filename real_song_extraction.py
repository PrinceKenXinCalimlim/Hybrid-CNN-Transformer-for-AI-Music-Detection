import pandas as pd
import subprocess
import os

csv_path = r"C:\Users\Xin\Downloads\ThesisData\real_songs.csv"
output_dir = r"C:\Users\Xin\Downloads\songs"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    youtube_id = row["youtube_id"]
    title = row["title"]
     

    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|')
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_template = os.path.join(output_dir, f"{safe_title}.%(ext)s")

    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "--ignore-errors",
        "--no-overwrites",     #  prevents duplicates
        "--no-playlist",
        "-o", output_template,
        url
    ]

    subprocess.run(command)
