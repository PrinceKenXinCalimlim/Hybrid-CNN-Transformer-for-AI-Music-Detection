import os
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress pydub's common ffmpeg warning if missing (optional)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

folders_to_convert = [
    r"D:\RealSong"
]

MIN_WAV_SIZE_BYTES = 50_000  # ~50 KB — consider file corrupt/empty below this

def convert_to_wav():
    for folder in folders_to_convert:
        print(f"\n>>> Processing folder: {folder}")
        path = Path(folder)

        if not path.exists() or not path.is_dir():
            print("❌ Folder does NOT exist or is not a directory")
            continue 

        files = []
        # Search recursively for supported audio formats (non-WAV)
        for ext in ['*.mp3', '*.webm', '*.mp4', '*.m4a', '*.ogg', '*.flac', '*.aac']:
            found = list(path.rglob(ext))
            files.extend(found)

        if not files:
            print("❌ No supported audio files found in folder (or subfolders)")
            print("   Checked extensions: mp3, webm, mp4, m4a, ogg, flac, aac")
            continue

        print(f"✅ Found {len(files)} potential files to process")

        skipped = 0
        converted = 0
        deleted = 0
        failed = 0

        for file_path in tqdm(files, desc="Processing"):
            try:
                output_path = file_path.with_suffix('.wav')

                # Skip if target WAV already exists
                if output_path.exists():
                    skipped += 1
                    continue

                print(f"  → Loading: {file_path.name}")
                audio = AudioSegment.from_file(str(file_path))

                print(f"  → Exporting to: {output_path.name}")
                audio.export(str(output_path), format="wav")

                # Safety check: make sure the output file was actually created and isn't tiny/empty
                if output_path.exists() and output_path.stat().st_size >= MIN_WAV_SIZE_BYTES:
                    converted += 1
                    print(f"  ✓ Converted: {file_path.name} → {output_path.name}")

                    # Delete the original file after successful conversion
                    try:
                        file_path.unlink()
                        deleted += 1
                        print(f"  🗑 Deleted original: {file_path.name}")
                    except Exception as del_err:
                        print(f"  ⚠ Could not delete original {file_path.name}: {del_err}")

                else:
                    failed += 1
                    print(f"  ✗ Output file missing or too small: {output_path.name}")

            except Exception as e:
                failed += 1
                print(f"✗ Failed {file_path.name}: {type(e).__name__} → {e}")

        print(f"\nSummary for {folder}:")
        print(f"  Converted & kept: {converted}")
        print(f"  Originals deleted: {deleted}")
        print(f"  Skipped (WAV already existed): {skipped}")
        print(f"  Failed: {failed}")
        print(f"  Total processed: {len(files)}")

if __name__ == "__main__":
    convert_to_wav()