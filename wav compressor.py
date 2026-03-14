import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
INPUT_DIR = r"D:\RealSong"
OUTPUT_DIR = r"C:\Users\vince\Downloads\3000plus Songs"

# Audio Specs for your Thesis Model
SAMPLE_RATE = "16000"
CHANNELS = "1" 
BIT_DEPTH = "pcm_s16le" 

def compress_and_delete(file_info):
    input_file, output_file = file_info
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Force .wav extension
    output_file = output_file.with_suffix('.wav')
    
    # ffmpeg command: -vn removes video if present (important for .webm)
    cmd = [
        'ffmpeg', '-y', '-i', str(input_file), 
        '-vn',                   # No video stream
        '-ar', SAMPLE_RATE, 
        '-ac', CHANNELS, 
        '-acodec', BIT_DEPTH, 
        str(output_file), 
        '-loglevel', 'error'     # Hide verbose output
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Only delete if output exists and has reasonable size
        if output_file.exists() and output_file.stat().st_size > 10000:  # >10KB
            os.remove(input_file)
            print(f"✓ Converted & deleted: {input_file.name} → {output_file.name}")
        else:
            print(f"✗ Output too small/missing for {input_file.name}")
            
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg error on {input_file.name}: {e.stderr.decode() if e.stderr else e}")
    except Exception as e:
        print(f"Error processing {input_file.name}: {type(e).__name__} → {e}")

def main():
    extensions = (".wav", ".webm")  # You can add more if needed: ".mp3", ".m4a", etc.
    all_files = []
    input_path = Path(INPUT_DIR)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    for ext in extensions:
        all_files.extend(list(input_path.rglob(f"*{ext}")))

    if not all_files:
        print(f"No .{extensions} files found in {INPUT_DIR} or subfolders!")
        return

    print(f"Found {len(all_files)} files (.wav / .webm)")
    print(f"Output folder: {OUTPUT_DIR}")
    print("!!! WARNING !!! Original files will be DELETED after successful conversion.")
    confirm = input("Type 'DELETE' (case-sensitive) to proceed: ").strip()

    if confirm == "DELETE":
        tasks = [(f, Path(OUTPUT_DIR) / f.relative_to(INPUT_DIR)) for f in all_files]
        print(f"\nStarting parallel conversion (8 workers)...\n")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(
                executor.map(compress_and_delete, tasks),
                total=len(tasks),
                desc="Processing",
                unit="file"
            ))
        print("\nDone!")
    else:
        print("Cancelled — no files were touched.")

if __name__ == "__main__":
    main()