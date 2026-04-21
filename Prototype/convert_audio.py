import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
INPUT_DIR = r"C:\Users\Xin\Downloads\songs"
OUTPUT_DIR = r"C:\Users\Xin\Downloads\songs_Compressed"

# Audio Specs for your Thesis Model
SAMPLE_RATE = "16000"
CHANNELS = "1" 
BIT_DEPTH = "pcm_s16le" 

def compress_and_delete(file_info):
    input_file, output_file = file_info
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # We force the output extension to be .wav regardless of input
    output_file = output_file.with_suffix('.wav')
    
    # ffmpeg command: -vn removes video if the webm has a video stream
    cmd = [
        'ffmpeg', '-y', '-i', str(input_file), 
        '-vn',                   # Disable video (crucial for .webm)
        '-ar', SAMPLE_RATE, 
        '-ac', CHANNELS, 
        '-acodec', BIT_DEPTH, 
        str(output_file), 
        '-loglevel', 'error'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # Verify the new file exists and isn't empty before deleting original
        if output_file.exists() and output_file.stat().st_size > 1000:
            os.remove(input_file)
    except Exception as e:
        print(f"Error processing {input_file.name}: {e}")

def main():
    # Look for both .wav and .webm files
    extensions = ("*.wav", "*.webm")
    all_files = []
    for ext in extensions:
        all_files.extend(list(Path(INPUT_DIR).rglob(ext)))

    if not all_files:
        print("No files found!")
        return

    print(f"!!! WARNING !!!")
    print(f"Found {len(all_files)} files. Original .webm/.wav will be DELETED.")
    confirm = input("Type 'DELETE' to proceed: ")
    
    if confirm == "DELETE":
        tasks = [(f, Path(OUTPUT_DIR) / f.relative_to(INPUT_DIR)) for f in all_files]
        # Using 8 workers is good for your CPU/SSD speed
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(compress_and_delete, tasks), total=len(tasks), desc="Converting to Wav"))
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()