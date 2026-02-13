import os
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# The folders where your new 100k files are located
folders_to_convert = [
    r"C:\Users\Xin\Downloads\ThesisData\fake_songs",
    r"C:\Users\Xin\Downloads\ThesisData\real_songs"
]

def convert_to_wav():
    for folder in folders_to_convert:
        print(f"\n>>> Processing folder: {folder}")
        
        # 1. Collect all non-wav files
        path = Path(folder)
        # Add any extensions you found in your new 100k data here
        files = []
        for ext in ['*.mp3', '*.webm', '*.mp4', '*.m4a', '*.ogg']:
            files.extend(list(path.rglob(ext)))

        if not files:
            print(f"No files to convert in {folder}.")
            continue

        print(f"Found {len(files)} files to convert.")

        # 2. Conversion Loop
        for file_path in tqdm(files, desc="Converting"):
            try:
                # Load the audio (pydub handles webm/mp3 automatically via ffmpeg)
                audio = AudioSegment.from_file(str(file_path))

                # 3. Standardization (Crucial for 99% accuracy)
                # We set it to 16kHz and Mono right here to save training time later
                audio = audio.set_frame_rate(16000).set_channels(1)

                # Create the new filename (e.g., song.mp3 -> song.wav)
                target_path = file_path.with_suffix('.wav')

                # Export as WAV
                audio.export(str(target_path), format="wav")

                # 4. Cleanup: Delete the original non-wav file to save space
                os.remove(file_path)
                
            except Exception as e:
                print(f"Error converting {file_path.name}: {e}")

if __name__ == "__main__":
    convert_to_wav()
    print("\n>>> All files converted and standardized to 16kHz WAV!")