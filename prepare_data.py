from pathlib import Path
import pandas as pd

# 1. SETUP: Define which audio formats to look for
AUDIO_EXTS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.webm', '.opus']

def collect_audio_paths(dir_path: str | Path):
    """Recursively finds all audio files in a folder and its subfolders."""
    p = Path(dir_path)
    files = []
    for ext in AUDIO_EXTS:
        # Search for lowercase extensions (e.g., .mp3)
        files.extend(p.rglob(f'*{ext}'))
        # Search for uppercase extensions (e.g., .MP3)
        files.extend(p.rglob(f'*{ext.upper()}'))
    return [str(f) for f in files if f.is_file()]

# 2. PATHS: Points to where your 'Real' vs 'AI-Generated' music lives
fake_dir = r"C:\Users\Xin\Downloads\ThesisData\fake_songs"
real_dir = r"C:\Users\Xin\Downloads\ThesisData\real_songs"

fake_paths = collect_audio_paths(fake_dir)
real_paths = collect_audio_paths(real_dir)

print(f"Found {len(real_paths):,} real audio files (all formats)")
print(f"Found {len(fake_paths):,} fake audio files (all formats)")

# 3. BALANCING: This prevents the AI from just guessing the "most common" type.
# It trims the larger set to match the size of the smaller set.
min_count = min(len(real_paths), len(fake_paths))
real_paths = real_paths[:min_count]
fake_paths = fake_paths[:min_count]

# 4. LABELING: 0 for Real, 1 for Fake
data = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
df = pd.DataFrame(data, columns=['path', 'label'])

# 5. SHUFFLING: Mixes the data up so the AI doesn't see all 'Reals' then all 'Fakes'
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. SPLITTING: Dividing the data into three buckets
# 80% for Training (Learning)
# 10% for Validation (Tuning during training)
# 10% for Testing (Final exam after training is done)
train_df = df.iloc[:int(0.8 * len(df))]
val_df   = df.iloc[int(0.8 * len(df)):int(0.9 * len(df))]
test_df  = df.iloc[int(0.9 * len(df)):]

# 7. OUTPUT: Save these maps as CSV files for the Trainer to read
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Dataset CSV files created successfully.")