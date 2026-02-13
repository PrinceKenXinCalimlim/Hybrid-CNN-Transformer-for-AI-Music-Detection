import os
from pathlib import Path
import pandas as pd

# --- CONFIGURATION ---
# Define which audio formats to look for
AUDIO_EXTS = ['.wav'] 

# Absolute paths to your raw audio folders
fake_dir = r"C:\Users\Xin\Downloads\ThesisData\fake_songs"
real_dir = r"C:\Users\Xin\Downloads\ThesisData\real_songs"

def collect_audio_paths(dir_path):
    """
    Recursively scans the directory for all files matching AUDIO_EXTS.
    Includes a check for both lowercase and uppercase extensions.
    """
    p = Path(dir_path)
    files = []
    for ext in AUDIO_EXTS:
        # rglob searches subfolders; *ext matches the file ending
        files.extend(p.rglob(f'*{ext}'))
        files.extend(p.rglob(f'*{ext.upper()}'))
    return [str(f) for f in files if f.is_file()]

def main():
    print(">>> Scanning folders for .wav files (this may take a minute for 180k files)...")
    fake_paths = collect_audio_paths(fake_dir)
    real_paths = collect_audio_paths(real_dir)

    print(f"Found {len(real_paths):,} Real and {len(fake_paths):,} Fake files.")

    # --- 1. THE BALANCING ACT ---
    # To prevent bias, we must have an equal number of Real (0) and Fake (1) samples.
    # We take the count of whichever folder is smaller.
    min_count = min(len(real_paths), len(fake_paths))
    
    # Trim the larger list to match the smaller list
    real_paths = real_paths[:min_count]
    fake_paths = fake_paths[:min_count]

    # --- 2. DATAFRAME CREATION ---
    # Create tuples of (file_path, label) where 0=Real and 1=Fake
    data = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    df = pd.DataFrame(data, columns=['path', 'label'])
    
    # --- 3. SHUFFLING ---
    # Mixing the data randomly so the model doesn't see all 'Real' then all 'Fake'.
    # random_state=42 ensures the "randomness" is the same every time you run the script.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 4. DATA SPLITTING (80/10/10 Rule) ---
    # 80% for Training: The model learns from this.
    # 10% for Validation: Used during training to check progress.
    # 10% for Testing: Used at the very end to get the "Final Grade".
    train_end = int(0.8 * len(df))
    val_end   = int(0.9 * len(df))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    # --- 5. SAVING OUTPUT ---
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("   DATASET PREP SUCCESSFUL")
    print("="*30)
    print(f"Total Balanced Samples: {len(df):,}")
    print(f"Training Set:         {len(train_df):,} files")
    print(f"Validation Set:       {len(val_df):,} files")
    print(f"Testing Set:          {len(test_df):,} files")
    print(f"Class Distribution:   50% Real / 50% Fake")
    print("="*30)

if __name__ == "__main__":
    main()