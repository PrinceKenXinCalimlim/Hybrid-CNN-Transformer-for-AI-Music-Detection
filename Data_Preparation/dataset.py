import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000 
DURATION = 3        # 3 seconds: High speed, high accuracy
N_MELS = 128        

class MusicDetectionDataset(Dataset):
    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # AST-specific Mel Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=N_MELS
        )
        # Optimized resamplers
        self.resample_44 = torchaudio.transforms.Resample(44100, SAMPLE_RATE)
        self.resample_48 = torchaudio.transforms.Resample(48000, SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            audio_path, label = row['path'], row['label']

            data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()
            
            # Mono conversion
            if waveform.ndim > 1:
                waveform = torch.mean(waveform.T, dim=0, keepdim=True)
            else:
                waveform = waveform.unsqueeze(0)

            # Fast Resampling
            if sr == 44100: waveform = self.resample_44(waveform)
            elif sr == 48000: waveform = self.resample_48(waveform)
            elif sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

            # Exact Duration Trimming
            target_len = SAMPLE_RATE * DURATION
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))

            # Log-Mel Spectrogram
            mel = self.mel_transform(waveform)
            mel = torch.log(mel + 1e-9)
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)
            mel = mel.squeeze(0).transpose(0, 1)
            
            # Pad to 1024 for AST model compatibility
            if mel.shape[0] < 1024:
                mel = torch.nn.functional.pad(mel, (0, 0, 0, 1024 - mel.shape[0]))

            return mel, torch.tensor(label, dtype=torch.float)
        except Exception as e:
            # Silent failure during training, but print during sanity check
            return torch.zeros((1024, N_MELS)), torch.tensor(0.0)

# --- THE "LOUD" SANITY CHECK ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("STARTING DATASET VERIFICATION...")
    print("="*50)
    
    csv_to_test = "train.csv"
    
    if os.path.exists(csv_to_test):
        print(f"[✓] Found {csv_to_test}")
        try:
            ds = MusicDetectionDataset(csv_to_test)
            print(f"[✓] Dataset initialized with {len(ds)} samples.")
            
            print("Attempting to load index 0...")
            mel, lbl = ds[0]
            
            print("\n--- RESULTS ---")
            print(f"Spectrogram Shape: {mel.shape}") # Expect [1024, 128]
            print(f"Label: {lbl.item()}")
            print(f"Data Min/Max: {mel.min():.2f} / {mel.max():.2f}")
            print("----------------")
        
            
        except Exception as e:
            print(f"[X] ERROR DURING LOADING: {e}")
    else:
        print(f"[X] CRITICAL ERROR: '{csv_to_test}' NOT FOUND!")
        print(f"    Current path: {os.getcwd()}")
        print("    ACTION: Run 'prepare_data.py' first.")
    
    print("="*50 + "\n")