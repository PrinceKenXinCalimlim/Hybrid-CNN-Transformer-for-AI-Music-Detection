import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import os
import soundfile as sf
import warnings

# Ignore specific warnings to keep the terminal output clean during training
warnings.filterwarnings("ignore", category=UserWarning)

# Global constants for consistency across training and inference
SAMPLE_RATE = 16000 # Standard frequency for speech/music models
DURATION = 10       # We want every audio clip to be exactly 10 seconds
N_MELS = 128        # The "height" of our spectrogram image (frequency bins)

class MusicDetectionDataset(Dataset):
    def __init__(self, csv_path):
        """Initializes the dataset by reading the CSV and setting up the Mel Transform."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load the manifest file containing [path, label]
        self.df = pd.read_csv(csv_path)
        
        # Define the mathematical transform that converts raw audio waves to Mel Spectrograms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,      # Size of the window used for Fourier Transform
            hop_length=512,  # Distance between windows (determines temporal resolution)
            n_mels=N_MELS    # Number of Mel filter banks
        )

    def __len__(self):
        """Returns the total number of samples in the CSV."""
        return len(self.df)

    def __getitem__(self, idx):
        """Loads, processes, and returns a single audio sample and its label."""
        row = self.df.iloc[idx]
        audio_path = row['path']
        label = row['label']

        try:
            # 1. LOAD AUDIO: Using soundfile is often more stable than torchaudio for various formats
            data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()
            
            # 2. SHAPE CORRECTION: Ensure shape is [Channels, Time]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # Add channel dim to mono
            else:
                waveform = waveform.T           # Flip [Time, Channel] to [Channel, Time]

            # 3. MONO DOWNMIX: If stereo, average the channels into one
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 4. RESAMPLING: Ensure the audio matches our target 16kHz
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

            # 5. DURATION STANDARDIZATION: Truncate if too long, Pad with zeros if too short
            target_len = SAMPLE_RATE * DURATION
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))

            # 6. GENERATE SPECTROGRAM: Convert raw wave to Mel Spectrogram
            mel = self.mel_transform(waveform) # Initial shape: [1, 128, 313]
            
            # 7. LOG SCALING & NORMALIZATION: Make the data more 'learnable' for the AI
            mel = torch.log(mel + 1e-9) # Use log scale because human hearing is logarithmic
            mel = (mel - mel.mean()) / (mel.std() + 1e-9) # Z-score normalization

            # 8. AST COMPATIBILITY FIXES:
            mel = mel.squeeze(0)          # Remove channel dim -> [128, 313]
            mel = mel.transpose(0, 1)     # Swap dims to [Time, Freq] -> [313, 128]
            
            # 9. FINAL PADDING: AST models often expect exactly 1024 time steps
            if mel.shape[0] < 1024:
                pad_amount = 1024 - mel.shape[0]
                mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_amount)) # Pad time axis

            return mel, torch.tensor(label, dtype=torch.float)

        except Exception as e:
            # Error handling: If a file is broken, return a silent (zero) tensor so training doesn't crash
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros((1024, N_MELS)), torch.tensor(0.0)

if __name__ == "__main__":
    # Quick sanity check to verify the output shape is exactly what the model expects
    if os.path.exists("train.csv"):
        ds = MusicDetectionDataset("train.csv")
        mel, lbl = ds[0]
        print(f"Dataset Test -> Shape: {mel.shape}, Label: {lbl}") 
        # Goal output: [1024, 128]