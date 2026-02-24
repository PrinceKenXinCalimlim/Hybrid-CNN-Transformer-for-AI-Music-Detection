import torch
import os
import numpy as np
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import MusicDetectionDataset  # Your custom data loader
from transformers import ASTModel           # Pre-trained Audio Spectrogram Transformer

# --- 1. GLOBAL CONFIGURATION ---
# These settings control the training environment and behavior
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4       # Number of songs processed at once (Keep small for RTX 3050 VRAM)
EPOCHS = 15          # How many times the model sees the entire dataset
LR = 1e-5            # Learning rate: How fast the model updates its knowledge
CHECKPOINT_PATH = "checkpoint.pth" # File to save progress

# --- 2. HYBRID MODEL ARCHITECTURE ---
# This class combines CNN (Local details) with Transformer (Global patterns)
class HybridASTDetector(torch.nn.Module):
    def __init__(self):
        super(HybridASTDetector, self).__init__()
        
        # A. CNN BLOCK: The "Microscope"
        # Per Chapter 3, Phase 2: Extracts local artifacts like AI-generated jitter or noise.
        self.cnn_extractor = torch.nn.Sequential(
            # Layer 1: Look for basic spectral shapes
            torch.nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(16), # Stabilizes training
            torch.nn.GELU(),          # Advanced activation function
            
            # Layer 2: Detect more complex textures
            torch.nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            
            # Layer 3: Refine the detected features
            torch.nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            
            # Layer 4: Match the input requirement for the AST Transformer
            torch.nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.GELU()
        )
        
        # B. TRANSFORMER BLOCK: The "Big Picture"
        # Uses the Audio Spectrogram Transformer (AST) to understand musical flow over time.
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # C. CLASSIFIER: The "Decision Maker"
        # Compresses the Transformer's complex output into a single "AI vs Human" score.
        self.classifier = torch.nn.Linear(self.ast.config.hidden_size, 1)

    def forward(self, x):
        # 1. Add channel dimension [Batch, 1, Freq, Time] for the CNN
        x = x.unsqueeze(1)    
        
        # 2. Extract local features using CNN layers
        x = self.cnn_extractor(x)
        
        # 3. Remove channel dimension to feed into Transformer
        x = x.squeeze(1)      
        
        # 4. Analyze global context using Transformer
        ast_outputs = self.ast(x)
        
        # 5. Extract the summary token ([CLS]) and run it through the final classifier
        # Result is a 'logit' which we will later turn into a percentage
        return self.classifier(ast_outputs.last_hidden_state[:, 0, :])

# --- 3. TRAINING FUNCTION ---
def train_model():
    # Initialize model and move it to your RTX 3050 GPU
    model = HybridASTDetector().to(device)
    
    # Define how the model learns (AdamW) and how it measures error (BCEWithLogits)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = BCEWithLogitsLoss()
    
    # Scaler allows the GPU to use 'Mixed Precision' (Makes training 2x faster)
    scaler = GradScaler() 

    # --- RESUME LOGIC ---
    # If training was interrupted, this loads the last saved state to continue.
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f">>> Found existing checkpoint. Loading weights...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f">>> Resuming from Epoch {start_epoch + 1}")

    # --- DATA PREPARATION ---
    # Loads your CSV files and prepares batches of audio spectrograms
    train_loader = DataLoader(
        MusicDetectionDataset('train.csv'), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True # Optimization for faster GPU data loading
    )
    val_loader = DataLoader(
        MusicDetectionDataset('val.csv'), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )

    print(f">>> Starting training on {device}...")

    # --- MAIN LOOP ---
    for epoch in range(start_epoch, EPOCHS):
        
        # A. TRAINING PHASE
        model.train()
        train_loss = 0.0
        # Progress bar (tqdm) to see training speed
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for mels, labels in train_loop:
            # Move data to GPU
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Clear previous math calculations
            optimizer.zero_grad(set_to_none=True)

            # Use Autocast to run calculations in float16 (GPU optimization)
            with autocast(): 
                outputs = model(mels).squeeze(-1)
                loss = criterion(outputs, labels)

            # Math magic: Calculate how much to change the 'brain' to fix errors
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        # B. VALIDATION PHASE (The "Exam")
        # Check how the model performs on music it has never seen before
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        print(f"\n>>> Running Validation...")
        with torch.no_grad(): # Don't learn during validation
            for mels, labels in tqdm(val_loader, desc="Validating"):
                mels = mels.to(device)
                labels = labels.to(device)
                
                with autocast():
                    logits = model(mels).squeeze(-1)
                
                # Turn raw output into a 0 (Human) or 1 (AI) prediction
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # C. METRICS CALCULATION
        avg_train_loss = train_loss / len(train_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        # Print visual summary of the epoch
        print("\n" + "="*40)
        print(f"EPOCH {epoch+1} COMPLETE")
        print(f"Loss (Lower is better): {avg_train_loss:.4f}")
        print(f"Accuracy (Higher is better): {val_acc:.2%}")
        print("="*40 + "\n")

        # D. SAVE PROGRESS
        # Save the 'brain' state so you don't lose progress if the PC restarts
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, CHECKPOINT_PATH)

if __name__ == '__main__':
    train_model()