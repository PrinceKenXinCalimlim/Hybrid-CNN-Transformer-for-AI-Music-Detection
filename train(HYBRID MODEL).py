import torch
import os
import numpy as np
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import MusicDetectionDataset
from transformers import ASTModel

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4      
EPOCHS = 15          
LR = 1e-5           
CHECKPOINT_PATH = "checkpoint.pth"

# --- HYBRID MODEL ARCHITECTURE ---
class HybridASTDetector(torch.nn.Module):
    def __init__(self):
        super(HybridASTDetector, self).__init__()
        # CNN component for local feature extraction
        self.cnn_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(1),
            torch.nn.GELU(),
            torch.nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.GELU()
        )
        # AST backbone for global audio structure
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.classifier = torch.nn.Linear(self.ast.config.hidden_size, 1)

    def forward(self, x):
        # Input shape: [Batch, 128, 1024]
        x = x.unsqueeze(1) # Add channel dim: [Batch, 1, 128, 1024]
        x = self.cnn_extractor(x)
        x = x.squeeze(1)   # Remove channel dim
        ast_outputs = self.ast(x)
        # Classification based on the pooler output (first token)
        return self.classifier(ast_outputs.last_hidden_state[:, 0, :])

def train_model():
    model = HybridASTDetector().to(device)
    
    # Optimizer and Loss function
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = BCEWithLogitsLoss()
    scaler = GradScaler() # For Mixed Precision (Speed)

    # Resume from checkpoint if it exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f">>> Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

    # Data Loaders
    train_loader = DataLoader(
        MusicDetectionDataset('train.csv'), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        MusicDetectionDataset('val.csv'), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    print(f">>> Starting training on {device}...")

    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for mels, labels in train_loop:
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with autocast(): # Use RTX 3050 Tensor Cores
                outputs = model(mels).squeeze(-1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDATION PHASE (End of Epoch) ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        print(f"\n>>> Running Validation for Epoch {epoch+1}...")
        with torch.no_grad():
            for mels, labels in tqdm(val_loader, desc="Validating"):
                mels = mels.to(device)
                labels = labels.to(device)
                
                with autocast():
                    logits = model(mels).squeeze(-1)
                    v_loss = criterion(logits, labels)
                
                val_loss += v_loss.item()
                
                # Convert logits to binary predictions
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # --- METRICS CALCULATION ---
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        print("\n" + "="*40)
        print(f"RESULTS FOR EPOCH {epoch+1}")
        print(f"Avg Training Loss:   {avg_train_loss:.4f}")
        print(f"Avg Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2%}")
        print("="*40 + "\n")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, CHECKPOINT_PATH)

if __name__ == '__main__':
    train_model()