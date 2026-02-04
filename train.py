import torch
import logging
import warnings
import os
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataset import MusicDetectionDataset
from transformers import AutoModelForAudioClassification

# 1. CLEANUP: Hide unnecessary warnings and messy library logs
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# 2. CONFIGURATION: Hardware and Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
batch_size = 4        # Number of samples processed before updating weights
epochs = 5            # Total times the model sees the entire dataset
lr = 2e-5             # Learning rate (how fast the AI 'adjusts' its brain)
CHECKPOINT_PATH = "checkpoint.pth" # Temporary file to save progress

def main():
    print(f"Starting training on {device}...")

    # 3. MODEL SETUP: Download MIT's AST and modify it for 1-label output (Music vs. Not Music)
    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=1,                   # We only need a single score (probability)
        ignore_mismatched_sizes=True    # Needed because we changed the output head size
    )
    model.to(device)

    # 4. TOOLS: Optimizer (updates weights) and Criterion (calculates the error)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss() # Best choice for binary classification
    
    # 5. RESUME LOGIC: Check if a previous run was interrupted
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint! Resuming from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch + 1}")

    # 6. DATA LOADERS: Feed the data into the model in small batches
    train_loader = DataLoader(MusicDetectionDataset('train.csv'), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MusicDetectionDataset('val.csv'), batch_size=batch_size, shuffle=False)

    # 7. MAIN TRAINING LOOP
    for epoch in range(start_epoch, epochs):
        model.train() # Tell the model it is in 'learning mode'
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for mels, labels in loop:
            mels = mels.to(device)
            labels = labels.to(device).float()

            # The Forward-Backward Pass
            optimizer.zero_grad()                        # Reset previous math
            outputs = model(mels).logits.squeeze(-1)     # Get model guess
            loss = criterion(outputs, labels)            # How wrong was the guess?
            loss.backward()                              # Calculate which weights to change
            optimizer.step()                             # Apply the changes

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 8. VALIDATION: Check performance on data the model hasn't seen yet
        model.eval() # Tell the model it is in 'testing mode'
        all_preds, all_true = [], []
        with torch.no_grad(): # Don't calculate gradients (saves memory)
            for mels, labels in val_loader:
                mels = mels.to(device)
                logits = model(mels).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy() # Convert to 0.0 - 1.0 range
                all_preds.extend(probs)
                all_true.extend(labels.numpy())

        # Convert probabilities (0.8) to binary labels (1)
        preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_true, preds_bin)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

        # 9. SAVE CHECKPOINT: Save progress after every epoch
        print(f"Saving checkpoint for epoch {epoch+1}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    # 10. FINAL SAVE: Export the finished "brain" and clean up
    torch.save(model.state_dict(), "ast_detector_final.pth")
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH) # Delete temp checkpoint
    print("Training Complete!")

if __name__ == '__main__':
    main()