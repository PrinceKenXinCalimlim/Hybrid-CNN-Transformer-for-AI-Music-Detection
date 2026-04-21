import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import ASTModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn as nn

from dataset import MusicDetectionDataset

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"   
TEST_CSV        = "test_10.csv"
BATCH_SIZE      = 4   

# ── MODEL ─────────────────────────────────────────────────────────────────────
class HybridASTDetector(nn.Module):
    def __init__(self, ast_backbone):
        super().__init__()
        self.ast = ast_backbone

        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.residual_downsample = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.MaxPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        residual = x.unsqueeze(1)
        cnn_out = self.cnn_block1(residual)
        cnn_out = self.cnn_block2(cnn_out)

        res = self.residual_downsample(residual)
        if res.shape != cnn_out.shape:
            res = torch.nn.functional.interpolate(
                res, size=cnn_out.shape[2:], mode='bilinear', align_corners=False
            )
        cnn_out = cnn_out + res
        cnn_out = cnn_out.squeeze(1)

        if cnn_out.shape[-1] != 1024 or cnn_out.shape[-2] != 128:
            cnn_out = torch.nn.functional.interpolate(
                cnn_out.unsqueeze(1), size=(128, 1024),
                mode='bilinear', align_corners=False
            ).squeeze(1)

        ast_out   = self.ast(cnn_out).last_hidden_state
        cls_token = ast_out[:, 0, :]
        mean_pool = ast_out[:, 1:, :].mean(dim=1)
        combined  = torch.cat([cls_token, mean_pool], dim=1)

        return self.classifier(combined)

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def save_confusion_matrix_plot(cm, target_names):
    """Generates and saves a heatmap of the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names,
                annot_kws={"size": 14})
    
    plt.title('Confusion Matrix: AI Music Detection', fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save as high-res PNG for the thesis
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close() # Close to free memory
    print("✅ Confusion matrix plot saved as 'confusion_matrix.png'")

# ── EVALUATION ────────────────────────────────────────────────────────────────
def run_final_evaluation():
    print(f"\n[SYSTEM] Starting evaluation on {DEVICE}...")

    if not os.path.exists(TEST_CSV):
        print(f"[ERROR] {TEST_CSV} not found.")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] {CHECKPOINT_PATH} not found.")
        return

    # 1. Load model
    print("[*] Loading model architecture...")
    ast_backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model        = HybridASTDetector(ast_backbone).to(DEVICE)

    # 2. Load weights
    print(f"[*] Loading weights from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[✓] Checkpoint: epoch {ckpt.get('epoch', 0)+1}, val_acc={ckpt.get('val_acc', 0):.2%}")

    # 3. Prepare test data
    test_dataset = MusicDetectionDataset(TEST_CSV)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    print(f"[*] Test samples: {len(test_dataset)}")

    all_preds    = []
    all_labels   = []

    # 4. Inference loop
    print("[*] Running inference on test set...")
    with torch.no_grad():
        for mels, labels in tqdm(test_loader, desc="Evaluating", colour="green"):
            mels = mels.to(DEVICE, non_blocking=True)

            with autocast():
                logits = model(mels).squeeze(-1)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Metrics
    final_acc = accuracy_score(all_labels, all_preds)
    target_names = ['Real Music', 'AI Music']
    report    = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate percentages for per-class detail
    precision_ai = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_ai    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_re = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_re    = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 6. Display results
    divider = "=" * 55
    print(f"\n{divider}")
    print("  FINAL TEST RESULTS")
    print(divider)
    print(f"\n{report}")
    print(f"  OVERALL ACCURACY : {final_acc:.2%}")
    print(f"\n{divider}")
    
    # 7. Generate and save visualization
    save_confusion_matrix_plot(cm, target_names)

    # 8. Save results to text file (FIXED ENCODING)
    output_text = f"""FINAL EVALUATION RESULTS
{'='*55}
Checkpoint : {CHECKPOINT_PATH}
Test Acc   : {final_acc:.2%}
Test Set   : {TEST_CSV} ({len(test_dataset)} samples)
{'='*55}

CLASSIFICATION REPORT
{report}

CONFUSION MATRIX
                Predicted Real   Predicted AI
Actual Real  :      {tn:>5}            {fp:>5}
Actual AI    :      {fn:>5}            {tp:>5}

Breakdown:
  True Negatives  (Real -> Real) : {tn}
  True Positives  (AI   -> AI)   : {tp}
  False Positives (Real -> AI)   : {fp}
  False Negatives (AI   -> Real) : {fn}

Per-class detail:
  Real Music -> Precision: {precision_re:.2%}  Recall: {recall_re:.2%}
  AI Music   -> Precision: {precision_ai:.2%}  Recall: {recall_ai:.2%}
{'='*55}
"""

    with open("thesis_final_results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    print("✅ Results saved to 'thesis_final_results.txt'")

if __name__ == "__main__":
    run_final_evaluation()