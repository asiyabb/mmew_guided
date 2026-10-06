import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from config.config import Config
from data.dataset import MMEWSequenceDataset
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from models.macro_guided_model import MacroGuidedModel
from losses.micro_losses import ContrastiveLoss, TemporalConsistencyLoss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_and_collect_history(model_type="baseline", epochs=Config.EPOCHS):
    set_seed(42)
    dataset = MMEWSequenceDataset(mode="macro")
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    vit = ViTFeatureExtractor().to(Config.DEVICE)
    macro_model = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    micro_encoder = None
    if model_type == "guided":
        micro_encoder = MotionMicroEncoder().to(Config.DEVICE)
        optimizer = torch.optim.Adam(
            list(micro_encoder.parameters()) + list(macro_model.parameters()), 
            lr=Config.LR
        )
        contrastive_criterion = ContrastiveLoss()
        temporal_criterion = TemporalConsistencyLoss()
    else:
        optimizer = torch.optim.Adam(macro_model.parameters(), lr=Config.LR)

    cls_criterion = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []

    print(f"\n--- Running Training Protocol: {model_type.upper()} ---")

    for epoch in range(epochs):
        macro_model.train()
        if micro_encoder:
            micro_encoder.train()

        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                vit_feats = vit(x)

            if model_type == "guided":
                latent_clues = micro_encoder(vit_feats)
                loss_c = contrastive_criterion(latent_clues)
                loss_t = temporal_criterion(latent_clues)
                logits = macro_model(vit_feats, latent_clues)
                loss_cls = cls_criterion(logits, y)
                loss = loss_cls + (Config.ALPHA_CONTRASTIVE * loss_c) + (Config.BETA_TEMPORAL * loss_t)
            else:
                dummy_clues = torch.zeros(x.size(0), 7, 256, device=Config.DEVICE)
                logits = macro_model(vit_feats, dummy_clues)
                loss = cls_criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == y).sum().item()
            total_samples += x.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds / total_samples
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc*100:.2f}%")

    # Final Evaluation for Confusion Matrix & Class Metrics
    macro_model.eval()
    if micro_encoder:
        micro_encoder.eval()

    all_preds, all_targets = [], []
    eval_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            vit_feats = vit(x)
            if model_type == "guided":
                latent_clues = micro_encoder(vit_feats)
            else:
                latent_clues = torch.zeros(x.size(0), 7, 256, device=Config.DEVICE)
            
            logits = macro_model(vit_feats, latent_clues)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    report = classification_report(all_targets, all_preds, target_names=Config.EMOTIONS, output_dict=True)

    return {
        "loss": loss_history,
        "acc": acc_history,
        "cm": cm,
        "report": report,
        "overall_acc": epoch_acc,
        "overall_f1": f1_score(all_targets, all_preds, average='macro')
    }

def generate_comparative_plots(baseline_res, guided_res, save_path="experiment_comparison.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12))

    # 1. Accuracy Curve
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(baseline_res["acc"], label="Baseline (Macro Only)", color="#1f77b4", linewidth=2.5, linestyle="--")
    ax1.plot(guided_res["acc"], label="Micro-Guided Macro", color="#d62728", linewidth=2.5)
    ax1.set_title("Training Accuracy Convergence", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.set_ylim(0, 1.05)

    # 2. Loss Curve
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(baseline_res["loss"], label="Baseline Loss", color="#1f77b4", linewidth=2.5, linestyle="--")
    ax2.plot(guided_res["loss"], label="Micro-Guided Loss", color="#d62728", linewidth=2.5)
    ax2.set_title("Training Loss Convergence", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)

    # 3. Class-Wise F1 Score Comparison
    ax3 = fig.add_subplot(2, 3, 3)
    emotions = Config.EMOTIONS
    b_f1s = [baseline_res["report"][e]["f1-score"] for e in emotions]
    g_f1s = [guided_res["report"][e]["f1-score"] for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    ax3.bar(x - width/2, b_f1s, width, label='Baseline', color='#72b7b2')
    ax3.bar(x + width/2, g_f1s, width, label='Micro-Guided', color='#e15759')
    ax3.set_title("Per-Class F1-Score Comparison", fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(emotions, rotation=30, ha='right')
    ax3.set_ylabel("F1-Score", fontsize=11)
    ax3.set_ylim(0, 1.1)
    ax3.legend(frameon=True)

    # 4. Confusion Matrix - Baseline
    ax4 = fig.add_subplot(2, 3, 4)
    sns.heatmap(baseline_res["cm"], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=emotions, yticklabels=emotions, ax=ax4)
    ax4.set_title("Baseline: Normalized Confusion Matrix", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Predicted Label", fontsize=10)
    ax4.set_ylabel("True Label", fontsize=10)

    # 5. Confusion Matrix - Guided
    ax5 = fig.add_subplot(2, 3, 5)
    sns.heatmap(guided_res["cm"], annot=True, fmt=".2f", cmap="Reds", cbar=False,
                xticklabels=emotions, yticklabels=emotions, ax=ax5)
    ax5.set_title("Micro-Guided: Normalized Confusion Matrix", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Predicted Label", fontsize=10)
    ax5.set_ylabel("True Label", fontsize=10)

    # 6. Overall Performance Metrics Summary Card
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = (
        "====================================\n"
        "       EXPERIMENTAL RESULTS         \n"
        "====================================\n\n"
        f"Baseline Model:\n"
        f"  • Overall Accuracy : {baseline_res['overall_acc']*100:.2f}%\n"
        f"  • Macro F1-Score   : {baseline_res['overall_f1']:.4f}\n\n"
        f"Micro-Guided Model:\n"
        f"  • Overall Accuracy : {guided_res['overall_acc']*100:.2f}%\n"
        f"  • Macro F1-Score   : {guided_res['overall_f1']:.4f}\n\n"
        "------------------------------------\n"
        f"Accuracy Gain : +{(guided_res['overall_acc'] - baseline_res['overall_acc'])*100:.2f}%\n"
        f"F1-Score Gain : +{(guided_res['overall_f1'] - baseline_res['overall_f1']):.4f}\n"
        "===================================="
    )
    ax6.text(0.1, 0.2, summary_text, fontsize=12, family='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f4f4", edgecolor="#cccccc"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Visual comparison plot saved successfully to: {save_path}")

if __name__ == "__main__":
    baseline_results = train_and_collect_history(model_type="baseline", epochs=Config.EPOCHS)
    guided_results = train_and_collect_history(model_type="guided", epochs=Config.EPOCHS)
    generate_comparative_plots(baseline_results, guided_results, save_path="experiment_comparison.png")