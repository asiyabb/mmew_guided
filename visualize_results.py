import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit

from config.config import Config
from data.dataset import MMEWSequenceDataset, AugmentedSubset
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from models.macro_guided_model import MacroGuidedModel, MacroBaselineModel
from losses.micro_losses import MicroTripletLoss, TemporalConsistencyLoss


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_subject_splits(dataset, test_size=0.2, seed=42):
    """Generates subject-grouped splits directly without relying on external import."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(dataset.samples, groups=dataset.subjects))

    raw_train = Subset(dataset, train_idx)
    raw_val = Subset(dataset, val_idx)

    train_dataset = AugmentedSubset(raw_train, transform=MMEWSequenceDataset.get_train_transforms())
    val_dataset = AugmentedSubset(raw_val, transform=MMEWSequenceDataset.get_val_transforms())

    return train_dataset, val_dataset


def train_and_collect_history(model_type="baseline", epochs=Config.EPOCHS):
    set_seed(42)
    full_dataset = MMEWSequenceDataset(mode="macro")
    train_dataset, val_dataset = create_subject_splits(full_dataset, test_size=0.2, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Enable last block fine-tuning
    vit = ViTFeatureExtractor(freeze=True, unfreeze_last_block=True).to(Config.DEVICE)

    if model_type == "guided":
        macro_model = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        micro_encoder = MotionMicroEncoder().to(Config.DEVICE)

        optimizer = torch.optim.Adam([
            {'params': [p for p in vit.parameters() if p.requires_grad], 'lr': 1e-5},
            {'params': micro_encoder.parameters(), 'lr': Config.LR},
            {'params': macro_model.parameters(), 'lr': Config.LR}
        ])
        triplet_criterion = MicroTripletLoss()
        temporal_criterion = TemporalConsistencyLoss()
    else:
        macro_model = MacroBaselineModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
        micro_encoder = None
        optimizer = torch.optim.Adam([
            {'params': [p for p in vit.parameters() if p.requires_grad], 'lr': 1e-5},
            {'params': macro_model.parameters(), 'lr': Config.LR}
        ])

    # Balanced Class Weights to combat class bias
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)
    cls_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss_history, acc_history = [], []
    best_val_acc = -1.0

    # Define paths for saving best model weights
    if model_type == "guided":
        best_macro_path = "guided_model_best.pth"
        best_micro_path = "micro_encoder_best.pth"
    else:
        best_macro_path = "baseline_best.pth"
        best_micro_path = None

    print(f"\n--- Running Training Protocol: {model_type.upper()} ---")

    for epoch in range(epochs):
        macro_model.train()
        vit.train()
        if micro_encoder:
            micro_encoder.train()

        running_loss, correct_preds, total_samples = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()

            vit_feats = vit(x)

            if model_type == "guided":
                latent_clues = micro_encoder(vit_feats)
                loss_trip = triplet_criterion(latent_clues)
                loss_temp = temporal_criterion(latent_clues)
                logits = macro_model(vit_feats, latent_clues)
                loss_cls = cls_criterion(logits, y)
                loss = loss_cls + (Config.ALPHA_TRIPLET * loss_trip) + (Config.BETA_TEMPORAL * loss_temp)
            else:
                logits = macro_model(vit_feats)
                loss = cls_criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == y).sum().item()
            total_samples += x.size(0)

        scheduler.step()
        epoch_loss = running_loss / total_samples
        loss_history.append(epoch_loss)

        # Validation Loop
        macro_model.eval()
        vit.eval()
        if micro_encoder:
            micro_encoder.eval()

        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                vit_feats = vit(x)
                if model_type == "guided":
                    latent_clues = micro_encoder(vit_feats)
                    logits = macro_model(vit_feats, latent_clues)
                else:
                    logits = macro_model(vit_feats)

                val_loss += cls_criterion(logits, y).item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        current_val_acc = val_correct / val_total
        acc_history.append(current_val_acc)

        # Save Best Model Checkpoint
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(macro_model.state_dict(), best_macro_path)
            if micro_encoder and best_micro_path:
                torch.save(micro_encoder.state_dict(), best_micro_path)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train Loss: {epoch_loss:.4f} - Val Acc: {current_val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    # Load Best Model Weights for Final Evaluation & Visualizations
    print(f"\nLoading best checkpoint for [{model_type.upper()}] evaluation (Val Acc: {best_val_acc*100:.2f}%)...")
    macro_model.load_state_dict(torch.load(best_macro_path))
    if model_type == "guided" and best_micro_path:
        micro_encoder.load_state_dict(torch.load(best_micro_path))

    macro_model.eval()
    vit.eval()
    if micro_encoder:
        micro_encoder.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            vit_feats = vit(x)
            logits = macro_model(vit_feats, micro_encoder(vit_feats)) if model_type == "guided" else macro_model(vit_feats)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    report = classification_report(all_targets, all_preds, target_names=Config.EMOTIONS, output_dict=True, zero_division=0)
    overall_f1 = f1_score(all_targets, all_preds, average='macro')

    return {
        "loss": loss_history,
        "acc": acc_history,
        "cm": cm,
        "report": report,
        "overall_acc": best_val_acc,
        "overall_f1": overall_f1
    }


def generate_comparative_plots(baseline_res, guided_res, save_path="experiment_comparison.png"):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(18, 12))

    # 1. Validation Accuracy Curve
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(baseline_res["acc"], label="Baseline (Macro)", color="#1f77b4", linewidth=2.5, linestyle="--")
    ax1.plot(guided_res["acc"], label="Micro-Guided Macro", color="#d62728", linewidth=2.5)
    ax1.set_title("Validation Accuracy Convergence", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.set_ylim(0, 1.05)

    # 2. Training Loss Curve
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
    ax3.set_title("Validation Per-Class F1-Score (Best Weights)", fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(emotions, rotation=30, ha='right')
    ax3.set_ylabel("F1-Score", fontsize=11)
    ax3.set_ylim(0, 1.1)
    ax3.legend(frameon=True)

    # 4. Confusion Matrix - Baseline
    ax4 = fig.add_subplot(2, 3, 4)
    sns.heatmap(baseline_res["cm"], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=emotions, yticklabels=emotions, ax=ax4)
    ax4.set_title("Baseline: Val Confusion Matrix (Best)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Predicted Label", fontsize=10)
    ax4.set_ylabel("True Label", fontsize=10)

    # 5. Confusion Matrix - Guided
    ax5 = fig.add_subplot(2, 3, 5)
    sns.heatmap(guided_res["cm"], annot=True, fmt=".2f", cmap="Reds", cbar=False,
                xticklabels=emotions, yticklabels=emotions, ax=ax5)
    ax5.set_title("Micro-Guided: Val Confusion Matrix (Best)", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Predicted Label", fontsize=10)
    ax5.set_ylabel("True Label", fontsize=10)

    # 6. Performance Summary Card
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    acc_diff = (guided_res['overall_acc'] - baseline_res['overall_acc']) * 100
    f1_diff = guided_res['overall_f1'] - baseline_res['overall_f1']

    summary_text = (
        "====================================\n"
        "     VALIDATION RESULTS SUMMARY     \n"
        "====================================\n\n"
        f"Baseline Model (Best Weights):\n"
        f"  • Overall Accuracy : {baseline_res['overall_acc']*100:.2f}%\n"
        f"  • Macro F1-Score   : {baseline_res['overall_f1']:.4f}\n\n"
        f"Micro-Guided Model (Best Weights):\n"
        f"  • Overall Accuracy : {guided_res['overall_acc']*100:.2f}%\n"
        f"  • Macro F1-Score   : {guided_res['overall_f1']:.4f}\n\n"
        "------------------------------------\n"
        f"Accuracy Gain : {'+' if acc_diff >= 0 else ''}{acc_diff:.2f}%\n"
        f"F1-Score Gain : {'+' if f1_diff >= 0 else ''}{f1_diff:.4f}\n"
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