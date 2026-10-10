import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config.config import Config
from data.dataset import MMEWSequenceDataset
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from models.macro_guided_model import MacroGuidedModel, MacroBaselineModel
from visualize_results import create_subject_splits


def run_evaluation(model, micro_encoder, vit, loader, mode_name="Model"):
    model.eval()
    vit.eval()
    if micro_encoder:
        micro_encoder.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            vit_feats = vit(x)

            if micro_encoder and isinstance(model, MacroGuidedModel):
                latent_clues = micro_encoder(vit_feats)
                logits = model(vit_feats, latent_clues)
            else:
                logits = model(vit_feats)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average='macro')

    print(f"\n================ {mode_name} EVALUATION ================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Macro F1 : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=Config.EMOTIONS, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    return acc, f1


if __name__ == "__main__":
    full_dataset = MMEWSequenceDataset(mode="macro")
    
    # Recreate exact subject-grouped validation split used in visualize_results.py (seed=42, test_size=0.2)
    _, val_dataset = create_subject_splits(full_dataset, test_size=0.2, seed=42)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    vit = ViTFeatureExtractor(freeze=True, unfreeze_last_block=True).to(Config.DEVICE)

    # 1. Evaluate Baseline Model Checkpoint
    print("Evaluating Baseline Model on Unseen Validation Subjects...")
    baseline_model = MacroBaselineModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    baseline_checkpoint = "baseline_best.pth"
    
    if os.path.exists(baseline_checkpoint):
        baseline_model.load_state_dict(torch.load(baseline_checkpoint))
        run_evaluation(baseline_model, None, vit, val_loader, mode_name="Macro Baseline")
    else:
        print(f"[Error] Could not find checkpoint file: {baseline_checkpoint}")

    # 2. Evaluate Guided Model Checkpoint
    print("\nEvaluating Guided Model on Unseen Validation Subjects...")
    guided_model = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    micro_encoder = MotionMicroEncoder().to(Config.DEVICE)
    guided_checkpoint = "guided_model_best.pth"
    micro_checkpoint = "micro_encoder_best.pth"

    if os.path.exists(guided_checkpoint) and os.path.exists(micro_checkpoint):
        guided_model.load_state_dict(torch.load(guided_checkpoint))
        micro_encoder.load_state_dict(torch.load(micro_checkpoint))
        run_evaluation(guided_model, micro_encoder, vit, val_loader, mode_name="Micro-Guided Macro")
    else:
        print(f"[Error] Could not find guided checkpoints: {guided_checkpoint} or {micro_checkpoint}")