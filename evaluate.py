import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from config.config import Config
from data.dataset import MMEWSequenceDataset, get_subject_splits
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from models.macro_guided_model import MacroGuidedModel, MacroBaselineModel

def run_evaluation(model, micro_encoder, vit, loader, mode_name="Model"):
    model.eval()
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

    print(f"\n================ {mode_name} Evaluation ================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Macro F1 : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=Config.EMOTIONS, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    return acc, f1

if __name__ == "__main__":
    full_dataset = MMEWSequenceDataset(mode="macro")
    _, val_dataset = get_subject_splits(full_dataset)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    vit = ViTFeatureExtractor().to(Config.DEVICE)

    # Evaluate Baseline
    print("Evaluating Baseline Model...")
    baseline_model = MacroBaselineModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    if os.path.exists("baseline.pth"):
        baseline_model.load_state_dict(torch.load("baseline.pth"))
    run_evaluation(baseline_model, None, vit, val_loader, mode_name="Macro Baseline")

    # Evaluate Guided Model
    print("Evaluating Guided Model...")
    guided_model = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    micro_encoder = MotionMicroEncoder().to(Config.DEVICE)
    if os.path.exists("guided_model.pth") and os.path.exists("micro_encoder.pth"):
        guided_model.load_state_dict(torch.load("guided_model.pth"))
        micro_encoder.load_state_dict(torch.load("micro_encoder.pth"))
    run_evaluation(guided_model, micro_encoder, vit, val_loader, mode_name="Micro-Guided Macro")