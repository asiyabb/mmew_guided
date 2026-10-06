import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
from data.dataset import MMEWSequenceDataset
from models.vit_extractor import ViTFeatureExtractor
from models.micro_encoder import MotionMicroEncoder
from models.macro_guided_model import MacroGuidedModel
from losses.micro_losses import ContrastiveLoss, TemporalConsistencyLoss
from utils.metrics import compute_accuracy

def train_guided_pipeline():
    dataset = MMEWSequenceDataset(mode="macro")
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    vit = ViTFeatureExtractor().to(Config.DEVICE)
    micro_encoder = MotionMicroEncoder().to(Config.DEVICE)
    macro_model = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    optimizer = torch.optim.Adam(
        list(micro_encoder.parameters()) + list(macro_model.parameters()), 
        lr=Config.LR
    )
    
    cls_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss()
    temporal_criterion = TemporalConsistencyLoss()

    print("--- Starting Micro-guided Macro Recognition Training ---")
    for epoch in range(Config.EPOCHS):
        micro_encoder.train()
        macro_model.train()
        total_loss, total_acc = 0.0, 0.0

        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                vit_feats = vit(x)

            # Short overlapping windows -> Motion/temporal encoder -> Latent micro-clues
            latent_clues = micro_encoder(vit_feats)

            # Losses
            loss_contrastive = contrastive_criterion(latent_clues)
            loss_temporal = temporal_criterion(latent_clues)

            # Macro encoder + Classifier
            logits = macro_model(vit_feats, latent_clues)
            loss_cls = cls_criterion(logits, y)

            total_loss_step = loss_cls + (Config.ALPHA_CONTRASTIVE * loss_contrastive) + (Config.BETA_TEMPORAL * loss_temporal)
            
            total_loss_step.backward()
            optimizer.step()

            total_loss += total_loss_step.item()
            total_acc += compute_accuracy(logits, y)

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Combined Loss: {total_loss/len(loader):.4f} | Acc: {total_acc/len(loader):.4f}")

if __name__ == "__main__":
    train_guided_pipeline()