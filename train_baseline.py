import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
from data.dataset import MMEWSequenceDataset
from models.vit_extractor import ViTFeatureExtractor
from models.macro_guided_model import MacroGuidedModel
from utils.metrics import compute_accuracy

def train_baseline():
    dataset = MMEWSequenceDataset(mode="macro")
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    vit = ViTFeatureExtractor().to(Config.DEVICE)
    model = MacroGuidedModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()

    print("--- Starting Macro Baseline Training ---")
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            
            with torch.no_grad():
                vit_feats = vit(x)
            
            # Dummy micro clues for baseline initialization
            dummy_clues = torch.zeros(x.size(0), 7, 256, device=Config.DEVICE)
            logits = model(vit_feats, dummy_clues)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, y)

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {total_acc/len(loader):.4f}")

if __name__ == "__main__":
    train_baseline()