import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTFeatureExtractor(nn.Module):
    def __init__(self, freeze=True, unfreeze_last_block=True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        vit = vit_b_16(weights=weights)
        self.backbone = vit
        self.backbone.heads = nn.Identity()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Unfreeze final encoder layer for facial feature adaptation
        if unfreeze_last_block:
            for param in self.backbone.encoder.layers[-1].parameters():
                param.requires_grad = True

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        features = self.backbone(x_reshaped)  # [B*T, D]
        return features.view(B, T, -1)