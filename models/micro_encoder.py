import torch
import torch.nn as nn
from config.config import Config

class MotionMicroEncoder(nn.Module):
    def __init__(self, in_dim=768, latent_dim=256):
        super().__init__()
        self.window_size = Config.WINDOW_SIZE
        self.stride = Config.WINDOW_STRIDE
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, vit_features):
        # vit_features: [B, T, D]
        B, T, D = vit_features.shape
        windows = []
        
        for start in range(0, T - self.window_size + 1, self.stride):
            win = vit_features[:, start:start+self.window_size, :]  # [B, window_size, D]
            win = win.transpose(1, 2)  # [B, D, window_size]
            latent = self.temporal_conv(win).squeeze(-1)  # [B, latent_dim]
            windows.append(latent)
            
        latent_clues = torch.stack(windows, dim=1)  # [B, Num_Windows, latent_dim]
        return latent_clues