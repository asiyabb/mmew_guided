import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroTripletLoss(nn.Module):
    """
    Replaces contradictory contrastive loss.
    Enforces adjacent windows (anchors & positives) to be closer in distance 
    than non-adjacent windows (negatives).
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, latent_clues):
        # latent_clues: [B, W, D]
        B, W, D = latent_clues.shape
        if W < 3:
            return torch.tensor(0.0, device=latent_clues.device)
            
        anchor = latent_clues[:, :-2, :]    # Windows 0 to W-3
        positive = latent_clues[:, 1:-1, :] # Windows 1 to W-2 (adjacent)
        negative = latent_clues[:, 2:, :]   # Windows 2 to W-1 (distant)

        return self.triplet_loss(
            anchor.reshape(-1, D), 
            positive.reshape(-1, D), 
            negative.reshape(-1, D)
        )

class TemporalConsistencyLoss(nn.Module):
    """Smooths out temporal feature fluctuations across consecutive windows."""
    def __init__(self):
        super().__init__()

    def forward(self, latent_clues):
        # latent_clues: [B, W, D]
        diff = latent_clues[:, 1:, :] - latent_clues[:, :-1, :]
        return torch.mean(torch.norm(diff, p=2, dim=-1))