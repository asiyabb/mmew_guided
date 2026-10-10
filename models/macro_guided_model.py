import torch
import torch.nn as nn

class MacroBaselineModel(nn.Module):
    """Pure Macro Baseline Model."""
    def __init__(self, feature_dim=768, latent_dim=256, num_classes=6):
        super().__init__()
        self.macro_rnn = nn.GRU(feature_dim, latent_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, vit_features):
        _, h_n = self.macro_rnn(vit_features)
        macro_feat = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, latent_dim * 2]
        return self.classifier(macro_feat)


class MacroGuidedModel(nn.Module):
    """Micro-Guided Macro Model using Cross-Attention & Adaptive Gating."""
    def __init__(self, feature_dim=768, latent_dim=256, num_classes=6):
        super().__init__()
        self.macro_rnn = nn.GRU(feature_dim, latent_dim, batch_first=True, bidirectional=True)
        
        # Project micro window clues into macro embedding space
        self.micro_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.Dropout(0.4)
        )
        
        # Cross-Attention: Macro queries Micro window clues
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim * 2, num_heads=4, batch_first=True)
        
        # Dynamic Channel Gate: Learns how much micro guidance to admit
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(latent_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, vit_features, latent_micro_clues):
        # vit_features: [B, T, D]
        # latent_micro_clues: [B, W, latent_dim]
        
        # Extract Macro Global Features
        _, h_n = self.macro_rnn(vit_features)
        macro_feat = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, latent_dim * 2]
        
        # Project Micro Sequence Clues
        micro_seq = self.micro_proj(latent_micro_clues)   # [B, W, latent_dim * 2]
        
        # Cross-Attention: Macro Queries Micro Sequence
        query = macro_feat.unsqueeze(1)                   # [B, 1, latent_dim * 2]
        micro_guided_clues, _ = self.cross_attn(query=query, key=micro_seq, value=micro_seq)
        micro_guided_clues = micro_guided_clues.squeeze(1) # [B, latent_dim * 2]
        
        # Compute Adaptive Gate (Range: 0 to 1)
        gate_input = torch.cat([macro_feat, micro_guided_clues], dim=-1)
        g = self.gate(gate_input)                         # [B, latent_dim * 2]
        
        # Guided Macro Representation: Macro + (Gated Micro Guidance)
        fused_feat = self.norm(macro_feat + (g * micro_guided_clues))
        
        return self.classifier(fused_feat)