import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroMacroModel(nn.Module):
    """
    A model for breast lesion classification using micro (ROI) and macro (context) crops,
    with manufacturer conditioning via an embedding vector.
    """

    def __init__(self, out_channels=2, num_manufacturers=2, weight=None, manu_emb_dim=8):
        super(MicroMacroModel, self).__init__()
        self.weight = weight
        self.out_channels = out_channels
        self.num_manufacturers = num_manufacturers
        self.manu_emb_dim = manu_emb_dim

        # --- feature extractors ---
        self.features_micro = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> [B, 16, 1, 1]
        )

        self.features_macro = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # -> [B, 16, 1, 1]
        )

        # manufacturer embedding (learned)
        self.manu_embed = nn.Embedding(self.num_manufacturers, self.manu_emb_dim)

        micro_feat_dim = 16
        macro_feat_dim = 16
        in_dim = micro_feat_dim + macro_feat_dim + self.manu_emb_dim  # 16 + 16 + 8

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.out_channels)
        )

    def forward(self, micro, macro, manu_id=None):
        # --- Per-sample, per-channel normalization ---
        # micro/macro shape: [B, 1, H, W]
        micro = (micro - micro.mean(dim=(2,3), keepdim=True)) / (micro.std(dim=(2,3), keepdim=True) + 1e-8)
        macro = (macro - macro.mean(dim=(2,3), keepdim=True)) / (macro.std(dim=(2,3), keepdim=True) + 1e-8)

        # --- Feature extraction ---
        micro = self.features_micro(micro).flatten(1)  # [B, 16]
        macro = self.features_macro(macro).flatten(1)  # [B, 16]

        # --- Manufacturer vector ---
        if manu_id is None:
            manu_vec = torch.zeros(micro.size(0), self.manu_emb_dim, device=micro.device, dtype=torch.float32)
        else:
            # ensure manu_id is on the SAME device and correct dtype
            manu_id = manu_id.to(micro.device).long()
            manu_vec = self.manu_embed(manu_id)  # [B, manu_emb_dim]

        x = torch.cat([micro, macro, manu_vec], dim=1)
        logits = self.classifier(x)
        return logits
