import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from eb_jepa.architectures import RNNPredictor

# ==========================================
# VIDEO-JEPA (Temporal Next-Frame Prediction)
# ==========================================
class EEGVideoJEPAEncoder(nn.Module):
    def __init__(self, in_channels=19, base_filters=64, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
            nn.Conv1d(base_filters, base_filters*2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_filters*2),
            nn.GELU(),
            nn.Conv1d(base_filters*2, out_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def represent(self, x):
        features = self.net(x)
        return self.pool(features).squeeze(-1)

    def forward(self, x):
        return self.represent(x)

class VideoJEPASSL(nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.predictor = RNNPredictor(
            hidden_size=cfg.out_dim,
            action_dim=1,
            final_ln=nn.Identity()
        )
        self.ema_momentum = cfg.get("ema_momentum", 0.99)

    @torch.no_grad()
    def _update_ema(self):
        for p_c, p_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(self.ema_momentum).add_(p_c.data, alpha=1.0 - self.ema_momentum)

    def compute_loss(self, batch):
        self._update_ema()
        v1 = batch[0]
        B, C, T = v1.shape
        frame_size = 200 # 1 second
        n_frames = T // frame_size
        
        frames = v1.view(B, n_frames, C, frame_size)
        
        context_frames = frames[:, :-1].reshape(B * (n_frames - 1), C, frame_size)
        context_reprs = self.context_encoder(context_frames).view(B, n_frames - 1, -1)
        
        with torch.no_grad():
            target_frames = frames[:, 1:].reshape(B * (n_frames - 1), C, frame_size)
            target_reprs = self.target_encoder(target_frames).view(B, n_frames - 1, -1)
            
        dummy_actions = torch.zeros(B, n_frames - 1, 1, device=v1.device)
        
        B_new = B * (n_frames - 1)
        state_in = context_reprs.reshape(B_new, -1, 1, 1, 1)
        action_in = dummy_actions.reshape(B_new, 1, 1)
        
        predictions_raw = self.predictor(state_in, action_in)
        predictions = predictions_raw.view(B, n_frames - 1, -1)
        
        loss = F.mse_loss(predictions, target_reprs)
        logs = f"mse: {loss.item():.4f}"
        return loss, logs

# ==========================================
# IMAGE-JEPA (Masked Grid Modeling)
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=200, embed_dim=128):
        super().__init__()
        self.proj = nn.Linear(patch_size, embed_dim)
        
    def forward(self, x):
        return self.proj(x)

class EEGImageJEPAEncoder(nn.Module):
    def __init__(self, patch_size=200, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 190, embed_dim)) # 19 channels x 10 frames
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x, mask_indices=None):
        B = x.shape[0]
        # x is [B, 19, 2000] -> [B, 19, 10, 200] -> [B, 190, 200]
        x = x.view(B, 19, 10, 200).contiguous().view(B, 190, 200)
        
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        if mask_indices is not None:
            x_kept = []
            for i in range(B):
                x_kept.append(x[i, mask_indices[i]])
            x = torch.stack(x_kept, dim=0)
            
        return self.transformer(x)

class ImageJEPASSL(nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.embed_dim = cfg.get("out_dim", 128)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        predictor_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=2)
        
        self.ema_momentum = cfg.get("ema_momentum", 0.99)
        self.mask_ratio = cfg.get("mask_ratio", 0.6)

    @torch.no_grad()
    def _update_ema(self):
        for p_c, p_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(self.ema_momentum).add_(p_c.data, alpha=1.0 - self.ema_momentum)

    def compute_loss(self, batch):
        self._update_ema()
        # View 1 is target, View 2 is augmented context
        v1, v2 = batch
        B = v1.shape[0]
        num_patches = 190
        num_mask = int(self.mask_ratio * num_patches)
        num_keep = num_patches - num_mask
        
        noise = torch.rand(B, num_patches, device=v1.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        keep_indices = ids_shuffle[:, :num_keep]
        mask_indices = ids_shuffle[:, num_keep:]
        
        with torch.no_grad():
            target_reprs = self.target_encoder(v1) # [B, 190, D]
            
        context_reprs = self.context_encoder(v2, mask_indices=keep_indices) # [B, num_keep, D]
        
        mask_tokens = self.mask_token.repeat(B, num_mask, 1) # [B, num_mask, D]
        pos_embed = self.context_encoder.pos_embed.expand(B, -1, -1)
        pos_mask = torch.gather(pos_embed, 1, mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        mask_tokens = mask_tokens + pos_mask
        
        full_seq = torch.cat([context_reprs, mask_tokens], dim=1)
        preds_full = self.predictor(full_seq)
        preds_mask = preds_full[:, num_keep:]
        
        targets_mask = torch.gather(target_reprs, 1, mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        loss = F.mse_loss(preds_mask, targets_mask)
        logs = f"mse: {loss.item():.4f}"
        return loss, logs
