import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.restormer import TransformerBlock

# ── Illumination Estimation ──

class L_net(nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(3, num, 3, 1, 0), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(num, num, 3, 1, 0), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(num, num, 3, 1, 0), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(num, num, 3, 1, 0), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

# ── Attention Aggregation (permutation-invariant over N images) ──

class AttentionAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1, bias=True),
        )

    def forward(self, features):
        # features: (B, N, C, H, W)
        B, N, C, H, W = features.shape
        x = features.reshape(B * N, C, H, W)
        scores = self.attention(x).reshape(B, N, 1, H, W)
        weights = F.softmax(scores, dim=1)
        aggregated = (features * weights).sum(dim=1)
        return aggregated, weights

# ── Shared Reflectance Estimator (flexible N-input) ──

class SRE(nn.Module):
    def __init__(self, out_channels=3, dim=32, heads=None,
                 ffn_expansion_factor=2, bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        if heads is None:
            heads = [8, 8, 8]
        self.patch_embed = nn.Conv2d(3, dim, 3, 1, 1, bias=bias)
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(2)
        ])
        self.aggregator = AttentionAggregation(dim)
        self.output = nn.Sequential(
            nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias),
        )
        self.act = nn.Sigmoid()

    def forward(self, images):
        # images: (B, N, 3, H, W)
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)
        feats = self.encoder(self.patch_embed(x)).reshape(B, N, -1, H, W)
        R_feat, attn_weights = self.aggregator(feats)
        Rhat = self.act(self.output(R_feat))
        return Rhat, R_feat, attn_weights

# ── Retinex Prior Gate (blend SRE output with quality-weighted prior) ──

class RetinexPriorGate(nn.Module):
    def __init__(self, channels=3, hidden=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, Rhat, retinex_prior):
        gate = self.gate_net(torch.cat([Rhat, retinex_prior], dim=1))
        return Rhat * (1 - gate) + retinex_prior * gate

# ── MEF Quality Weights ──

def compute_mef_quality(L_maps, sat_thresh=0.88, bloom_k=None, bloom_sigma=15.0):
    """
    Three-tier quality weight for MEF:
      1. EV-aware bell-curve: soft-normalize L by global mean before scoring
         so extreme dark/bright pairs are handled correctly.
      2. Sigmoid saturation penalty on raw L: detects blown-out pixels.
      3. Bloom proximity (adaptive kernel): suppresses regions near blown areas.
    """
    B, N, C, H, W = L_maps.shape

    # 1. EV-aware bell-curve
    L_mean = L_maps.mean(dim=[3, 4], keepdim=True).clamp(min=0.05)
    ev_scale = (0.40 / L_mean).pow(0.5)
    L_ev = (L_maps * ev_scale).clamp(0, 1)
    base_q = 1.0 - (2.0 * L_ev - 1.0).pow(2)

    # 2. Sigmoid saturation penalty
    sat_pen = torch.sigmoid(-15.0 * (L_maps - sat_thresh))

    # 2b. Dark noise penalty — suppresses very dark pixels (L < 0.04) with poor SNR
    dark_pen = torch.sigmoid(25.0 * (L_maps - 0.04))

    # 3. Bloom proximity (adaptive kernel size ~25% of shortest side, max 151)
    if bloom_k is None:
        bloom_k = max(41, min(H, W) // 4)
        bloom_k = min(bloom_k, 151)
        if bloom_k % 2 == 0:
            bloom_k += 1
    L_flat = L_maps.reshape(B * N, 1, H, W)
    blown_ind = torch.relu(L_flat - sat_thresh)
    pad = bloom_k // 2
    bloom_nbhd = F.avg_pool2d(blown_ind, kernel_size=bloom_k, stride=1, padding=pad)
    bloom_pen = torch.exp(-bloom_sigma * bloom_nbhd).reshape(B, N, 1, H, W)

    quality = (base_q * sat_pen * dark_pen * bloom_pen).clamp(min=0.02)
    return quality

# ── Laplacian Pyramid MEF for L maps (Mertens 2007) ──

class PyramidLFusion(nn.Module):
    """
    Fuse N illumination maps via Laplacian Pyramid MEF.

    Weighted-sum blending causes visible transition rings at region boundaries
    because weights change abruptly between blown/clear areas.
    Pyramid blending processes each frequency band independently with
    Gaussian-smoothed quality weights, eliminating transition artifacts.

    No learnable parameters. Uses quality from compute_mef_quality().
    """
    def __init__(self, n_levels=5, hidden_dim=None):  # hidden_dim kept for API compatibility
        super().__init__()
        self.n_levels = n_levels

    @staticmethod
    def _blur(x):
        return F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)

    @staticmethod
    def _down(x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    @staticmethod
    def _up(x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

    def _gauss_pyr(self, x):
        pyr, cur = [x], x
        for _ in range(self.n_levels - 1):
            cur = self._down(self._blur(cur))
            pyr.append(cur)
        return pyr

    def _lap_pyr(self, x):
        gp = self._gauss_pyr(x)
        pyr = []
        for i in range(self.n_levels - 1):
            up = self._up(gp[i + 1], gp[i].shape[-2], gp[i].shape[-1])
            pyr.append(gp[i] - up)
        pyr.append(gp[-1])
        return pyr

    def _collapse(self, pyr):
        img = pyr[-1]
        for i in range(self.n_levels - 2, -1, -1):
            img = self._up(img, pyr[i].shape[-2], pyr[i].shape[-1]) + pyr[i]
        return img

    def forward(self, L_maps):
        # L_maps: (B, N, 1, H, W)
        B, N, C, H, W = L_maps.shape

        quality = compute_mef_quality(L_maps)  # (B, N, 1, H, W)
        w = quality / (quality.sum(dim=1, keepdim=True) + 1e-6)

        L_flat = L_maps.reshape(B * N, 1, H, W)
        w_flat = w.reshape(B * N, 1, H, W)

        L_pyrs = self._lap_pyr(L_flat)
        w_pyrs = self._gauss_pyr(w_flat)

        fused_pyr = []
        for lvl in range(self.n_levels):
            H2, W2 = L_pyrs[lvl].shape[-2:]
            L_l = L_pyrs[lvl].reshape(B, N, 1, H2, W2)
            w_l = w_pyrs[lvl].reshape(B, N, 1, H2, W2)
            # Re-normalize: Gaussian blur alters the sum of weights per level
            w_l = w_l / (w_l.sum(dim=1, keepdim=True) + 1e-6)
            fused_pyr.append((L_l * w_l).sum(dim=1))        # (B, 1, H', W')

        L_fused = self._collapse(fused_pyr).clamp(0, 1)     # (B, 1, H, W)
        return L_fused, w

# ── Statistics Pooling (mean + std) ──

class StatisticsPooling(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3]).clamp(min=1e-6)
        return torch.cat([mean, std], dim=1)

# ── ToneNet (global tone/colour correction) ──

class ToneNet(nn.Module):
    def __init__(self, base_dim=32, n_curve_iters=8):
        super().__init__()
        self.n_curve_iters = n_curve_iters
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_dim, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool = StatisticsPooling()
        pool_dim = base_dim * 4 * 2

        self.fc = nn.Sequential(
            nn.Linear(pool_dim + 1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 13),  # 3 curve params + 9 CCM + 1 brightness
        )
        self._init_identity()

    def _init_identity(self):
        nn.init.zeros_(self.fc[-1].weight)
        self.fc[-1].bias.data.copy_(torch.tensor([
            0., 0., 0.,
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.,
            0.,
        ]))

    def forward(self, I_rec):
        B = I_rec.shape[0]

        input_lum = (0.299*I_rec[:,0] + 0.587*I_rec[:,1] + 0.114*I_rec[:,2])
        input_lum_mean = input_lum.mean(dim=[1,2])  # (B,)

        feat = self.pool(self.encoder(I_rec))
        feat = torch.cat([feat, input_lum_mean.unsqueeze(1)], dim=1)
        raw = self.fc(feat)

        curve_A = torch.tanh(raw[:, :3])
        ccm = raw[:, 3:12].reshape(B, 3, 3)
        brightness = torch.exp(raw[:, 12]).clamp(0.1, 10.0)

        x = (I_rec * brightness.view(B,1,1,1)).clamp(0, 1)
        A_4d = curve_A.unsqueeze(-1).unsqueeze(-1)
        for _ in range(self.n_curve_iters):
            x = (x + A_4d * x * (1.0 - x)).clamp(0, 1)

        px = x.reshape(B, 3, -1)
        I_corrected = torch.bmm(ccm, px).reshape(B, 3, *x.shape[2:]).clamp(0, 1)

        return I_corrected, {
            'curve_A':    curve_A,
            'ccm':        ccm,
            'brightness': brightness,
        }

# ── Fusion Network (flexible N-input Retinex) ──

class FusionNet(nn.Module):
    def __init__(self, out_channels=3, dim=64, num_blocks=3,
                 heads=None, ffn_expansion_factor=2, bias=False,
                 LayerNorm_type='WithBias', sre_dim=32,
                 use_retinex_prior=True, l_fusion_hidden=16):
        super().__init__()
        if heads is None:
            heads = [8, 8, 8]
        self.use_retinex_prior = use_retinex_prior

        self.SRE = SRE(dim=sre_dim, heads=heads,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias, LayerNorm_type=LayerNorm_type)

        if use_retinex_prior:
            self.retinex_gate = RetinexPriorGate(channels=out_channels)

        self.l_fusion = PyramidLFusion(n_levels=5)

        self.patch_embed2 = nn.Conv2d(3, dim, 3, 1, 1, bias=bias)
        self.decoder = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             crossatt=True)
            for _ in range(2 * num_blocks)
        ])
        self.output2 = nn.Sequential(
            nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias),
        )

    def forward(self, images, L_maps=None, L_override=None):
        """
        Args:
            images:     (B, N, 3, H, W)
            L_maps:     (B, N, 1, H, W) — used for L fusion and retinex_prior
            L_override: (B, 1, H, W)    — override L output (cross-eval mode)
                        If both provided: L_override for output,
                        L_maps for quality-weighted retinex_prior.
        Returns:
            output, R_refined, Rhat, L_fused, L_weights
        """
        if L_override is not None:
            L_fused, L_weights = L_override, None
        else:
            assert L_maps is not None
            L_fused, L_weights = self.l_fusion(L_maps)

        Rhat_raw, _, _ = self.SRE(images)

        if self.use_retinex_prior:
            if L_maps is not None:
                # Quality-weighted mean image as retinex_prior
                quality = compute_mef_quality(L_maps)
                w = quality / (quality.sum(dim=1, keepdim=True) + 1e-6)
                retinex_prior = (images * w).sum(dim=1).clamp(0, 1)  # (B, 3, H, W)
            else:
                mean_exposed = images.mean(dim=1)
                L_safe = L_fused.clamp(min=0.05)
                retinex_prior = (mean_exposed / L_safe).clamp(0, 1)
            Rhat = self.retinex_gate(Rhat_raw, retinex_prior)
        else:
            Rhat = Rhat_raw

        # Use a smoothed L for cross-attention to prevent L_fused artifacts
        # from propagating through attention (causing banding in R_refined).
        # The final output still uses the original L_fused for correct brightness.
        L_for_crossatt = F.avg_pool2d(L_fused, kernel_size=21, stride=1, padding=10)

        R = self.patch_embed2(Rhat)
        for block in self.decoder:
            R = block(R, crossatt=L_for_crossatt)
        R_refined = torch.clamp(Rhat + self.output2(R), 0, 1)
        output = R_refined * L_fused

        return output, R_refined, Rhat, L_fused, L_weights

    def load_pretrained(self, ckpt_path):
        """Load weights from original 2-input model or flexible checkpoint."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        old_state = ckpt.get('net_fusion', ckpt)
        new_state = self.state_dict()
        loaded, skipped = [], []

        for name, param in old_state.items():
            new_name = name
            if 'SRE.patch_embed1' in name:
                new_name = name.replace('patch_embed1', 'patch_embed')
                if 'weight' in name and param.shape[1] == 6:
                    param = (param[:, :3] + param[:, 3:]) / 2
                if new_name in new_state and param.shape == new_state[new_name].shape:
                    new_state[new_name] = param
                    loaded.append(f"{name} -> {new_name}")
                else:
                    skipped.append(f"{name} -> {new_name} (mismatch)")
                continue
            if 'SRE.output1.' in name:
                new_name = name.replace('SRE.output1.', 'SRE.output.')
            if new_name in new_state and param.shape == new_state[new_name].shape:
                new_state[new_name] = param
                loaded.append(f"{name} -> {new_name}")
            else:
                skipped.append(f"{name} (skip)")

        self.load_state_dict(new_state, strict=False)
        print(f"[Pretrained] Loaded {len(loaded)}, skipped {len(skipped)}")
        for s in skipped:
            print(f"  SKIP: {s}")
        return loaded, skipped