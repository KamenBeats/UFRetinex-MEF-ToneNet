import math

import torch
import torch.nn.functional as F
import cv2
import numpy as np

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ('RGB', 'Gray', 'YCrCb'), f'Unknown mode: {mode}'
    if mode == 'RGB':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'Gray':
        return np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

def convert2gray(input_tensor):
    if input_tensor.size(1) == 3:
        r = input_tensor[:, 0:1, :, :]
        g = input_tensor[:, 1:2, :, :]
        b = input_tensor[:, 2:3, :, :]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b
    return input_tensor

def illu_smooth(illu, img, c=10):
    """Structure-guided illumination smoothness loss."""
    img = convert2gray(img)
    grad_y_illu = torch.abs(illu[:, :, 1:, :] - illu[:, :, :-1, :])
    grad_x_illu = torch.abs(illu[:, :, :, 1:] - illu[:, :, :, :-1])
    grad_y_img = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    grad_x_img = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    eps = torch.tensor([0.01]).to(illu.device)
    x_tv = grad_x_illu / torch.max(eps, torch.max(torch.abs(grad_x_img)))
    y_tv = grad_y_illu / torch.max(eps, torch.max(torch.abs(grad_y_img)))
    return torch.mean(x_tv) + torch.mean(y_tv)

def gradient_loss(output, target):
    """L1 loss on image gradients to penalize over-smoothing."""
    out_dx = torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])
    out_dy = torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])
    tgt_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    tgt_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    return F.l1_loss(out_dx, tgt_dx) + F.l1_loss(out_dy, tgt_dy)

# ── Phase 2 losses ──

def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def ssim_loss(img1, img2, window_size=11, sigma=1.5, size_average=True,
              data_range=1.0, K=(0.01, 0.03)):
    """Differentiable SSIM loss: returns 1 - SSIM(img1, img2)."""
    C1 = (K[0] * data_range) ** 2
    C2 = (K[1] * data_range) ** 2

    g = _fspecial_gauss_1d(window_size, sigma)
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.size(1), -1, -1, -1).contiguous().to(img1.device)

    pad = window_size // 2
    C = img1.size(1)

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1.0 - ssim_map.mean()
    return 1.0 - ssim_map.mean(dim=[1, 2, 3])

def psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(data_range ** 2 / mse.item())

def histogram_loss(output, target):
    """Sort-based Wasserstein-1 histogram matching loss per channel."""
    B, C = output.shape[:2]
    loss = 0.0
    for c in range(C):
        out_sorted = torch.sort(output[:, c].reshape(B, -1), dim=-1)[0]
        tgt_sorted = torch.sort(target[:, c].reshape(B, -1), dim=-1)[0]
        loss = loss + F.l1_loss(out_sorted, tgt_sorted)
    return loss / C

def style_loss(output, target):
    """Multi-scale Gram-matrix style loss (scales 1x, 2x, 4x)."""
    loss = 0.0
    for scale in [1, 2, 4]:
        out_s = F.avg_pool2d(output, scale) if scale > 1 else output
        tgt_s = F.avg_pool2d(target, scale) if scale > 1 else target
        B, C, H, W = out_s.shape
        out_feat = out_s.reshape(B, C, -1)
        tgt_feat = tgt_s.reshape(B, C, -1)
        out_gram = torch.bmm(out_feat, out_feat.transpose(1, 2)) / (C * H * W)
        tgt_gram = torch.bmm(tgt_feat, tgt_feat.transpose(1, 2)) / (C * H * W)
        loss = loss + F.mse_loss(out_gram, tgt_gram)
    return loss / 3.0

def lut_smoothness_loss(lut):
    """
    Total variation smoothness for a 3D LUT.
    lut shape: (B, 3, D, D, D)
    """
    tv_d = torch.mean((lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :]) ** 2)
    tv_h = torch.mean((lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :]) ** 2)
    tv_w = torch.mean((lut[:, :, :, :, 1:] - lut[:, :, :, :, :-1]) ** 2)
    return tv_d + tv_h + tv_w

def color_angle_loss(output, target):
    """
    Cosine angle loss between RGB color vectors.
    Effective at suppressing color cast in reconstructed bright regions.
    """
    eps = 1e-6
    out_norm = output / (torch.norm(output, p=2, dim=1, keepdim=True) + eps)
    tgt_norm = target / (torch.norm(target, p=2, dim=1, keepdim=True) + eps)
    cos_sim = torch.sum(out_norm * tgt_norm, dim=1)
    return (1.0 - cos_sim).mean()

# ── Phase 1: Color & Chrominance losses ──

def chrominance_consistency_loss(R, L_list, images):
    """
    Enforce consistent chrominance of R across all exposures.
    For each exposure: R * L_n should match the chrominance of I_n.
    Only evaluated in well-exposed regions (avoids noisy dark/blown pixel artifacts).
    """
    eps = 1e-6
    B, N = images.shape[:2]
    loss = 0.0
    count = 0

    for n in range(N):
        img_n = images[:, n]                  # (B, 3, H, W)
        reconstructed = R * L_list[n]         # (B, 3, H, W)

        img_sum = img_n.sum(dim=1, keepdim=True).clamp(min=eps)
        rec_sum = reconstructed.sum(dim=1, keepdim=True).clamp(min=eps)
        chroma_img = img_n / img_sum
        chroma_rec = reconstructed / rec_sum

        luminance = img_n.mean(dim=1, keepdim=True)
        weight = (luminance > 0.05).float() * (luminance < 0.95).float()
        weight_sum = weight.sum().clamp(min=1.0)

        loss += (weight * (chroma_img - chroma_rec).abs()).sum() / weight_sum
        count += 1

    return loss / max(count, 1)

def luminance_mean_loss(output, target):
    """
    Match global mean luminance (misalignment-safe, per-image comparison).
    Primary loss for teaching ToneNet how much brightness adjustment is needed.
    """
    out_lum = (0.299*output[:,0] + 0.587*output[:,1] + 0.114*output[:,2])
    tgt_lum = (0.299*target[:,0] + 0.587*target[:,1] + 0.114*target[:,2])
    return F.l1_loss(out_lum.mean(dim=[1,2]), tgt_lum.mean(dim=[1,2]))

def lum_histogram_loss(output, target):
    """
    Wasserstein-1 loss on luminance histogram.
    Matches the full brightness distribution (shadows/midtones/highlights).
    Misalignment-safe: sort-based, requires no pixel correspondence.
    """
    out_lum = (0.299*output[:,0] + 0.587*output[:,1] + 0.114*output[:,2])
    tgt_lum = (0.299*target[:,0] + 0.587*target[:,1] + 0.114*target[:,2])
    B = output.shape[0]
    out_s = torch.sort(out_lum.reshape(B, -1), dim=1)[0]
    tgt_s = torch.sort(tgt_lum.reshape(B, -1), dim=1)[0]
    return F.l1_loss(out_s, tgt_s)

def structure_loss(output, i_rec):
    """Gradient-based structure preservation: output should retain edges of i_rec."""
    out_dx = output[:,:,:,1:] - output[:,:,:,:-1]
    out_dy = output[:,:,1:,:] - output[:,:,:-1,:]
    rec_dx = i_rec[:,:,:,1:]  - i_rec[:,:,:,:-1]
    rec_dy = i_rec[:,:,1:,:]  - i_rec[:,:,:-1,:]
    return F.l1_loss(out_dx, rec_dx) + F.l1_loss(out_dy, rec_dy)

def conditional_hue_loss(output, target, i_rec, threshold=0.15):
    """
    Penalize global hue deviation only when the input has a significant color cast.
    The cast magnitude is measured from i_rec vs. neutral gray.
    Below the threshold, the loss weight fades to zero automatically.
    """
    eps = 1e-6

    rec_mean = i_rec.mean(dim=[2, 3])  # (B, 3)
    rec_lum = rec_mean.mean(dim=1, keepdim=True)  # (B, 1)
    rec_chroma = rec_mean / (rec_lum + eps)  # (B, 3) normalized

    neutral = torch.ones_like(rec_chroma)
    color_cast = (rec_chroma - neutral).norm(dim=1)  # (B,)

    weight = (color_cast - threshold).clamp(min=0) / (1.0 - threshold + eps)
    weight = weight.clamp(0, 1)  # (B,)

    out_mean = output.mean(dim=[2, 3])
    tgt_mean = target.mean(dim=[2, 3])
    out_norm = out_mean / (out_mean.norm(dim=1, keepdim=True) + eps)
    tgt_norm = tgt_mean / (tgt_mean.norm(dim=1, keepdim=True) + eps)
    hue_error = 1.0 - (out_norm * tgt_norm).sum(dim=1)  # (B,)

    return (weight * hue_error).mean()