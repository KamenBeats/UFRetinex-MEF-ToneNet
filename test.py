"""
Test script for Retinex-MEF.

Usage:
    python test.py --test_dir dataset/test --model_phase_1 model/ckpt.pth
    python test.py --input scene1 scene2 --model_phase_1 model/ckpt.pth
    python test.py --test_dir dataset/test --model_phase_1 model/ckpt.pth --model_phase_2 tonenet.pth
    python test.py --test_dir dataset/test --model_phase_1 model/ckpt.pth --fp16
    python test.py --test_dir dataset/test --model_phase_1 model/ckpt.pth --align
"""

import os
import sys
import argparse
import time
import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.append(os.getcwd())
from nets.net import L_net, FusionNet, ToneNet, SRE, compute_mef_quality
from nets.restormer import Attention
from utils import image_read

# FusionNet FeedForward expands to 256 channels → 32-bit indexing limit
MAX_SAFE_PIXELS = (2**31 - 1) // 256


# ── Memory-optimized forward passes ──

def _attn_forward_opt(self, x, crossatt=None):
    b, c, h, w = x.shape
    k = self.k_dwconv(self.k(x))
    v = self.v_dwconv(self.v(x))
    q = self.q_dwconv(self.q(x if crossatt is None else crossatt))
    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * self.temperature
    attn = attn.softmax(dim=-1)
    out = attn @ v
    del q, k, attn
    out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                    head=self.num_heads, h=h, w=w)
    return self.project_out(out)


def _sre_forward_seq(self, images):
    B, N, C, H, W = images.shape
    feat_list = []
    for n in range(N):
        feat_list.append(self.encoder(self.patch_embed(images[:, n])))
    feats = torch.stack(feat_list, dim=1)
    del feat_list
    R_feat, attn_weights = self.aggregator(feats)
    del feats
    Rhat = self.act(self.output(R_feat))
    return Rhat, R_feat, attn_weights


def _fusion_forward_opt(self, images, L_maps=None, L_override=None):
    if L_override is not None:
        L_fused, L_weights = L_override, None
    else:
        assert L_maps is not None
        L_fused, L_weights = self.l_fusion(L_maps)

    Rhat_raw, _, _ = self.SRE(images)

    if self.use_retinex_prior:
        if L_maps is not None:
            quality = compute_mef_quality(L_maps)
            w = quality / (quality.sum(dim=1, keepdim=True) + 1e-6)
            # Exposure-compensate images before weighted sum to prevent
            # brightness transition halo at blown boundaries
            I_mean = images.mean(dim=[3, 4], keepdim=True).clamp(min=0.01)  # (B,N,3,1,1)
            target_I = (I_mean * w.mean(dim=[3, 4], keepdim=True)).sum(dim=1, keepdim=True) \
                       / (w.mean(dim=[3, 4], keepdim=True).sum(dim=1, keepdim=True) + 1e-6)
            images_comp = (images * (target_I / I_mean)).clamp(0, 1)
            retinex_prior = (images_comp * w).sum(dim=1).clamp(0, 1)
            del quality, w, images_comp
        else:
            mean_exposed = images.mean(dim=1)
            L_safe = L_fused.clamp(min=0.05)
            retinex_prior = (mean_exposed / L_safe).clamp(0, 1)
            del mean_exposed, L_safe
        Rhat = self.retinex_gate(Rhat_raw, retinex_prior)
        del Rhat_raw, retinex_prior
    else:
        Rhat = Rhat_raw
        del Rhat_raw

    L_for_crossatt = F.avg_pool2d(L_fused, kernel_size=21, stride=1, padding=10)
    R = self.patch_embed2(Rhat)
    for block in self.decoder:
        R = block(R, crossatt=L_for_crossatt)
    del L_for_crossatt
    R_refined = torch.clamp(Rhat + self.output2(R), 0, 1)
    del R
    output = R_refined * L_fused
    return output, R_refined, Rhat, L_fused, L_weights


def apply_memory_patches():
    Attention.forward = _attn_forward_opt
    SRE.forward = _sre_forward_seq
    FusionNet.forward = _fusion_forward_opt


# ── Auto-downscale for 32-bit safety ──

def compute_safe_size(H, W, max_side=0):
    """Return (new_H, new_W) if downscale needed, else None."""
    scale = 1.0
    if max_side > 0 and max(H, W) > max_side:
        scale = min(scale, max_side / max(H, W))
    if H * W > MAX_SAFE_PIXELS:
        scale = min(scale, (MAX_SAFE_PIXELS / (H * W)) ** 0.5)
    if scale >= 1.0:
        return None
    new_H = max(int(H * scale) // 8 * 8, 8)
    new_W = max(int(W * scale) // 8 * 8, 8)
    return (new_H, new_W)


def resize_tensor(tensor, max_size):
    """Resize so longer side <= max_size. Returns (resized, orig_size or None)."""
    if max_size <= 0:
        return tensor, None

    if tensor.dim() == 5:
        B, N, C, H, W = tensor.shape
        scale = min(max_size / H, max_size / W, 1.0)
        if scale >= 1.0:
            return tensor, None
        new_H, new_W = int(H * scale), int(W * scale)
        flat = tensor.reshape(B * N, C, H, W)
        flat = F.interpolate(
            flat, size=(new_H, new_W), mode='bilinear', align_corners=False)
        return flat.reshape(B, N, C, new_H, new_W), (H, W)
    else:
        B, C, H, W = tensor.shape
        scale = min(max_size / H, max_size / W, 1.0)
        if scale >= 1.0:
            return tensor, None
        new_H, new_W = int(H * scale), int(W * scale)
        return F.interpolate(
            tensor, size=(new_H, new_W), mode='bilinear',
            align_corners=False), (H, W)


def resize_to(tensor, size, mode='bilinear'):
    """Resize 4D or 5D tensor to exact (H, W)."""
    if tensor.dim() == 5:
        B, N, C, H, W = tensor.shape
        flat = tensor.reshape(B * N, C, H, W)
        flat = F.interpolate(flat, size=size, mode=mode, align_corners=False)
        return flat.reshape(B, N, C, *size)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


# ── Scene loading ──

def load_scene_images(scene_dir):
    """Load all images from a scene directory, sorted numerically. Returns (1, N, 3, H, W) on CPU."""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    paths = []
    for f in os.listdir(scene_dir):
        if f.lower().endswith(extensions) and not f.lower().startswith('gt'):
            paths.append(os.path.join(scene_dir, f))

    def _key(p):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(base)
        except ValueError:
            return base

    paths.sort(key=_key)
    if len(paths) == 0:
        raise ValueError(f"No images found in {scene_dir}")

    images = []
    for p in paths:
        img = image_read(p, 'RGB').transpose(2, 0, 1)[np.newaxis, ...] / 255.
        images.append(torch.FloatTensor(img))

    images = torch.cat(images, dim=0).unsqueeze(0)
    print(f"  Loaded {len(paths)} images from {scene_dir}")
    return images, paths


def _validate_homography(H_mat, W, H, max_shift_frac=0.25, max_scale=2.0):
    """
    Reject degenerate homographies that cause starburst/extreme distortion.
    Returns True if homography is safe to use.
    """
    if H_mat is None:
        return False

    # Check determinant (must be positive → no inversion/flip)
    det = np.linalg.det(H_mat)
    if det <= 0:
        return False

    # Check perspective component — the main cause of starburst
    # h7, h8 in [h7, h8, 1] should be near zero for near-planar alignment
    h7, h8 = H_mat[2, 0], H_mat[2, 1]
    persp_magnitude = np.sqrt(h7 ** 2 + h8 ** 2) * max(W, H)
    if persp_magnitude > max_shift_frac * max(W, H):
        return False

    # Check scale (top-left 2x2 submatrix scale)
    scale = np.sqrt(abs(np.linalg.det(H_mat[:2, :2])))
    if scale < 1.0 / max_scale or scale > max_scale:
        return False

    # Check corner displacement: each corner should not drift too far
    max_drift = max_shift_frac * max(W, H)
    corners = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]).reshape(-1, 1, 2)
    try:
        warped_corners = cv2.perspectiveTransform(corners, H_mat)
    except cv2.error:
        return False

    for (ox, oy), (wc,) in zip(corners[:, 0], warped_corners):
        wx, wy = wc
        if abs(wx - ox) > max_drift or abs(wy - oy) > max_drift:
            return False

    return True


def align_images_sift(images_tensor, ratio_thresh=0.65, ransac_thresh=4.0,
                      min_inliers=15, min_inlier_ratio=0.25):
    """
    Align all images to the smart-selected reference using SIFT + validated homography.
    images_tensor: (1, N, 3, H, W) float32 [0,1] on CPU
    Returns: aligned (1, N, 3, H, W) float32 [0,1] (unchanged on failure)
    """
    B, N, C, H, W = images_tensor.shape
    imgs_np = (images_tensor[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.04)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Pre-compute grayscale + keypoints for all exposures
    gray_list, kp_list, des_list = [], [], []
    for i in range(N):
        g = cv2.cvtColor(imgs_np[i], cv2.COLOR_RGB2GRAY)
        # Normalize exposure for more uniform feature detection across dark/bright images
        g = cv2.equalizeHist(g)
        gray_list.append(g)
        kp, des = sift.detectAndCompute(g, None)
        kp_list.append(kp)
        des_list.append(des)

    # Smart reference: pick exposure with most keypoints (most trackable)
    counts = [len(k) if k is not None else 0 for k in kp_list]
    max_c, min_c = max(counts), min(counts)
    if max_c - min_c < 50:
        ref_idx = N // 2  # All similar → use middle exposure
    else:
        ref_idx = int(np.argmax(counts))
    print(f"    [align] Reference: exposure {ref_idx + 1}/{N} "
          f"({counts[ref_idx]} keypoints, feature counts: {counts})")

    kp_ref = kp_list[ref_idx]
    des_ref = des_list[ref_idx]
    if des_ref is None or len(kp_ref) < min_inliers:
        print("    [align] Reference has too few keypoints — skipping alignment")
        return images_tensor

    aligned = list(imgs_np)

    for i in range(N):
        if i == ref_idx:
            continue
        des_src = des_list[i]
        kp_src = kp_list[i]
        if des_src is None or len(kp_src) < min_inliers:
            print(f"    [align] Image {i + 1}: too few keypoints ({len(kp_src) if kp_src else 0}), skipping")
            continue

        raw_matches = matcher.knnMatch(des_src, des_ref, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n2 = pair
                if m.distance < ratio_thresh * n2.distance:
                    good.append(m)

        if len(good) < min_inliers:
            print(f"    [align] Image {i + 1}: only {len(good)} good matches (need {min_inliers}), skipping")
            continue

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H_mat, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=0.999,
            maxIters=2000,
        )

        if mask is not None:
            n_inliers = int(mask.sum())
            inlier_ratio = n_inliers / len(good)
        else:
            n_inliers, inlier_ratio = 0, 0.0

        # Validate: reject degenerate homography (main fix for starburst)
        if not _validate_homography(H_mat, W, H):
            print(f"    [align] Image {i + 1}: homography invalid/degenerate, skipping "
                  f"({n_inliers} inliers, ratio={inlier_ratio:.2f})")
            continue

        if n_inliers < min_inliers or inlier_ratio < min_inlier_ratio:
            print(f"    [align] Image {i + 1}: too few inliers ({n_inliers}, ratio={inlier_ratio:.2f}), skipping")
            continue

        warped = cv2.warpPerspective(
            imgs_np[i], H_mat, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        aligned[i] = warped
        print(f"    [align] Image {i + 1} -> ref {ref_idx + 1}: "
              f"{len(good)} matches, {n_inliers} inliers ({inlier_ratio:.0%}) ✓")

    aligned_np = np.stack(aligned, axis=0)
    aligned_tensor = torch.from_numpy(aligned_np.astype(np.float32) / 255.0)
    aligned_tensor = aligned_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, N, 3, H, W)
    return aligned_tensor


def discover_scenes(root_dir):
    """Find all scene subdirectories containing images."""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    scenes = []
    for entry in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(scene_dir):
            continue
        has_images = any(
            f.lower().endswith(extensions) for f in os.listdir(scene_dir))
        if has_images:
            scenes.append(scene_dir)
    return scenes


def main():
    parser = argparse.ArgumentParser(description='Retinex-MEF: Test')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Directory of scene folders (e.g. dataset/test)')
    parser.add_argument('--input', nargs='+', default=None,
                        help='Specific scene folder(s)')
    parser.add_argument('--model_phase_1', type=str, default='model/phase_1/final.pth',
                        help='Path to Phase 1 checkpoint')
    parser.add_argument('--model_phase_2', type=str, default='model/phase_2/final_with_tonenet.pth',
                        help='Path to ToneNet checkpoint (Phase 2)')
    parser.add_argument('--output', type=str, default='test_case/results')
    parser.add_argument('--resize', type=int, default=0,
                        help='Resize longer side (0 = no resize)')
    parser.add_argument('--max_size', type=int, default=0,
                        help='Max long side for auto-downscale (0 = auto from 32-bit limit only)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 inference (reduces ~50%% VRAM)')
    parser.add_argument('--save_intermediates', action='store_true')
    parser.add_argument('--align', action='store_true',
                        help='Align exposures with SIFT + homography before fusion')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = args.fp16 and device == 'cuda'

    apply_memory_patches()

    # ── Load models ──
    net_fusion = FusionNet().to(device)
    net_L = L_net().to(device)

    print(f"Loading Phase 1 model from {args.model_phase_1}")
    ckpt = torch.load(args.model_phase_1, map_location=device, weights_only=False)

    if 'net_fusion' in ckpt:
        try:
            net_fusion.load_state_dict(ckpt['net_fusion'])
        except RuntimeError:
            print("[Info] Key mismatch — loading via load_pretrained()")
            net_fusion.load_pretrained(args.model_phase_1)
        if 'net_L' in ckpt:
            net_L.load_state_dict(ckpt['net_L'])
    else:
        net_fusion.load_pretrained(args.model_phase_1)
        if 'net_L' in ckpt:
            net_L.load_state_dict(ckpt['net_L'])

    net_fusion.eval()
    net_L.eval()

    if use_fp16:
        net_fusion.half()
        net_L.half()

    os.makedirs(args.output, exist_ok=True)

    # ── Load ToneNet if provided ──
    tone_net = None
    if args.model_phase_2:
        print(f"Loading ToneNet from {args.model_phase_2}")
        tn_ckpt = torch.load(args.model_phase_2, map_location=device,
                              weights_only=False)
        if 'tone_net' in tn_ckpt:
            tone_net = ToneNet().to(device)
            tone_net.load_state_dict(tn_ckpt['tone_net'])
            tone_net.eval()
            if use_fp16:
                tone_net.half()
            print(f"  ToneNet loaded ({sum(p.numel() for p in tone_net.parameters()):,} params)")
        else:
            print("  WARNING: 'tone_net' key not found in checkpoint")

    print(f"Device: {device}")
    if device == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    if use_fp16:
        print("FP16: enabled")
    print(f"Output: {args.output}")
    if args.resize > 0:
        print(f"Resize: longer side -> {args.resize}px")
    if args.align:
        print("Align: SIFT + homography (RANSAC)")
    print()

    # ── Determine scenes ──
    scene_dirs = []
    if args.test_dir:
        scene_dirs = discover_scenes(args.test_dir)
        print(f"Found {len(scene_dirs)} scenes in {args.test_dir}\n")
    elif args.input:
        scene_dirs = args.input

    if not scene_dirs:
        print("Error: Provide --test_dir or --input")
        sys.exit(1)

    # ── Inference ──
    with torch.no_grad():
        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir.rstrip('/\\'))
            print(f"Processing scene: {scene_name}")

            images, paths = load_scene_images(scene_dir)

            # SIFT alignment (--align)
            if args.align:
                print(f"  Aligning {images.shape[1]} images with SIFT...")
                images = align_images_sift(images)

            # Manual resize (--resize)
            images, resize_orig = resize_tensor(images, args.resize)

            # Auto-downscale for 32-bit safety
            B, N, C, H, W = images.shape
            safe_size = compute_safe_size(H, W, args.max_size)
            safe_orig = None
            if safe_size is not None:
                safe_orig = (H, W)
                images = resize_to(images, safe_size)
                _, _, _, H, W = images.shape
                print(f"  Auto-downscale: {safe_orig[0]}x{safe_orig[1]} -> {H}x{W}")

            print(f"  Input: {N} images @ {H}x{W}")
            images = images.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_fp16):
                L_list = [net_L(images[:, n]) for n in range(N)]
                L_maps = torch.stack(L_list, dim=1)
                del L_list

                t0 = time.time()
                output, R, Rhat, L_used, L_weights = net_fusion(
                    images, L_maps=L_maps)
                dt = time.time() - t0
                print(f"  Fusion: {dt:.2f}s")

                del images, L_maps
                torch.cuda.empty_cache()

                if tone_net is not None:
                    output, tn_params = tone_net(output)
                    print(f"  ToneNet: curve_A={tn_params['curve_A'][0].tolist()}")

            output = output.float()

            # Upscale back from auto-downscale
            if safe_orig is not None:
                output = F.interpolate(
                    output, size=safe_orig, mode='bicubic', align_corners=False).clamp(0, 1)
                Rhat = F.interpolate(Rhat, size=safe_orig, mode='bilinear', align_corners=False)
                L_used = F.interpolate(L_used, size=safe_orig, mode='bilinear', align_corners=False)
                print(f"  Upscale: -> {safe_orig[0]}x{safe_orig[1]}")

            # Upscale back from --resize
            if resize_orig is not None:
                output = F.interpolate(
                    output, size=resize_orig, mode='bicubic', align_corners=False).clamp(0, 1)
                Rhat = F.interpolate(Rhat, size=resize_orig, mode='bilinear', align_corners=False)
                L_used = F.interpolate(L_used, size=resize_orig, mode='bilinear', align_corners=False)

            # Save output
            imgout = np.uint8(
                np.squeeze(output.cpu().numpy()).transpose(1, 2, 0)
                .clip(0, 1) * 255)
            save_path = os.path.join(args.output, f"{scene_name}.png")
            cv2.imwrite(save_path, cv2.cvtColor(imgout, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {save_path}")

            if args.save_intermediates:
                final_H, final_W = output.shape[2], output.shape[3]
                rhat_out = np.uint8(
                    np.squeeze(Rhat.cpu().numpy()).transpose(1, 2, 0)
                    .clip(0, 1) * 255)
                cv2.imwrite(
                    os.path.join(args.output, f"{scene_name}_R.png"),
                    cv2.cvtColor(rhat_out, cv2.COLOR_RGB2BGR))

                L_out = np.uint8(
                    np.squeeze(L_used.cpu().numpy()).clip(0, 1) * 255)
                cv2.imwrite(
                    os.path.join(args.output, f"{scene_name}_L.png"),
                    L_out)

                if L_weights is not None:
                    for wi in range(L_weights.shape[1]):
                        w_out = np.uint8(
                            np.squeeze(L_weights[0, wi].cpu().numpy()
                                       ).clip(0, 1) * 255)
                        if safe_orig is not None or resize_orig is not None:
                            w_out = cv2.resize(
                                w_out, (final_W, final_H),
                                interpolation=cv2.INTER_LINEAR)
                        w_color = cv2.applyColorMap(w_out, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(args.output,
                                         f"{scene_name}_w{wi+1}.png"),
                            w_color)

            # Cleanup
            del output, Rhat, L_used, R, L_weights
            if device == 'cuda':
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  VRAM peak: {peak:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nDone! Results saved to: {args.output}")


if __name__ == '__main__':
    main()