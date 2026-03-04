"""
Test script for Retinex-MEF.

Usage:
    python test.py --test_dir dataset/test --model_phase_1 model/ckpt.pth
    python test.py --input scene1 scene2 --model_phase_1 model/ckpt.pth
    python test.py --test_dir dataset/test --model_phase_1 model/ckpt.pth --model_phase_2 tonenet.pth
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
from nets.net import L_net, FusionNet, ToneNet
from utils import image_read


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
        flat = torch.nn.functional.interpolate(
            flat, size=(new_H, new_W), mode='bilinear', align_corners=False)
        return flat.reshape(B, N, C, new_H, new_W), (H, W)
    else:
        B, C, H, W = tensor.shape
        scale = min(max_size / H, max_size / W, 1.0)
        if scale >= 1.0:
            return tensor, None
        new_H, new_W = int(H * scale), int(W * scale)
        return torch.nn.functional.interpolate(
            tensor, size=(new_H, new_W), mode='bilinear',
            align_corners=False), (H, W)


def load_scene_images(scene_dir, device):
    """Load all images from a scene directory, sorted numerically. Returns (1, N, 3, H, W)."""
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
        images.append(torch.FloatTensor(img).to(device))

    images = torch.cat(images, dim=0).unsqueeze(0)
    print(f"  Loaded {len(paths)} images from {scene_dir}")
    return images, paths


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
    parser.add_argument('--save_intermediates', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            print(f"  ToneNet loaded ({sum(p.numel() for p in tone_net.parameters()):,} params)")
        else:
            print("  WARNING: 'tone_net' key not found in checkpoint")

    print(f"Device: {device}")
    print(f"Output: {args.output}")
    if args.resize > 0:
        print(f"Resize: longer side -> {args.resize}px")
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

            images, paths = load_scene_images(scene_dir, device)
            images, orig_size = resize_tensor(images, args.resize)
            B, N, C, H, W = images.shape
            print(f"  Input: {N} images @ {H}x{W}")

            L_list = [net_L(images[:, n]) for n in range(N)]
            L_maps = torch.stack(L_list, dim=1)

            t0 = time.time()
            output, R, Rhat, L_used, L_weights = net_fusion(
                images, L_maps=L_maps)
            dt = time.time() - t0
            print(f"  Fusion: {dt:.2f}s")

            if tone_net is not None:
                output, tn_params = tone_net(output)
                print(f"  ToneNet: curve_A={tn_params['curve_A'][0].tolist()}")

            # Resize back
            if orig_size is not None:
                output = torch.nn.functional.interpolate(
                    output, size=orig_size, mode='bilinear', align_corners=False)
                Rhat = torch.nn.functional.interpolate(
                    Rhat, size=orig_size, mode='bilinear', align_corners=False)
                L_used = torch.nn.functional.interpolate(
                    L_used, size=orig_size, mode='bilinear', align_corners=False)

            # Save output
            imgout = np.uint8(
                np.squeeze(output.cpu().numpy()).transpose(1, 2, 0)
                .clip(0, 1) * 255)
            save_path = os.path.join(args.output, f"{scene_name}.png")
            cv2.imwrite(save_path, cv2.cvtColor(imgout, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {save_path}")

            if args.save_intermediates:
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
                        if orig_size is not None:
                            w_out = cv2.resize(
                                w_out, (orig_size[1], orig_size[0]),
                                interpolation=cv2.INTER_LINEAR)
                        w_color = cv2.applyColorMap(w_out, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(args.output,
                                         f"{scene_name}_w{wi+1}.png"),
                            w_color)

    print(f"\nDone! Results saved to: {args.output}")


if __name__ == '__main__':
    main()