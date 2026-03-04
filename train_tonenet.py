import os
import sys
import time
import random
import datetime
import argparse

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nets.net import L_net, FusionNet, ToneNet
from utils import histogram_loss, style_loss, psnr, conditional_hue_loss, luminance_mean_loss, lum_histogram_loss, structure_loss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

@torch.no_grad()
def validate(tone_net, net_fusion, net_L, val_loader, device, w_hist, w_style, w_struct, w_hue, w_lum_mean, w_lum_hist):
    """Evaluate ToneNet on validation set. Returns dict with psnr and all losses."""
    tone_net.eval()

    psnr_list = []
    loss_hist_list = []
    loss_style_list = []
    loss_struct_list = []
    loss_hue_list = []
    loss_lum_mean_list = []
    loss_lum_hist_list = []

    for images, gts, indices in val_loader:
        images = images.to(device)
        gts = gts.to(device)
        B, N, C, H, W = images.shape

        all_L_list = [net_L(images[:, n]) for n in range(N)]
        L_maps = torch.stack(all_L_list, dim=1)
        I_rec, _, _, L_fused, _ = net_fusion(images, L_maps=L_maps)

        I_corrected, _ = tone_net(I_rec)

        for b in range(B):
            psnr_list.append(psnr(I_corrected[b:b+1], gts[b:b+1]))

        loss_hist_list.append(histogram_loss(I_corrected, gts).item())
        loss_style_list.append(style_loss(I_corrected, gts).item())
        loss_struct_list.append(structure_loss(I_corrected, I_rec).item())
        loss_hue_list.append(conditional_hue_loss(I_corrected, gts, I_rec.detach()).item())
        loss_lum_mean_list.append(luminance_mean_loss(I_corrected, gts).item())
        loss_lum_hist_list.append(lum_histogram_loss(I_corrected, gts).item())

    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_hist = np.mean(loss_hist_list) if loss_hist_list else 0.0
    avg_style = np.mean(loss_style_list) if loss_style_list else 0.0
    avg_struct = np.mean(loss_struct_list) if loss_struct_list else 0.0
    avg_hue = np.mean(loss_hue_list) if loss_hue_list else 0.0
    avg_lum_mean = np.mean(loss_lum_mean_list) if loss_lum_mean_list else 0.0
    avg_lum_hist = np.mean(loss_lum_hist_list) if loss_lum_hist_list else 0.0

    return {
        'psnr': avg_psnr,
        'loss_histogram': avg_hist,
        'loss_style': avg_style,
        'loss_struct': avg_struct,
        'loss_hue': avg_hue,
        'loss_lum_mean': avg_lum_mean,
        'loss_lum_hist': avg_lum_hist,
        'loss_total': (w_hist*avg_hist + w_style*avg_style + w_struct*avg_struct
                       + w_hue*avg_hue + w_lum_mean*avg_lum_mean + w_lum_hist*avg_lum_hist),
    }

def train_tonenet():
    parser = argparse.ArgumentParser(description='Phase 2: Train ToneNet')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--phase1_ckpt', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)

    cfg = load_config(args.config)
    mcfg = cfg['model']
    dcfg = cfg['data']
    tcfg_base = cfg['training']
    tcfg = cfg.get('tonenet', {})

    # ── Build Retinex models (frozen) ──
    net_fusion = FusionNet(
        dim=mcfg['decoder_dim'],
        num_blocks=mcfg['num_blocks'],
        heads=[mcfg['num_heads']] * 3,
        ffn_expansion_factor=mcfg['ffn_expansion_factor'],
        sre_dim=mcfg['sre_dim'],
        use_retinex_prior=mcfg.get('use_retinex_prior', True),
        l_fusion_hidden=mcfg.get('l_fusion_hidden', 16),
    ).to(device)
    net_L = L_net().to(device)

    print(f"\n[Phase2] Loading Retinex from: {args.phase1_ckpt}")
    ckpt = torch.load(args.phase1_ckpt, map_location='cpu', weights_only=False)
    net_fusion.load_state_dict(ckpt['net_fusion'])
    if 'net_L' in ckpt:
        net_L.load_state_dict(ckpt['net_L'])
    print("[Phase2] Retinex loaded. Freezing.")

    freeze_model(net_fusion)
    freeze_model(net_L)

    # ── Build ToneNet (trainable) ──
    tone_net = ToneNet(
        base_dim=tcfg.get('base_dim', 32),
        n_curve_iters=tcfg.get('n_curve_iters', 8),
    ).to(device)

    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"[Phase2] Resuming ToneNet from: {args.resume}")
        tn_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        tone_net.load_state_dict(tn_ckpt['tone_net'])
        start_epoch = tn_ckpt.get('epoch', 0)
        best_psnr = tn_ckpt.get('best_psnr', 0.0)

    # ── Datasets ──
    from data.dataset import MultiExposureGTDataset, create_gt_collate_fn

    train_dataset = MultiExposureGTDataset(
        data_dir=dcfg['train_dir'],
        gt_dir=dcfg.get('gt_dir', None),
        patch_size=dcfg.get('patch_size', 0),
        augment=dcfg.get('augment', True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg.get('batch_size', tcfg_base['batch_size']),
        shuffle=True,
        num_workers=dcfg.get('num_workers', 0),
        collate_fn=create_gt_collate_fn(n_max=dcfg['n_max']),
        drop_last=True,
    )

    if len(train_dataset) == 0:
        print("[Phase2] ERROR: No scenes with GT found! Check data_dir "
              "and gt_dir in config.")
        sys.exit(1)

    val_dir = dcfg.get('val_dir', None)
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = MultiExposureGTDataset(
            data_dir=val_dir,
            gt_dir=dcfg.get('gt_dir', None),
            patch_size=0,
            augment=False,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=tcfg.get('batch_size', tcfg_base['batch_size']),
                shuffle=False,
                num_workers=dcfg.get('num_workers', 0),
                collate_fn=create_gt_collate_fn(n_max=dcfg['n_max']),
                drop_last=False,
            )
            print(f"[Phase2] Validation: {len(val_dataset)} scenes")
        else:
            print(f"[Phase2] WARNING: val_dir '{val_dir}' has no GT scenes. "
                  f"Validation disabled.")
    else:
        print(f"[Phase2] WARNING: No val_dir configured or not found. "
              f"Validation disabled — best model will be saved by loss.")

    # ── Optimizer & Scheduler ──
    lr = tcfg.get('lr', 5e-4)
    num_epochs = tcfg.get('epochs', 30)
    optimizer = torch.optim.Adam(
        tone_net.parameters(), lr=lr,
        weight_decay=tcfg.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)

    if args.resume and os.path.exists(args.resume):
        tn_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if 'optimizer' in tn_ckpt:
            optimizer.load_state_dict(tn_ckpt['optimizer'])
        if 'scheduler' in tn_ckpt:
            scheduler.load_state_dict(tn_ckpt['scheduler'])

    # Loss weights
    w_hist = tcfg.get('w_histogram', 3.0)
    w_style = tcfg.get('w_style', 0.5)
    w_struct = tcfg.get('w_structure', 1.0)
    w_hue = tcfg.get('w_global_hue', 1.0)
    w_lum_mean = tcfg.get('w_lum_mean', 8.0)
    w_lum_hist = tcfg.get('w_lum_hist', 5.0)

    val_every = tcfg.get('val_every', 1)

    # ── Experiment directory ──
    exp_path = os.path.join(
        "exp", "tonenet_" + time.strftime("%m_%d_%H_%M", time.localtime()))
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)

    cfg['tonenet_training'] = {
        'phase1_ckpt': args.phase1_ckpt,
        'lr': lr, 'epochs': num_epochs,
        'w_histogram': w_hist, 'w_style': w_style,
        'n_curve_iters': tcfg.get('n_curve_iters', 8),
    }
    with open(os.path.join(exp_path, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Wandb ──
    wcfg = cfg.get('wandb', {})
    use_wandb = (
        WANDB_AVAILABLE and not args.no_wandb
        and wcfg.get('enabled', True)
    )
    if use_wandb:
        exp_name = os.path.basename(exp_path)
        wandb_project = (args.wandb_project
                         or wcfg.get('project', 'UFRetinex-MEF'))
        wandb_run_name = (args.wandb_run_name
                          or f"ToneNet-{exp_name}")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=cfg,
            tags=['phase2', 'tonenet'],
            dir=exp_path,
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.watch(tone_net, log='gradients', log_freq=100)

    # ── Print summary ──
    print(f"\n{'=' * 65}")
    print(f"  Phase 2: ToneNet Training")
    print(f"  Retinex frozen ({count_params(net_fusion):,} + "
          f"{count_params(net_L):,} params)")
    print(f"  ToneNet trainable: {count_params(tone_net):,} params")
    print(f"  Epochs: {start_epoch+1} → {num_epochs}, LR: {lr}")
    print(f"  Loss: w_hist={w_hist}, w_style={w_style}")
    print(f"  Train scenes: {len(train_dataset)}")
    val_info = (f"{len(val_loader.dataset)} scenes"
                if val_loader else "DISABLED")
    print(f"  Val scenes:   {val_info}")
    print(f"  Best model:   {'PSNR on val set' if val_loader else 'total loss (fallback)'}")
    print(f"  Device: {device}")
    print(f"  Exp: {exp_path}")
    print(f"{'=' * 65}\n")

    # ── Training loop ──
    prev_time = time.time()
    save_every = tcfg.get('save_every', 5)
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        tone_net.train()

        loss_acc = {k: [] for k in
                    ['total', 'histogram', 'style', 'struct', 'hue', 'lum_mean', 'lum_histogram']}

        for i, (images, gts, indices) in enumerate(train_loader):
            images = images.to(device)
            gts = gts.to(device)
            B, N, C, H, W = images.shape

            optimizer.zero_grad()

            with torch.no_grad():
                all_L_list = [net_L(images[:, n]) for n in range(N)]
                L_maps = torch.stack(all_L_list, dim=1)
                I_rec, _, _, L_fused, _ = net_fusion(
                    images, L_maps=L_maps)
                I_rec = I_rec.detach()

            I_corrected, params = tone_net(I_rec)

            loss_hist = histogram_loss(I_corrected, gts)
            loss_sty = style_loss(I_corrected, gts)
            loss_struct = structure_loss(I_corrected, I_rec.detach())
            loss_hue = conditional_hue_loss(I_corrected, gts, I_rec.detach())
            loss_lum_mean = luminance_mean_loss(I_corrected, gts)
            loss_lum_hist = lum_histogram_loss(I_corrected, gts)

            loss_total = (w_hist * loss_hist
                        + w_style * loss_sty
                        + w_struct * loss_struct
                        + w_hue * loss_hue
                        + w_lum_mean * loss_lum_mean
                        + w_lum_hist * loss_lum_hist)

            loss_total.backward()
            optimizer.step()

            loss_acc['total'].append(loss_total.item())
            loss_acc['histogram'].append(loss_hist.item())
            loss_acc['style'].append(loss_sty.item())
            loss_acc['struct'].append(loss_struct.item())
            loss_acc['hue'].append(loss_hue.item())
            loss_acc['lum_mean'].append(loss_lum_mean.item())
            loss_acc['lum_histogram'].append(loss_lum_hist.item())

            batches_done = epoch * len(train_loader) + i
            batches_left = num_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if (i + 1) % 20 == 0 or i == 0:
                print(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"[Batch {i+1}/{len(train_loader)}] "
                    f"[total: {np.mean(loss_acc['total']):.4f}] "
                    f"[hist: {np.mean(loss_acc['histogram']):.4f}] "
                    f"[style: {np.mean(loss_acc['style']):.4f}] "
                    f"[struct: {np.mean(loss_acc['struct']):.4f}] "
                    f"[hue: {np.mean(loss_acc['hue']):.4f}] "
                    f"[lum_mean: {np.mean(loss_acc['lum_mean']):.4f}] "
                    f"[lum_hist: {np.mean(loss_acc['lum_histogram']):.4f}] "
                    f"[curve_A: {params['curve_A'][0].tolist()}] "
                    f"ETA: {str(time_left)[:10]}"
                )

            if use_wandb:
                wandb.log({
                    'train/step': batches_done,
                    'train/loss_total': loss_total.item(),
                    'train/loss_histogram': loss_hist.item(),
                    'train/loss_style': loss_sty.item(),
                    'train/loss_struct': loss_struct.item(),
                    'train/loss_hue': loss_hue.item(),
                    'train/brightness_mean': params['brightness'].mean().item(),
                    'train/brightness_min':  params['brightness'].min().item(),
                    'train/brightness_max':  params['brightness'].max().item(),
                    'train/curve_A_r': params['curve_A'][0, 0].item(),
                    'train/curve_A_g': params['curve_A'][0, 1].item(),
                    'train/curve_A_b': params['curve_A'][0, 2].item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                })

        # ── Epoch summary ──
        avg = {k: np.mean(v) for k, v in loss_acc.items()}
        print(f"\n  [Phase2] Epoch {epoch+1} Train: "
              f"total={avg['total']:.4f}  hist={avg['histogram']:.4f}")

        epoch_log = {}
        if use_wandb:
            epoch_log = {
                'epoch': epoch + 1,
                'epoch/train_loss_total': avg['total'],
                'epoch/train_loss_histogram': avg['histogram'],
                'epoch/train_loss_style': avg['style'],
                'epoch/train_loss_struct': avg['struct'],
                'epoch/train_loss_hue': avg['hue'],
                'epoch/train_loss_lum_mean': avg['lum_mean'],
                'epoch/train_loss_lum_hist': avg['lum_histogram'],
                'epoch/lr': optimizer.param_groups[0]['lr'],
            }

        # ── Validation ──
        if val_loader and (epoch + 1) % val_every == 0:
            val_metrics = validate(
                tone_net, net_fusion, net_L, val_loader, device,
                w_hist, w_style, w_struct, w_hue, w_lum_mean, w_lum_hist)

            print(f"  [Phase2] Epoch {epoch+1} Val: "
                  f"PSNR={val_metrics['psnr']:.2f} dB  "
                  f"loss={val_metrics['loss_total']:.4f}")

            if use_wandb:
                epoch_log.update({
                    'epoch/val_psnr': val_metrics['psnr'],
                    'epoch/val_loss_total': val_metrics['loss_total'],
                    'epoch/val_loss_histogram': val_metrics['loss_histogram'],
                    'epoch/val_loss_style': val_metrics['loss_style'],
                    'epoch/val_loss_struct': val_metrics['loss_struct'],
                    'epoch/train_loss_hue': avg['hue'],
                    'epoch/train_loss_lum_mean': avg['lum_mean'],
                    'epoch/train_loss_lum_hist': avg['lum_histogram'],
                })

            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                torch.save({
                    'tone_net': tone_net.state_dict(),
                    'epoch': epoch + 1,
                    'best_psnr': best_psnr,
                }, os.path.join(exp_path, 'model', 'best_tonenet.pth'))
                print(f"  ★ New best! PSNR={best_psnr:.2f} dB ")

        elif not val_loader:
            if avg['total'] < best_loss:
                best_loss = avg['total']
                torch.save({
                    'tone_net': tone_net.state_dict(),
                    'epoch': epoch + 1,
                    'loss': best_loss,
                }, os.path.join(exp_path, 'model', 'best_tonenet.pth'))
                print(f"  ★ New best (by loss)! loss={best_loss:.4f}")

        # Log sample images
        if use_wandb:
            try:
                tone_net.eval()
                sample_loader = val_loader if val_loader else train_loader

                with torch.no_grad():
                    all_batches = list(sample_loader)
                    s_imgs, s_gts, _ = random.choice(all_batches)

                    sB = s_imgs.shape[0]
                    n_log = min(sB, int(wcfg.get('log_images', 4)))
                    chosen = random.sample(range(sB), n_log)

                    s_imgs = s_imgs[chosen].to(device)
                    s_gts = s_gts[chosen].to(device)

                    sN = s_imgs.shape[1]
                    s_L_list = [net_L(s_imgs[:, n]) for n in range(sN)]
                    s_L_maps = torch.stack(s_L_list, dim=1)
                    s_rec, _, _, _, _ = net_fusion(s_imgs, L_maps=s_L_maps)
                    s_corrected, _ = tone_net(s_rec)

                    wb_imgs = []
                    for b in range(n_log):
                        concat = torch.cat([
                            s_imgs[b, 0].cpu().clamp(0, 1),
                            s_rec[b].cpu().clamp(0, 1),
                            s_corrected[b].cpu().clamp(0, 1),
                            s_gts[b].cpu().clamp(0, 1),
                        ], dim=2)
                        wb_imgs.append(wandb.Image(
                            concat.permute(1, 2, 0).numpy(),
                            caption=f"[E{epoch+1}] Input | Retinex | ToneNet | GT"
                        ))
                    epoch_log['samples'] = wb_imgs

            except Exception as e:
                print(f"  [Wandb] Image error: {e}")

            wandb.log(epoch_log)
            tone_net.train()

        scheduler.step()

        if (epoch + 1) % save_every == 0:
            torch.save({
                'tone_net': tone_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_psnr': best_psnr,
            }, os.path.join(exp_path, 'model',
                           f'tonenet_{epoch+1}.pth'))

    # Final save (combined Retinex + ToneNet for inference)
    torch.save({
        'net_fusion': net_fusion.state_dict(),
        'net_L': net_L.state_dict(),
        'tone_net': tone_net.state_dict(),
        'epoch': num_epochs,
        'best_psnr': best_psnr,
    }, os.path.join(exp_path, 'model', 'final_with_tonenet.pth'))
    print(f"\nPhase 2 complete. Combined model: "
          f"{exp_path}/model/final_with_tonenet.pth")
    if val_loader:
        print(f"Best validation PSNR: {best_psnr:.2f} dB")
    else:
        print(f"Best train loss: {best_loss:.4f}")

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    train_tonenet()