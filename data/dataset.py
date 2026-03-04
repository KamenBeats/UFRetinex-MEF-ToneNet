import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as Data
from functools import partial

EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path)
    if img_BGR is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_BGR = img_BGR.astype('float32')
    if mode == 'RGB':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'Gray':
        return np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    raise ValueError(f"Unknown mode: {mode}")

def augment_images(images):
    """Apply consistent random flip/rotation to a list of images."""
    if random.random() > 0.5:
        images = [np.fliplr(img).copy() for img in images]
    if random.random() > 0.5:
        images = [np.flipud(img).copy() for img in images]
    k = random.randint(0, 3)
    if k > 0:
        images = [np.rot90(img, k).copy() for img in images]
    return images

def _numeric_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(base)
    except ValueError:
        return base

def _find_gt_image(scene_dir):
    for fname in os.listdir(scene_dir):
        base = os.path.splitext(fname)[0].lower()
        if base == 'gt' and fname.lower().endswith(EXTENSIONS):
            return os.path.join(scene_dir, fname)
    return None

def _scan_scenes(data_dir, require_gt=False, gt_dir=None):
    """Scan data_dir for scene folders. Returns list of scene dicts."""
    scenes = []
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory not found: {data_dir}")
        return scenes

    for scene_name in sorted(os.listdir(data_dir)):
        scene_dir = os.path.join(data_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        img_paths = []
        for fname in os.listdir(scene_dir):
            if os.path.splitext(fname)[0].lower() == 'gt':
                continue
            if fname.lower().endswith(EXTENSIONS):
                img_paths.append(os.path.join(scene_dir, fname))
        img_paths.sort(key=_numeric_key)

        if len(img_paths) < 1:
            continue

        scene = {
            'name': scene_name,
            'paths': img_paths,
            'n_images': len(img_paths),
        }

        if require_gt:
            gt_path = None
            if gt_dir and os.path.isdir(gt_dir):
                for ext in EXTENSIONS:
                    cand = os.path.join(gt_dir, scene_name + ext)
                    if os.path.exists(cand):
                        gt_path = cand
                        break
            if gt_path is None:
                gt_path = _find_gt_image(scene_dir)
            if gt_path is None:
                continue
            scene['gt_path'] = gt_path

        scenes.append(scene)
    return scenes

# ── Phase 1 dataset (no GT) ──

class MultiExposureDataset(Data.Dataset):
    def __init__(self, data_dir, patch_size=0, augment=True):
        self.patch_size = patch_size
        self.augment = augment
        self.scenes = _scan_scenes(data_dir)
        print(f"[Dataset] {len(self.scenes)} scenes in {data_dir}")
        if self.scenes:
            counts = [s['n_images'] for s in self.scenes]
            print(f"  Images/scene: min={min(counts)}, "
                  f"max={max(counts)}, mean={np.mean(counts):.1f}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        raw_images = [image_read(p) for p in scene['paths']]

        H, W = raw_images[0].shape[:2]
        for i in range(1, len(raw_images)):
            h, w = raw_images[i].shape[:2]
            if h != H or w != W:
                raw_images[i] = cv2.resize(raw_images[i], (W, H),
                                           interpolation=cv2.INTER_LINEAR)

        if self.patch_size > 0 and H >= self.patch_size and W >= self.patch_size:
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            ps = self.patch_size
            raw_images = [img[top:top+ps, left:left+ps] for img in raw_images]

        if self.augment:
            raw_images = augment_images(raw_images)

        tensors = [
            torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
            for img in raw_images
        ]
        return tensors, index

# ── Phase 2 dataset (with GT) ──

class MultiExposureGTDataset(Data.Dataset):
    def __init__(self, data_dir, gt_dir=None, patch_size=0, augment=True):
        self.patch_size = patch_size
        self.augment = augment
        self.scenes = _scan_scenes(data_dir, require_gt=True, gt_dir=gt_dir)
        print(f"[GT-Dataset] {len(self.scenes)} scenes with GT in {data_dir}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        raw_images = [image_read(p) for p in scene['paths']]
        gt_raw = image_read(scene['gt_path'])

        H, W = raw_images[0].shape[:2]
        for i in range(1, len(raw_images)):
            h, w = raw_images[i].shape[:2]
            if h != H or w != W:
                raw_images[i] = cv2.resize(raw_images[i], (W, H),
                                           interpolation=cv2.INTER_LINEAR)
        gh, gw = gt_raw.shape[:2]
        if gh != H or gw != W:
            gt_raw = cv2.resize(gt_raw, (W, H),
                                interpolation=cv2.INTER_LINEAR)

        if self.patch_size > 0 and H >= self.patch_size and W >= self.patch_size:
            top = random.randint(0, H - self.patch_size)
            left = random.randint(0, W - self.patch_size)
            ps = self.patch_size
            raw_images = [img[top:top+ps, left:left+ps] for img in raw_images]
            gt_raw = gt_raw[top:top+ps, left:left+ps]

        if self.augment:
            all_imgs = raw_images + [gt_raw]
            all_imgs = augment_images(all_imgs)
            raw_images = all_imgs[:-1]
            gt_raw = all_imgs[-1]

        tensors = [
            torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
            for img in raw_images
        ]
        gt_tensor = torch.from_numpy(
            gt_raw.transpose(2, 0, 1).copy()).float() / 255.0
        return tensors, gt_tensor, index

# ── Collate functions ──

def _sample_indices(K, N):
    """Select N indices from K images, always including first & last if N>=2."""
    if N >= 2 and K >= 2:
        selected = [0, K - 1]
        remaining = list(range(1, K - 1))
        need = N - 2
        if need > 0 and remaining:
            selected += random.sample(remaining, min(need, len(remaining)))
        while len(selected) < N:
            selected.append(random.randint(0, K - 1))
        return sorted(selected[:N])
    elif N == 1:
        return [random.randint(0, K - 1)]
    return random.sample(range(K), min(N, K))

def multi_collate_fn(batch, n_max=7):
    min_available = min(len(item[0]) for item in batch)
    N = random.randint(1, min(n_max, min_available))
    images_batch, indices = [], []
    for images_list, idx in batch:
        sel = _sample_indices(len(images_list), N)
        images_batch.append(torch.stack([images_list[i] for i in sel]))
        indices.append(idx)
    return torch.stack(images_batch), torch.tensor(indices)

def gt_collate_fn(batch, n_max=7):
    min_available = min(len(item[0]) for item in batch)
    N = random.randint(1, min(n_max, min_available))
    images_batch, gts, indices = [], [], []
    for images_list, gt, idx in batch:
        sel = _sample_indices(len(images_list), N)
        images_batch.append(torch.stack([images_list[i] for i in sel]))
        gts.append(gt)
        indices.append(idx)
    return torch.stack(images_batch), torch.stack(gts), torch.tensor(indices)

def create_collate_fn(n_max=7):
    return partial(multi_collate_fn, n_max=n_max)

def create_gt_collate_fn(n_max=7):
    return partial(gt_collate_fn, n_max=n_max)