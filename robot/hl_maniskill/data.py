"""torch Datasets over the materialized episodes."""


import threading
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from hl_maniskill.utils import load_json

CACHE_SIZE = 256


class EpisodeStore:
    """Lazy episode npz loader with a small LRU cache."""

    def __init__(self, ep_dir: str | Path, cache_size: int = CACHE_SIZE):
        self.ep_dir = Path(ep_dir)
        self.cache_size = cache_size
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        self._order: list[int] = []
        self._lock = threading.Lock()

    def __getitem__(self, ep_id: int) -> dict[str, np.ndarray]:
        with self._lock:
            if ep_id in self._cache:
                return self._cache[ep_id]
        path = self.ep_dir / f"traj_{int(ep_id):06d}.npz"
        with np.load(path, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        with self._lock:
            self._cache[ep_id] = data
            self._order.append(ep_id)
            while len(self._order) > self.cache_size:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
        return data


def preprocess_image(img: np.ndarray, size: int = 128,
                     mean: tuple[float, float, float] | None = None,
                     std: tuple[float, float, float] | None = None) -> torch.Tensor:
    """uint8 HWC -> float32 (3, size, size). Returns [0,1] unless ImageNet
    mean/std are given (then returns ImageNet-normalized input)."""
    img = np.asarray(img)
    if img.shape[0] != size or img.shape[1] != size:
        img = _resize(img, size)
    x = torch.from_numpy(img).float().permute(2, 0, 1).contiguous() / 255.0
    if mean is not None and std is not None:
        m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        x = (x - m) / s
    return x


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_norm_params(norm: str):
    """Map an image_norm config string to (mean, std) or (None, None)."""
    if str(norm).lower() in ("imagenet", "true", "1"):
        return IMAGENET_MEAN, IMAGENET_STD
    return None, None


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import torchvision.transforms.functional as F

    t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    t = F.resize(t, (size, size), antialias=True)
    return t.permute(1, 2, 0).numpy()


class TemporalDataset(Dataset):
    """Consecutive observations (t -> t+1) with the taken action."""

    def __init__(self, ep_dir: Path, episode_ids: list[int], lens: dict[int, int],
                 size: int = 128, mean: tuple | None = None, std: tuple | None = None):
        self.store = EpisodeStore(ep_dir)
        self.size = size
        self.mean = mean
        self.std = std
        self.items = []
        for ep in episode_ids:
            T = lens[ep]
            if T >= 2:
                self.items.extend((ep, t) for t in range(T - 1))
        if not self.items:
            raise RuntimeError("empty TemporalDataset")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ep, t = self.items[idx]
        data = self.store[ep]
        return dict(
            x_t=preprocess_image(data["image_base"][t], self.size, self.mean, self.std),
            x_t1=preprocess_image(data["image_base"][t + 1], self.size, self.mean, self.std),
            a_t=torch.from_numpy(data["action"][t].astype(np.float32)),
        )


class PairDataset(Dataset):
    """Matched pairs for the invariance objective.

    pair arrays:
        cross_traj  (N,4) [ep_a, t_a, ep_b, t_b]   base/base
        viewpoint   (N,2) [ep, t]                  base/side
    """

    def __init__(self, ep_dir: Path, pair_arrays: dict[str, np.ndarray], size: int = 128,
                 mean: tuple | None = None, std: tuple | None = None):
        self.store = EpisodeStore(ep_dir)
        self.size = size
        self.mean = mean
        self.std = std
        self.items = []
        for kind in ("cross_traj", "viewpoint"):
            arr = pair_arrays.get(kind)
            if arr is None or len(arr) == 0:
                continue
            if kind == "viewpoint":
                for ep, t in arr:
                    self.items.append((int(ep), int(t), int(ep), int(t), kind))
            else:
                for a, ta, b, tb in arr:
                    self.items.append((int(a), int(ta), int(b), int(tb), kind))
        if not self.items:
            raise RuntimeError("empty PairDataset")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a, ta, b, tb, kind = self.items[idx]
        da, db = self.store[a], self.store[b]
        cam_b = "image_side" if kind == "viewpoint" else "image_base"
        return dict(
            x_a=preprocess_image(da["image_base"][ta], self.size, self.mean, self.std),
            x_b=preprocess_image(db[cam_b][tb], self.size, self.mean, self.std),
            kind=kind,
        )


class BcDataset(Dataset):
    """(x_t, a_t) behavior-cloning samples from a subset of episodes.

    With stack_frames>1 the observation is the last `stack_frames` frames
    stacked along a new leading axis, padded by repeating the first frame.
    """

    def __init__(self, ep_dir: Path, episode_ids: list[int], lens: dict[int, int],
                 size: int = 128, stack_frames: int = 1):
        self.store = EpisodeStore(ep_dir)
        self.size = size
        self.stack = max(1, int(stack_frames))
        self.items = [(ep, t) for ep in episode_ids for t in range(lens[ep])]
        if not self.items:
            raise RuntimeError("empty BcDataset")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ep, t = self.items[idx]
        data = self.store[ep]
        imgs = data["image_base"]
        lo = max(0, t - self.stack + 1)
        frames = imgs[lo: t + 1]
        if len(frames) < self.stack:
            pad = np.repeat(imgs[0][None], self.stack - len(frames), axis=0)
            frames = np.concatenate([pad, frames], axis=0)
        xs = torch.stack([preprocess_image(f, self.size) for f in frames])  # (S,3,H,W)
        out = dict(
            x_t=xs if self.stack > 1 else xs[0],
            a_t=torch.from_numpy(data["action"][t].astype(np.float32)),
        )
        return out


def load_episode_lens(out_root: Path) -> dict[int, int]:
    index = load_json(out_root / "index.json")
    ids = index["episode_ids"]
    lengths = index["episode_lengths"]
    return {int(i): int(l) for i, l in zip(ids, lengths)}


def make_split_pair_arrays(out_root: Path, split: str) -> dict[str, np.ndarray]:
    import h5py

    path = out_root / "pairs.h5"
    out = {}
    with h5py.File(path, "r") as hf:
        g = hf[split]
        for k in ("cross_traj", "viewpoint"):
            if k in g:
                out[k] = g[k][:]
    return out



