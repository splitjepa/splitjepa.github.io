"""SplitJEPA and LeJEPA encoders, predictor and objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_backbone(name: str, pretrained: bool) -> nn.Module:
    import torchvision.models as tv

    if name == "resnet18":
        weights = tv.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv.resnet18(weights=weights)
    elif name == "resnet34":
        weights = tv.ResNet34_Weights.DEFAULT if pretrained else None
        model = tv.resnet34(weights=weights)
    elif name == "resnet50":
        weights = tv.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv.resnet50(weights=weights)
    else:
        raise ValueError(f"unknown backbone: {name}")
    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class VisualEncoder(nn.Module):
    """image -> shared trunk -> independent invariant and variant heads."""
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        zh_dim: int = 64,
        zl_dim: int = 64,
        pool: str = "avg",
        normalize: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.zh_dim = zh_dim
        self.zl_dim = zl_dim
        self.normalize = normalize

        net, feat_dim = _build_backbone(backbone, pretrained)
        self.shared_trunk = net
        self.feat_dim = feat_dim
        if pool != "avg":
            raise ValueError("the compact release supports average pooling only")

        self.invariant_head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, zh_dim),
        )
        self.variant_head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, zl_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_trunk(x)
        zh = self.invariant_head(features)
        zl = self.variant_head(features)
        if self.normalize:
            zh = torch.nn.functional.normalize(zh, dim=-1)
            zl = torch.nn.functional.normalize(zl, dim=-1)
        return zh, zl




class LeJEPAEncoder(nn.Module):
    """Image -> pooled ResNet feature -> one projected latent ``z_enc``."""
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = False,
        projection_hidden: int = 512,
        projection_dim: int = 128,
        normalize: bool = False,
    ):
        super().__init__()
        self.backbone_name = str(backbone)
        self.projection_hidden = int(projection_hidden)
        self.projection_dim = int(projection_dim)
        self.normalize = bool(normalize)

        self.backbone, self.feat_dim = _build_backbone(self.backbone_name, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(self.feat_dim, self.projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_hidden, self.projection_dim),
        )

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the backbone's globally pooled feature before projection."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(self.pooled_features(x))
        if self.normalize:
            z = torch.nn.functional.normalize(z, dim=-1)
        return z



class LatentPredictor(nn.Module):
    """Predicts the next latent given current latent and action."""

    def __init__(self, z_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z: (B, z_dim), a: (B, action_dim) -> (B, z_dim)."""
        return self.net(torch.cat([z, a], dim=-1))

def predictive_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE against a stop-gradient target (BYOL-style stability)."""
    return F.mse_loss(pred, target.detach())


def invariance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    """||zH_a - zH_b||^2 for matched pairs."""
    return F.mse_loss(za, zb)


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance: keep std >= gamma along the batch dim."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance: zero the off-diagonal of the feature covariance."""
    n, d = z.shape
    if n < 2:
        return torch.zeros((), device=z.device)
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    off = cov - torch.diag(cov.diag())
    return (off ** 2).sum() / d


def compute_rep_loss(
    *,
    z_t, z_t1,
    zh_a, zh_b,
    pred: torch.Tensor,
    weights: dict,
    batch_z: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Aggregate all representation losses. Returns (total, per-term dict)."""
    losses = {}
    if weights.get("pred", 0.0) > 0 and pred is not None:
        losses["pred"] = predictive_loss(pred, z_t1)
    if weights.get("inv", 0.0) > 0:
        losses["inv"] = invariance_loss(zh_a, zh_b)
    if weights.get("variance", 0.0) > 0:
        losses["variance"] = variance_loss(batch_z)
    if weights.get("covariance", 0.0) > 0:
        losses["covariance"] = covariance_loss(batch_z)
    total = torch.zeros((), device=z_t.device)
    for k, v in losses.items():
        total = total + float(weights.get(k, 0.0)) * v
    return total, {k: float(v.detach()) for k, v in losses.items()}
