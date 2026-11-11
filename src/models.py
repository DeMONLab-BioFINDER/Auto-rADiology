# src/models.py (append these)
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, BasicUNet, DenseNet121, resnet

class AttentionPool3D(nn.Module):
    """Gated attention pooling (Ilse et al., 2018, "Attention-based Deep Multiple
    Instance Learning"), treating each spatial location of the final feature map as
    an instance in a bag. Learns which locations actually drive the prediction
    instead of averaging the whole volume uniformly - meant for focal pathology
    that global average pooling can dilute.
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_channels, hidden_dim)
        self.U = nn.Linear(in_channels, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.last_attn_weights = None  # [B, N] after a forward pass (detached), for visualization

    def forward(self, x):
        # x: [B, C, D, H, W]
        instances = x.flatten(2).transpose(1, 2)  # [B, N, C], N = D*H*W
        gated = torch.tanh(self.V(instances)) * torch.sigmoid(self.U(instances))  # [B, N, hidden_dim]
        scores = self.w(gated).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1)  # [B, N], sums to 1 per sample
        self.last_attn_weights = weights.detach()
        pooled = torch.bmm(weights.unsqueeze(1), instances).squeeze(1)  # [B, C]
        return pooled


class AvgMaxPool3D(nn.Module):
    """Global average and global max pooling side by side ([B, 2C]): the head sees both the
    whole-volume average of each feature and its single strongest location, so one focal spot
    can count without being diluted by the rest of the brain. No learnable parameters."""
    def forward(self, x):
        return torch.cat([x.mean(dim=(2, 3, 4)), x.amax(dim=(2, 3, 4))], dim=1)


class TopKPool3D(nn.Module):
    """Per feature, the mean of its strongest `frac` of spatial locations ([B, C]): between
    global average pooling (frac = 1) and max pooling (one location), closer to an "any raised
    region" rule than an average and more stable than a single max. No learnable parameters."""
    def __init__(self, frac: float = 0.05):
        super().__init__()
        assert 0 < frac <= 1
        # a buffer, so the checkpoint records it: without any saved tensor, a top-k checkpoint would
        # load into an average-pooling model without a mismatch (see checkpoints.load_state_dict_checked)
        self.register_buffer("frac", torch.tensor(float(frac)))

    def forward(self, x):
        flat = x.flatten(2)  # [B, C, N]
        k = max(1, int(round(float(self.frac) * flat.shape[-1])))
        return flat.topk(k, dim=-1).values.mean(dim=-1)


class CNN3D(nn.Module):
    """
    A small 3D CNN for scalar output.
    - Stack of Conv3d -> BN -> ReLU -> (optional MaxPool)
    - Global average pool (or attention/MIL pooling)
    - Linear head -> [B, 1]

    Args
    ----
    in_channels : int
        Input channels (default 1 for PET).
    widths : tuple[int, ...]
        Feature widths per conv stage, e.g. (16, 32, 64, 128).
    pool_every : int
        Apply MaxPool3d(2) after every N conv stages (set 1 to pool after each).
    n_pools : int or None
        At most this many MaxPool3d(2) steps, the first ones (None = no limit, i.e. every stage that
        pool_every picks; with 4 stages that is 4). n_pools=3 skips the last one: block 4's convolutions
        run on the same grid either way, so the final feature map is twice as fine per side
        (MNI: 11x13x11 at 16 mm instead of 5x6x5 at 32 mm) at almost no extra cost - finer attention
        maps. The weights have the same shapes either way, so the setting is stored in the checkpoint
        (buffer "n_pools") and a mismatched checkpoint fails to load instead of loading silently.
    dropout : float
        Dropout before the final linear head.
    norm : str
        "batch", "instance", or "group" normalization.
    gn_groups : int
        Number of groups for GroupNorm (only used when norm="group"). Falls back to
        1 group for any conv stage whose channel count isn't divisible by gn_groups.
    pool : str
        "avg" (global average pooling, default), "attention" (gated attention/MIL
        pooling over spatial locations - see AttentionPool3D), "avgmax" (average and max
        side by side - see AvgMaxPool3D) or "topk" (mean of the strongest topk_frac of
        locations - see TopKPool3D).
    attn_hidden : int
        Hidden dim of the attention scoring MLP (only used when pool="attention").
    topk_frac : float
        Share of locations averaged by pool="topk" (default 0.05).
    """
    def __init__(self, in_channels: int = 1, widths=(16, 32, 64, 128), pool_every: int = 1,
                 dropout: float = 0.2, norm: str = "batch", num_classes=1, extra_dim=0,
                 gn_groups: int = 8, pool: str = "avg", attn_hidden: int = 128, topk_frac: float = 0.05,
                 n_pools: int | None = None):
        super().__init__()
        assert pool_every >= 1
        assert n_pools is None or n_pools >= 0
        assert norm in {"batch", "instance", "group"}
        assert pool in {"avg", "attention", "avgmax", "topk"}
        print(f'in_channels: {in_channels}, widths: {widths}, pool_every: {pool_every}, n_pools: {n_pools}, dropout: {dropout}, norm: {norm}, num_classes: {num_classes}, pool: {pool}')
        def make_norm(c):
            if norm == "batch":
                return nn.BatchNorm3d(c)  # ok if batch_size >= ~8
            elif norm == "instance":
                return nn.InstanceNorm3d(c, affine=True, track_running_stats=False)
            else:  # "group"
                groups = gn_groups if c % gn_groups == 0 else 1
                return nn.GroupNorm(groups, c)

        layers = []
        c_in = in_channels
        pools_done = 0
        for i, c_out in enumerate(widths, start=1):
            layers += [
                nn.Conv3d(c_in, c_out, 3, 1, 1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True),
            ]
            if (i % pool_every) == 0 and (n_pools is None or pools_done < n_pools):
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
                pools_done += 1
            c_in = c_out

        self.features = nn.Sequential(*layers)
        if n_pools is not None:
            # only when set, so checkpoints trained before this option still load into the default model
            self.register_buffer("n_pools", torch.tensor(int(n_pools)))
        self._attention_pool = pool == "attention"
        if self._attention_pool:
            self.pool = AttentionPool3D(widths[-1], hidden_dim=attn_hidden)
        elif pool == "avgmax":
            self.pool = AvgMaxPool3D()
        elif pool == "topk":
            self.pool = TopKPool3D(topk_frac)
        else:
            self.pool = nn.AdaptiveAvgPool3d(1)
        in_fc = widths[-1] * (2 if pool == "avgmax" else 1) + extra_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_fc, num_classes))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # both sides store n_pools but with different values (e.g. 3 vs 2): the weights would load
        # silently into a model with a different number of poolings - refuse instead
        key = prefix + "n_pools"
        if key in state_dict and hasattr(self, "n_pools") and int(state_dict[key]) != int(self.n_pools):
            raise RuntimeError(f"Checkpoint was trained with n_pools={int(state_dict[key])}, this model has "
                               f"n_pools={int(self.n_pools)}. Pass the same --model_kwargs (NPOOLS=...).")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x, extra):
        x = self.features(x)       # [B, C, D, H, W]
        x = self.pool(x).flatten(1)  # [B, C] ([B, 2C] for avgmax)

        if extra is not None and not torch.isnan(extra).all():
            if extra.ndim == 1: extra = extra.unsqueeze(0)
            x = torch.cat([x, extra], dim=1)

        y = self.head(x)           # [B, num_classes]
        return y
    

class UNet3D(nn.Module):
    """
    MONAI 3D U-Net for volumetric prediction (e.g., segmentation).
    Returns logits of shape [B, out_channels, D, H, W].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64, num_classes: int = 1, 
                 channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), 
                 num_res_units: int = 2, norm: str = "instance", dropout: float = 0.0, 
                 use_basic: bool = False, extra_dim: int = 0): #up_kernel_size: int = 2, 
    
        super().__init__()
        self.num_classes = num_classes
        self.extra_dim = extra_dim

        if use_basic:
            # BasicUNet has a simplified API
            self.net = BasicUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                features=list(channels), dropout=dropout)  # BasicUNet uses "features"
        else:
            self.net = UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                            channels=list(channels), strides=list(strides), num_res_units=num_res_units,
                            norm=norm, dropout=dropout) #, up_kernel_size=up_kernel_size
        
        in_fc = out_channels + extra_dim

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_fc, num_classes),)

    def forward(self, x, extra):
        feat = self.net(x)          # [B, C, D, H, W]
        feat = self.gap(feat).flatten(1)  # [B, C]

        if extra is not None and torch.isfinite(extra).any():
            if extra.ndim == 1: extra = extra.unsqueeze(0)
            feat = torch.cat([feat, extra], dim=1)

        out = self.head(feat)              # [B, num_classes]
    
        return out


class DenseNet121_3D(nn.Module):
    """
    MONAI 3D DenseNet121 for scalar prediction (classification or regression).
    Returns logits of shape [B, 1].
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1,   # keep =1 for single-output head
                 dropout_prob: float = 0.0):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,  # classifier head -> [B, out_channels]
            dropout_prob=dropout_prob)

    def forward(self, x):
        # DenseNet121 already includes GAP + Linear -> [B, out_channels]
        y = self.backbone(x)
        # Ensure shape [B, 1] for BCEWithLogitsLoss/MSE downstream
        if y.ndim == 2 and y.size(1) == 1:
            return y
        return y.view(y.size(0), -1)[:, :1]


class ResNet50_3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.backbone = resnet.ResNet(block=resnet.Bottleneck, layers=[3,4,6,3],
                                      block_inplanes=resnet.get_inplanes(),
                                      n_input_channels=in_channels,
                                      conv1_t_size=7, conv1_t_stride=2,
                                      no_max_pool=False, shortcut_type='B',
                                      widen_factor=1.0, num_classes=out_channels,    # -> [B, out_channels]
                                      spatial_dims=3)
    def forward(self, x):
        y = self.backbone(x)
        return y if y.ndim == 2 else y.view(y.size(0), -1)[:, :1]