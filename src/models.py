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
    dropout : float
        Dropout before the final linear head.
    norm : str
        "batch", "instance", or "group" normalization.
    gn_groups : int
        Number of groups for GroupNorm (only used when norm="group"). Falls back to
        1 group for any conv stage whose channel count isn't divisible by gn_groups.
    pool : str
        "avg" (global average pooling, default) or "attention" (gated attention/MIL
        pooling over spatial locations - see AttentionPool3D).
    attn_hidden : int
        Hidden dim of the attention scoring MLP (only used when pool="attention").
    """
    def __init__(self, in_channels: int = 1, widths=(16, 32, 64, 128), pool_every: int = 1,
                 dropout: float = 0.2, norm: str = "batch", num_classes=1, extra_dim=0,
                 gn_groups: int = 8, pool: str = "avg", attn_hidden: int = 128):
        super().__init__()
        assert pool_every >= 1
        assert norm in {"batch", "instance", "group"}
        assert pool in {"avg", "attention"}
        print(f'in_channels: {in_channels}, widths: {widths}, pool_every: {pool_every}, dropout: {dropout}, norm: {norm}, num_classes: {num_classes}, pool: {pool}')
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
        for i, c_out in enumerate(widths, start=1):
            layers += [
                nn.Conv3d(c_in, c_out, 3, 1, 1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True),
                nn.Conv3d(c_out, c_out, 3, 1, 1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True),
            ]
            if (i % pool_every) == 0:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            c_in = c_out

        self.features = nn.Sequential(*layers)
        self._attention_pool = pool == "attention"
        if self._attention_pool:
            self.pool = AttentionPool3D(widths[-1], hidden_dim=attn_hidden)
        else:
            self.pool = nn.AdaptiveAvgPool3d(1)
        in_fc = widths[-1] + extra_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_fc, num_classes))

    def forward(self, x, extra):
        x = self.features(x)       # [B, C, D, H, W]
        x = self.pool(x) if self._attention_pool else self.pool(x).flatten(1)  # [B, C]

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