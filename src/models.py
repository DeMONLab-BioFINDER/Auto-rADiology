# src/models.py (append these)
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet, BasicUNet, DenseNet121, resnet


def get_norm_layer(norm: str, c: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm3d(c)
    if norm == "instance":
        return nn.InstanceNorm3d(c, affine=True, track_running_stats=False)
    raise ValueError(f"Unknown norm='{norm}'. Use 'batch' or 'instance'.")


class BasicBlock3D(nn.Module):
    """
    Conv block or residual block.

    If residual=False:
        Conv -> Norm -> ReLU -> Conv -> Norm -> ReLU

    If residual=True:
        Conv -> Norm -> ReLU -> Conv -> Norm -> Add(skip) -> ReLU
    """

    def __init__(self, c_in: int, c_out: int, norm: str = "batch", residual: bool = False):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm, c_out)

        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm, c_out)

        self.relu = nn.ReLU(inplace=True)

        if residual:
            self.skip = nn.Identity() if c_in == c_out else nn.Conv3d(c_in, c_out, kernel_size=1, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.residual:
            out = out + self.skip(x)

        out = self.relu(out)

        return out
    

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradReverse.apply(x, self.lambda_grl)
    

class CNN3D(nn.Module):
    """
    3D CNN for scalar or multi-class output.

    Architecture switches:
    - block: "conv" or "res"
    - downsample: "pool" or "stride"

    Args
    ----
    in_channels : int. Input channels (default 1 for PET).
    widths : tuple[int, ...]. Feature widths per conv stage, e.g. (16, 32, 64, 128) or (32, 64, 128, 256).
    pool_every : int. Apply MaxPool3d(2) after every N conv stages (set 1 to pool after each).
    dropout : float. Dropout before the final linear head.
    norm : str. "batch" or "instance" normalization.
    block: "conv" = standard conv block. "res" = residual block.
    downsample: "pool" = MaxPool3d(2). "stride" = Conv3d(stride=2).
    num_classes: Output dimension.
    extra_dim: Extra tabular features concatenated after global average pooling.
    """
    def __init__(self, in_channels: int = 1, widths: tuple[int, ...] =(16, 32, 64, 128), 
                 pool_every: int = 1, dropout: float = 0.2, norm: str = "batch", 
                 num_classes: int = 1, extra_dim: int = 0, block: str = "conv", downsample: str = "pool",
                 num_domains: int = 0, lambda_grl: float = 1.0):
        super().__init__()
        assert pool_every >= 1
        assert norm in {"batch", "instance"}
        assert block in {"conv", "res"}
        assert downsample in {"pool", "stride"}

        print(f"in_channels={in_channels}, widths={widths}, pool_every={pool_every}, "
              f"dropout={dropout}, norm={norm}, num_classes={num_classes}, "
              f"extra_dim={extra_dim}, block={block}, downsample={downsample}")

        residual = block == "res"
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(widths, start=1):
            layers.append(BasicBlock3D(c_in=c_in, c_out=c_out, norm=norm, residual=residual))

            if (i % pool_every) == 0:
                if downsample == "pool":
                    layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
                else:
                    layers.extend([
                        nn.Conv3d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                        get_norm_layer(norm, c_out),
                        nn.ReLU(inplace=True)])

            c_in = c_out

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1] + extra_dim, num_classes))
        self.num_domains = num_domains
        self.grl = GradientReversalLayer(lambda_grl=lambda_grl)
        self.dataset_head = nn.Linear(widths[-1], num_domains) if num_domains > 0 else None

    def forward(self, x, extra):
        feat = self.features(x)       # [B, C, D, H, W]
        feat = self.gap(feat).flatten(1) # [B, C]

        head_input = feat
        if extra is not None:
            if extra.ndim == 1: extra = extra.unsqueeze(1)
            if not torch.isnan(extra).all(): head_input = torch.cat([head_input, extra], dim=1)

        y = self.head(head_input)           # [B, num_classes]

        if self.dataset_head is not None:
            dataset_logit = self.dataset_head(self.grl(feat))
            return {"logit": y, "dataset_logit": dataset_logit, "feats": feat}
        
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