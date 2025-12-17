# Model Conv Next
# import torch
import torch.nn as nn
import torch.nn.functional as F

#   ConvNeXt-like block 
class CNextBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)  # depthwise
        self.norm = nn.GroupNorm(1, dim)  # LayerNorm-like
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual



#   Basic Down Block
class DownStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, kernel_size):
        super().__init__()
        self.blocks = nn.Sequential(*[
            CNextBlock(in_dim, kernel_size=kernel_size) for _ in range(num_blocks)
        ])
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.blocks(x)
        down = self.down(skip)
        return skip, down




#   Basic Up Block
class UpStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_blocks, kernel_size):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(*[
            CNextBlock(out_dim, kernel_size=kernel_size) for _ in range(num_blocks)
        ])

    def forward(self, x, skip):
        x = self.up(x)

        # pad skip if size mismatch
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]))

        x = x + skip  # skip connection
        x = self.blocks(x)
        return x


class CNextUNet(nn.Module):
    def __init__(
        self,
        in_channels=44,
        out_channels=11,
        base_dim=42,
        num_stages=4,
        blocks_per_stage=2,
        bottleneck_blocks=1,
        kernel_size=7
    ):
        super().__init__()

        dims = [base_dim * (2 ** i) for i in range(num_stages)]

        # initial projection
        self.in_proj = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)

        # Down path
        self.downs = nn.ModuleList([
            DownStage(
                dims[i], dims[i + 1],
                num_blocks=blocks_per_stage,
                kernel_size=kernel_size
            ) for i in range(num_stages - 1)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            CNextBlock(dims[-1], kernel_size=kernel_size)
            for _ in range(bottleneck_blocks)
        ])

        # Up path (reverse dims)
        self.ups = nn.ModuleList([
            UpStage(
                dims[i + 1], dims[i],
                num_blocks=blocks_per_stage,
                kernel_size=kernel_size
            ) for i in reversed(range(num_stages - 1))
        ])

        # output projection
        self.out_proj = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def forward(self, x):

        Bsize = x.shape[0]
        # Enters as: [B, 4, 256, 256, 11]
        # Move channels last → channels first
        x = x.permute(0, 1, 4, 2, 3)   # (B,4,11,256,256)
        x = x.reshape(Bsize, 4*11, 256, 256)  # (B,44,256,256)

        skips = []
        x = self.in_proj(x)

        # Down path
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path (reverse order)
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        # # Remove time dimension of target
        # # torch.Size([B, 11, 256, 256])
        # self.out_proj(x) = self.out_proj(x)   # (B,1,11,256,256)
        # y = y.reshape(11, 256, 256)     # (B,11,256,256)

        return (self.out_proj(x).permute(0, 2, 3, 1)).unsqueeze(1)