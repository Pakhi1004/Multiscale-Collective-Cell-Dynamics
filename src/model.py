"""Step 4a: CvT (convolutional transformer) from the paper's Materials & Methods.

  per frame  : Conv(1->2)->Conv(2->4)->Conv(4->6), k=4 s=2 p=1, BN + GELU
               128 -> 64 -> 32 -> 16 px ; flatten 6*16*16 -> FC 128 -> FC 64
  sequence   : + learned time embedding (8 tokens of dim 64)
  attention  : 3 layers with 2, 4, 6 heads; Q,K,V via FC 64 -> 256,
               softmax(QK^T/sqrt(d))V, then FC -> 64 (residual + LayerNorm)
  head       : mean over time -> FC -> n_out (1 = Ka, 2 = Ka, alpha)
Output is linear (no sigmoid) so the Ka-model is able to extrapolate.
"""
import math
import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    def __init__(self, img=128, d=64):
        super().__init__()
        ch = [1, 2, 4, 6]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        flat = ch[-1] * (img // 8) ** 2
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.GELU(),
                                nn.Linear(128, d), nn.GELU())

    def forward(self, x):                 # (N,1,H,W) -> (N,d)
        return self.fc(self.conv(x))


class Attention(nn.Module):
    def __init__(self, d=64, heads=2, inner=256):
        super().__init__()
        self.h = heads
        self.dh = inner // heads
        inner = self.h * self.dh
        self.q, self.k, self.v = (nn.Linear(d, inner) for _ in range(3))
        self.out = nn.Linear(inner, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.last_attn = None

    def forward(self, x):                 # (B,T,d)
        B, T, _ = x.shape
        z = self.norm1(x)
        sp = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = sp(self.q(z)), sp(self.k(z)), sp(self.v(z))
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), -1)
        self.last_attn = a.detach()
        y = (a @ v).transpose(1, 2).reshape(B, T, self.h * self.dh)
        x = x + self.out(y)
        return x + self.ff(self.norm2(x))


class CvT(nn.Module):
    def __init__(self, n_out=1, seq_len=8, img=128, d=64, heads=(2, 4, 6)):
        super().__init__()
        self.enc = FrameEncoder(img, d)
        self.time = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.normal_(self.time, std=0.02)
        self.attn = nn.ModuleList(Attention(d, h) for h in heads)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)

    def forward(self, x):                 # (B,T,1,H,W)
        B, T = x.shape[:2]
        z = self.enc(x.flatten(0, 1)).view(B, T, -1) + self.time
        for layer in self.attn:
            z = layer(z)
        return self.head(self.norm(z).mean(1))


def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n in (1, 2):
        m = CvT(n_out=n)
        y = m(torch.rand(4, 8, 1, 128, 128))
        print(f"n_out={n}  params={n_params(m):,}  out={tuple(y.shape)}")
