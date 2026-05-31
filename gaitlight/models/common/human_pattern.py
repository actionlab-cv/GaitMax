import einops
import torch
import torch.nn.functional as func
from torch import nn


class HumanPattern(nn.Module):
    """
    Slot-attention-based body region discovery module.

    :param s_dim: DINO feature dimension.
    :param k: number of DINO intermediate layers fused.
    :param m: number of body region slots.
    :param n_iter: number of slot attention iterations.
    """

    def __init__(self, s_dim: int = 384, k: int = 1, m: int = 5, n_iter: int = 3):
        super().__init__()
        self.m = m
        self.n_iter = n_iter
        self.slot_dim = s_dim

        # project DINO features
        self.enc = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm2d(s_dim * k),
            nn.Conv2d(s_dim * k, s_dim, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(s_dim),
            nn.ReLU(inplace=True),
        )

        # slot attention components
        self.slots_mu = nn.Parameter(torch.randn(1, m, s_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, m, s_dim))

        self.norm_input = nn.LayerNorm(s_dim)
        self.norm_slots = nn.LayerNorm(s_dim)

        self.proj_k = nn.Linear(s_dim, s_dim, bias=False)
        self.proj_v = nn.Linear(s_dim, s_dim, bias=False)
        self.proj_q = nn.Linear(s_dim, s_dim, bias=False)

        self.gru = nn.GRUCell(s_dim, s_dim)
        self.mlp = nn.Sequential(
            nn.Linear(s_dim, s_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(s_dim * 2, s_dim),
        )
        self.norm_mlp = nn.LayerNorm(s_dim)

        self.scale = s_dim ** -0.5

    def _init_slots(self, bt: int) -> torch.Tensor:
        """Sample initial slots from learned distribution. Returns [bt, m, d]."""
        mu = self.slots_mu.expand(bt, -1, -1)
        sigma = self.slots_log_sigma.exp().expand(bt, -1, -1)
        return mu + sigma * torch.randn_like(sigma)

    def forward(self, f_v: torch.Tensor) -> dict:
        """
        :param f_v: DINO features [k, b, t, c, h, w]
        :return:
            - slots: region feature vectors [b, t, m, d]
            - attn: soft assignment maps [b, t, m, h, w]
        """
        k, b, t, c, h, w = f_v.shape

        # [k, b, t, c, h, w] -> [bt, ck, h, w] -> [bt, s_dim, h, w]
        f_s = einops.rearrange(f_v, 'k b t c h w -> (b t) (c k) h w')
        f_s = self.enc(f_s)

        # flatten spatial: [bt, s_dim, h, w] -> [bt, hw, s_dim]
        bt = b * t
        inputs = einops.rearrange(f_s, 'bt d h w -> bt (h w) d')
        inputs = self.norm_input(inputs)

        # pre-compute k, v (shared across iterations)
        kk = self.proj_k(inputs)  # [bt, hw, d]
        vv = self.proj_v(inputs)  # [bt, hw, d]

        # init slots
        slots = self._init_slots(bt)  # [bt, m, d]

        # iterative attention
        for _ in range(self.n_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # attention: slots query, pixels respond
            qq = self.proj_q(slots)  # [bt, m, d]
            attn_logits = torch.bmm(qq, kk.transpose(1, 2)) * self.scale  # [bt, m, hw]
            attn = func.softmax(attn_logits, dim=1)  # compete over slots (dim=m)

            # weighted mean
            attn_norm = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)  # [bt, m, hw]
            updates = torch.bmm(attn_norm, vv)  # [bt, m, d]

            # update slots
            slots = self.gru(
                updates.flatten(0, 1),  # [bt*m, d]
                slots_prev.flatten(0, 1),
            ).unflatten(0, (bt, self.m))

            slots = slots + self.mlp(self.norm_mlp(slots))

        # final attention map for output
        qq = self.proj_q(self.norm_slots(slots))
        attn_logits = torch.bmm(qq, kk.transpose(1, 2)) * self.scale
        attn = func.softmax(attn_logits, dim=1)  # [bt, m, hw]
        attn = einops.rearrange(attn, '(b t) m (h w) -> b t m h w', b=b, t=t, h=h, w=w)

        slots = einops.rearrange(slots, '(b t) m d -> b t m d', b=b, t=t)

        return {
            'slots': slots,
            'attn': attn,
        }
