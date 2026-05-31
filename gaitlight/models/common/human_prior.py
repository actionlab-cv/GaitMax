from torch import nn, Tensor

from gaitlight.models.common.human_mask import HumanMask
from gaitlight.models.common.human_pattern import HumanPattern

BG_INDEX = 10  # must match dino_pattern.py


class HumanPrior(nn.Module):
    """
    Unified human prior module combining mask estimation and body pattern discovery.
    Inference only — accepts DINO features, outputs mask + pattern + slots.

    :param s_dim: DINO feature dimension
    :param n_slots: number of body region slots (including background)
    :param n_iter: slot attention iterations
    """

    def __init__(self, s_dim: int = 384, n_slots: int = 11, n_iter: int = 3):
        super().__init__()

        self.mask = HumanMask(s_dim=s_dim, k=1)
        self.pattern = HumanPattern(s_dim=s_dim, k=1, m=n_slots, n_iter=n_iter)

        # freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, f_v: Tensor, coverage: float = 0.8) -> dict[str, Tensor]:
        """
        :param f_v: DINO features [k, b, t, c, h, w] (typically k=1, last layer only)
        :param coverage: attention mass coverage for seg (0.8 = keep top 80%)
        :return:
            - mask: binary foreground mask [b, t, h, w] bool
            - seg:  pattern segmentation [b, t, h, w] int, class indices (BG_INDEX for uncertain)
            - conf: per-slot confidence [b, t, m] float
            - slot: slot feature vectors [b, t, m, d] float
            - attn: soft attention maps [b, t, m, h, w] float
        """
        k, b, t, c, h, w = f_v.shape

        # --- mask branch ---
        mask_out = self.mask(f_v)
        mask = mask_out['mask'][:, :, 0] > 0.5  # [b, t, h, w] bool

        # --- mask features before pattern (match training: frame * mask → DINO) ---
        mask_feat = mask.unsqueeze(0).unsqueeze(3).float()  # [1, b, t, 1, h, w]
        f_v_masked = f_v * mask_feat

        # --- pattern branch ---
        pattern_out = self.pattern(f_v_masked)
        attn = pattern_out['attn']  # [b, t, m, h, w]
        slot = pattern_out['slots']  # [b, t, m, d]

        # hard segmentation with coverage thresholding
        seg = self._threshold_seg(attn, coverage)

        # per-slot confidence from attention sharpness
        pixel_max = attn.max(dim=2).values
        attn_sum = attn.sum(dim=(-2, -1)).clamp(min=1e-8)
        conf = (attn * pixel_max.unsqueeze(2)).sum(dim=(-2, -1)) / attn_sum

        return {
            'mask': mask,
            'seg': seg,
            'conf': conf,
            'slot': slot,
            'attn': attn,
        }

    @staticmethod
    def _threshold_seg(attn: Tensor, coverage: float) -> Tensor:
        """
        Produce clean seg by only keeping the top `coverage` fraction of each slot's
        attention mass. Pixels not confidently assigned → BG_INDEX.

        :param attn: [b, t, m, h, w] softmax over slots
        :param coverage: fraction of attention mass to keep per slot (e.g. 0.8)
        :return: [b, t, h, w] int class indices
        """
        b, t, m, h, w = attn.shape
        n = h * w

        # reshape to [bt, m, n]
        a = attn.reshape(b * t, m, n)

        # for each slot, find threshold that captures `coverage` of its total mass
        a_sorted, _ = a.sort(dim=-1, descending=True)  # [bt, m, n]
        cumsum = a_sorted.cumsum(dim=-1)  # [bt, m, n]
        total = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [bt, m, 1]

        # threshold: smallest value in the top-coverage set
        above_coverage = (cumsum / total) <= coverage  # [bt, m, n] bool
        # index of last element within coverage (+1 because we want to include it)
        n_keep = above_coverage.sum(dim=-1).clamp(min=1)  # [bt, m]
        thresh = a_sorted.gather(-1, (n_keep - 1).unsqueeze(-1)).squeeze(-1)  # [bt, m]

        # mask: keep pixels above threshold for each slot
        keep = a >= thresh.unsqueeze(-1)  # [bt, m, n]

        # zero out attention below threshold, then argmax
        a_thresh = a * keep.float()
        seg = a_thresh.argmax(dim=1)  # [bt, n]

        # pixels where no slot has enough attention → background
        any_kept = keep.any(dim=1)  # [bt, n]
        seg[~any_kept] = BG_INDEX

        return seg.reshape(b, t, h, w)
