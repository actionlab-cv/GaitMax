"""
Efficient tensor-based visualization for mask and pattern overlays.
No matplotlib — pure tensor ops + kornia morphology.
"""

import torch
import torch.nn.functional as func
from kornia.morphology import dilation, erosion
from torch import Tensor

# Maximally separated palette for 10 body regions + background (RGB 0-255)
# Colors chosen for maximum perceptual distance across hue wheel
# fmt: off
PATTERN_COLORS = torch.tensor([
    [244, 63, 94],  # 0: head        — rose
    [99, 102, 241],  # 1: l_upper_arm — indigo
    [245, 158, 11],  # 2: l_forearm   — amber
    [14, 165, 233],  # 3: r_upper_arm — sky
    [217, 70, 239],  # 4: r_forearm   — fuchsia
    [34, 197, 94],  # 5: torso       — green
    [251, 146, 60],  # 6: l_thigh     — orange
    [6, 182, 212],  # 7: l_calf      — cyan
    [163, 230, 53],  # 8: r_thigh     — lime
    [168, 85, 247],  # 9: r_calf      — violet
    [0, 0, 0],  # 10: background
], dtype=torch.uint8)
# fmt: on

MASK_COLOR = torch.tensor([34, 211, 238], dtype=torch.uint8)  # cyan-400


def clean_seg(seg: Tensor, kernel_size: int = 5, bg_idx: int = 10) -> Tensor:
    """
    Remove salt-and-pepper noise from a label map using mode (majority) filter.
    Only cleans foreground pixels; background pixels are preserved.

    :param seg: [h, w] int — class indices
    :param kernel_size: neighborhood size (odd)
    :param bg_idx: background class index (preserved as-is)
    :return: [h, w] int — cleaned label map
    """
    n_cls = seg.max().item() + 1
    h, w = seg.shape
    pad = kernel_size // 2
    bg_mask = seg == bg_idx

    # one-hot → sum in neighborhood → argmax = majority vote
    one_hot = torch.zeros(1, n_cls, h, w, device=seg.device)
    one_hot.scatter_(1, seg.long().reshape(1, 1, h, w), 1.0)

    # exclude background from voting: zero out bg channel so fg classes always win
    one_hot[0, bg_idx] = 0

    votes = func.avg_pool2d(
        func.pad(one_hot, [pad] * 4, mode='replicate'),
        kernel_size=kernel_size, stride=1,
    )  # [1, n_cls, h, w]

    result = votes.squeeze(0).argmax(dim=0)  # [h, w]

    # restore background pixels
    result[bg_mask] = bg_idx
    return result


def _soft_edge(mask_hw: Tensor, width: int) -> Tensor:
    """
    Compute soft anti-aliased edge from a binary mask via morphological ops + Gaussian blur.

    :param mask_hw: [h, w] bool or float
    :param width: edge thickness
    :return: [h, w] float in [0, 1], soft edge
    """
    device = mask_hw.device
    m = mask_hw.float().unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    kernel = torch.ones(width, width, device=device)

    dilated = dilation(m, kernel)
    eroded = erosion(m, kernel)
    edge = (dilated - eroded).squeeze()  # [h, w] hard edge

    # soften with Gaussian blur for anti-aliasing
    k = max(width * 2 - 1, 3)
    if k % 2 == 0:
        k += 1
    edge_4d = edge.unsqueeze(0).unsqueeze(0)
    edge_soft = func.avg_pool2d(
        func.pad(edge_4d, [k // 2] * 4, mode='replicate'),
        kernel_size=k, stride=1,
    ).squeeze()

    return edge_soft.clamp(0, 1)


def overlay_mask(
        frame: Tensor,
        mask: Tensor,
        color: Tensor = MASK_COLOR,
        fill_alpha: float = 0.3,
        edge_width: int = 3,
        edge_alpha: float = 0.9,
) -> Tensor:
    """
    Overlay binary mask on frame with colored fill and soft anti-aliased edges.

    :param frame: [3, h, w] uint8
    :param mask: [h, w] bool
    :param color: [3] uint8 RGB
    :param fill_alpha: fill transparency
    :param edge_width: edge thickness
    :param edge_alpha: edge opacity
    :return: [3, h, w] uint8
    """
    out = frame.clone().float()
    c = color.to(frame.device).float().view(3, 1, 1)

    # fill
    m = mask.float().unsqueeze(0)  # [1, h, w]
    out = out * (1 - m * fill_alpha) + c * m * fill_alpha

    # soft edge
    edge = _soft_edge(mask, edge_width)  # [h, w]
    e = edge.unsqueeze(0) * edge_alpha  # [1, h, w]
    out = out * (1 - e) + c * e

    return out.clamp(0, 255).to(torch.uint8)


_TORSO_IDX = 5
_HEAD_IDX = 0

# default per-slot coverage: fraction of 2D Gaussian mass the rectangle encloses
DEFAULT_COVERAGE = 0.30
SLOT_COVERAGE = {
    _HEAD_IDX: 0.80,
}


def _coverage_to_k(coverage: float) -> float:
    """Convert coverage (0-1) to scale factor k for rectangle half-extent.

    A rectangle with half-widths k*σ along each principal axis of a 2D Gaussian
    captures erf(k/√2)² of the mass. Invert: k = √2 * erfinv(√coverage).
    """
    from scipy.special import erfinv
    import math
    return math.sqrt(2) * float(erfinv(math.sqrt(coverage)))


def fit_slots(attn: Tensor) -> dict[str, Tensor]:
    """
    Fit 2D Gaussian parameters (mean, eigenvalues, eigenvectors) for each slot.

    :param attn: [m, h, w] slot attention maps
    :return: dict with:
        - mu: [m, 2] centers
        - lam: [m, 2] eigenvalues (lam1 >= lam2)
        - v1: [m, 2] major eigenvector
    """
    m, h, w = attn.shape
    device = attn.device

    fy, fx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij',
    )
    feat_coords = torch.stack([fx, fy], dim=-1)  # [h, w, 2]

    mu_all = torch.zeros(m, 2, device=device)
    lam_all = torch.zeros(m, 2, device=device)
    v1_all = torch.zeros(m, 2, device=device)

    for i in range(m):
        a = attn[i]
        if a.sum() < 1e-6:
            continue

        alpha = a / a.sum().clamp(min=1e-8)

        mu = (alpha.unsqueeze(-1) * feat_coords).sum(dim=(0, 1))
        diff = feat_coords - mu
        cov = torch.einsum('hwi,hwj,hw->ij', diff, diff, alpha)
        cov = cov + torch.eye(2, device=device) * 0.2

        ca, cb, cd = cov[0, 0], cov[0, 1], cov[1, 1]
        disc = ((ca - cd).pow(2) + 4 * cb.pow(2)).clamp(min=1e-8).sqrt()
        lam1 = (ca + cd + disc) / 2
        lam2 = (ca + cd - disc) / 2

        v1 = torch.stack([cb, lam1 - ca])
        v1 = v1 / v1.norm().clamp(min=1e-8)

        mu_all[i] = mu
        lam_all[i] = torch.stack([lam1, lam2])
        v1_all[i] = v1

    return {'mu': mu_all, 'lam': lam_all, 'v1': v1_all}


def smooth_slot_sizes(
        all_lam: Tensor,
        alpha: float = 0.3,
) -> Tensor:
    """
    Temporal EMA smoothing on eigenvalues (rectangle sizes) only.

    :param all_lam: [t, m, 2] eigenvalues per frame per slot
    :param alpha: EMA weight for current frame (lower = smoother)
    :return: [t, m, 2] smoothed eigenvalues
    """
    t = all_lam.shape[0]
    smoothed = torch.empty_like(all_lam)
    smoothed[0] = all_lam[0]
    for i in range(1, t):
        smoothed[i] = alpha * all_lam[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def overlay_pattern(
        frame: Tensor,
        attn: Tensor,
        conf: Tensor | None = None,
        slot_params: dict[str, Tensor] | None = None,
        colors: Tensor = PATTERN_COLORS,
        dim_factor: float = 0.5,
        fill_alpha: float = 0.3,
        edge_alpha: float = 0.8,
        coverage: dict[int, float] | None = None,
        edge_band: float = 0.4,
        corner_radius: float = 0.3,
        center_radius: float = 4.0,
        conf_thresh: float = 0.5,
        bg_idx: int = 10,
        skip: tuple[int, ...] = (_TORSO_IDX,),
) -> Tensor:
    """
    Overlay fitted rounded rectangles for each body region based on attention maps.

    1. Dim entire image
    2. For each foreground slot, fit a rotated rectangle from attention's principal axes
    3. Rectangle size = per-slot coverage of Gaussian mass (head=80%, limbs=70%)
    4. Draw border and center point in bright region color

    :param frame: [3, h, w] uint8
    :param attn: [m, h_feat, w_feat] float — slot attention maps (softmax over m)
    :param colors: [n_cls, 3] uint8
    :param dim_factor: darken factor
    :param fill_alpha: rectangle fill transparency
    :param edge_alpha: border + center opacity
    :param coverage: per-slot coverage override {slot_idx: fraction}
    :param edge_band: border thickness in feature-space units
    :param corner_radius: rounded corner radius as fraction of shorter half-extent
    :param center_radius: center dot radius in frame pixels
    :param bg_idx: background slot index (skipped)
    :param skip: additional slot indices to skip
    :return: [3, h, w] uint8
    """
    cov_map = {**SLOT_COVERAGE, **(coverage or {})}
    device = frame.device
    colors = colors.to(device)
    _, h_frame, w_frame = frame.shape
    m, h_feat, w_feat = attn.shape

    # fit slot params if not pre-computed
    if slot_params is None:
        slot_params = fit_slots(attn)

    # coordinate grid at frame resolution, mapped to feature space
    gy, gx = torch.meshgrid(
        torch.linspace(0, h_feat - 1, h_frame, device=device),
        torch.linspace(0, w_feat - 1, w_frame, device=device),
        indexing='ij',
    )
    coords = torch.stack([gx, gy], dim=-1)  # [h_frame, w_frame, 2]

    # dim entire image
    out = frame.float() * dim_factor

    skip_set = set(skip) | {bg_idx}

    for slot_idx in range(m):
        if slot_idx in skip_set:
            continue
        if conf is not None and conf[slot_idx] < conf_thresh:
            continue

        mu = slot_params['mu'][slot_idx]
        lam = slot_params['lam'][slot_idx]
        v1 = slot_params['v1'][slot_idx]

        lam1, lam2 = lam[0], lam[1]
        if lam1 < 1e-6:
            continue
        v2 = torch.stack([-v1[1], v1[0]])

        # half-extents: k * σ along each principal axis, k from coverage
        k = _coverage_to_k(cov_map.get(slot_idx, DEFAULT_COVERAGE))
        e1 = k * lam1.clamp(min=0).sqrt()
        e2 = k * lam2.clamp(min=0).sqrt()

        # rounded corner radius
        r = corner_radius * torch.minimum(e1, e2)

        # project frame coords onto principal axes (relative to center)
        d = coords - mu  # [h_frame, w_frame, 2]
        p1 = (d * v1).sum(dim=-1)  # [h_frame, w_frame]
        p2 = (d * v2).sum(dim=-1)

        # rounded rectangle SDF
        # q = |p| - half_extent + radius
        q1 = p1.abs() - e1 + r
        q2 = p2.abs() - e2 + r
        q1_pos = q1.clamp(min=0)
        q2_pos = q2.clamp(min=0)
        outside_dist = (q1_pos.pow(2) + q2_pos.pow(2)).sqrt()
        inside_dist = torch.minimum(torch.maximum(q1, q2), torch.zeros_like(q1))
        sdf = outside_dist + inside_dist - r  # negative inside, positive outside

        # fill: smooth falloff
        fill_smooth = (1 - (sdf / (edge_band * 0.5)).clamp(0, 1))

        # edge: ring at border (sdf ≈ 0)
        edge_smooth = (1 - (sdf.abs() / (edge_band * 0.5)).clamp(0, 1))

        # center dot
        center_d = ((coords - mu).pow(2).sum(dim=-1)).sqrt()
        # convert center_radius from frame pixels to feature-space units
        cr_feat = center_radius * (h_feat / h_frame)
        center_dot = (1 - (center_d / cr_feat).clamp(0, 1))

        c = colors[slot_idx].float().view(3, 1, 1)

        # apply fill
        f = fill_smooth.unsqueeze(0) * fill_alpha
        out = out * (1 - f) + c * f

        # apply edge
        e = edge_smooth.unsqueeze(0) * edge_alpha
        out = out * (1 - e) + c * e

        # apply center dot
        cd_mask = center_dot.unsqueeze(0) * edge_alpha
        out = out * (1 - cd_mask) + c * cd_mask

    return out.clamp(0, 255).to(torch.uint8)
