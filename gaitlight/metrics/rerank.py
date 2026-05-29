import torch


@torch.inference_mode()
def re_ranking(
        q_g_dist: torch.Tensor,
        q_q_dist: torch.Tensor,
        g_g_dist: torch.Tensor,
        k1: int = 20,
        k2: int = 6,
        lambda_value: float = 0.3,
) -> torch.Tensor:
    """
    Re-ranking by k-reciprocal encoding.

    :param q_g_dist: query-to-gallery distance [nq, ng]
    :param q_q_dist: query-to-query distance   [nq, nq]
    :param g_g_dist: gallery-to-gallery distance [ng, ng]
    :param k1: k-reciprocal neighborhood size
    :param k2: query expansion neighborhood size
    :param lambda_value: interpolation weight for original distance (0 = pure Jaccard, 1 = original)
    :return: re-ranked distance matrix [nq, ng]
    """
    nq, ng = q_g_dist.shape
    n = nq + ng
    device = q_g_dist.device

    # --- build full distance matrix [n, n]: rows/cols are [query | gallery] ---
    dist = torch.cat([
        torch.cat([q_q_dist, q_g_dist], dim=1),
        torch.cat([q_g_dist.T, g_g_dist], dim=1),
    ], dim=0).float()
    dist /= dist.max().clamp(min=1e-6)

    # --- k-nearest neighbors (self is always rank-0 since dist[i,i]=0) ---
    k1 = min(k1 + 1, n)  # +1 to include self, clamp to n
    k1h = min(k1 // 2 + 1, n)  # half-k version

    _, nn_k1 = dist.topk(k1, dim=1, largest=False)  # [n, k1]
    _, nn_k1h = dist.topk(k1h, dim=1, largest=False)  # [n, k1h]

    # --- boolean knn matrices (self excluded via [:, 1:]) ---
    knn = torch.zeros(n, n, dtype=torch.bool, device=device)
    knn.scatter_(1, nn_k1[:, 1:], True)
    knh = torch.zeros(n, n, dtype=torch.bool, device=device)
    knh.scatter_(1, nn_k1h[:, 1:], True)

    # --- k-reciprocal neighbors: R(i) = N(i) ∩ Nᵀ(i) ---
    R = knn & knn.T  # [n, n]  j ∈ R(i)  iff  j ∈ N(i) and i ∈ N(j)
    Rh = knh & knh.T  # [n, n]  half-k version

    # --- expansion: add Rh(j) to R*(i) when j ∈ R(i) and |R(i) ∩ Rh(j)| ≥ 2/3|R(i)| ---
    # overlap[i,j] = |R(i) ∩ Rh(j)|
    overlap = R.float() @ Rh.float().T  # [n, n]
    r_size = R.sum(dim=1, keepdim=True).float()  # [n, 1]
    should_expand = R & (overlap >= 2 / 3 * r_size)  # [n, n]
    R_star = R | (should_expand.float() @ Rh.float() > 0)  # [n, n]

    # --- soft membership V[i,j] = exp(-d(i,j)) / Z_i  for j ∈ R*(i) ---
    weights = torch.where(R_star, torch.exp(-dist), torch.zeros_like(dist))
    weights /= weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    V = weights.half()  # store as fp16 to save memory

    # --- query expansion: average V over k2 nearest neighbors (including self) ---
    if k2 > 1:
        k2 = min(k2, n)
        V = V[nn_k1[:, :k2]].mean(dim=1)  # [n, n]

    # --- Jaccard distance via L1 identity ---
    # inter(a, b) = Σ min(a_k, b_k) = (‖a‖₁ + ‖b‖₁ − ‖a−b‖₁) / 2
    # union(a, b) = Σ max(a_k, b_k) = (‖a‖₁ + ‖b‖₁ + ‖a−b‖₁) / 2
    # jaccard_dist = 1 − inter/union = 1 − (sa+sb−L1) / (sa+sb+L1)
    V_q = V[:nq].float()  # [nq, n]
    V_g = V[nq:].float()  # [ng, n]
    sa = V_q.sum(dim=1)  # [nq]
    sb = V_g.sum(dim=1)  # [ng]
    l1 = torch.cdist(V_q, V_g, p=1)  # [nq, ng]
    sab = sa.unsqueeze(1) + sb.unsqueeze(0)  # [nq, ng]
    jaccard_dist = 1 - (sab - l1) / (sab + l1).clamp(min=1e-6)

    # --- interpolate with original distance ---
    orig = q_g_dist.float() / q_g_dist.max().clamp(min=1e-6)
    return (1 - lambda_value) * jaccard_dist + lambda_value * orig
