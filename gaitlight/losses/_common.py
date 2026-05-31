"""Shared numerical constant for all loss functions.

EPS is the single epsilon used across losses for: sqrt-distance floors (finite
gradient at zero distance), division/std guards, log guards, and matrix-inverse
regularization. 1e-6 matches torch.nn.functional.pairwise_distance's default.
"""

EPS = 1e-6
