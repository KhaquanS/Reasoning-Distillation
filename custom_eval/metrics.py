"""Pass@k estimator, shared-sample protocol from the HumanEval evaluation."""

from math import comb


def estimate_pass_at_k(n, c, k):
    """Probability that a uniformly selected k-subset contains a correct sample."""
    if not (1 <= k <= n and 0 <= c <= n):
        raise ValueError("Require 1 <= k <= n and 0 <= c <= n.")
    return 1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
