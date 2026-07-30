#!/usr/bin/env python3
"""
Unit tests for the rationality cost function.
"""

import numpy as np
from sr_gb import rationality_cost

def test_rationality_cost():
    # Test with simple rational coefficients
    # Coefficients: [1, 2, -3] -> infinity norm = 3 -> normalized: [1/3, 2/3, -1]
    c = np.array([1.0, 2.0, -3.0])
    cost = rationality_cost(c, max_denom=16, eps=1e-3)
    # Expected normalized costs:
    # 1/3: p=1, q=3 => log2(1)+log2(3)+1 = 0 + 1.58496 + 1 = 2.58496
    # 2/3: p=2, q=3 => log2(2)+log2(3)+1 = 1 + 1.58496 + 1 = 3.58496
    # -1: p=-1, q=1 => log2(1)+log2(1)+1 = 1
    expected = (np.log2(3) + 1) + (np.log2(2) + np.log2(3) + 1) + 1
    # Simplify: (log2(3)+1) + (1+log2(3)+1) + 1 = 2*log2(3) + 4
    # = 2*1.58496 + 4 = 3.16992 + 4 = 7.16992
    assert abs(cost - expected) < 1e-3, f"Expected {expected}, got {cost}"

    # Test with half-integer
    c2 = np.array([0.5, 1.0])
    cost2 = rationality_cost(c2, max_denom=16, eps=1e-3)
    # 0.5 -> p=1, q=2 => log2(1)+log2(2)+1 = 0+1+1=2
    # 1.0 -> p=1, q=1 => log2(1)+log2(1)+1 = 1
    expected2 = 2 + 1
    assert abs(cost2 - expected2) < 1e-3, f"Expected {expected2}, got {cost2}"

    # Test with irrational that cannot be approximated within eps
    c3 = np.array([np.sqrt(2), 1.0])
    cost3 = rationality_cost(c3, max_denom=16, eps=1e-6)
    # sqrt(2) ≈ 1.4142, not representable with denom <=16 within 1e-6, so cost inf
    assert cost3 == float('inf'), f"Expected inf, got {cost3}"

    print("All rationality cost tests passed.")

if __name__ == "__main__":
    test_rationality_cost()