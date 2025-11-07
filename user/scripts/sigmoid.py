"""シンプルなシグモイド関数の実装."""

from __future__ import annotations

import numpy as np


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """ロジスティックシグモイド σ(x)=1/(1+e^{-x}) を返す."""
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    sample = np.linspace(-6, 6, 5)
    print(sigmoid(sample))
