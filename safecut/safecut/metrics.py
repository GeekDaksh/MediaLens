import numpy as np


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n (works for small n and for 0 or n)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def fmt_ci(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={n})"


def bootstrap_mean_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def box_coverage(gt, boxes):
    """Fraction of the gt box (x0,y0,x1,y1) covered by the union of boxes (pixel mask, exact)."""
    x0, y0, x1, y1 = [int(v) for v in gt]
    if x1 <= x0 or y1 <= y0:
        return 0.0
    m = np.zeros((y1 - y0, x1 - x0), bool)
    for bx0, by0, bx1, by1 in boxes:
        a0, b0 = max(int(bx0), x0) - x0, max(int(by0), y0) - y0
        a1, b1 = min(int(bx1), x1) - x0, min(int(by1), y1) - y0
        if a1 > a0 and b1 > b0:
            m[b0:b1, a0:a1] = True
    return float(m.mean())
