import numpy as np
from scipy.stats import norm, binomtest

def call_with_timeout(fn, args=(), kwargs=None, timeout_s=30):
    """Run fn(*args, **kwargs) under a SIGALRM wall-clock cap, the same
    pattern as benchmark_gb_scalability.py's _timed_groebner. Raises
    TimeoutError if fn does not return within timeout_s, instead of
    blocking a seed loop indefinitely on one pathological call; callers
    should catch TimeoutError explicitly and record it as a distinct
    outcome from a normal None/empty result, not conflate the two.
    """
    import signal

    def _raise(_sig, _frm):
        raise TimeoutError(f"call_with_timeout: exceeded {timeout_s}s")

    old_handler = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(timeout_s)
    try:
        return fn(*args, **(kwargs or {}))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def wilson_interval(k, n, alpha=0.05):
    """
    Wilson score interval for a binomial proportion.
    Returns (lower, upper) as floats.
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    z = norm.ppf(1 - alpha/2)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2*n)) / denom
    radius = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return (max(0, centre - radius), min(1, centre + radius))


def mcnemar_exact(correct_a, correct_b):
    """
    Exact McNemar's test on paired per-trial binary outcomes (e.g. two
    methods' exact-recovery indicators on the same (equation, sigma, seed)
    rows). Uses the discordant-pair exact binomial test rather than the
    chi-square approximation, since some baseline comparisons here have
    small discordant counts where the approximation is unreliable.
    Returns (b, c, pvalue): b = #(a correct, b wrong), c = #(a wrong, b
    correct), pvalue from a two-sided binomial test on min(b,c) of (b+c)
    trials under p=0.5. pvalue is 1.0 (no evidence of a difference) when
    b+c == 0, i.e. the two methods never disagree.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))
    if b + c == 0:
        return (b, c, 1.0)
    pvalue = binomtest(min(b, c), b + c, 0.5, alternative='two-sided').pvalue
    return (b, c, pvalue)