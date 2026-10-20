"""Mechanical application of the pre-registered keep criteria (reports/experiments/PREREGISTRATION.md)."""


def mc(r, min_seasons=5):
    """Main criterion: improves in >= min_seasons seasons AND pooled paired CI excludes zero in the favorable direction."""
    return bool(r['seasons_improved'] >= min_seasons and r['ci_excludes_zero_in_favor'])


def nh(r, tol=0.005):
    """No-harm: pooled point estimate not worse than baseline by more than tol (relative)."""
    return bool(r['diff'] <= tol * abs(r['base_mean']))


def describe(r):
    return (f"diff {r['diff']:+.5f} (base {r['base_mean']:.4f}), 95% CI [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}], "
            f"{r['seasons_improved']}/{r['seasons']} seasons improved")
