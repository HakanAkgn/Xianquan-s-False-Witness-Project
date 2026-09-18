"""Exact finite reductions: do not mistake these identities for model validation."""
import numpy as np
from scipy.special import softmax, logsumexp


def witness_mass(scores, witness_mask):
    a = softmax(np.asarray(scores, dtype=np.float64))
    return float(a[np.asarray(witness_mask, dtype=bool)].sum())


def score_susceptibility(scores, values):
    """Gradient of the fixed linear readout sum_i softmax(s)_i b_i."""
    a = softmax(np.asarray(scores, dtype=np.float64))
    b = np.asarray(values, dtype=np.float64)
    return a * (b - a @ b)


def factorial_effects(d00, d10, d01, d11):
    """Exact 2x2 selection/payload decomposition at fixed recipient residual/Q."""
    selection, payload = d10 - d00, d01 - d00
    interaction = d11 - d10 - d01 + d00
    return {"selection": selection, "payload": payload,
            "interaction": interaction, "total": d11 - d00}


def orbit_average(predictor, orbit):
    """Uniform finite-orbit projection; oracle generator, not a learned defense."""
    return np.mean([np.asarray(predictor(x), float) for x in orbit], axis=0)


def theory_checks(seed=20260918, trials=1000):
    rng = np.random.default_rng(seed)
    gradient_error = mass_error = decomposition_error = 0.0
    eps = 1e-5
    for _ in range(trials):
        n = int(rng.integers(2, 40))
        s, b = rng.normal(size=n), rng.normal(size=n)
        j = int(rng.integers(n))
        plus, minus = s.copy(), s.copy()
        plus[j] += eps
        minus[j] -= eps
        fd = (softmax(plus) @ b - softmax(minus) @ b)/(2*eps)
        gradient_error = max(gradient_error, abs(fd-score_susceptibility(s,b)[j]))
        m = int(rng.integers(1, 50))
        st, sw = rng.normal(size=2)
        exact = m*np.exp(sw)/(np.exp(st)+m*np.exp(sw))
        direct = softmax(np.r_[st, np.repeat(sw,m)])[1:].sum()
        mass_error = max(mass_error, abs(exact-direct))
        d = rng.normal(size=4)
        e = factorial_effects(*d)
        decomposition_error = max(decomposition_error,
            abs(e['total']-e['selection']-e['payload']-e['interaction']))
    return {"trials": trials, "seed": seed,
            "max_gradient_finite_difference_error": gradient_error,
            "max_softmax_mass_identity_error": mass_error,
            "max_factorial_identity_error": decomposition_error,
            "interpretation": "Numerical identity checks, not transformer predictions or novelty evidence"}
