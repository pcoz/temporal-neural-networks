"""
Unit tests for the self-tuning latency feedback (adaptive leaky integrator).

Assert-based, runnable directly:  python tests/test_self_tuning.py
Covers: backward compatibility (off by default), best-of-both on a step,
self-calibrated change-vs-noise discrimination, and the network switch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn.temporal_neuron import TemporalNeuron, NeuronType
from tnn import ClassicalNetwork, convert_to_temporal


def _run_neuron(params, signal, dt=0.1, seed=1, noise=0.0):
    """Run one neuron over a signal (optionally adding noise); return outputs."""
    rng = np.random.default_rng(seed)
    n = TemporalNeuron(NeuronType.LEAKY_INTEGRATOR, params=dict(params))
    out = []
    for x in signal:
        n.step(dt=dt, inputs=float(x) + (rng.normal(0, noise) if noise else 0.0))
        out.append(n.state.output)
    return np.array(out)


def test_off_by_default_is_identical():
    """Adaptive defaults to False -> behaviour must match an explicit fixed tau."""
    sig = np.sin(np.linspace(0, 6, 120))
    a = _run_neuron({'tau': 20.0}, sig)                       # plain
    b = _run_neuron({'tau': 20.0, 'adaptive': False}, sig)    # explicitly off
    assert np.allclose(a, b), "adaptive=False must be byte-identical to fixed tau"
    print("  PASS  off-by-default is identical to fixed tau (backward compatible)")


def test_best_of_both_on_step():
    """Adaptive should keep ~slow stability AND beat slow latency on a real step."""
    T = 240
    sig = np.where(np.arange(T) < 120, 1.5, -1.5)             # step after warmup
    def stab_lat(params):
        o = _run_neuron(params, sig, seed=1, noise=0.8)
        stab = float(o[60:120].std())                         # jitter in a steady window
        cr = np.where(o[120:] < 0)[0]
        lat = int(cr[0]) if len(cr) else 10**9                # steps to flip after the step
        return stab, lat
    s_stab, s_lat = stab_lat({'tau': 20.0})                   # slow
    a_stab, a_lat = stab_lat({'tau': 20.0, 'adaptive': True}) # adaptive
    assert a_lat < s_lat, f"adaptive latency {a_lat} should beat slow {s_lat}"
    assert a_stab <= s_stab + 1e-3, f"adaptive jitter {a_stab} should stay ~slow {s_stab}"
    print(f"  PASS  best-of-both on step (latency {a_lat} < slow {s_lat}; jitter ~slow)")


def test_change_vs_noise_discrimination():
    """Fires on a real step; stays (relatively) calm through a noisy-but-steady
    stretch; and the online sigma reflects the higher noise."""
    rng = np.random.default_rng(11)
    seg = [(0.5, 0.04, 100), (1.8, 0.04, 80), (1.8, 0.55, 90)]  # calm, STEP, noisy-steady
    sig, bounds, t = [], [], 0
    for level, nz, n in seg:
        bounds.append((t, t + n)); t += n
        sig += [level + rng.normal(0, nz) for _ in range(n)]
    nu = TemporalNeuron(NeuronType.LEAKY_INTEGRATOR, params={'tau': 20.0, 'adaptive': True})
    chasing, sigma = [], []
    for x in sig:
        nu.step(dt=0.1, inputs=float(x)); chasing.append(nu.surprise); sigma.append(nu.noise_scale)
    chasing, sigma = np.array(chasing), np.array(sigma)
    step_pct = chasing[bounds[1][0]:bounds[1][1]].mean()
    noisy_pct = chasing[bounds[2][0]:bounds[2][1]].mean()
    calm_sigma = np.median(sigma[10:100]); noisy_sigma = np.median(sigma[bounds[2][0]:bounds[2][1]])
    assert step_pct > 0.05, "the real step must trigger a chase"
    assert noisy_pct < step_pct, "noisy-but-steady must chase less than the real step"
    assert noisy_sigma > 3 * calm_sigma, "online sigma must reveal the higher noise floor"
    print(f"  PASS  change-vs-noise (step {step_pct:.0%} > noisy {noisy_pct:.0%}; sigma reveals noise)")


def test_network_enable_self_tuning():
    """network.enable_self_tuning flips leaky neurons and reports the count."""
    net = ClassicalNetwork([3, 8, 3], activation='tanh')
    temporal = convert_to_temporal(net, {})
    n = temporal.enable_self_tuning(tau_min=2.0)
    assert n > 0, "should switch at least one leaky neuron to adaptive"
    flagged = [nu.params.get('adaptive') for L in temporal.layers for nu in L.neurons]
    assert any(flagged), "neurons must have adaptive set"
    print(f"  PASS  network.enable_self_tuning switched {n} neurons")


def _trace(params, sig):
    nu = TemporalNeuron(NeuronType.LEAKY_INTEGRATOR, params=dict(params, adaptive=True))
    z, sg, ch = [], [], []
    for x in sig:
        nu.step(dt=1.0, inputs=float(x)); z.append(nu.zscore); sg.append(nu.noise_scale); ch.append(nu.surprise)
    return np.array(z), np.array(sg), np.array(ch)


def test_zscore_not_capped_by_sigma_window():
    """z must scale with the size of a change. Scoring against a variance that
    already includes the current bp^2 capped z at sqrt(sigma_steps)."""
    rng = np.random.default_rng(3)
    p = {'tau': 5.0, 'fast_steps': 1, 'slow_steps': 10, 'sigma_steps': 25}
    base = 1.0 + rng.normal(0, 0.05, 60)
    z_small, _, _ = _trace(p, np.r_[base, base[:5] + 0.5])
    z_big, _, _ = _trace(p, np.r_[base, base[:5] + 5.0])
    cap = 25 ** 0.5
    assert z_big[60] > 2 * cap, f"big step z={z_big[60]:.1f} should exceed the old cap {cap}"
    assert z_big[60] > 5 * z_small[60], "z must grow with the change magnitude"
    print(f"  PASS  z not capped (big step z={z_big[60]:.0f} vs old cap {cap:.0f})")


def test_robust_sigma_after_event():
    """After a large event, a second, smaller-but-real step must still be
    detected, and sigma must inflate less than with the unclipped estimator."""
    rng = np.random.default_rng(5)
    p = {'tau': 5.0, 'fast_steps': 1, 'slow_steps': 10, 'sigma_steps': 25}
    # big 3-day spike (80 sigma), back to baseline, then a modest real step (10 sigma)
    sig = np.r_[np.full(40, 1.0), np.full(3, 5.0), np.full(27, 1.0), np.full(15, 1.5)] + rng.normal(0, 0.05, 85)
    z, sg, ch = _trace(p, sig)
    # the 2nd step must be significant on its own (it may fall inside the
    # still-latched chase of the first event, so test z, not chase onsets)
    assert z[70] > 3.0, f"second step must be significant (z={z[70]:.1f})"
    z_raw, sg_raw, _ = _trace(dict(p, sigma_clip=None), sig)
    assert z_raw[70] < 3.0, "control: the unclipped estimator misses it"
    assert np.max(sg_raw[40:70]) > 2 * np.max(sg[40:70]), "clipping must curb inflation"
    print(f"  PASS  robust sigma: 2nd step z={z[70]:.1f} (unclipped {z_raw[70]:.1f}); "
          f"peak sigma {np.max(sg[40:70]):.2f} vs unclipped {np.max(sg_raw[40:70]):.2f}")


if __name__ == "__main__":
    print("Self-tuning latency feedback — unit tests")
    test_off_by_default_is_identical()
    test_best_of_both_on_step()
    test_change_vs_noise_discrimination()
    test_network_enable_self_tuning()
    test_zscore_not_capped_by_sigma_window()
    test_robust_sigma_after_event()
    print("ALL TESTS PASSED.")
