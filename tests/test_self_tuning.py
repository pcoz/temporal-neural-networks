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


if __name__ == "__main__":
    print("Self-tuning latency feedback — unit tests")
    test_off_by_default_is_identical()
    test_best_of_both_on_step()
    test_change_vs_noise_discrimination()
    test_network_enable_self_tuning()
    print("ALL TESTS PASSED.")
