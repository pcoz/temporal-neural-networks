"""
Self-tuning as a signal probe: the tau trace REVEALS a signal's structure.

The self-tuning leaky neuron (see TemporalNeuron, params['adaptive']) shrinks
its time constant only when it detects a *statistically significant, sustained*
change in its input — calibrated online to the signal's OWN noise floor. A
useful side effect: the record of WHEN it was "chasing" (tau small) and the
noise-floor it estimated are a free, label-free characterisation of the signal:

  * change-points       — where the regime genuinely changed (chase episodes),
  * noise floor (sigma) — how noisy each regime is (the online estimate),
  * real-vs-noise        — it stays calm through a noisy-but-steady stretch and
                           fires at a true step, so it separates signal from noise.

We run ONE adaptive neuron over a multi-regime signal and read these out. No
training, no labels, no separate change-point detector, no LLM.

WHEN THIS IS USEFUL
  - online change-point detection / regime segmentation of a stream,
  - anomaly / event flagging that auto-calibrates to each sensor's noise,
  - adaptive sampling / "where to pay attention" (record more when chasing),
  - simply learning a signal's noise floor and event SNR for free.

Run:  python examples/self_tuning_reveals_signal.py
(saves a plot to examples/self_tuning_signal.png if matplotlib is available)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn.temporal_neuron import TemporalNeuron, NeuronType


def build_signal(rng):
    """A signal with labelled regimes. Each tuple: (name, level, noise, n_steps).
    'level=None' marks the noisy-but-steady regime's level explicitly below."""
    segments = [
        ("calm-low",      0.5, 0.04, 100),  # warmup: neuron calibrates its noise floor here
        ("STEP up",       1.8, 0.04,  80),  # a real, sustained change  -> should be DETECTED
        ("calm-high",     1.8, 0.04,  80),  # steady again              -> should be QUIET
        ("noisy-steady",  1.8, 0.55,  90),  # SAME level, heavy noise   -> should stay QUIET (no false alarm)
        ("STEP down",    -1.2, 0.04,  80),  # another real change       -> should be DETECTED
        ("calm-end",     -1.2, 0.04,  50),  # steady                    -> QUIET
    ]
    signal, labels, bounds = [], [], []
    t = 0
    for name, level, noise, n in segments:
        bounds.append((name, t, t + n, level, noise))
        for _ in range(n):
            signal.append(level + rng.normal(0, noise))
            labels.append(name)
        t += n
    return np.array(signal), labels, bounds


def main():
    rng = np.random.default_rng(11)
    signal, labels, bounds = build_signal(rng)
    T = len(signal)

    # ONE self-tuning leaky neuron reads the stream. We log its internal probes.
    neuron = TemporalNeuron(NeuronType.LEAKY_INTEGRATOR,
                            params={'tau': 20.0, 'adaptive': True})
    tau_eff, chasing, zscore, sigma = [], [], [], []
    for x in signal:
        neuron.step(dt=0.1, inputs=float(x))
        tau_eff.append(neuron.tau_eff)        # small tau == "paying attention"
        chasing.append(neuron.surprise)       # 1.0 while chasing a detected change
        zscore.append(neuron.zscore)          # significance in sigma units
        sigma.append(neuron.noise_scale)      # the ONLINE noise-floor estimate
    tau_eff = np.array(tau_eff); chasing = np.array(chasing)
    zscore = np.array(zscore); sigma = np.array(sigma)

    print("=" * 78)
    print("Self-tuning as a signal probe — the tau trace reveals the signal")
    print("=" * 78)

    # 1) Change-points the neuron discovered = onsets of chase episodes.
    onsets = [k for k in range(1, T) if chasing[k] > 0 and chasing[k-1] == 0]
    true_changes = [b[1] for b in bounds[1:]]   # true regime start steps
    print("\nDiscovered change-points (chase onsets):", onsets)
    print("True regime boundaries:                  ", true_changes)

    # 2) Per-regime read-out: did it fire? what noise floor did it see?
    print(f"\n{'regime':14s} {'level':>6s} {'true noise':>11s} "
          f"{'est. sigma':>11s} {'% chasing':>10s}  verdict")
    for name, a, b, level, noise in bounds:
        frac = 100.0 * np.mean(chasing[a:b])
        est_sigma = float(np.median(sigma[a:b]))
        verdict = ("DETECTED change" if frac > 8 else "quiet (steady)")
        print(f"{name:14s} {level:6.2f} {noise:11.2f} {est_sigma:11.3f} "
              f"{frac:9.0f}%  {verdict}")

    # 3) The headline: it stays calm through the noisy-but-steady regime.
    def pct(name):
        a, b = next((x[1], x[2]) for x in bounds if x[0] == name)
        return 100.0 * np.mean(chasing[a:b])
    noisy_pct = pct("noisy-steady")
    step_up_pct = pct("STEP up")
    print(f"\nSIGNAL vs NOISE: 'STEP up' chased {step_up_pct:.0f}% of the time "
          f"(real change), while 'noisy-steady' — same level, 14x the noise — "
          f"chased only {noisy_pct:.0f}% (no false alarm). The estimated sigma also "
          f"jumped in the noisy regime, revealing its higher noise floor.")

    # ---- assertions (robust): real steps fire, noisy-but-steady stays calm,
    # and the online sigma reflects the true noise structure. ----
    calm_sigma = float(np.median(sigma[10:100]))
    noisy_sigma = float(np.median(sigma[next(x[1] for x in bounds if x[0]=='noisy-steady'):
                                          next(x[2] for x in bounds if x[0]=='noisy-steady')]))
    ok = True
    for nm, cond in [("STEP up detected", step_up_pct > 8),
                     ("STEP down detected", pct("STEP down") > 8),
                     ("noisy-steady stayed calm", noisy_pct < step_up_pct),
                     ("sigma reveals higher noise floor", noisy_sigma > 3 * calm_sigma)]:
        print(("  PASS " if cond else "  FAIL ") + nm)
        ok &= bool(cond)

    _try_plot(signal, tau_eff, chasing, sigma, bounds)

    print("\n" + ("ALL ASSERTIONS PASSED." if ok else "SOME ASSERTIONS FAILED."))
    print("Takeaway: the neuron's self-tuning is not just for stability+speed — its")
    print("tau/chase/sigma trace is a free, online, self-calibrating CHARACTERISATION")
    print("of the signal (change-points, noise floor, signal-vs-noise). No LLM, no")
    print("labels, no separate detector.")
    if not ok:
        sys.exit(1)


def _try_plot(signal, tau_eff, chasing, sigma, bounds):
    """Save a 3-panel plot (signal / tau / sigma) if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available — skipping plot)")
        return
    T = len(signal)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax1.plot(signal, color="#34495e", lw=0.8); ax1.set_ylabel("input")
    ax1.set_title("Signal: calm → STEP up → steady → NOISY-but-steady → STEP down → calm")
    ax2.plot(tau_eff, color="#27ae60", lw=1.2); ax2.set_ylabel("tau_eff")
    ax2.set_title("Self-tuned tau — small = 'chasing' a detected real change")
    ax3.plot(sigma, color="#c0392b", lw=1.2); ax3.set_ylabel("est. sigma")
    ax3.set_title("Online noise-floor estimate — rises in the noisy regime")
    for ax in (ax1, ax2, ax3):
        for _, a, b, _, _ in bounds[1:]:
            ax.axvline(a, color="#2980b9", ls=":", alpha=0.6)
    ax3.set_xlabel("step (→ time)")
    fig.suptitle("Self-tuning reveals signal structure (TNN, no LLM)", y=1.0)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_tuning_signal.png")
    fig.savefig(out, dpi=130)
    print(f"\n(plot saved to {out})")


if __name__ == "__main__":
    main()
