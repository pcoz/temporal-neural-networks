# TNN Parameter Reference

Every `TemporalNeuron` carries a `params` dict (see
`tnn/temporal_neuron.py::_default_params`). This document lists **all** tuning
parameters, their defaults, units, role, and guidance — with a detailed section
on the **self-tuning latency feedback** added to the leaky integrator.

## How to set parameters

```python
from tnn.temporal_neuron import TemporalNeuron, NeuronType

# directly, per neuron
n = TemporalNeuron(NeuronType.LEAKY_INTEGRATOR,
                   params={'tau': 20.0, 'adaptive': True, 'tau_min': 2.0})

# convenience: turn on self-tuning (leaky neurons)
n.enable_self_tuning(tau_min=2.0)              # one neuron
network.enable_self_tuning(tau_min=2.0)        # every leaky neuron in a network
```

`dt` is **not** a neuron parameter — it is passed to `step(dt, inputs)` /
`run(..., dt=...)` and sets the integration step. Time constants below in
"ms" interact with `dt` through `dt/tau`; the self-tuning detector windows are
in **steps** (dt-independent) by design.

---

## Leaky integrator — base dynamics (`NeuronType.LEAKY_INTEGRATOR`)

`dV/dt = -(V - v_rest)/tau + gain * I`,  output `= tanh(V)`.

| Param | Default | Units | Role | Guidance |
|---|---|---|---|---|
| `tau` | `20.0` | ms | Time constant of integration (also the **stable / max** tau when self-tuning is on) | Larger = more stable + more low-pass smoothing, slower to respond; smaller = faster, jitterier. Effective memory ≈ `tau/dt` steps. |
| `v_rest` | `0.0` | mV | Resting potential the voltage leaks toward | Usually 0. |
| `gain` | `1.0` | — | Scales input drive `I` | Raise if inputs are small; lower if saturating `tanh` too hard. |

---

## Self-tuning latency feedback (leaky integrator, opt-in)

Turn on with `params['adaptive'] = True` (or `enable_self_tuning(...)`). Off by
default, so classic fixed-`tau` behaviour is unchanged.

**Mechanism (see `_self_tuning_tau`).** A closed loop:
1. **Onset detector (difference-of-EMAs band-pass + online variance).** The raw
   input drive is passed through a fast EMA and a slow EMA; their difference
   `bp = m_fast - m_slow` is a band-pass change signal (≈ 0 at steady DC, spikes
   at a step, rejects single-sample noise). It is z-scored by an **online
   estimate of the signal's own band-pass variance**, giving `z` in *sigma
   units*. Raw drive (not the `tanh` output) is used so a step between two large
   values is still visible.
2. **Latch.** When `z > z_thresh` (a statistically significant change), `tau`
   drops to `tau_min` — the neuron "chases".
3. **Feedback termination.** The chase ends only when the change is fully
   accommodated: the band-pass has subsided (`z < z_exit`) **and** the output
   has caught up (`out_lag < caught_thresh`). A 1-step noise spike trips `z` for
   an instant but creates no real output lag, so the chase dies immediately.

| Param | Default | Units | Role | Guidance |
|---|---|---|---|---|
| `adaptive` | `False` | bool | Master switch for self-tuning | `True` to enable; `False` = classic fixed `tau`. |
| `tau_min` | `2.0` | ms | The **fast** tau used while chasing a detected change | Lower = faster catch-up (more responsive), but the responsiveness floor is ≈ `tau_min/dt` steps. Keep `< tau`. |
| `fast_steps` | `8` | steps | Band-pass **fast** mean window | Smaller = more sensitive to abrupt change but lets through more single-sample noise; larger = smoother onset. Keep `< slow_steps`. |
| `slow_steps` | `40` | steps | Band-pass **slow** mean window (the "recent normal") | Larger = the band-pass stays nonzero longer after a step (chase sustained longer); smaller = sharper, more transient change-point spikes. |
| `sigma_steps` | `60` | steps | Window of the ONLINE band-pass variance estimate; also the **warmup** length before `tau` is allowed to shrink | Span enough samples to estimate the noise during a steady period. The first `sigma_steps` are warmup (variance calibrating; `tau` held at `tau`). |
| `z_thresh` | `3.0` | σ (sigma units) | Onset threshold: a change is "real" past this many sigma. **Scale-free** — the same value works across signals of any magnitude/noise | Raise (e.g. 4–5) for fewer false triggers in heavy noise; lower (e.g. 2) to catch subtler changes. This is the closest thing to a universal knob. |
| `z_exit` | `1.5` | σ | Hysteresis: the chase may end once `z` falls back below this | Keep `< z_thresh` to avoid chattering at the boundary. |
| `caught_thresh` | `0.2` | output units (`tanh`, 0–2) | The output-space lag at which the neuron is deemed "caught up", ending the chase | Smaller = wait until the output has settled very precisely (longer chases); larger = end sooner. |

> **Known limitation / future extension.** A *very gradual drift* can slip under
> the instantaneous z-bar, because the drift's own band-pass slowly inflates the
> online variance it is measured against. Robust slow-drift detection is best
> added by accumulating the normalised band-pass **CUSUM-style** (a running sum
> that trips on small persistent deviations). The current detector targets
> abrupt regime changes with strong noise rejection.

### Important: there is no universal "best" setting

The **time-scales** (`baseline_steps`, `sigma_steps`, `tau`, `tau_min`) are
inherently signal- and application-dependent — what counts as "fast", "the
recent normal", or "long enough to be real" depends on the signal. The
*scale* problem is removed (σ is estimated online; `z_thresh` is in sigma
units), but the *time-scale* problem is genuine. Rather than pretend one
setting fits all, use the self-tuning trace itself to **reveal** a signal's
scales — see `examples/self_tuning_reveals_signal.py`, which reads out the
discovered change-points and the online noise floor.

### Inspecting the self-tuning state

After each `step`, the neuron exposes (for logging / the worked examples):

| Attribute | Meaning |
|---|---|
| `neuron.tau_eff` | the effective tau used this step (`tau_min` while chasing, else `tau`) |
| `neuron.surprise` | `1.0` while chasing a detected change, else `0.0` |
| `neuron.zscore` | current change significance, in sigma units |
| `neuron.noise_scale` | the online noise-floor (σ) estimate |
| `neuron.out_lag` | the output-space tracking lag feeding the chase-termination |

---

## Other neuron types

| Type | Params (default) |
|---|---|
| `INTEGRATOR` | `gain` (1.0) |
| `OSCILLATOR` | `frequency` (10.0 Hz), `amplitude` (1.0), `baseline` (0.0) |
| `RESONATOR` | `resonant_freq` (10.0 Hz), `damping` (0.1), `gain` (1.0) |
| `ADAPTING` | `tau_v` (20.0), `tau_adapt` (100.0), `adapt_strength` (0.5), `v_rest` (0.0) |
| `BURSTING` | `tau_fast` (10.0), `tau_slow` (100.0), `burst_threshold` (0.8), `quiescent_threshold` (0.2) |
| `CUSTOM` | supplied by the PPF-discovered form (`custom_form`) |

Self-tuning latency feedback currently applies to the **leaky integrator**.
