# TNN: Temporal Neural Networks

**A Dynamical Systems Approach to Stable and Robust Neural Computation**

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author

**Edward Chalk** - Independent Researcher

Exploring biologically-inspired computation, temporal dynamics in neural systems, and interpretable machine learning.

- Email: edward@fleetingswallow.com
- GitHub: [@pcoz](https://github.com/pcoz)

## Overview

Temporal Neural Networks (TNNs) model each neuron as a continuous-time dynamical system rather than an instantaneous function:

```
Classical:  y = f(Wx + b)                    # Instantaneous
TNN:        dV/dt = (1/τ) * (-V + f(Wx+b))   # Evolves over time
```

This simple change provides **dramatic improvements in stability and robustness**:

| Metric | Classical | TNN | Improvement |
|--------|-----------|-----|-------------|
| Prediction flips (noise=0.5) | 3.7 | 0.9 | **75% fewer** |
| Prediction flips (noise=1.0) | 11.0 | 1.0 | **91% fewer** |
| Accuracy (noise=0.5) | 93.0% | 99.0% | **+6%** |
| Accuracy (40% dropout) | 86.0% | 94.4% | **+8.4%** |

## Key Results

- **Matches classical accuracy** on clean benchmarks (95.1% vs 95.3%)
- **75-91% fewer prediction flips** under noisy conditions
- **Superior robustness** to missing data (33% less degradation)
- **Biologically plausible** (based on Leaky Integrate-and-Fire model)
- **Interpretable dynamics** via symbolic regression (PPF)
- **Optional self-tuning `tau`** — closed-loop latency feedback gives both stability *and* responsiveness, and its trace doubles as a self-calibrating change-point / noise-floor probe (see below)

## Quick Start

```python
from tnn import ClassicalNetwork, convert_to_temporal, NeuronType

# Train a classical network
classical = ClassicalNetwork([input_size, 128, 64, n_classes], activation='tanh')
# ... training code ...

# Convert to temporal
temporal = convert_to_temporal(classical, {}, default_type=NeuronType.LEAKY_INTEGRATOR)

# Inference with temporal dynamics
temporal.reset()
for _ in range(settle_steps):
    output = temporal.step(dt=0.1, input_signal=x)
prediction = np.argmax(output)
```

## Three-Phase Pipeline

1. **Phase 1: Classical Training** - Train a standard feedforward network
2. **Phase 2: Form Discovery** - Use PPF (symbolic regression) to discover temporal dynamics
3. **Phase 3: Temporal Conversion** - Convert to TNN with discovered or default dynamics

## Self-Tuning Latency Feedback (optional)

A leaky neuron can **self-tune its time constant** via closed-loop feedback —
recovering both the stability of a large `tau` and the responsiveness of a small
one. It is opt-in (off by default; classic fixed-`tau` behaviour is unchanged):

```python
temporal.enable_self_tuning()          # every leaky neuron self-tunes its tau
```

How it works: a **high-pass + low-pass** detector flags a *sustained,
statistically-significant* change in the input — normalised by an **online
estimate of the signal's own noise floor**, so the threshold is scale-free (in
sigma units) and needs no per-signal tuning. On a detected change the neuron
drops to a fast `tau` and **chases**; the chase is then *sustained and
terminated by the neuron's own tracking-lag feedback* (a noise spike creates no
real lag, so spurious chases die immediately). Result: steady/noisy input stays
stable; a real shift is tracked quickly.

A useful side effect — the self-tuning trace (`tau_eff`, `surprise`, online
`sigma`) is a **free, label-free characterisation of the signal**: change-points,
noise floor, and signal-vs-noise. See `examples/self_tuning_reveals_signal.py`.
Every knob is documented in [docs/PARAMETERS.md](docs/PARAMETERS.md).

## Where this sits among signal-analysis tools

The self-tuning leaky neuron is a **causal, online, O(1)-state adaptive
low-pass filter with a change detector that is *coupled* to the filter** —
the detector retunes the filter rather than just reporting. That places it at
the intersection of a few classical families:

| Family | What it does | How the self-tuning TNN relates |
|---|---|---|
| **Exponential smoothing / EWMA, fixed low-pass** | One-pole smoothing with a fixed rate | A fixed-`tau` TNN neuron *is* a one-pole low-pass; self-tuning makes the rate adaptive instead of fixed. |
| **Adaptive filters (LMS / RLS)** | Adapt filter *coefficients* to track a target | TNN adapts its *time constant* (responsiveness), not weights — simpler, and aimed at the stability↔latency trade-off. |
| **Kalman / Bayesian filters** | Statistically optimal for a *known* linear-Gaussian model, with covariance tracking | TNN is model-light, nonlinear (`tanh`), no covariance bookkeeping, far cheaper, and self-calibrates its noise floor — but it is *not* statistically optimal and assumes no generative model. Use a Kalman filter when you have a good model. |
| **Change-point detectors (CUSUM, Page-Hinkley, Bayesian online CPD)** | Flag when a stream's regime changes | TNN's `surprise`/`sigma`/`z` trace is a lightweight, self-calibrating online change detector — but it also *acts* (retunes), and is O(1) state with no priors to set beyond a sigma threshold. Dedicated detectors are more principled if flagging is all you need. |
| **Wavelets / multiresolution, STFT** | Decompose a signal across scales/frequencies (often batch) | TNN is a single causal stream with one adaptive scale, not a frequency decomposition. |
| **Matched / band-pass filters** | Detect a *known* template or frequency band | TNN's high-pass+low-pass is a crude, *self-calibrating* "sustained-change" band — not frequency-selective like a designed filter. |
| **Spiking neurons / adaptive-threshold LIF** | Biological integrate-and-fire with adaptation | Same lineage; here the adaptation modulates the *time constant* via closed-loop feedback. |

**The distinctive niche:** it is *not just* a filter and *not just* a detector —
it is one self-tuning object that does adaptive temporal integration **and**
self-calibrating, feedback-coupled change response, and composes as a neuron
inside a network (so the same mechanism that filters a sensor can be a hidden
unit in a classifier). That makes it a natural fit for **cheap, streaming,
edge / online** settings, and for cases where you want the smoother, the
change-detector, and the downstream model to be the *same* differentiable
building block. It is **not** a replacement for a Kalman filter when you have a
model, nor for offline optimal estimators.

### Why this matters

- **It dissolves the stability–latency dilemma.** Every fixed filter forces a
  choice: smooth (stable, sluggish) *or* responsive (fast, jittery). Here the
  signal picks the time constant, online, per event — stable when quiet, fast on
  a real change. That trade-off is fundamental to monitoring, control, and
  perception, and is usually "solved" by hand-tuning a compromise.
- **Self-calibration removes per-deployment tuning.** The noise floor is learned
  online and the threshold is in sigma units, so the *same* configuration works
  across sensors of different scale and noisiness — decisive when you deploy a
  *fleet* of heterogeneous edge sensors and cannot tune each one.
- **One mechanism does three jobs** (smoothing + change detection + a composable
  network unit) instead of a filter + detector + model pipeline that must be
  co-tuned and kept consistent. Fewer moving parts, fewer failure modes, and the
  smoother and the detector can never disagree — they are the same object.
- **It is differentiable and composable.** Because it is a neuron, the same
  self-tuning that cleans a sensor stream can be a hidden unit in a classifier,
  so a stable perception front-end protects downstream decisions from
  noise-driven thrash (see `examples/tnn_in_reasoning_stack.py`).
- **Cheap, causal, O(1) state, and interpretable.** It runs on tiny hardware
  with no batch and no model identification, and its `tau`/`surprise`/`sigma`
  trace is a readable account of *what the signal did and when the unit paid
  attention* — not a black box.

## Project Structure

```
temporal-neural-networks/
├── tnn/                        # Core library
│   ├── __init__.py             # Package exports
│   ├── classical_phase.py      # Phase 1: Classical network training
│   ├── form_discovery.py       # Phase 2: PPF integration
│   ├── conversion.py           # Phase 3: Temporal conversion
│   ├── temporal_neuron.py      # Temporal neuron (+ self-tuning latency feedback)
│   ├── temporal_network.py     # Temporal network (+ enable_self_tuning)
│   └── temporal_training.py    # Temporal training utilities
├── examples/                   # Example experiments
│   ├── har_experiment.py             # UCI HAR baseline experiment
│   ├── tnn_advantage_test.py         # Stability and robustness tests
│   ├── ecg_experiment.py             # ECG analysis example
│   ├── tnn_streaming_v2.py           # Raw signal streaming test
│   ├── self_tuning_reveals_signal.py # Self-tuning as a signal probe (change-points/noise)
│   └── tnn_in_reasoning_stack.py     # TNN stability protects a downstream decision layer
├── tests/                      # Test suite
│   ├── test_tnn.py             # Core functionality tests
│   └── test_self_tuning.py     # Self-tuning latency-feedback tests
├── docs/                       # Documentation
│   ├── TNN_REPORT.md           # Full technical report
│   ├── PARAMETERS.md           # All tuning parameters (incl. self-tuning)
│   └── arxiv_submission/       # arXiv paper (LaTeX + PDF)
├── setup.py                    # Package installation
└── README.md
```

## Installation

```bash
git clone https://github.com/pcoz/temporal-neural-networks.git
cd temporal-neural-networks
pip install -e .
```

## Experiments

### Run the main experiment (UCI HAR)

```bash
python examples/har_experiment.py
```

### Run the advantage tests (stability, robustness)

```bash
python examples/tnn_advantage_test.py
```

### Run raw signal streaming test

```bash
python examples/tnn_streaming_v2.py
```

### Self-tuning: signal-probe and stack demos

```bash
python examples/self_tuning_reveals_signal.py   # tau trace reveals change-points / noise floor
python examples/tnn_in_reasoning_stack.py        # stability protects a downstream decision layer
```

### Run tests

```bash
python tests/test_tnn.py
python tests/test_self_tuning.py                 # self-tuning latency-feedback tests
```

## Why TNNs Matter

### Clinical Relevance

- **Alarm Fatigue**: 75-91% fewer prediction flips = fewer false alarms
- **Sensor Reliability**: +8.4% accuracy at 40% dropout = robust to real-world sensors
- **Graceful Degradation**: Predictable behavior under stress

### Biological Plausibility

The TNN equation is the **Leaky Integrate-and-Fire (LIF) model** used throughout computational neuroscience:
- Neurons integrate inputs over time
- Leak toward resting potential
- Natural temporal filtering

## Citation

```bibtex
@article{chalk2025tnn,
  title={Temporal Neural Networks: A Dynamical Systems Approach to Stable and Robust Neural Computation},
  author={Chalk, Edward},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

Edward Chalk - edward@fleetingswallow.com
