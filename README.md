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
    output = temporal.step(dt=0.1, external_input=x)
prediction = np.argmax(output)
```

## Three-Phase Pipeline

1. **Phase 1: Classical Training** - Train a standard feedforward network
2. **Phase 2: Form Discovery** - Use PPF (symbolic regression) to discover temporal dynamics
3. **Phase 3: Temporal Conversion** - Convert to TNN with discovered or default dynamics

## Project Structure

```
temporal-neural-networks/
├── tnn/                        # Core library
│   ├── __init__.py             # Package exports
│   ├── classical_phase.py      # Phase 1: Classical network training
│   ├── form_discovery.py       # Phase 2: PPF integration
│   ├── conversion.py           # Phase 3: Temporal conversion
│   ├── temporal_neuron.py      # Temporal neuron implementation
│   ├── temporal_network.py     # Temporal network implementation
│   └── temporal_training.py    # Temporal training utilities
├── examples/                   # Example experiments
│   ├── har_experiment.py       # UCI HAR baseline experiment
│   ├── tnn_advantage_test.py   # Stability and robustness tests
│   ├── ecg_experiment.py       # ECG analysis example
│   └── tnn_streaming_v2.py     # Raw signal streaming test
├── tests/                      # Test suite
│   └── test_tnn.py             # Core functionality tests
├── docs/                       # Documentation
│   ├── TNN_REPORT.md           # Full technical report
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

### Run tests

```bash
python tests/test_tnn.py
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
