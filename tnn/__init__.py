"""
TNN - Temporal Neural Networks

A Dynamical Systems Approach to Stable and Robust Neural Computation

Author: Edward Chalk
Email: edward@fleetingswallow.com
GitHub: https://github.com/pcoz/temporal-neural-networks
License: MIT

A neural network architecture where each neuron is a temporal process,
not an instantaneous function. Uses PPF (Partial Form Finding) to discover
the mathematical forms governing neuron dynamics.

Key Results:
- 75-91% fewer prediction flips under noise
- +8.4% accuracy at 40% feature dropout
- Matches classical accuracy on clean benchmarks

Three-phase approach:
1. Classical training - learn structure with standard backprop
2. Form discovery - use PPF to find temporal dynamics in activations
3. Temporal implementation - neurons become continuous processes

Example:
    from tnn import ClassicalNetwork, convert_to_temporal, NeuronType

    # Phase 1: Classical training
    classical = ClassicalNetwork([input_size, 128, 64, n_classes])
    train_classical(classical, x_train, y_train)

    # Phase 2: Discover dynamics (optional)
    forms = discover_forms(activation_history, config)

    # Phase 3: Convert to temporal
    temporal = convert_to_temporal(classical, forms)

    # Run with temporal dynamics
    temporal.reset()
    for _ in range(settle_steps):
        output = temporal.step(dt=0.1, external_input=x)
"""

__version__ = "0.1.0"
__author__ = "Edward Chalk"
__email__ = "edward@fleetingswallow.com"
__license__ = "MIT"

from .temporal_neuron import TemporalNeuron, NeuronType, NeuronState
from .temporal_network import (
    TemporalNetwork, TemporalLayer, LayerConfig,
    create_heterogeneous_network, create_uniform_network
)
from .classical_phase import (
    ClassicalNetwork, train_classical, TrainingConfig,
    create_sequence_data
)
from .form_discovery import (
    discover_forms, analyze_activations,
    DiscoveredForm, FormDiscoveryConfig,
    summarize_discoveries
)
from .conversion import (
    convert_to_temporal, convert_with_neuron_types,
    analyze_conversion_quality
)
