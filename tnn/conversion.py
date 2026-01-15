"""
Conversion - Transform classical networks into temporal networks.

Phase 3 of the TNN approach: Take a trained classical network and
discovered forms, create a TemporalNetwork where each neuron is
governed by its discovered mathematical dynamics.

The result is a network that truly evolves in time, not just computes.
"""

import numpy as np
from typing import Dict, List, Optional

from .temporal_neuron import TemporalNeuron, NeuronType, NeuronState
from .temporal_network import TemporalNetwork, TemporalLayer, LayerConfig
from .classical_phase import ClassicalNetwork
from .form_discovery import DiscoveredForm, create_temporal_form_function


def convert_to_temporal(
    classical: ClassicalNetwork,
    discovered_forms: Dict[int, List[DiscoveredForm]],
    default_type: NeuronType = NeuronType.LEAKY_INTEGRATOR
) -> TemporalNetwork:
    """
    Convert a classical network to a temporal network using discovered forms.

    Args:
        classical: Trained ClassicalNetwork
        discovered_forms: Forms discovered in Phase 2, keyed by layer index
        default_type: Fallback neuron type when no form is discovered

    Returns:
        TemporalNetwork with neurons governed by discovered dynamics
    """
    # Get layer sizes (excluding input and output)
    hidden_sizes = classical.layer_sizes[1:-1]
    input_size = classical.layer_sizes[0]
    output_size = classical.layer_sizes[-1]

    # Build layer configurations
    layer_configs = []

    for layer_idx, size in enumerate(hidden_sizes):
        # Check if we have discovered forms for this layer
        forms = discovered_forms.get(layer_idx, [])

        if forms:
            # Create heterogeneous layer with discovered forms
            config = LayerConfig(
                size=size,
                neuron_type=NeuronType.CUSTOM,  # Will be overridden per-neuron
                params=None
            )
        else:
            # No forms discovered - use default type
            config = LayerConfig(
                size=size,
                neuron_type=default_type,
                params=None
            )

        layer_configs.append(config)

    # Create the temporal network structure
    temporal_net = TemporalNetwork(
        layer_configs=layer_configs,
        input_size=input_size,
        output_size=output_size
    )

    # Transfer weights from classical network
    temporal_net.set_weights(classical.weights, classical.biases)

    # Now customize individual neurons with discovered forms
    for layer_idx, forms in discovered_forms.items():
        if layer_idx >= len(temporal_net.layers):
            continue

        layer = temporal_net.layers[layer_idx]

        # Build neuron-to-form mapping
        neuron_to_form = {}
        for form in forms:
            for neuron_id in form.neuron_ids:
                if neuron_id < layer.size:
                    neuron_to_form[neuron_id] = form

        # Update neurons with their discovered forms
        for neuron_id in range(layer.size):
            if neuron_id in neuron_to_form:
                form = neuron_to_form[neuron_id]
                custom_fn = create_temporal_form_function(form)

                # Replace the neuron with a custom one
                layer.neurons[neuron_id] = TemporalNeuron(
                    neuron_type=NeuronType.CUSTOM,
                    custom_form=custom_fn,
                    params={'is_derivative': False, 'input_weight': 0.3}
                )
            else:
                # No form discovered - keep default type
                layer.neurons[neuron_id] = TemporalNeuron(
                    neuron_type=default_type
                )

    return temporal_net


def convert_with_neuron_types(
    classical: ClassicalNetwork,
    layer_types: List[NeuronType],
    layer_params: Optional[List[dict]] = None
) -> TemporalNetwork:
    """
    Convert a classical network using specified neuron types.

    Alternative to form discovery - manually specify what kind of
    temporal dynamics each layer should have.

    Args:
        classical: Trained ClassicalNetwork
        layer_types: NeuronType for each hidden layer
        layer_params: Optional params for each layer

    Returns:
        TemporalNetwork with specified dynamics
    """
    hidden_sizes = classical.layer_sizes[1:-1]
    input_size = classical.layer_sizes[0]
    output_size = classical.layer_sizes[-1]

    if len(layer_types) != len(hidden_sizes):
        raise ValueError(f"Expected {len(hidden_sizes)} layer types, got {len(layer_types)}")

    if layer_params is None:
        layer_params = [None] * len(hidden_sizes)

    layer_configs = [
        LayerConfig(size=size, neuron_type=ntype, params=params)
        for size, ntype, params in zip(hidden_sizes, layer_types, layer_params)
    ]

    temporal_net = TemporalNetwork(
        layer_configs=layer_configs,
        input_size=input_size,
        output_size=output_size
    )

    temporal_net.set_weights(classical.weights, classical.biases)

    return temporal_net


def analyze_conversion_quality(
    classical: ClassicalNetwork,
    temporal: TemporalNetwork,
    test_inputs: np.ndarray,
    dt: float = 0.01,
    steps_per_input: int = 10
) -> Dict[str, float]:
    """
    Compare classical and temporal network outputs.

    Args:
        classical: Original classical network
        temporal: Converted temporal network
        test_inputs: Test inputs
        dt: Time step for temporal network
        steps_per_input: Steps to run for each input

    Returns:
        Quality metrics
    """
    # Get classical predictions (instantaneous)
    classical_outputs = classical.predict(test_inputs)

    # Get temporal predictions (after settling)
    temporal_outputs = temporal.run_on_sequence(
        test_inputs,
        dt=dt,
        steps_per_input=steps_per_input
    )

    # Compute metrics
    mse = np.mean((classical_outputs - temporal_outputs) ** 2)
    correlation = np.corrcoef(
        classical_outputs.flatten(),
        temporal_outputs.flatten()
    )[0, 1]

    # Output variance ratio
    classical_var = np.var(classical_outputs)
    temporal_var = np.var(temporal_outputs)
    var_ratio = temporal_var / classical_var if classical_var > 0 else 0

    return {
        'mse': float(mse),
        'correlation': float(correlation),
        'variance_ratio': float(var_ratio),
        'classical_mean': float(np.mean(classical_outputs)),
        'temporal_mean': float(np.mean(temporal_outputs))
    }


def create_hybrid_network(
    classical: ClassicalNetwork,
    discovered_forms: Dict[int, List[DiscoveredForm]],
    temporal_layers: List[int],
    static_layers: List[int]
) -> TemporalNetwork:
    """
    Create a hybrid network with some temporal and some static layers.

    Useful for experimenting with which layers benefit most from
    temporal dynamics.

    Args:
        classical: Trained ClassicalNetwork
        discovered_forms: PPF-discovered forms
        temporal_layers: Layer indices to make temporal
        static_layers: Layer indices to keep static-ish

    Returns:
        Hybrid TemporalNetwork
    """
    hidden_sizes = classical.layer_sizes[1:-1]
    input_size = classical.layer_sizes[0]
    output_size = classical.layer_sizes[-1]

    layer_configs = []

    for layer_idx, size in enumerate(hidden_sizes):
        if layer_idx in temporal_layers:
            # Use discovered form or oscillator for temporal richness
            forms = discovered_forms.get(layer_idx, [])
            if forms:
                config = LayerConfig(size=size, neuron_type=NeuronType.CUSTOM)
            else:
                config = LayerConfig(size=size, neuron_type=NeuronType.OSCILLATOR)
        else:
            # Use fast leaky integrator (nearly instantaneous)
            config = LayerConfig(
                size=size,
                neuron_type=NeuronType.LEAKY_INTEGRATOR,
                params={'tau': 1.0}  # Very fast time constant
            )

        layer_configs.append(config)

    temporal_net = TemporalNetwork(
        layer_configs=layer_configs,
        input_size=input_size,
        output_size=output_size
    )

    temporal_net.set_weights(classical.weights, classical.biases)

    # Apply discovered forms to temporal layers
    for layer_idx in temporal_layers:
        if layer_idx >= len(temporal_net.layers):
            continue

        forms = discovered_forms.get(layer_idx, [])
        if not forms:
            continue

        layer = temporal_net.layers[layer_idx]
        neuron_to_form = {}
        for form in forms:
            for neuron_id in form.neuron_ids:
                if neuron_id < layer.size:
                    neuron_to_form[neuron_id] = form

        for neuron_id, form in neuron_to_form.items():
            custom_fn = create_temporal_form_function(form)
            layer.neurons[neuron_id] = TemporalNeuron(
                neuron_type=NeuronType.CUSTOM,
                custom_form=custom_fn,
                params={'is_derivative': False}
            )

    return temporal_net
