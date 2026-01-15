"""
Temporal Network - A network of coupled temporal neurons.

All neurons share the same time base and evolve together as a
coupled dynamical system. The network can be run continuously,
producing outputs that unfold over time.
"""

import numpy as np
from typing import List, Optional, Dict, Callable, Tuple
from dataclasses import dataclass
from .temporal_neuron import TemporalNeuron, NeuronType, NeuronState


@dataclass
class LayerConfig:
    """Configuration for a temporal layer."""
    size: int
    neuron_type: NeuronType
    params: Optional[dict] = None


class TemporalLayer:
    """
    A layer of temporal neurons.

    All neurons in a layer can share the same type and parameters,
    or each can be individually configured.
    """

    def __init__(
        self,
        size: int,
        neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATOR,
        params: Optional[dict] = None,
        neurons: Optional[List[TemporalNeuron]] = None
    ):
        self.size = size

        if neurons:
            self.neurons = neurons
        else:
            self.neurons = [
                TemporalNeuron(neuron_type=neuron_type, params=params)
                for _ in range(size)
            ]

    def step(self, dt: float, inputs: np.ndarray) -> np.ndarray:
        """
        Advance all neurons by one time step.

        Args:
            dt: Time step
            inputs: Array of inputs, one per neuron

        Returns:
            Array of outputs
        """
        outputs = np.zeros(self.size)
        for i, neuron in enumerate(self.neurons):
            outputs[i] = neuron.step(dt, inputs[i])
        return outputs

    def reset(self):
        """Reset all neurons."""
        for neuron in self.neurons:
            neuron.reset()

    def get_outputs(self) -> np.ndarray:
        """Get current outputs of all neurons."""
        return np.array([n.state.output for n in self.neurons])

    def get_states(self) -> List[NeuronState]:
        """Get current states of all neurons."""
        return [n.state for n in self.neurons]


class TemporalNetwork:
    """
    A network of temporal layers, all evolving on shared time.

    The network processes inputs continuously, with each neuron
    maintaining its own temporal dynamics while being influenced
    by its connections to other neurons.

    Architecture:
        input -> [temporal layers] -> output

    All layers share the same dt and evolve synchronously.
    """

    def __init__(
        self,
        layer_configs: List[LayerConfig],
        input_size: int,
        output_size: int
    ):
        """
        Create a temporal network.

        Args:
            layer_configs: Configuration for each hidden layer
            input_size: Dimension of input
            output_size: Dimension of output
        """
        self.input_size = input_size
        self.output_size = output_size

        # Create layers
        self.layers: List[TemporalLayer] = []
        for config in layer_configs:
            layer = TemporalLayer(
                size=config.size,
                neuron_type=config.neuron_type,
                params=config.params
            )
            self.layers.append(layer)

        # Initialize weights
        self._init_weights()

        # Time tracking
        self.t = 0.0
        self.history: List[Dict] = []

    def _init_weights(self):
        """Initialize connection weights between layers."""
        self.weights = []
        self.biases = []

        sizes = [self.input_size] + [l.size for l in self.layers] + [self.output_size]

        for i in range(len(sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (sizes[i] + sizes[i+1]))
            w = np.random.randn(sizes[i], sizes[i+1]) * scale
            b = np.zeros(sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def set_weights(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        """Set weights from a trained classical network."""
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]

    def step(self, dt: float, input_signal: np.ndarray) -> np.ndarray:
        """
        Advance the network by one time step.

        Args:
            dt: Time step size
            input_signal: Current input values

        Returns:
            Current output values
        """
        # Forward pass through layers
        x = input_signal

        for i, layer in enumerate(self.layers):
            # Weighted input to this layer
            weighted = x @ self.weights[i] + self.biases[i]

            # Each neuron processes its input with temporal dynamics
            x = layer.step(dt, weighted)

        # Final output layer (simple weighted sum for now)
        output = x @ self.weights[-1] + self.biases[-1]

        # Record state
        self.history.append({
            't': self.t,
            'input': input_signal.copy(),
            'layer_outputs': [l.get_outputs().copy() for l in self.layers],
            'output': output.copy()
        })

        self.t += dt
        return output

    def run(
        self,
        input_signal: Callable[[float], np.ndarray],
        duration: float,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the network continuously for a duration.

        Args:
            input_signal: Function t -> input array
            duration: How long to run (in time units)
            dt: Time step

        Returns:
            Tuple of (time_points, outputs)
        """
        self.reset()

        times = []
        outputs = []

        t = 0.0
        while t < duration:
            inp = input_signal(t)
            out = self.step(dt, inp)

            times.append(t)
            outputs.append(out)

            t += dt

        return np.array(times), np.array(outputs)

    def run_on_sequence(
        self,
        inputs: np.ndarray,
        dt: float = 0.01,
        steps_per_input: int = 10
    ) -> np.ndarray:
        """
        Run the network on a sequence of inputs.

        Each input is held for `steps_per_input` time steps,
        allowing temporal dynamics to unfold.

        Args:
            inputs: Array of shape (seq_len, input_size)
            dt: Time step
            steps_per_input: How many dt steps per input

        Returns:
            Array of outputs (one per input)
        """
        self.reset()
        outputs = []

        for inp in inputs:
            # Hold this input for several time steps
            for _ in range(steps_per_input):
                out = self.step(dt, inp)

            # Record output after dynamics have settled
            outputs.append(out)

        return np.array(outputs)

    def reset(self):
        """Reset all layers and time."""
        for layer in self.layers:
            layer.reset()
        self.t = 0.0
        self.history = []

    def get_activation_histories(self) -> Dict[int, np.ndarray]:
        """Get activation histories for all layers."""
        histories = {}
        for i, layer in enumerate(self.layers):
            layer_history = []
            for neuron in layer.neurons:
                layer_history.append(neuron.get_activation_history())
            histories[i] = np.array(layer_history)
        return histories


def create_heterogeneous_network(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    neuron_types: Optional[List[NeuronType]] = None
) -> TemporalNetwork:
    """
    Create a network with different neuron types in different layers.

    Args:
        input_size: Input dimension
        hidden_sizes: Size of each hidden layer
        output_size: Output dimension
        neuron_types: Type for each layer (defaults to varied types)

    Returns:
        TemporalNetwork with heterogeneous dynamics
    """
    if neuron_types is None:
        # Default: vary the types
        available_types = [
            NeuronType.LEAKY_INTEGRATOR,
            NeuronType.OSCILLATOR,
            NeuronType.ADAPTING,
            NeuronType.RESONATOR
        ]
        neuron_types = [
            available_types[i % len(available_types)]
            for i in range(len(hidden_sizes))
        ]

    configs = [
        LayerConfig(size=size, neuron_type=ntype)
        for size, ntype in zip(hidden_sizes, neuron_types)
    ]

    return TemporalNetwork(configs, input_size, output_size)


def create_uniform_network(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATOR,
    params: Optional[dict] = None
) -> TemporalNetwork:
    """
    Create a network with the same neuron type throughout.

    Args:
        input_size: Input dimension
        hidden_sizes: Size of each hidden layer
        output_size: Output dimension
        neuron_type: Type for all neurons
        params: Parameters for neurons

    Returns:
        TemporalNetwork with uniform dynamics
    """
    configs = [
        LayerConfig(size=size, neuron_type=neuron_type, params=params)
        for size in hidden_sizes
    ]

    return TemporalNetwork(configs, input_size, output_size)
