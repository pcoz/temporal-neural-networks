"""
Classical Phase - Standard neural network training.

Phase 1 of the TNN approach: Train a classical feedforward network
using backpropagation to learn the structure of the task.

The trained network provides:
1. Weights and biases that encode learned relationships
2. Activation patterns that reveal temporal dynamics (for Phase 2)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for classical training."""
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    momentum: float = 0.9
    activation: str = 'tanh'  # 'tanh', 'relu', 'sigmoid'
    record_activations: bool = True  # For Phase 2


class ClassicalNetwork:
    """
    A standard feedforward neural network.

    This is Phase 1: learn the structure using classical backpropagation.
    After training, activation patterns can be extracted for PPF analysis.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'tanh'
    ):
        """
        Create a classical feedforward network.

        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation

        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

        # For momentum
        self.weight_velocities = [np.zeros_like(w) for w in self.weights]
        self.bias_velocities = [np.zeros_like(b) for b in self.biases]

        # Activation recording (for Phase 2)
        self.activation_history: List[List[np.ndarray]] = []
        self.recording = False

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return np.tanh(x)

    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.activation_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        else:
            return 1 - np.tanh(x) ** 2

    def forward(self, x: np.ndarray, record: bool = False) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_size) or (input_size,)
            record: Whether to record activations for this pass

        Returns:
            Output array
        """
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self.layer_inputs = [x]  # For backprop
        self.layer_outputs = [x]

        activations_this_pass = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self.layer_outputs[-1] @ w + b
            self.layer_inputs.append(z)

            # Last layer: no activation (for regression) or softmax (for classification)
            if i == len(self.weights) - 1:
                a = z  # Linear output
            else:
                a = self._activation(z)

            self.layer_outputs.append(a)

            if record and self.recording:
                activations_this_pass.append(a.copy())

        if record and self.recording:
            self.activation_history.append(activations_this_pass)

        return self.layer_outputs[-1]

    def backward(self, y_true: np.ndarray, learning_rate: float, momentum: float = 0.9):
        """
        Backward pass - compute gradients and update weights.

        Args:
            y_true: True labels
            learning_rate: Learning rate
            momentum: Momentum coefficient
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)

        batch_size = y_true.shape[0]

        # Output layer error (MSE gradient)
        delta = (self.layer_outputs[-1] - y_true) / batch_size

        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dw = self.layer_outputs[i].T @ delta
            db = np.sum(delta, axis=0)

            # Update with momentum
            self.weight_velocities[i] = momentum * self.weight_velocities[i] - learning_rate * dw
            self.bias_velocities[i] = momentum * self.bias_velocities[i] - learning_rate * db

            self.weights[i] += self.weight_velocities[i]
            self.biases[i] += self.bias_velocities[i]

            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._activation_derivative(self.layer_inputs[i])

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        momentum: float = 0.9
    ) -> float:
        """
        Single training step.

        Returns:
            Loss for this batch
        """
        # Forward
        y_pred = self.forward(x, record=True)

        # Compute loss (MSE)
        loss = np.mean((y_pred - y) ** 2)

        # Backward
        self.backward(y, learning_rate, momentum)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions without recording."""
        return self.forward(x, record=False)

    def start_recording(self):
        """Start recording activations for Phase 2."""
        self.recording = True
        self.activation_history = []

    def stop_recording(self):
        """Stop recording activations."""
        self.recording = False

    def get_activation_history(self) -> List[List[np.ndarray]]:
        """
        Get recorded activation history.

        Returns:
            List of [time_step][layer] -> activation array
        """
        return self.activation_history

    def get_neuron_timeseries(self, layer: int, neuron: int) -> np.ndarray:
        """
        Get activation time series for a specific neuron.

        Args:
            layer: Layer index (0 = first hidden layer)
            neuron: Neuron index within layer

        Returns:
            Array of activations over time
        """
        return np.array([
            step[layer][0, neuron] if step[layer].shape[0] == 1 else step[layer][:, neuron].mean()
            for step in self.activation_history
        ])


def train_classical(
    model: ClassicalNetwork,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig = None,
    x_val: np.ndarray = None,
    y_val: np.ndarray = None,
    verbose: bool = True
) -> dict:
    """
    Train a classical network using backpropagation.

    Args:
        model: ClassicalNetwork to train
        x_train: Training inputs
        y_train: Training labels
        config: Training configuration
        x_val: Validation inputs (optional)
        y_val: Validation labels (optional)
        verbose: Print progress

    Returns:
        Training history dict
    """
    if config is None:
        config = TrainingConfig()

    n_samples = x_train.shape[0]
    n_batches = max(1, n_samples // config.batch_size)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Start recording activations if requested
    if config.record_activations:
        model.start_recording()

    for epoch in range(config.epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0

        for batch in range(n_batches):
            start = batch * config.batch_size
            end = min(start + config.batch_size, n_samples)

            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            loss = model.train_step(
                x_batch, y_batch,
                config.learning_rate,
                config.momentum
            )
            epoch_loss += loss

        epoch_loss /= n_batches
        history['train_loss'].append(epoch_loss)

        # Validation
        if x_val is not None and y_val is not None:
            val_pred = model.predict(x_val)
            val_loss = np.mean((val_pred - y_val) ** 2)
            history['val_loss'].append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch + 1}/{config.epochs} - Loss: {epoch_loss:.6f}"
            if x_val is not None:
                msg += f" - Val Loss: {history['val_loss'][-1]:.6f}"
            print(msg)

    if config.record_activations:
        model.stop_recording()

    return history


def create_sequence_data(
    sequence_fn: Callable[[float], float],
    n_sequences: int,
    sequence_length: int,
    input_window: int,
    output_horizon: int = 1,
    dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence prediction data from a function.

    This is useful for testing temporal dynamics discovery.

    Args:
        sequence_fn: Function t -> value
        n_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        input_window: Number of past values as input
        output_horizon: How far ahead to predict
        dt: Time step

    Returns:
        (x_train, y_train) arrays
    """
    x_list = []
    y_list = []

    for seq in range(n_sequences):
        # Random starting point
        t_start = np.random.uniform(0, 100)

        # Generate sequence
        t_points = t_start + np.arange(sequence_length) * dt
        values = np.array([sequence_fn(t) for t in t_points])

        # Create input-output pairs
        for i in range(input_window, sequence_length - output_horizon):
            x = values[i - input_window:i]
            y = values[i + output_horizon - 1:i + output_horizon]
            x_list.append(x)
            y_list.append(y)

    return np.array(x_list), np.array(y_list)
