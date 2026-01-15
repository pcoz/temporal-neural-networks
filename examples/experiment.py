"""
TNN Experiment - Full pipeline demonstration.

This script demonstrates the complete three-phase TNN approach:
1. Train a classical network on a temporal task
2. Use PPF to discover the mathematical forms in activations
3. Convert to a temporal network and compare

The task: Predict the next value in oscillating signals.
We specifically record activations during continuous sequence processing
to capture how neurons respond over time.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn import (
    ClassicalNetwork, train_classical, TrainingConfig,
    discover_forms, FormDiscoveryConfig, summarize_discoveries,
    convert_to_temporal, analyze_conversion_quality,
    create_uniform_network, NeuronType
)


def generate_temporal_data(
    n_samples: int = 1000,
    input_window: int = 10,
    noise_level: float = 0.05
) -> tuple:
    """
    Generate data with temporal structure.

    Creates sequences combining oscillations at different frequencies.
    """
    print("Generating temporal training data...")

    x_list = []
    y_list = []

    for _ in range(n_samples):
        # Random parameters for this sequence
        freq1 = np.random.uniform(0.2, 0.5)
        phase1 = np.random.uniform(0, 2 * np.pi)
        amp1 = np.random.uniform(0.5, 1.0)

        # Generate sequence
        t = np.arange(input_window + 1)
        signal = amp1 * np.sin(freq1 * t + phase1)

        # Add small noise
        signal += np.random.randn(len(signal)) * noise_level

        x_list.append(signal[:-1])
        y_list.append([signal[-1]])

    return np.array(x_list), np.array(y_list)


def record_continuous_activations(
    model: ClassicalNetwork,
    duration: int = 500,
    freq: float = 0.3
):
    """
    Record activations while processing a continuous signal.

    This captures how neurons actually respond over TIME, not just
    to different samples. Essential for PPF form discovery.
    """
    print(f"Recording activations for continuous signal (duration={duration})...")

    model.start_recording()

    # Process a continuous oscillating signal step by step
    input_size = model.layer_sizes[0]

    for t in range(duration):
        # Create input that represents a sliding window of a continuous signal
        signal = np.array([
            np.sin(freq * (t - i) + 0.5)
            for i in range(input_size)
        ])
        model.forward(signal.reshape(1, -1), record=True)

    model.stop_recording()

    return model.get_activation_history()


def run_experiment():
    """Run the complete TNN experiment."""
    print("=" * 60)
    print("TNN EXPERIMENT: Temporal Neural Networks via Form Discovery")
    print("=" * 60)

    # Configuration
    input_window = 10
    hidden_sizes = [32, 16]
    output_size = 1

    # ================================================================
    # PHASE 1: Generate data and train classical network
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Classical Training")
    print("=" * 60)

    # Generate training and test data
    x_train, y_train = generate_temporal_data(n_samples=2000, input_window=input_window)
    x_test, y_test = generate_temporal_data(n_samples=200, input_window=input_window)

    print(f"Training data shape: {x_train.shape} -> {y_train.shape}")
    print(f"Test data shape: {x_test.shape} -> {y_test.shape}")

    # Create and train classical network
    layer_sizes = [input_window] + hidden_sizes + [output_size]
    classical = ClassicalNetwork(layer_sizes=layer_sizes, activation='tanh')

    config = TrainingConfig(
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        momentum=0.9,
        record_activations=True  # Critical for Phase 2!
    )

    print(f"\nTraining classical network: {layer_sizes}")
    history = train_classical(
        classical, x_train, y_train,
        config=config,
        x_val=x_test, y_val=y_test,
        verbose=True
    )

    # Evaluate classical network
    classical_pred = classical.predict(x_test)
    classical_mse = np.mean((classical_pred - y_test) ** 2)
    print(f"\nClassical network test MSE: {classical_mse:.6f}")

    # ================================================================
    # PHASE 2: Discover temporal forms with PPF
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Form Discovery with PPF")
    print("=" * 60)

    # Record activations during CONTINUOUS signal processing
    # This is key - we need to see how neurons respond over time,
    # not just to different independent samples
    activation_history = record_continuous_activations(
        classical, duration=500, freq=0.3
    )
    print(f"Recorded {len(activation_history)} time steps")

    # Configure form discovery
    discovery_config = FormDiscoveryConfig(
        population_size=200,
        generations=20,
        max_depth=4,
        parsimony_coefficient=0.002,
        min_r_squared=0.5,
        cluster_neurons=True,
        n_clusters=5  # 5 clusters per layer
    )

    print("\nDiscovering mathematical forms in activations...")
    try:
        discovered_forms = discover_forms(
            activation_history,
            config=discovery_config,
            verbose=True
        )

        print("\n" + summarize_discoveries(discovered_forms))

    except Exception as e:
        print(f"\nForm discovery encountered an error: {e}")
        print("Continuing with default neuron types...")
        discovered_forms = {}

    # ================================================================
    # PHASE 3: Convert to temporal network
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Temporal Conversion")
    print("=" * 60)

    if discovered_forms:
        print("Converting to temporal network with discovered forms...")
        temporal = convert_to_temporal(
            classical,
            discovered_forms,
            default_type=NeuronType.LEAKY_INTEGRATOR
        )
    else:
        print("Using default oscillator dynamics (no forms discovered)...")
        temporal = create_uniform_network(
            input_size=input_window,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            neuron_type=NeuronType.OSCILLATOR
        )
        temporal.set_weights(classical.weights, classical.biases)

    # Compare outputs
    print("\nComparing classical vs temporal network...")
    quality = analyze_conversion_quality(
        classical, temporal, x_test[:50],  # Use subset for speed
        dt=0.1,
        steps_per_input=20
    )

    print(f"  MSE between classical/temporal: {quality['mse']:.6f}")
    print(f"  Correlation: {quality['correlation']:.4f}")
    print(f"  Variance ratio: {quality['variance_ratio']:.4f}")

    # ================================================================
    # Run temporal network continuously
    # ================================================================
    print("\n" + "=" * 60)
    print("TEMPORAL DYNAMICS DEMONSTRATION")
    print("=" * 60)

    # Create a continuous input signal
    def input_signal(t):
        """Oscillating input signal."""
        return np.array([
            np.sin(0.2 * t) * 0.5,   # Slow oscillation
            np.cos(0.5 * t) * 0.3,   # Faster oscillation
            np.sin(0.1 * t + 1.0),   # Phase-shifted
        ] + [0.0] * (input_window - 3))  # Pad to input_window

    print("Running temporal network for 100 time units...")
    temporal.reset()
    times, outputs = temporal.run(
        input_signal=input_signal,
        duration=100.0,
        dt=0.1
    )

    print(f"  Generated {len(times)} time points")
    print(f"  Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"  Output std: {outputs.std():.3f}")

    # Show temporal evolution
    print("\nOutput evolution (first 20 steps):")
    for i in range(min(20, len(outputs))):
        bar = "#" * int(abs(outputs[i, 0]) * 20)
        sign = "+" if outputs[i, 0] >= 0 else "-"
        print(f"  t={times[i]:5.1f}: {sign}{bar:20s} ({outputs[i, 0]:+.3f})")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"""
Classical Network:
  - Architecture: {layer_sizes}
  - Test MSE: {classical_mse:.6f}
  - Activations recorded: {len(activation_history)}

Form Discovery:
  - Forms discovered: {sum(len(f) for f in discovered_forms.values())}
  - Layers analyzed: {len(discovered_forms)}

Temporal Network:
  - Conversion quality (correlation): {quality['correlation']:.4f}
  - Can run continuously: YES
  - Time-evolving outputs: YES

The temporal network is now a continuous dynamical system where
each neuron evolves according to its discovered (or default)
mathematical form. Unlike the classical network which computes
instantaneously, this network truly exists in time.
""")

    return {
        'classical': classical,
        'temporal': temporal,
        'discovered_forms': discovered_forms,
        'quality': quality
    }


if __name__ == "__main__":
    results = run_experiment()
