"""
Test the TNN - Does the temporal network actually work?

Compare classical vs temporal on the prediction task.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn import (
    ClassicalNetwork, train_classical, TrainingConfig,
    discover_forms, FormDiscoveryConfig,
    convert_to_temporal, NeuronType,
    create_uniform_network
)


def generate_test_sequence(length=100, freq=0.3, phase=0.0):
    """Generate a test sine wave sequence."""
    t = np.arange(length)
    return np.sin(freq * t + phase)


def test_classical_prediction(model, sequence, input_window=10):
    """Test classical network on sequence prediction."""
    predictions = []
    actuals = []

    for i in range(input_window, len(sequence)):
        x = sequence[i-input_window:i].reshape(1, -1)
        pred = model.predict(x)[0, 0]
        actual = sequence[i]
        predictions.append(pred)
        actuals.append(actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = np.mean((predictions - actuals) ** 2)

    return predictions, actuals, mse


def test_temporal_prediction(model, sequence, input_window=10, dt=0.1, settle_steps=5):
    """Test temporal network on sequence prediction."""
    predictions = []
    actuals = []

    model.reset()

    for i in range(input_window, len(sequence)):
        x = sequence[i-input_window:i]

        # Let the temporal network settle
        for _ in range(settle_steps):
            out = model.step(dt, x)

        predictions.append(out[0])
        actuals.append(sequence[i])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = np.mean((predictions - actuals) ** 2)

    return predictions, actuals, mse


def main():
    print("=" * 60)
    print("TNN TEST: Does the temporal network actually work?")
    print("=" * 60)

    # Parameters
    input_window = 10
    hidden_sizes = [32, 16]

    # Generate training data
    print("\n1. Generating training data...")
    x_train = []
    y_train = []

    for _ in range(2000):
        freq = np.random.uniform(0.2, 0.5)
        phase = np.random.uniform(0, 2*np.pi)
        t = np.arange(input_window + 1)
        signal = np.sin(freq * t + phase)
        x_train.append(signal[:-1])
        y_train.append([signal[-1]])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(f"   Training samples: {len(x_train)}")

    # Train classical network
    print("\n2. Training classical network...")
    classical = ClassicalNetwork([input_window, 32, 16, 1], activation='tanh')

    config = TrainingConfig(
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        record_activations=False
    )

    train_classical(classical, x_train, y_train, config=config, verbose=False)
    print("   Done.")

    # Record activations for PPF
    print("\n3. Recording continuous activations for PPF...")
    classical.start_recording()
    for t in range(300):
        signal = np.array([np.sin(0.3 * (t - i)) for i in range(input_window)])
        classical.forward(signal.reshape(1, -1), record=True)
    classical.stop_recording()
    activation_history = classical.get_activation_history()
    print(f"   Recorded {len(activation_history)} time steps")

    # Discover forms
    print("\n4. Discovering forms with PPF...")
    discovery_config = FormDiscoveryConfig(
        population_size=150,
        generations=15,
        min_r_squared=0.5,
        cluster_neurons=True,
        n_clusters=4
    )

    discovered_forms = discover_forms(activation_history, config=discovery_config, verbose=False)
    total_forms = sum(len(f) for f in discovered_forms.values())
    print(f"   Discovered {total_forms} forms")

    # Convert to temporal
    print("\n5. Converting to temporal network...")
    temporal = convert_to_temporal(classical, discovered_forms, default_type=NeuronType.LEAKY_INTEGRATOR)
    print("   Done.")

    # TEST: Generate test sequences
    print("\n" + "=" * 60)
    print("TESTING ON NEW SEQUENCES")
    print("=" * 60)

    test_freqs = [0.25, 0.3, 0.4]  # Different frequencies

    for freq in test_freqs:
        print(f"\n--- Test frequency: {freq} ---")

        test_seq = generate_test_sequence(length=50, freq=freq)

        # Test classical
        c_pred, c_actual, c_mse = test_classical_prediction(classical, test_seq, input_window)
        print(f"   Classical MSE: {c_mse:.6f}")

        # Test temporal with different settle times
        for settle in [1, 5, 10, 20]:
            t_pred, t_actual, t_mse = test_temporal_prediction(
                temporal, test_seq, input_window, dt=0.1, settle_steps=settle
            )
            print(f"   Temporal MSE (settle={settle:2d}): {t_mse:.6f}")

    # Show prediction comparison
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (freq=0.3)")
    print("=" * 60)

    test_seq = generate_test_sequence(length=30, freq=0.3)
    c_pred, c_actual, _ = test_classical_prediction(classical, test_seq, input_window)
    temporal.reset()
    t_pred, t_actual, _ = test_temporal_prediction(temporal, test_seq, input_window, settle_steps=10)

    print(f"\n{'Step':>4} | {'Actual':>8} | {'Classical':>10} | {'Temporal':>10} | {'C_err':>8} | {'T_err':>8}")
    print("-" * 65)

    for i in range(min(15, len(c_pred))):
        c_err = abs(c_pred[i] - c_actual[i])
        t_err = abs(t_pred[i] - t_actual[i])
        print(f"{i:4d} | {c_actual[i]:8.4f} | {c_pred[i]:10.4f} | {t_pred[i]:10.4f} | {c_err:8.4f} | {t_err:8.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    c_total_mse = np.mean((c_pred - c_actual) ** 2)
    t_total_mse = np.mean((t_pred - t_actual) ** 2)

    print(f"\nClassical network MSE: {c_total_mse:.6f}")
    print(f"Temporal network MSE:  {t_total_mse:.6f}")

    if t_total_mse < c_total_mse * 2:
        print("\nTemporal network is working - predictions are reasonable!")
    else:
        print("\nTemporal network needs tuning - MSE is higher than expected.")

    print(f"\nKey insight: The classical network computes instantly.")
    print(f"The temporal network EVOLVES - it takes time to settle.")
    print(f"This is fundamentally different - it exists in time.")

    # Show what temporal networks CAN do that classical can't
    print("\n" + "=" * 60)
    print("TEMPORAL ADVANTAGE: Continuous Evolution")
    print("=" * 60)

    print("\nRunning temporal network on a continuous changing input...")
    print("(Classical network can only compute snapshots)\n")

    temporal.reset()

    # Run with smoothly changing input
    print("Time  | Input[0] | Output  | (Network is continuously evolving)")
    print("-" * 60)

    for step in range(30):
        t = step * 0.5
        # Smoothly varying input
        inp = np.array([np.sin(0.2 * t + i * 0.1) for i in range(input_window)])

        # Temporal network evolves each step
        out = temporal.step(0.1, inp)

        if step % 2 == 0:
            print(f"{t:5.1f} | {inp[0]:8.4f} | {out[0]:7.4f} |")

    print("\nThe temporal network maintains state and evolves continuously.")
    print("This enables processing of streams, not just snapshots.")


if __name__ == "__main__":
    main()
