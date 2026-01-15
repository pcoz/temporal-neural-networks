"""
ECG Classification Experiment - TNN on real medical data.

Train a temporal neural network to classify heartbeat arrhythmias
from the MIT-BIH ECG dataset.

This is a real-world temporal task where the shape of the waveform
over time determines the classification.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn import (
    ClassicalNetwork, train_classical, TrainingConfig,
    discover_forms, FormDiscoveryConfig, summarize_discoveries,
    convert_to_temporal, NeuronType
)


def load_ecg_data(n_train=5000, n_test=1000):
    """
    Load MIT-BIH ECG dataset.

    Using a subset for faster experimentation.
    """
    data_path = os.path.expanduser('~/.cache/kagglehub/datasets/shayanfazeli/heartbeat/versions/1/')

    print("Loading ECG data...")
    train_df = pd.read_csv(data_path + 'mitbih_train.csv', header=None)
    test_df = pd.read_csv(data_path + 'mitbih_test.csv', header=None)

    # Features and labels
    x_train_full = train_df.iloc[:, :-1].values
    y_train_full = train_df.iloc[:, -1].values.astype(int)

    x_test_full = test_df.iloc[:, :-1].values
    y_test_full = test_df.iloc[:, -1].values.astype(int)

    # Balance the training set by sampling from each class
    x_train_balanced = []
    y_train_balanced = []

    samples_per_class = n_train // 5

    for class_id in range(5):
        class_indices = np.where(y_train_full == class_id)[0]
        if len(class_indices) > samples_per_class:
            selected = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            selected = np.random.choice(class_indices, samples_per_class, replace=True)

        x_train_balanced.append(x_train_full[selected])
        y_train_balanced.append(y_train_full[selected])

    x_train = np.vstack(x_train_balanced)
    y_train = np.hstack(y_train_balanced)

    # Shuffle
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # Test set - random sample
    test_indices = np.random.choice(len(x_test_full), min(n_test, len(x_test_full)), replace=False)
    x_test = x_test_full[test_indices]
    y_test = y_test_full[test_indices]

    return x_train, y_train, x_test, y_test


def one_hot_encode(labels, n_classes=5):
    """Convert labels to one-hot encoding."""
    one_hot = np.zeros((len(labels), n_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def accuracy(predictions, labels):
    """Compute classification accuracy."""
    pred_classes = np.argmax(predictions, axis=1)
    return np.mean(pred_classes == labels)


def record_ecg_activations(model, x_data, n_samples=100):
    """
    Record activations while processing ECG sequences.

    Feed time points sequentially using a sliding window
    to capture how neurons respond over time.
    """
    print(f"Recording temporal activations (sliding window)...")

    model.start_recording()
    input_size = model.layer_sizes[0]

    # Take a few ECG samples and process them with sliding windows
    for sample_idx in range(min(5, len(x_data))):
        ecg = x_data[sample_idx]

        # Slide through the ECG with overlapping windows
        window_size = input_size
        for start in range(0, len(ecg) - window_size + 1, 3):  # Step by 3
            window = ecg[start:start + window_size]
            if len(window) == window_size:
                model.forward(window.reshape(1, -1), record=True)

    model.stop_recording()
    print(f"  Recorded {len(model.get_activation_history())} time steps")
    return model.get_activation_history()


def test_temporal_classification(temporal_net, x_test, y_test, dt=0.1, settle_steps=10):
    """Test temporal network on classification."""
    predictions = []

    for i in range(len(x_test)):
        temporal_net.reset()

        # Run temporal network on this ECG
        for _ in range(settle_steps):
            out = temporal_net.step(dt, x_test[i])

        predictions.append(out)

    predictions = np.array(predictions)
    pred_classes = np.argmax(predictions, axis=1)
    acc = np.mean(pred_classes == y_test)

    return acc, predictions


def main():
    print("=" * 70)
    print("TNN ECG EXPERIMENT: Heartbeat Arrhythmia Classification")
    print("=" * 70)

    # Configuration
    n_train = 5000  # Use subset for speed
    n_test = 1000
    input_size = 187  # ECG time points
    hidden_sizes = [64, 32]
    n_classes = 5

    # ================================================================
    # Load Data
    # ================================================================
    x_train, y_train, x_test, y_test = load_ecg_data(n_train, n_test)
    y_train_onehot = one_hot_encode(y_train, n_classes)
    y_test_onehot = one_hot_encode(y_test, n_classes)

    print(f"\nTraining samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input dimension: {input_size}")
    print(f"Classes: {n_classes}")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # ================================================================
    # PHASE 1: Classical Training
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Classical Network Training")
    print("=" * 70)

    layer_sizes = [input_size] + hidden_sizes + [n_classes]
    classical = ClassicalNetwork(layer_sizes=layer_sizes, activation='tanh')

    config = TrainingConfig(
        learning_rate=0.005,
        epochs=50,
        batch_size=64,
        momentum=0.9,
        record_activations=False
    )

    print(f"\nArchitecture: {layer_sizes}")
    history = train_classical(
        classical, x_train, y_train_onehot,
        config=config,
        x_val=x_test, y_val=y_test_onehot,
        verbose=True
    )

    # Evaluate classical network
    classical_pred = classical.predict(x_test)
    classical_acc = accuracy(classical_pred, y_test)
    print(f"\nClassical Network Test Accuracy: {classical_acc:.4f} ({classical_acc*100:.1f}%)")

    # Per-class accuracy
    pred_classes = np.argmax(classical_pred, axis=1)
    print("\nPer-class accuracy (Classical):")
    class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc = np.mean(pred_classes[mask] == c)
            print(f"  {class_names[c]}: {class_acc:.3f} ({mask.sum()} samples)")

    # ================================================================
    # PHASE 2: Form Discovery with PPF
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Form Discovery with PPF")
    print("=" * 70)

    # Record activations during inference
    activation_history = record_ecg_activations(classical, x_train, n_samples=300)
    print(f"Recorded {len(activation_history)} activation snapshots")

    # Discover forms
    discovery_config = FormDiscoveryConfig(
        population_size=150,
        generations=15,
        max_depth=4,
        min_r_squared=0.3,  # Lower threshold for real data
        cluster_neurons=True,
        n_clusters=4
    )

    print("\nDiscovering temporal forms...")
    try:
        discovered_forms = discover_forms(
            activation_history,
            config=discovery_config,
            verbose=True
        )
        total_forms = sum(len(f) for f in discovered_forms.values())
        print(f"\nTotal forms discovered: {total_forms}")

        if total_forms > 0:
            print("\n" + summarize_discoveries(discovered_forms))
    except Exception as e:
        print(f"Form discovery error: {e}")
        discovered_forms = {}

    # ================================================================
    # PHASE 3: Temporal Network
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Temporal Network Conversion & Testing")
    print("=" * 70)

    # Use leaky integrator dynamics - discovered forms need more data points
    # to be reliable (the 5-point rational functions overfit)
    print("Using leaky integrator dynamics (temporal processing)...")
    temporal = convert_to_temporal(
        classical, {},
        default_type=NeuronType.LEAKY_INTEGRATOR
    )

    # Copy weights
    temporal.set_weights(classical.weights, classical.biases)

    # Test temporal network with different settle times
    print("\nTesting temporal network...")

    for settle in [1, 5, 10, 20]:
        temporal_acc, _ = test_temporal_classification(
            temporal, x_test[:200], y_test[:200],  # Subset for speed
            dt=0.1, settle_steps=settle
        )
        print(f"  Settle steps={settle:2d}: Accuracy={temporal_acc:.4f} ({temporal_acc*100:.1f}%)")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    # Full temporal test
    temporal_acc_full, temporal_pred = test_temporal_classification(
        temporal, x_test, y_test, dt=0.1, settle_steps=10
    )

    print(f"""
Dataset: MIT-BIH ECG Heartbeat Classification
  - Training samples: {len(x_train)}
  - Test samples: {len(x_test)}
  - Time points per heartbeat: {input_size}
  - Classes: {n_classes} (Normal + 4 arrhythmia types)

Classical Network:
  - Architecture: {layer_sizes}
  - Test Accuracy: {classical_acc:.4f} ({classical_acc*100:.1f}%)

Form Discovery:
  - Forms discovered: {sum(len(f) for f in discovered_forms.values())}

Temporal Network:
  - Test Accuracy: {temporal_acc_full:.4f} ({temporal_acc_full*100:.1f}%)
  - Each neuron now evolves over time
  - Network can process continuous ECG streams

The temporal network processes each heartbeat as a temporal signal,
allowing neurons to accumulate and integrate information over time
rather than computing instantaneously.
""")

    # Show a few example predictions
    print("Sample Predictions:")
    print("-" * 50)
    for i in range(10):
        true_class = class_names[y_test[i]]
        classical_class = class_names[np.argmax(classical_pred[i])]
        temporal_class = class_names[np.argmax(temporal_pred[i])]

        match_c = "OK" if classical_class == true_class else "X"
        match_t = "OK" if temporal_class == true_class else "X"

        print(f"  Sample {i}: True={true_class:15s} "
              f"Classical={classical_class:15s}[{match_c}] "
              f"Temporal={temporal_class:15s}[{match_t}]")

    return {
        'classical_acc': classical_acc,
        'temporal_acc': temporal_acc_full,
        'discovered_forms': discovered_forms
    }


if __name__ == "__main__":
    results = main()
