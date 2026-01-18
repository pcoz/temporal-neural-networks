"""
TNN Streaming Test v2 - Proper Architecture

Key insight: The temporal advantage comes from ACCUMULATING information
over time, not from training on tiny slices.

Approach:
1. Classical network: sees full window at once
2. Temporal network: accumulates timesteps, makes predictions from partial windows

Both use the same underlying weights, but temporal can decide EARLY.
"""

import numpy as np
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn import ClassicalNetwork


def load_raw_signals():
    """Load raw inertial signals."""
    base_path = os.path.expanduser('~/.cache/uci_har_raw/extracted/UCI HAR Dataset/')

    signal_names = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]

    train_signals = []
    for name in signal_names:
        data = np.loadtxt(f"{base_path}train/Inertial Signals/{name}_train.txt")
        train_signals.append(data)
    x_train = np.stack(train_signals, axis=2)  # (7352, 128, 9)

    test_signals = []
    for name in signal_names:
        data = np.loadtxt(f"{base_path}test/Inertial Signals/{name}_test.txt")
        test_signals.append(data)
    x_test = np.stack(test_signals, axis=2)

    y_train = np.loadtxt(f"{base_path}train/y_train.txt").astype(int) - 1
    y_test = np.loadtxt(f"{base_path}test/y_test.txt").astype(int) - 1

    # Normalize
    for c in range(9):
        mean = x_train[:, :, c].mean()
        std = x_train[:, :, c].std() + 1e-8
        x_train[:, :, c] = (x_train[:, :, c] - mean) / std
        x_test[:, :, c] = (x_test[:, :, c] - mean) / std

    return x_train, y_train, x_test, y_test


class TemporalAccumulator:
    """
    Temporal network that accumulates features with leaky integration.

    As timesteps arrive, it updates an accumulated feature vector.
    This accumulated vector is then fed through a classifier.
    """

    def __init__(self, n_channels: int, n_timesteps: int, classifier: ClassicalNetwork,
                 tau: float = 10.0):
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.classifier = classifier
        self.tau = tau

        # Accumulated feature state (same size as classifier input)
        self.feature_state = np.zeros(n_timesteps * n_channels)
        self.current_step = 0

    def reset(self):
        self.feature_state = np.zeros(self.n_timesteps * self.n_channels)
        self.current_step = 0

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """
        Receive one timestep (n_channels values).
        Update accumulated features with leaky integration.
        Return current prediction.
        """
        # Position in feature vector
        start_idx = self.current_step * self.n_channels
        end_idx = start_idx + self.n_channels

        if end_idx <= len(self.feature_state):
            # Leaky integration: gradually fill in this timestep's features
            decay = 1.0 / self.tau
            self.feature_state[start_idx:end_idx] += decay * (x_t - self.feature_state[start_idx:end_idx])

        self.current_step += 1

        # Get prediction from current accumulated state
        pred = self.classifier.predict(self.feature_state.reshape(1, -1))
        return pred[0]

    def get_prediction(self) -> int:
        pred = self.classifier.predict(self.feature_state.reshape(1, -1))
        return np.argmax(pred)


class SimpleStreamingNet:
    """
    Simpler streaming approach: just fill in features as they arrive.
    No leaky integration - direct comparison baseline.
    """

    def __init__(self, n_channels: int, n_timesteps: int, classifier: ClassicalNetwork):
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.classifier = classifier
        self.features = np.zeros(n_timesteps * n_channels)
        self.current_step = 0

    def reset(self):
        self.features = np.zeros(self.n_timesteps * self.n_channels)
        self.current_step = 0

    def step(self, x_t: np.ndarray):
        start_idx = self.current_step * self.n_channels
        end_idx = start_idx + self.n_channels
        if end_idx <= len(self.features):
            self.features[start_idx:end_idx] = x_t
        self.current_step += 1

    def get_prediction(self) -> int:
        pred = self.classifier.predict(self.features.reshape(1, -1))
        return np.argmax(pred)


def train_classifier(x_train: np.ndarray, y_train: np.ndarray,
                     epochs: int = 60, lr: float = 0.003) -> ClassicalNetwork:
    """Train classifier on full windows."""
    n_samples = len(x_train)
    n_classes = 6
    input_size = x_train.shape[1] * x_train.shape[2]  # 128 * 9 = 1152

    x_flat = x_train.reshape(n_samples, -1)

    model = ClassicalNetwork([input_size, 256, 64, n_classes], activation='tanh')

    batch_size = 64
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            x_batch = x_flat[batch_idx]
            y_batch = y_train[batch_idx]

            pred = model.forward(x_batch)

            y_onehot = np.zeros((len(y_batch), n_classes))
            y_onehot[np.arange(len(y_batch)), y_batch] = 1

            model.backward(y_onehot, lr)

        if (epoch + 1) % 20 == 0:
            acc = np.mean(np.argmax(model.predict(x_flat[:500]), axis=1) == y_train[:500])
            print(f"  Epoch {epoch+1}: Train acc = {acc:.3f}")

    return model


def main():
    print("=" * 70)
    print("TNN STREAMING TEST v2 - Early Decision on Raw Signals")
    print("=" * 70)

    # Load data
    print("\nLoading raw signals...")
    x_train, y_train, x_test, y_test = load_raw_signals()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    n_timesteps = x_train.shape[1]  # 128
    n_channels = x_train.shape[2]   # 9

    # Train classifier
    print("\nTraining classifier on full windows...")
    classifier = train_classifier(x_train, y_train)

    # Baseline
    x_test_flat = x_test.reshape(len(x_test), -1)
    baseline_pred = np.argmax(classifier.predict(x_test_flat), axis=1)
    baseline_acc = np.mean(baseline_pred == y_test)
    print(f"\nBaseline accuracy (full window): {baseline_acc:.3f}")

    # Create streaming networks
    simple_streamer = SimpleStreamingNet(n_channels, n_timesteps, classifier)
    temporal_streamer = TemporalAccumulator(n_channels, n_timesteps, classifier, tau=5.0)

    # ==========================================================================
    # TEST: Early Decision
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EARLY DECISION TEST")
    print("=" * 70)
    print("\nMeasure accuracy at different observation percentages")

    checkpoints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_samples = 500

    simple_results = []
    temporal_results = []

    for pct in checkpoints:
        n_steps = int(pct * n_timesteps)

        simple_correct = 0
        temporal_correct = 0

        for i in range(min(n_samples, len(x_test))):
            signal = x_test[i]
            y_true = y_test[i]

            simple_streamer.reset()
            temporal_streamer.reset()

            for t in range(n_steps):
                simple_streamer.step(signal[t])
                temporal_streamer.step(signal[t])

            if simple_streamer.get_prediction() == y_true:
                simple_correct += 1
            if temporal_streamer.get_prediction() == y_true:
                temporal_correct += 1

        simple_results.append(simple_correct / n_samples)
        temporal_results.append(temporal_correct / n_samples)

    print(f"\n{'% Window':>10} | {'Steps':>6} | {'Simple':>8} | {'Temporal':>10} | {'Delta':>8}")
    print("-" * 55)
    for i, pct in enumerate(checkpoints):
        n_steps = int(pct * n_timesteps)
        delta = temporal_results[i] - simple_results[i]
        marker = " <--" if delta > 0.01 else ""
        print(f"{pct*100:>9.0f}% | {n_steps:>6} | {simple_results[i]:>8.3f} | {temporal_results[i]:>10.3f} | {delta:>+8.3f}{marker}")

    # ==========================================================================
    # TEST: Noise Robustness in Streaming
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STREAMING NOISE TEST")
    print("=" * 70)

    noise_levels = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]
    n_samples = 300

    print(f"\n{'Noise':>6} | {'Simple':>8} | {'Temporal':>10} | {'Simple Flips':>12} | {'TNN Flips':>10}")
    print("-" * 60)

    for noise in noise_levels:
        simple_correct = 0
        temporal_correct = 0
        simple_flips = 0
        temporal_flips = 0

        for i in range(min(n_samples, len(x_test))):
            signal = x_test[i]
            y_true = y_test[i]

            simple_streamer.reset()
            temporal_streamer.reset()

            simple_history = []
            temporal_history = []

            for t in range(n_timesteps):
                noisy_t = signal[t] + np.random.randn(n_channels) * noise
                simple_streamer.step(noisy_t)
                temporal_streamer.step(noisy_t)

                simple_history.append(simple_streamer.get_prediction())
                temporal_history.append(temporal_streamer.get_prediction())

            if simple_history[-1] == y_true:
                simple_correct += 1
            if temporal_history[-1] == y_true:
                temporal_correct += 1

            for j in range(1, len(simple_history)):
                if simple_history[j] != simple_history[j-1]:
                    simple_flips += 1
                if temporal_history[j] != temporal_history[j-1]:
                    temporal_flips += 1

        print(f"{noise:>6.1f} | {simple_correct/n_samples:>8.3f} | {temporal_correct/n_samples:>10.3f} | "
              f"{simple_flips/n_samples:>12.1f} | {temporal_flips/n_samples:>10.1f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find when temporal reaches 90% of final
    final_temporal = temporal_results[-1]
    threshold = 0.9 * final_temporal
    early_pct = None
    for i, acc in enumerate(temporal_results):
        if acc >= threshold:
            early_pct = checkpoints[i]
            break

    print(f"""
RESULTS:

1. BASELINE (full 128-step window): {baseline_acc:.3f}

2. EARLY DECISION
   - At 50% window: Simple {simple_results[4]:.3f}, Temporal {temporal_results[4]:.3f}
   - Temporal reaches 90% of final accuracy at {early_pct*100:.0f}% of window

3. KEY INSIGHT
   The temporal accumulator with leaky integration provides
   smoother evidence accumulation, potentially enabling
   earlier confident decisions.

4. STREAMING ADVANTAGE
   - Process data as it arrives
   - Don't need to wait for full window
   - Smoother predictions under noise
""")


if __name__ == "__main__":
    main()
