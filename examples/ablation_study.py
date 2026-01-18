"""
Ablation Study for TNN Paper

Experiments requested by reviewer:
1. τ Ablation: τ ∈ {1, 2, 4, 8, 16} - trade-off between responsiveness and stability
2. TNN vs Classical + Post-hoc Smoothing comparison
3. Noise injection training comparison
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn import ClassicalNetwork


# =============================================================================
# STREAMING TEMPORAL NETWORK
# =============================================================================

class StreamingTemporalNet:
    """Temporal network with configurable tau."""

    def __init__(self, classical: ClassicalNetwork, tau: float = 8.0):
        self.classical = classical
        self.tau = tau
        self.states = [np.zeros(s) for s in classical.layer_sizes[1:]]

    def reset(self):
        self.states = [np.zeros(s) for s in self.classical.layer_sizes[1:]]

    def step(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(zip(self.classical.weights, self.classical.biases)):
            z = h @ w + b
            target = np.tanh(z) if i < len(self.classical.weights) - 1 else z
            decay = dt / self.tau
            self.states[i] = self.states[i] + decay * (target - self.states[i])
            h = self.states[i]
        return h

    def get_prediction(self) -> int:
        return np.argmax(self.states[-1])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_har_data():
    """Load UCI HAR with proper subject split."""
    data_path = os.path.expanduser(
        '~/.cache/kagglehub/datasets/uciml/human-activity-recognition-with-smartphones/versions/2/'
    )

    train_df = pd.read_csv(data_path + 'train.csv')
    test_df = pd.read_csv(data_path + 'test.csv')

    feature_cols = [c for c in train_df.columns if c not in ['subject', 'Activity']]

    x_train = train_df[feature_cols].values
    x_test = test_df[feature_cols].values

    activity_map = {
        'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
        'SITTING': 3, 'STANDING': 4, 'LAYING': 5
    }

    y_train = train_df['Activity'].map(activity_map).values
    y_test = test_df['Activity'].map(activity_map).values

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_test, y_test, mean, std


def train_model(model: ClassicalNetwork, x: np.ndarray, y: np.ndarray,
                epochs: int = 80, batch_size: int = 64, lr: float = 0.003,
                noise_std: float = 0.0):
    """Train classical network, optionally with noise injection."""
    n_classes = model.layer_sizes[-1]
    n_samples = len(x)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]

            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            # Optionally add noise during training
            if noise_std > 0:
                x_batch = x_batch + np.random.randn(*x_batch.shape) * noise_std

            pred = model.forward(x_batch)

            y_onehot = np.zeros((len(y_batch), n_classes))
            y_onehot[np.arange(len(y_batch)), y_batch] = 1

            model.backward(y_onehot, lr)


# =============================================================================
# POST-HOC SMOOTHING BASELINES
# =============================================================================

def moving_average_smooth(predictions: List[int], window: int) -> List[int]:
    """Apply moving average smoothing to predictions."""
    if window <= 1:
        return predictions

    smoothed = []
    for i in range(len(predictions)):
        start = max(0, i - window + 1)
        window_preds = predictions[start:i+1]
        # Majority vote in window
        counts = np.bincount(window_preds, minlength=6)
        smoothed.append(np.argmax(counts))
    return smoothed


def exponential_smooth(predictions: List[int], alpha: float, n_classes: int = 6) -> List[int]:
    """Apply exponential smoothing to class logits."""
    # Convert to one-hot and smooth
    smoothed_logits = np.zeros(n_classes)
    result = []

    for pred in predictions:
        one_hot = np.zeros(n_classes)
        one_hot[pred] = 1.0
        smoothed_logits = alpha * one_hot + (1 - alpha) * smoothed_logits
        result.append(np.argmax(smoothed_logits))

    return result


# =============================================================================
# ABLATION 1: TAU SWEEP
# =============================================================================

def run_tau_ablation(classical: ClassicalNetwork, x_test: np.ndarray, y_test: np.ndarray,
                     n_samples: int = 300, n_timesteps: int = 50, noise_std: float = 0.5):
    """Test different tau values for stability vs responsiveness trade-off."""
    print("\n" + "=" * 70)
    print("ABLATION 1: TAU SWEEP (tau in {1, 2, 4, 8, 16})")
    print("=" * 70)

    tau_values = [1, 2, 4, 8, 16]

    results = {
        'tau': tau_values,
        'flips': [],
        'accuracy': [],
        'settle_time': []  # Steps to reach stable prediction
    }

    for tau in tau_values:
        temporal = StreamingTemporalNet(classical, tau=tau)
        total_flips = 0
        correct = 0
        total_settle_time = 0

        for i in range(min(n_samples, len(x_test))):
            x_base = x_test[i]
            y_true = y_test[i]

            history = []
            temporal.reset()

            # Track when prediction first matches final
            settle_step = n_timesteps

            for t in range(n_timesteps):
                noise = np.random.randn(*x_base.shape) * noise_std
                x_noisy = x_base + noise
                temporal.step(x_noisy)
                pred = temporal.get_prediction()
                history.append(pred)

            # Count flips
            for j in range(1, len(history)):
                if history[j] != history[j-1]:
                    total_flips += 1

            # Find settle time (last change)
            final_pred = history[-1]
            for j in range(len(history)-1, -1, -1):
                if history[j] != final_pred:
                    settle_step = j + 1
                    break
            else:
                settle_step = 0

            total_settle_time += settle_step

            if final_pred == y_true:
                correct += 1

        results['flips'].append(total_flips / n_samples)
        results['accuracy'].append(correct / n_samples)
        results['settle_time'].append(total_settle_time / n_samples)

    # Print results
    print(f"\n{'tau':>4} | {'Flips':>8} | {'Accuracy':>10} | {'Settle Steps':>12}")
    print("-" * 45)
    for i, tau in enumerate(tau_values):
        print(f"{tau:>4} | {results['flips'][i]:>8.1f} | {results['accuracy'][i]:>10.1%} | {results['settle_time'][i]:>12.1f}")

    return results


# =============================================================================
# ABLATION 2: TNN vs POST-HOC SMOOTHING
# =============================================================================

def run_smoothing_comparison(classical: ClassicalNetwork, x_test: np.ndarray, y_test: np.ndarray,
                             n_samples: int = 300, n_timesteps: int = 50, noise_std: float = 0.5):
    """Compare TNN temporal dynamics vs post-hoc smoothing techniques."""
    print("\n" + "=" * 70)
    print("ABLATION 2: TNN vs POST-HOC SMOOTHING")
    print("=" * 70)

    methods = ['Classical', 'MA-3', 'MA-5', 'MA-10', 'Exp-0.3', 'Exp-0.1', 'TNN tau=8']

    results = {
        'method': methods,
        'flips': [],
        'accuracy': []
    }

    temporal = StreamingTemporalNet(classical, tau=8.0)

    # Collect predictions for all methods
    for method_idx, method in enumerate(methods):
        total_flips = 0
        correct = 0

        for i in range(min(n_samples, len(x_test))):
            x_base = x_test[i]
            y_true = y_test[i]

            classical_history = []
            temporal.reset()
            temporal_history = []

            for t in range(n_timesteps):
                noise = np.random.randn(*x_base.shape) * noise_std
                x_noisy = x_base + noise

                # Classical prediction
                c_out = classical.predict(x_noisy.reshape(1, -1))
                c_pred = np.argmax(c_out)
                classical_history.append(c_pred)

                # Temporal prediction
                temporal.step(x_noisy)
                temporal_history.append(temporal.get_prediction())

            # Apply smoothing based on method
            if method == 'Classical':
                history = classical_history
            elif method == 'MA-3':
                history = moving_average_smooth(classical_history, 3)
            elif method == 'MA-5':
                history = moving_average_smooth(classical_history, 5)
            elif method == 'MA-10':
                history = moving_average_smooth(classical_history, 10)
            elif method == 'Exp-0.3':
                history = exponential_smooth(classical_history, 0.3)
            elif method == 'Exp-0.1':
                history = exponential_smooth(classical_history, 0.1)
            elif method == 'TNN tau=8':
                history = temporal_history

            # Count flips
            for j in range(1, len(history)):
                if history[j] != history[j-1]:
                    total_flips += 1

            if history[-1] == y_true:
                correct += 1

        results['flips'].append(total_flips / n_samples)
        results['accuracy'].append(correct / n_samples)

    # Print results
    print(f"\n{'Method':>12} | {'Flips':>8} | {'Accuracy':>10} | {'Notes':>25}")
    print("-" * 65)
    for i, method in enumerate(methods):
        notes = ""
        if method == 'Classical':
            notes = "Baseline"
        elif 'MA' in method:
            notes = "Post-hoc majority vote"
        elif 'Exp' in method:
            notes = "Post-hoc exp. smoothing"
        elif 'TNN' in method:
            notes = "Integrated dynamics"
        print(f"{method:>12} | {results['flips'][i]:>8.1f} | {results['accuracy'][i]:>10.1%} | {notes:>25}")

    return results


# =============================================================================
# ABLATION 3: NOISE INJECTION TRAINING
# =============================================================================

def run_noise_injection_comparison(x_train: np.ndarray, y_train: np.ndarray,
                                   x_test: np.ndarray, y_test: np.ndarray,
                                   n_samples: int = 300, n_timesteps: int = 50,
                                   test_noise: float = 0.5):
    """Compare standard training vs noise-injection training vs TNN."""
    print("\n" + "=" * 70)
    print("ABLATION 3: NOISE INJECTION TRAINING COMPARISON")
    print("=" * 70)

    input_size = x_train.shape[1]

    methods = ['Standard', 'Noise s=0.1', 'Noise s=0.3', 'Noise s=0.5', 'TNN (Standard)']
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.0]  # Last is for TNN

    results = {
        'method': methods,
        'flips': [],
        'accuracy': []
    }

    for method_idx, (method, train_noise) in enumerate(zip(methods, noise_levels)):
        print(f"\nTraining {method}...")

        # Train fresh model
        model = ClassicalNetwork(
            layer_sizes=[input_size, 128, 64, 6],
            activation='tanh'
        )
        train_model(model, x_train, y_train, epochs=80, noise_std=train_noise)

        # Test with noise
        total_flips = 0
        correct = 0

        use_temporal = 'TNN' in method
        if use_temporal:
            temporal = StreamingTemporalNet(model, tau=8.0)

        for i in range(min(n_samples, len(x_test))):
            x_base = x_test[i]
            y_true = y_test[i]

            history = []
            if use_temporal:
                temporal.reset()

            for t in range(n_timesteps):
                noise = np.random.randn(*x_base.shape) * test_noise
                x_noisy = x_base + noise

                if use_temporal:
                    temporal.step(x_noisy)
                    pred = temporal.get_prediction()
                else:
                    out = model.predict(x_noisy.reshape(1, -1))
                    pred = np.argmax(out)

                history.append(pred)

            # Count flips
            for j in range(1, len(history)):
                if history[j] != history[j-1]:
                    total_flips += 1

            if history[-1] == y_true:
                correct += 1

        results['flips'].append(total_flips / n_samples)
        results['accuracy'].append(correct / n_samples)

    # Print results
    print(f"\n{'Method':>15} | {'Flips':>8} | {'Accuracy':>10}")
    print("-" * 40)
    for i, method in enumerate(methods):
        print(f"{method:>15} | {results['flips'][i]:>8.1f} | {results['accuracy'][i]:>10.1%}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TNN ABLATION STUDY")
    print("Experiments for Reviewer Response")
    print("=" * 70)

    # Load data
    print("\nLoading UCI HAR data...")
    x_train, y_train, x_test, y_test, mean, std = load_har_data()
    input_size = x_train.shape[1]
    print(f"Train: {len(x_train)}, Test: {len(x_test)}, Features: {input_size}")

    # Train base classical network
    print("\nTraining base classical network...")
    classical = ClassicalNetwork(
        layer_sizes=[input_size, 128, 64, 6],
        activation='tanh'
    )
    train_model(classical, x_train, y_train, epochs=80)

    # Baseline accuracy
    baseline_pred = np.argmax(classical.predict(x_test), axis=1)
    baseline_acc = np.mean(y_test == baseline_pred)
    print(f"Baseline test accuracy: {baseline_acc:.1%}")

    # Run ablations
    tau_results = run_tau_ablation(classical, x_test, y_test)
    smoothing_results = run_smoothing_comparison(classical, x_test, y_test)
    noise_results = run_noise_injection_comparison(x_train, y_train, x_test, y_test)

    # ==========================================================================
    # SUMMARY FOR PAPER
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER (LaTeX-ready)")
    print("=" * 70)

    print("\n% TAU ABLATION TABLE")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("$\\tau$ & Flips & Accuracy & Settle Steps \\\\")
    print("\\midrule")
    for i, tau in enumerate(tau_results['tau']):
        print(f"{tau} & {tau_results['flips'][i]:.1f} & {tau_results['accuracy'][i]*100:.1f}\\% & {tau_results['settle_time'][i]:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Effect of time constant $\\tau$ on stability and accuracy under noise ($\\sigma=0.5$).}")
    print("\\label{tab:tau_ablation}")
    print("\\end{table}")

    print("\n% SMOOTHING COMPARISON TABLE")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Method & Flips & Accuracy \\\\")
    print("\\midrule")
    for i, method in enumerate(smoothing_results['method']):
        print(f"{method} & {smoothing_results['flips'][i]:.1f} & {smoothing_results['accuracy'][i]*100:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{TNN vs post-hoc smoothing under noise ($\\sigma=0.5$). MA-$k$ = moving average with window $k$, Exp-$\\alpha$ = exponential smoothing.}")
    print("\\label{tab:smoothing}")
    print("\\end{table}")

    print("\n% NOISE INJECTION TABLE")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Training Method & Flips & Accuracy \\\\")
    print("\\midrule")
    for i, method in enumerate(noise_results['method']):
        print(f"{method} & {noise_results['flips'][i]:.1f} & {noise_results['accuracy'][i]*100:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Noise injection training vs TNN under test noise ($\\sigma=0.5$).}")
    print("\\label{tab:noise_injection}")
    print("\\end{table}")

    return {
        'tau': tau_results,
        'smoothing': smoothing_results,
        'noise_injection': noise_results
    }


if __name__ == "__main__":
    results = main()
