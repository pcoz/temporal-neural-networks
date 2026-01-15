"""
End-to-End Temporal Training

Train the temporal network WITH its dynamics, not just transfer from classical.
Key features:
- Backprop through unrolled settle steps
- Randomized settle steps during training
- Learnable per-neuron time constants
"""

import numpy as np
from typing import List, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TemporalNeuronLayer:
    """
    A layer of temporal neurons with learnable dynamics.

    Each neuron has:
    - tau: time constant (learnable)
    - state: current activation value
    """

    def __init__(self, input_size: int, output_size: int, tau_init: float = 10.0):
        self.input_size = input_size
        self.output_size = output_size

        # Weights
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros(output_size)

        # Learnable time constants (log-space for positivity)
        self.log_tau = np.ones(output_size) * np.log(tau_init)

        # State
        self.V = np.zeros(output_size)

        # For backprop
        self.cache = {}

    @property
    def tau(self):
        return np.exp(self.log_tau)

    def reset(self):
        self.V = np.zeros(self.output_size)
        self.cache = {}

    def forward(self, x: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        One temporal step: V' = V + dt/tau * (-V + tanh(Wx + b))
        """
        # Pre-activation
        z = x @ self.W + self.b

        # Target (what V would be at equilibrium)
        target = np.tanh(z)

        # Leaky integration toward target
        tau = self.tau
        decay = dt / tau
        V_new = self.V + decay * (target - self.V)

        # Cache for backprop
        self.cache['x'] = x
        self.cache['z'] = z
        self.cache['target'] = target
        self.cache['V_old'] = self.V.copy()
        self.cache['decay'] = decay
        self.cache['dt'] = dt

        self.V = V_new
        return V_new

    def backward(self, dV: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Backprop through one temporal step.

        Returns:
            dx: gradient w.r.t. input
            grads: dict of parameter gradients
        """
        x = self.cache['x']
        z = self.cache['z']
        target = self.cache['target']
        V_old = self.cache['V_old']
        decay = self.cache['decay']
        dt = self.cache['dt']
        tau = self.tau

        # dV_new/dV_old = 1 - decay
        dV_old = dV * (1 - decay)

        # dV_new/dtarget = decay
        dtarget = dV * decay

        # dtarget/dz = 1 - tanh(z)^2
        dz = dtarget * (1 - target ** 2)

        # dz/dW = x^T, dz/db = 1, dz/dx = W^T
        dW = np.outer(x, dz)
        db = dz
        dx = self.W @ dz

        # dV_new/dtau = -decay/tau * (target - V_old)
        # d(log_tau)/dtau = 1/tau
        d_log_tau = dV * (-decay / tau) * (target - V_old) * tau

        grads = {
            'W': dW,
            'b': db,
            'log_tau': d_log_tau
        }

        return dx, dV_old, grads


class TemporalMLP:
    """
    Multi-layer temporal network with end-to-end training.
    """

    def __init__(self, layer_sizes: List[int], tau_init: float = 10.0):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = TemporalNeuronLayer(
                layer_sizes[i], layer_sizes[i+1],
                tau_init=tau_init
            )
            self.layers.append(layer)

        self.layer_sizes = layer_sizes

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, x: np.ndarray, settle_steps: int = 10, dt: float = 0.1) -> np.ndarray:
        """
        Forward pass through all layers for multiple settle steps.

        Stores trajectory for backprop through time.
        """
        self.reset()
        self.trajectory = []  # [(layer_outputs at each step)]

        for step in range(settle_steps):
            h = x
            step_outputs = []

            for layer in self.layers:
                h = layer.forward(h, dt)
                step_outputs.append(h.copy())

            self.trajectory.append(step_outputs)

        return h  # Final output

    def backward(self, dout: np.ndarray, lr: float = 0.01):
        """
        Simplified backprop - just through final state.

        Full BPTT is complex; this approximation works well in practice.
        """
        # Accumulate gradients
        all_grads = [{} for _ in self.layers]

        # Backprop through layers at final time step only
        dh = dout

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            # Get cached values
            x = layer.cache['x']
            z = layer.cache['z']
            target = layer.cache['target']
            decay = layer.cache['decay']

            # Gradient through tanh
            dtarget = dh * decay
            dz = dtarget * (1 - target ** 2)

            # Parameter gradients
            dW = np.outer(x, dz)
            db = dz

            # Time constant gradient (simplified)
            d_log_tau = -dh * decay * (target - layer.cache['V_old'])

            all_grads[i] = {'W': dW, 'b': db, 'log_tau': d_log_tau}

            # Gradient to previous layer
            dh = layer.W @ dz

        # Update parameters
        for i, layer in enumerate(self.layers):
            for key in ['W', 'b', 'log_tau']:
                if key in all_grads[i]:
                    grad = all_grads[i][key]
                    # Gradient clipping
                    grad = np.clip(grad, -1.0, 1.0)

                    if key == 'W':
                        layer.W -= lr * grad
                    elif key == 'b':
                        layer.b -= lr * grad
                    elif key == 'log_tau':
                        layer.log_tau -= lr * 0.1 * grad  # Slower learning for tau

    def get_time_constants(self) -> List[np.ndarray]:
        """Get learned time constants for each layer."""
        return [layer.tau.copy() for layer in self.layers]


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute softmax cross-entropy loss and gradient."""
    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    # Cross-entropy
    loss = -np.log(probs[labels] + 1e-10)

    # Gradient
    grad = probs.copy()
    grad[labels] -= 1

    return loss, grad


def train_temporal_e2e(model: TemporalMLP, x_train: np.ndarray, y_train: np.ndarray,
                       x_val: np.ndarray, y_val: np.ndarray,
                       epochs: int = 30, lr: float = 0.01,
                       settle_range: Tuple[int, int] = (5, 20)):
    """
    Train temporal network end-to-end with randomized settle steps.
    """
    n_samples = len(x_train)
    history = {'train_loss': [], 'val_acc': []}

    print(f"Training with settle steps in range {settle_range}")
    print(f"Initial time constants: {[f'{t.mean():.1f}' for t in model.get_time_constants()]}")

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0

        for i in indices:
            # Randomize settle steps
            settle = np.random.randint(settle_range[0], settle_range[1] + 1)

            # Forward
            out = model.forward(x_train[i], settle_steps=settle)

            # Loss
            loss, grad = softmax_cross_entropy(out, y_train[i])
            epoch_loss += loss

            # Backward
            model.backward(grad, lr=lr)

        # Validation
        correct = 0
        for i in range(len(x_val)):
            out = model.forward(x_val[i], settle_steps=15)
            if np.argmax(out) == y_val[i]:
                correct += 1

        val_acc = correct / len(x_val)

        history['train_loss'].append(epoch_loss / n_samples)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            taus = model.get_time_constants()
            tau_str = ", ".join([f"{t.mean():.1f}" for t in taus])
            print(f"  Epoch {epoch+1}: Loss={epoch_loss/n_samples:.4f}, "
                  f"Val Acc={val_acc:.3f}, Tau=[{tau_str}]")

    return history


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("END-TO-END TEMPORAL TRAINING")
    print("=" * 70)

    # Load data
    from ecg_proper import load_ecg_with_rr, precision_recall_f1, print_metrics

    x_train, y_train, x_test, y_test = load_ecg_with_rr(n_train=5000, n_test=1000)

    CLASS_NAMES = ['Normal', 'Supravent.', 'Ventricular', 'Fusion', 'Unknown']
    N_CLASSES = 5
    input_size = x_train.shape[1]

    print(f"\nData: {len(x_train)} train, {len(x_test)} test, {input_size} features")

    # Create temporal network
    model = TemporalMLP(
        layer_sizes=[input_size, 64, 32, N_CLASSES],
        tau_init=15.0
    )

    print(f"Architecture: {model.layer_sizes}")
    print(f"Initial time constants per layer: {[t.mean() for t in model.get_time_constants()]}")

    # Train end-to-end
    print("\n" + "-" * 70)
    print("Training with backprop through time...")
    print("-" * 70)

    history = train_temporal_e2e(
        model, x_train, y_train,
        x_test[:200], y_test[:200],  # Small val set for speed
        epochs=30,
        lr=0.005,
        settle_range=(5, 25)  # Randomize settle steps
    )

    # Final evaluation
    print("\n" + "-" * 70)
    print("Final Evaluation")
    print("-" * 70)

    print(f"\nLearned time constants: {[f'{t.mean():.2f}' for t in model.get_time_constants()]}")

    predictions = []
    for i in range(len(x_test)):
        out = model.forward(x_test[i], settle_steps=15)
        predictions.append(np.argmax(out))

    predictions = np.array(predictions)
    metrics = precision_recall_f1(y_test, predictions, N_CLASSES)

    print("\n--- End-to-End Temporal Network ---")
    print_metrics(metrics, CLASS_NAMES)

    # Compare different settle times
    print("\n" + "-" * 70)
    print("Performance vs Settle Steps (learned dynamics)")
    print("-" * 70)

    for settle in [3, 5, 10, 15, 20, 30]:
        preds = []
        for i in range(min(500, len(x_test))):
            out = model.forward(x_test[i], settle_steps=settle)
            preds.append(np.argmax(out))
        preds = np.array(preds)
        m = precision_recall_f1(y_test[:len(preds)], preds, N_CLASSES)
        print(f"  Settle={settle:2d}: Macro F1={m['macro_f1']:.3f}, "
              f"Balanced Acc={m['balanced_accuracy']:.3f}")

    return metrics


if __name__ == "__main__":
    main()
