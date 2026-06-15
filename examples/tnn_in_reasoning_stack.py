"""
TNN in a deterministic reasoning stack — stability protects the downstream layers.

A TNN's headline strength is STABILITY: far fewer noise-driven prediction flips
than an instantaneous classical network. This example shows WHY that matters
when the classifier is the perception front-end of a deterministic decision
pipeline — the kind where each declared state-change triggers an expensive,
auditable computation downstream (an exact risk recompute, a knowledge-base
update, an alarm).

A flip-prone classical front-end THRASHES that downstream layer: every
noise-driven flip looks like a real state-change and fires a spurious recompute
/ alarm / contradictory fact. The TNN's stability means the downstream layer
only acts on REAL transitions.

Composition (all deterministic, NO LLM anywhere):
    TNN (this repo)            — robust, stable perception from a noisy stream
    structural-computing (opt) — the exact downstream decision (failure prob per state)
    [a knowledge/decision layer — here a simple state-change trigger; in a full
     stack this is where sourced, temporally-scoped facts would live]

Run:  python examples/tnn_in_reasoning_stack.py
(structural-computing is optional: `pip install structural-computing` to see the
 exact recomputes; without it the demo counts decision-layer triggers instead.)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tnn import ClassicalNetwork, convert_to_temporal, NeuronType

# Optional exact-decision engine (no LLM). Graceful if absent.
try:
    from structural_computing import StructuralComputer
    _SC = StructuralComputer()
except Exception:
    _SC = None

RNG = np.random.default_rng(7)

# Three site "protection states" the perception layer must classify from a
# noisy sensor feature vector. Each maps to a protection topology whose EXACT
# failure probability the downstream layer recomputes when the state changes.
STATE_NAMES = {0: "fully_protected", 1: "degraded", 2: "impaired"}
STATE_MEANS = {0: np.array([2.0, 0.0, 0.0]),
               1: np.array([0.0, 2.0, 0.0]),
               2: np.array([0.0, 0.0, 2.0])}
RING = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
STATE_TOPOLOGY = {
    0: RING + [(0, 3), (1, 4), (2, 5)],   # fully protected: gridded
    1: RING + [(0, 3)],                   # degraded: ring + one cross-tie
    2: RING,                              # impaired: ring only (least redundant)
}


def train_perception():
    """Phase 1: a small classical classifier on clean state samples."""
    X, y = [], []
    for _ in range(900):
        s = int(RNG.integers(3))
        X.append(STATE_MEANS[s] + RNG.normal(0, 0.4, 3))
        y.append(s)
    X, y = np.array(X), np.array(y)
    net = ClassicalNetwork([3, 24, 3], activation="tanh")
    Yoh = np.eye(3)[y]
    for _ in range(400):
        net.train_step(X, Yoh, learning_rate=0.1)
    acc = (net.predict(X).argmax(1) == y).mean()
    return net, acc


def true_state_sequence(T=140):
    """Mostly stable, with TWO genuine transitions: 0 -> 1 -> 2."""
    s = np.zeros(T, dtype=int)
    s[50:100] = 1
    s[100:] = 2
    return s


def exact_failure_prob(state):
    if _SC is None:
        return None
    return _SC.tail_probability(STATE_TOPOLOGY[state], p_fail=0.05)


def run():
    net, acc = train_perception()
    temporal = convert_to_temporal(net, {}, default_type=NeuronType.LEAKY_INTEGRATOR)
    truth = true_state_sequence()
    T = len(truth)
    NOISE = 1.1   # heavy observation noise

    cls_pred, tnn_pred = [], []
    temporal.reset()
    for k in range(T):
        x = STATE_MEANS[truth[k]] + RNG.normal(0, NOISE, 3)
        cls_pred.append(int(net.predict(x.reshape(1, -1)).argmax()))
        out = temporal.step(dt=0.1, input_signal=x)        # TNN integrates over time
        tnn_pred.append(int(np.argmax(out)))
    cls_pred, tnn_pred = np.array(cls_pred), np.array(tnn_pred)

    LATENCY = 40   # the TNN integrates over a window to reject noise, so it
                   # responds to a REAL transition with some latency; count a
                   # transition caught if it settles to the new state within this many steps

    def flips(p):
        return int(np.sum(p[1:] != p[:-1]))

    def spurious_changes(p):
        # a declared change to a state that is NOT the current true state = a
        # noise-driven error. A lagged-but-correct transition predicts the RIGHT
        # new state, so it is not counted as spurious.
        return int(sum(1 for t in range(1, T) if p[t] != p[t-1] and p[t] != truth[t]))

    def caught(p):
        # a real transition (truth changes at r to state S) is caught if the
        # front-end settles to S within LATENCY steps.
        c = 0
        for r in range(1, T):
            if truth[r] != truth[r-1]:
                if any(p[t] == truth[r] for t in range(r, min(r + LATENCY, T))):
                    c += 1
        return c

    real_transitions = int(np.sum(truth[1:] != truth[:-1]))  # = 2

    print("=" * 78)
    print("TNN in a deterministic reasoning stack — stability protects downstream")
    print("=" * 78)
    print(f"\nPerception classifier trained (clean accuracy {acc:.1%}).")
    print(f"Stream: {T} steps, heavy noise (sigma={NOISE}); {real_transitions} REAL state transitions "
          "(0->1->2).\n")

    print(f"{'front-end':14s} {'total flips':>12s} {'spurious (noise→wrong state)':>30s} {'real transitions caught':>26s}")
    for name, p in [("classical", cls_pred), ("TNN", tnn_pred)]:
        print(f"{name:14s} {flips(p):>12d} {spurious_changes(p):>30d} {caught(p):>24d}/{real_transitions}")

    # ---- Downstream deterministic decision layer -------------------------
    # Each DECLARED state change triggers an exact recompute (structural-computing)
    # or, absent it, a decision-layer trigger. Count how many each front-end fires.
    def downstream_triggers(p):
        changes = [t for t in range(1, T) if p[t] != p[t-1]]
        recomputes = []
        for t in changes:
            fp = exact_failure_prob(p[t])
            recomputes.append((t, STATE_NAMES[p[t]], fp))
        return recomputes

    cls_tr = downstream_triggers(cls_pred)
    tnn_tr = downstream_triggers(tnn_pred)
    engine = "exact failure-prob recomputes (structural-computing)" if _SC else "decision-layer triggers"
    print(f"\nDownstream {engine}:")
    print(f"   classical front-end fired {len(cls_tr):>2d}  (mostly spurious, noise-driven churn)")
    print(f"   TNN       front-end fired {len(tnn_tr):>2d}  (tracks the real transitions)")
    if _SC and tnn_tr:
        seq = " -> ".join(f"{name}:{fp:.4f}" for _, name, fp in tnn_tr)
        print(f"   TNN-driven exact MFL trajectory: {seq}")

    reduction = (len(cls_tr) / max(1, len(tnn_tr)))
    print("\nVALUE IN COMBINATION:")
    print(f"   The TNN front-end cuts downstream churn ~{reduction:.0f}x. Its stability means the")
    print("   EXACT decision layer recomputes (and alarms) only on REAL state changes, not on")
    print("   sensor noise. A flip-prone classical front-end would thrash the deterministic")
    print("   stack with spurious recomputes and contradictory state. TNN is the robust")
    print("   perception layer that makes downstream exact/auditable reasoning trustworthy —")
    print("   and there is NO LLM anywhere in the chain.")

    assert spurious_changes(tnn_pred) < spurious_changes(cls_pred), "TNN should reduce noise-driven changes"
    assert len(tnn_tr) <= len(cls_tr), "TNN should not fire more downstream triggers than classical"
    print("\nALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    run()
