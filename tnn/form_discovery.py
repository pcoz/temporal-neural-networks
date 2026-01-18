"""
Form Discovery - Use PPF to discover temporal dynamics.

Phase 2 of the TNN approach: Analyze activation patterns from a trained
classical network and discover the mathematical forms that describe
how neurons behave over time.

This module uses PPF (Partial Form Finding) for symbolic regression,
discovering expressions like:
- Damped oscillations: A·sin(ωt)·exp(-λt)
- Adaptation: sigmoid(accumulated - θ)
- Resonance: driven harmonic oscillators
- Novel forms we wouldn't have designed
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans

# PPF imports
from ppf import SymbolicRegressor
from ppf.symbolic_types import (
    DiscoveryMode,
    SymbolicRegressionResult,
    SymbolicFitResult,
    ExprNode
)


@dataclass
class DiscoveredForm:
    """A mathematical form discovered for a neuron or cluster."""
    expression: ExprNode
    expression_string: str
    r_squared: float
    complexity: int
    neuron_ids: List[int]  # Which neurons this form applies to
    layer: int


@dataclass
class FormDiscoveryConfig:
    """Configuration for form discovery."""
    population_size: int = 300
    generations: int = 30
    max_depth: int = 5
    parsimony_coefficient: float = 0.001
    mode: DiscoveryMode = DiscoveryMode.AUTO
    min_r_squared: float = 0.7  # Minimum acceptable fit
    cluster_neurons: bool = True  # Group similar neurons
    n_clusters: int = 10  # Number of neuron clusters per layer


def discover_forms(
    activation_history: List[List[np.ndarray]],
    config: FormDiscoveryConfig = None,
    verbose: bool = True
) -> Dict[int, List[DiscoveredForm]]:
    """
    Discover mathematical forms from recorded activations.

    Args:
        activation_history: List of [time_step][layer] -> activation array
            from ClassicalNetwork.get_activation_history()
        config: Discovery configuration
        verbose: Print progress

    Returns:
        Dict mapping layer index to list of discovered forms
    """
    if config is None:
        config = FormDiscoveryConfig()

    if not activation_history:
        raise ValueError("No activation history provided")

    # Determine number of layers
    n_layers = len(activation_history[0])
    n_timesteps = len(activation_history)

    if verbose:
        print(f"Analyzing {n_layers} layers over {n_timesteps} time steps")

    # Create symbolic regressor
    regressor = SymbolicRegressor(
        population_size=config.population_size,
        generations=config.generations,
        max_depth=config.max_depth,
        parsimony_coefficient=config.parsimony_coefficient
    )

    discovered = {}
    time_points = np.arange(n_timesteps).astype(float)

    for layer_idx in range(n_layers):
        if verbose:
            print(f"\n--- Layer {layer_idx} ---")

        # Extract activation time series for all neurons in this layer
        # Shape: (n_neurons, n_timesteps)
        layer_activations = _extract_layer_timeseries(activation_history, layer_idx)
        n_neurons = layer_activations.shape[0]

        if verbose:
            print(f"  {n_neurons} neurons")

        if config.cluster_neurons and n_neurons > config.n_clusters:
            # Cluster neurons by their temporal patterns
            forms = _discover_clustered(
                layer_activations, time_points, layer_idx,
                regressor, config, verbose
            )
        else:
            # Discover form for each neuron individually
            forms = _discover_individual(
                layer_activations, time_points, layer_idx,
                regressor, config, verbose
            )

        discovered[layer_idx] = forms

    return discovered


def _extract_layer_timeseries(
    activation_history: List[List[np.ndarray]],
    layer_idx: int
) -> np.ndarray:
    """Extract time series for all neurons in a layer."""
    timeseries = []

    for t, step in enumerate(activation_history):
        layer_output = step[layer_idx]
        # Handle both batched and single outputs
        if layer_output.ndim > 1:
            layer_output = layer_output.mean(axis=0)  # Average over batch
        timeseries.append(layer_output)

    return np.array(timeseries).T  # Shape: (n_neurons, n_timesteps)


def _discover_clustered(
    activations: np.ndarray,
    time_points: np.ndarray,
    layer_idx: int,
    regressor: SymbolicRegressor,
    config: FormDiscoveryConfig,
    verbose: bool
) -> List[DiscoveredForm]:
    """Discover forms by clustering similar neurons."""
    n_neurons = activations.shape[0]
    n_clusters = min(config.n_clusters, n_neurons)

    if verbose:
        print(f"  Clustering into {n_clusters} groups")

    # Cluster by activation pattern similarity
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(activations)

    forms = []

    for cluster_id in range(n_clusters):
        neuron_ids = np.where(cluster_labels == cluster_id)[0].tolist()

        if not neuron_ids:
            continue

        # Use mean activation of cluster
        cluster_mean = activations[neuron_ids].mean(axis=0)

        # Discover form
        try:
            result = regressor.discover(
                time_points,
                cluster_mean,
                mode=config.mode,
                verbose=False
            )

            if result.best_tradeoff and result.best_tradeoff.r_squared >= config.min_r_squared:
                form = DiscoveredForm(
                    expression=result.best_tradeoff.expression,
                    expression_string=result.best_tradeoff.expression_string,
                    r_squared=result.best_tradeoff.r_squared,
                    complexity=result.best_tradeoff.complexity,
                    neuron_ids=neuron_ids,
                    layer=layer_idx
                )
                forms.append(form)

                if verbose:
                    print(f"  Cluster {cluster_id} ({len(neuron_ids)} neurons): "
                          f"{result.best_tradeoff.expression_string} "
                          f"(R²={result.best_tradeoff.r_squared:.3f})")
            else:
                if verbose:
                    r2 = result.best_tradeoff.r_squared if result.best_tradeoff else 0
                    print(f"  Cluster {cluster_id}: No good fit found (R²={r2:.3f})")

        except Exception as e:
            if verbose:
                print(f"  Cluster {cluster_id}: Discovery failed - {e}")

    return forms


def _discover_individual(
    activations: np.ndarray,
    time_points: np.ndarray,
    layer_idx: int,
    regressor: SymbolicRegressor,
    config: FormDiscoveryConfig,
    verbose: bool
) -> List[DiscoveredForm]:
    """Discover forms for each neuron individually."""
    forms = []
    n_neurons = activations.shape[0]

    for neuron_id in range(n_neurons):
        neuron_series = activations[neuron_id]

        try:
            result = regressor.discover(
                time_points,
                neuron_series,
                mode=config.mode,
                verbose=False
            )

            if result.best_tradeoff and result.best_tradeoff.r_squared >= config.min_r_squared:
                form = DiscoveredForm(
                    expression=result.best_tradeoff.expression,
                    expression_string=result.best_tradeoff.expression_string,
                    r_squared=result.best_tradeoff.r_squared,
                    complexity=result.best_tradeoff.complexity,
                    neuron_ids=[neuron_id],
                    layer=layer_idx
                )
                forms.append(form)

                if verbose:
                    print(f"  Neuron {neuron_id}: {result.best_tradeoff.expression_string} "
                          f"(R²={result.best_tradeoff.r_squared:.3f})")

        except Exception as e:
            if verbose:
                print(f"  Neuron {neuron_id}: Discovery failed - {e}")

    return forms


def analyze_activations(
    activation_history: List[List[np.ndarray]],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Analyze activation patterns without full form discovery.

    Quick analysis to understand what kinds of dynamics are present.

    Args:
        activation_history: Recorded activations from classical network
        verbose: Print analysis

    Returns:
        Analysis dict with statistics
    """
    if not activation_history:
        return {}

    n_layers = len(activation_history[0])
    n_timesteps = len(activation_history)

    analysis = {
        'n_layers': n_layers,
        'n_timesteps': n_timesteps,
        'layer_stats': {}
    }

    for layer_idx in range(n_layers):
        layer_ts = _extract_layer_timeseries(activation_history, layer_idx)

        stats = {
            'n_neurons': layer_ts.shape[0],
            'mean_activation': float(np.mean(layer_ts)),
            'std_activation': float(np.std(layer_ts)),
            'temporal_variance': float(np.mean(np.var(layer_ts, axis=1))),
            'has_oscillation': _detect_oscillation(layer_ts),
            'has_drift': _detect_drift(layer_ts)
        }

        analysis['layer_stats'][layer_idx] = stats

        if verbose:
            print(f"Layer {layer_idx}: {stats['n_neurons']} neurons, "
                  f"mean={stats['mean_activation']:.3f}, "
                  f"temporal_var={stats['temporal_variance']:.3f}, "
                  f"oscillation={stats['has_oscillation']}, "
                  f"drift={stats['has_drift']}")

    return analysis


def _detect_oscillation(timeseries: np.ndarray, threshold: float = 0.1) -> bool:
    """Detect if time series shows oscillatory behavior."""
    # Simple zero-crossing based detection
    for neuron_ts in timeseries:
        centered = neuron_ts - np.mean(neuron_ts)
        if np.std(centered) < 0.01:
            continue

        crossings = np.sum(np.diff(np.sign(centered)) != 0)
        if crossings > len(neuron_ts) * threshold:
            return True
    return False


def _detect_drift(timeseries: np.ndarray, threshold: float = 0.5) -> bool:
    """Detect if time series shows systematic drift."""
    for neuron_ts in timeseries:
        if len(neuron_ts) < 10:
            continue

        # Compare first and last quarters
        first_quarter = np.mean(neuron_ts[:len(neuron_ts)//4])
        last_quarter = np.mean(neuron_ts[-len(neuron_ts)//4:])

        if np.abs(last_quarter - first_quarter) > threshold * np.std(neuron_ts):
            return True
    return False


def extract_oscillation_params(form: DiscoveredForm) -> dict:
    """
    Extract oscillation parameters from a discovered form.

    PPF discovers forms like: A*exp(-d*t)*sin(w*t + p)
    We extract the frequency (w) and use it to parameterize
    proper temporal dynamics.
    """
    expr_str = form.expression_string.lower()

    # Default parameters
    params = {
        'frequency': 0.1,
        'damping': 0.01,
        'amplitude': 1.0
    }

    # Try to extract frequency from sin/cos terms
    import re

    # Look for patterns like sin(0.3*x) or cos(5.983*x)
    freq_match = re.search(r'(sin|cos)\(([0-9.]+)\*x', expr_str)
    if freq_match:
        params['frequency'] = float(freq_match.group(2))

    # Look for exp(-d*x) damping
    damp_match = re.search(r'exp\(-([0-9.e-]+)\*x\)', expr_str)
    if damp_match:
        params['damping'] = float(damp_match.group(1))

    # Look for amplitude
    amp_match = re.search(r'^(-?[0-9.]+)\*', expr_str)
    if amp_match:
        params['amplitude'] = abs(float(amp_match.group(1)))

    return params


def create_temporal_form_function(form: DiscoveredForm) -> callable:
    """
    Create a callable function from a discovered form.

    Instead of using the form directly (which describes activation patterns),
    we extract the oscillation frequency and create proper dynamics:
    - A resonator that responds to the discovered frequency
    - Input-driven with the learned temporal characteristics

    Args:
        form: DiscoveredForm from discovery

    Returns:
        Callable for use in TemporalNeuron
    """
    osc_params = extract_oscillation_params(form)

    def temporal_dynamics(t: float, V: float, I: float, params: dict) -> float:
        """
        Proper temporal dynamics based on discovered form.

        Uses a leaky integrator that responds to input,
        with time constant derived from discovered frequency.
        """
        # Time constant from discovered frequency
        # Higher frequency = faster response
        freq = osc_params.get('frequency', 0.1)
        tau = max(1.0, 10.0 / (freq + 0.01))  # Avoid division by zero

        # Leaky integration toward input
        decay = np.exp(-1.0 / tau)
        V_new = decay * V + (1 - decay) * I

        return V_new

    return temporal_dynamics


def summarize_discoveries(
    discovered_forms: Dict[int, List[DiscoveredForm]]
) -> str:
    """Generate a summary of discovered forms."""
    lines = ["=== Discovered Temporal Forms ===\n"]

    for layer_idx in sorted(discovered_forms.keys()):
        forms = discovered_forms[layer_idx]
        lines.append(f"Layer {layer_idx}: {len(forms)} forms discovered")

        for i, form in enumerate(forms):
            lines.append(f"  [{i}] {form.expression_string}")
            lines.append(f"      R²={form.r_squared:.3f}, "
                        f"complexity={form.complexity}, "
                        f"neurons={len(form.neuron_ids)}")

        lines.append("")

    return "\n".join(lines)
