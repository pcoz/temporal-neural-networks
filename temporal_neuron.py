"""
Temporal Neuron - A neuron that exists in time.

Each TemporalNeuron is a dynamical process governed by a mathematical form.
The form can be discovered via PPF or specified directly.
"""

import numpy as np
from enum import Enum
from typing import Optional, Callable, List
from dataclasses import dataclass


class NeuronType(Enum):
    """Types of temporal dynamics a neuron can exhibit."""
    INTEGRATOR = "integrator"           # Accumulates input over time
    LEAKY_INTEGRATOR = "leaky"          # Accumulates with decay
    OSCILLATOR = "oscillator"           # Intrinsic rhythm
    RESONATOR = "resonator"             # Responds to specific frequencies
    ADAPTING = "adapting"               # Threshold changes with activity
    BURSTING = "bursting"               # Alternates active/quiescent
    CUSTOM = "custom"                   # PPF-discovered form


@dataclass
class NeuronState:
    """Complete state of a temporal neuron at a moment."""
    voltage: float              # Primary state variable
    threshold: float            # Firing threshold (for adapting neurons)
    refractory: float          # Time remaining in refractory period
    phase: float               # Phase for oscillatory neurons
    adaptation: float          # Adaptation variable
    output: float              # Current output (post-activation)


class TemporalNeuron:
    """
    A neuron that evolves over time according to its dynamics.

    Unlike standard neurons (output = activation(weights @ inputs + bias)),
    a TemporalNeuron maintains state that evolves according to a
    mathematical form:

        dV/dt = f(V, inputs, t)  or  V(t) = form(t, inputs, params)

    The form can be:
    - A built-in type (INTEGRATOR, OSCILLATOR, etc.)
    - A custom PPF-discovered expression
    """

    def __init__(
        self,
        neuron_type: NeuronType = NeuronType.LEAKY_INTEGRATOR,
        custom_form: Optional[Callable] = None,
        params: Optional[dict] = None,
        initial_state: Optional[NeuronState] = None
    ):
        self.neuron_type = neuron_type
        self.custom_form = custom_form
        self.params = params or self._default_params(neuron_type)

        # Initialize state
        if initial_state:
            self.state = initial_state
        else:
            self.state = NeuronState(
                voltage=0.0,
                threshold=1.0,
                refractory=0.0,
                phase=0.0,
                adaptation=0.0,
                output=0.0
            )

        self.t = 0.0  # Internal time
        self.history: List[NeuronState] = []

    def _default_params(self, neuron_type: NeuronType) -> dict:
        """Default parameters for each neuron type."""
        defaults = {
            NeuronType.INTEGRATOR: {
                'gain': 1.0,
            },
            NeuronType.LEAKY_INTEGRATOR: {
                'tau': 20.0,        # Time constant (ms)
                'v_rest': 0.0,      # Resting potential
                'gain': 1.0,
            },
            NeuronType.OSCILLATOR: {
                'frequency': 10.0,  # Hz
                'amplitude': 1.0,
                'baseline': 0.0,
            },
            NeuronType.RESONATOR: {
                'resonant_freq': 10.0,
                'damping': 0.1,
                'gain': 1.0,
            },
            NeuronType.ADAPTING: {
                'tau_v': 20.0,      # Voltage time constant
                'tau_adapt': 100.0, # Adaptation time constant
                'adapt_strength': 0.5,
                'v_rest': 0.0,
            },
            NeuronType.BURSTING: {
                'tau_fast': 10.0,
                'tau_slow': 100.0,
                'burst_threshold': 0.8,
                'quiescent_threshold': 0.2,
            },
            NeuronType.CUSTOM: {},
        }
        return defaults.get(neuron_type, {})

    def step(self, dt: float, inputs: float) -> float:
        """
        Advance the neuron by one time step.

        Args:
            dt: Time step size
            inputs: Summed weighted input from other neurons

        Returns:
            Current output value
        """
        if self.neuron_type == NeuronType.CUSTOM and self.custom_form:
            return self._step_custom(dt, inputs)

        # Built-in dynamics
        method = {
            NeuronType.INTEGRATOR: self._step_integrator,
            NeuronType.LEAKY_INTEGRATOR: self._step_leaky,
            NeuronType.OSCILLATOR: self._step_oscillator,
            NeuronType.RESONATOR: self._step_resonator,
            NeuronType.ADAPTING: self._step_adapting,
            NeuronType.BURSTING: self._step_bursting,
        }.get(self.neuron_type, self._step_leaky)

        output = method(dt, inputs)
        self.t += dt

        # Record history
        self.history.append(NeuronState(
            voltage=self.state.voltage,
            threshold=self.state.threshold,
            refractory=self.state.refractory,
            phase=self.state.phase,
            adaptation=self.state.adaptation,
            output=output
        ))

        return output

    def _step_integrator(self, dt: float, inputs: float) -> float:
        """Pure integrator - accumulates input."""
        gain = self.params.get('gain', 1.0)
        self.state.voltage += gain * inputs * dt
        self.state.output = np.tanh(self.state.voltage)  # Bounded output
        return self.state.output

    def _step_leaky(self, dt: float, inputs: float) -> float:
        """Leaky integrator - accumulates with decay."""
        tau = self.params.get('tau', 20.0)
        v_rest = self.params.get('v_rest', 0.0)
        gain = self.params.get('gain', 1.0)

        # dV/dt = -(V - V_rest)/tau + gain * I
        dv = (-(self.state.voltage - v_rest) / tau + gain * inputs) * dt
        self.state.voltage += dv

        self.state.output = np.tanh(self.state.voltage)
        return self.state.output

    def _step_oscillator(self, dt: float, inputs: float) -> float:
        """Intrinsic oscillator with input modulation."""
        freq = self.params.get('frequency', 10.0)
        amp = self.params.get('amplitude', 1.0)
        baseline = self.params.get('baseline', 0.0)

        # Advance phase
        self.state.phase += 2 * np.pi * freq * dt / 1000  # Convert to radians

        # Oscillation + input modulation
        oscillation = amp * np.sin(self.state.phase)
        self.state.voltage = baseline + oscillation + 0.3 * inputs

        self.state.output = np.tanh(self.state.voltage)
        return self.state.output

    def _step_resonator(self, dt: float, inputs: float) -> float:
        """Resonator - responds to specific frequencies."""
        omega = 2 * np.pi * self.params.get('resonant_freq', 10.0) / 1000
        damping = self.params.get('damping', 0.1)
        gain = self.params.get('gain', 1.0)

        # Damped harmonic oscillator driven by input
        # d²V/dt² + 2*damping*omega*dV/dt + omega²*V = gain*I
        # Discretized as two first-order equations
        v = self.state.voltage
        dv = self.state.adaptation  # Using adaptation as velocity

        ddv = -2 * damping * omega * dv - omega**2 * v + gain * inputs

        self.state.adaptation += ddv * dt  # Update velocity
        self.state.voltage += self.state.adaptation * dt  # Update position

        self.state.output = np.tanh(self.state.voltage)
        return self.state.output

    def _step_adapting(self, dt: float, inputs: float) -> float:
        """Adapting neuron - threshold increases with activity."""
        tau_v = self.params.get('tau_v', 20.0)
        tau_adapt = self.params.get('tau_adapt', 100.0)
        adapt_strength = self.params.get('adapt_strength', 0.5)
        v_rest = self.params.get('v_rest', 0.0)

        # Voltage dynamics with adaptation
        dv = (-(self.state.voltage - v_rest) / tau_v + inputs - self.state.adaptation) * dt
        self.state.voltage += dv

        # Adaptation increases with activity, decays to zero
        activity = max(0, self.state.voltage)
        da = (-self.state.adaptation / tau_adapt + adapt_strength * activity) * dt
        self.state.adaptation += da

        self.state.output = np.tanh(self.state.voltage)
        return self.state.output

    def _step_bursting(self, dt: float, inputs: float) -> float:
        """Bursting neuron - alternates between active and quiescent."""
        tau_fast = self.params.get('tau_fast', 10.0)
        tau_slow = self.params.get('tau_slow', 100.0)
        burst_thresh = self.params.get('burst_threshold', 0.8)
        quiet_thresh = self.params.get('quiescent_threshold', 0.2)

        # Fast variable (voltage-like)
        dv = (-self.state.voltage / tau_fast + inputs + self.state.adaptation) * dt
        self.state.voltage += dv

        # Slow variable (determines bursting/quiescent)
        # Increases during activity, decreases during quiescence
        if self.state.voltage > burst_thresh:
            da = (1.0 - self.state.adaptation) / tau_slow * dt
        elif self.state.voltage < quiet_thresh:
            da = -self.state.adaptation / tau_slow * dt
        else:
            da = 0

        self.state.adaptation += da

        # Adaptation inhibits when high (ends burst), excites when low (starts burst)
        effective_adapt = 2 * (self.state.adaptation - 0.5)

        self.state.output = np.tanh(self.state.voltage - effective_adapt)
        return self.state.output

    def _step_custom(self, dt: float, inputs: float) -> float:
        """Custom dynamics from PPF-discovered form."""
        if self.custom_form is None:
            return self._step_leaky(dt, inputs)

        # The custom form receives: t, V, I, params
        # And returns: new V (or dV/dt depending on form type)
        try:
            result = self.custom_form(
                self.t,
                self.state.voltage,
                inputs,
                self.params
            )

            # If result is dV/dt, integrate
            if self.params.get('is_derivative', False):
                self.state.voltage += result * dt
            else:
                self.state.voltage = result

            self.state.output = np.tanh(self.state.voltage)
            return self.state.output

        except Exception as e:
            # Fallback to leaky integrator
            return self._step_leaky(dt, inputs)

    def reset(self):
        """Reset neuron to initial state."""
        self.state = NeuronState(
            voltage=0.0,
            threshold=1.0,
            refractory=0.0,
            phase=np.random.uniform(0, 2*np.pi),  # Random initial phase
            adaptation=0.0,
            output=0.0
        )
        self.t = 0.0
        self.history = []

    def get_activation_history(self) -> np.ndarray:
        """Return the history of outputs as numpy array."""
        return np.array([s.output for s in self.history])

    def get_voltage_history(self) -> np.ndarray:
        """Return the history of voltages as numpy array."""
        return np.array([s.voltage for s in self.history])


def create_neuron_from_expression(expr, params: dict = None) -> TemporalNeuron:
    """
    Create a TemporalNeuron from a PPF expression.

    Args:
        expr: PPF ExprNode or callable
        params: Optional parameters for the expression

    Returns:
        TemporalNeuron with custom dynamics
    """
    if callable(expr):
        custom_form = expr
    else:
        # Assume it's a PPF ExprNode
        def custom_form(t, V, I, p):
            # PPF expressions typically take just x (time)
            # We'll extend to include V and I in params
            return expr.evaluate(t)

    return TemporalNeuron(
        neuron_type=NeuronType.CUSTOM,
        custom_form=custom_form,
        params=params or {}
    )
