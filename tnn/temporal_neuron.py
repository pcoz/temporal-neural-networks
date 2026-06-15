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
                # --- self-tuning latency feedback (opt-in; off = v1 behaviour) ---
                'adaptive': False,  # when True, tau self-adjusts per step
                'tau_min': 2.0,     # fast tau while chasing a detected change
                # Detector windows are in STEPS (intuitive, dt-independent). The
                # noise floor is estimated ONLINE, so the only scale-free knob is
                # z_thresh (sigma units). These time-SCALES are inherently
                # signal-dependent — there is no universal best; the self-tuning
                # trace itself helps reveal a signal's scales (see the worked
                # example examples/self_tuning_reveals_signal.py). Full reference:
                # docs/PARAMETERS.md.
                'fast_steps': 8,        # BAND-PASS fast mean (rejects single-sample noise)
                'slow_steps': 40,       # BAND-PASS slow mean (the recent 'normal')
                'sigma_steps': 60,      # online band-pass variance + warmup window
                'z_thresh': 3.0,        # onset: change is 'real' past ~this many sigma
                'z_exit': 1.5,          # hysteresis: chase may end once z falls below this
                'caught_thresh': 0.2,   # output-space lag at which the chase ends (feedback)
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

    def enable_self_tuning(self, tau_min: Optional[float] = None,
                           adapt_gain: Optional[float] = None,
                           surprise_tau: Optional[float] = None) -> "TemporalNeuron":
        """Turn on self-tuning latency feedback (leaky neurons).

        The time constant then adapts per step: stable under steady/noisy
        input, fast under a persistent shift. Returns self for chaining."""
        self.params['adaptive'] = True
        if tau_min is not None:
            self.params['tau_min'] = tau_min
        if adapt_gain is not None:
            self.params['adapt_gain'] = adapt_gain
        if surprise_tau is not None:
            self.params['surprise_tau'] = surprise_tau
        return self

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
        """Leaky integrator - accumulates with decay.

        With ``params['adaptive']`` set, the time constant SELF-TUNES via
        latency feedback (see ``_self_tuning_tau``): steady input keeps tau
        large (stable, noise-rejecting); a persistent shift shrinks tau so the
        neuron responds quickly, then tau relaxes back. Off by default, so the
        classic fixed-tau behaviour is unchanged.
        """
        tau = self.params.get('tau', 20.0)
        v_rest = self.params.get('v_rest', 0.0)
        gain = self.params.get('gain', 1.0)

        if self.params.get('adaptive', False):
            tau = self._self_tuning_tau(dt, inputs, tau, v_rest, gain)
        self.tau_eff = tau   # exposed for inspection / metrics

        # dV/dt = -(V - V_rest)/tau + gain * I
        dv = (-(self.state.voltage - v_rest) / tau + gain * inputs) * dt
        self.state.voltage += dv

        self.state.output = np.tanh(self.state.voltage)
        return self.state.output

    def _self_tuning_tau(self, dt: float, inputs: float,
                         tau_base: float, v_rest: float, gain: float) -> float:
        """Self-CALIBRATING latency feedback. The change detector is a proper
        difference-of-EMAs BAND-PASS, z-scored against an ONLINE variance whose
        estimate is FROZEN during a detected event (so events don't inflate the
        noise floor). The chase is then closed-loop: a significant change latches
        it, the neuron's own tracking lag sustains/terminates it.

          drive   = gain * inputs                 # RAW drive (no tanh: a step between
                                                  #   two large values must stay visible)
          m_fast  = EMA(drive, fast_steps)        # fast mean  (rejects single-sample noise)
          m_slow  = EMA(drive, slow_steps)        # slow mean  (the 'recent normal')
          bp      = m_fast - m_slow   [BAND-PASS] # signed change signal: 0 at steady DC,
                                                  #   spikes at a step, and stays NONZERO
                                                  #   during a ramp (fast mean leads slow)
          var     = EMA(bp^2, sigma_steps)        # online variance of the band-pass...
                                                  #   ...FROZEN while chasing (estimate noise,
                                                  #      not events)
          z       = |bp| / sqrt(var)              # significance in sigma units (scale-free)

        Versus a single high-pass against one baseline, the difference-of-EMAs
        band-pass (a) removes slow DC via the slow mean and (b) suppresses
        single-sample noise via the fast mean — a cleaner abrupt-change /
        noise-rejection detector. NOTE: a *very gradual* drift can slip under the
        instantaneous z-bar (its own band-pass slowly inflates the online
        variance); detecting slow drift robustly is best done by accumulating the
        normalised band-pass CUSUM-style — a natural future extension, see
        docs/PARAMETERS.md. `bp`, `zscore`, `noise_scale` are exposed; they reveal
        the signal's change-points and noise floor
        (examples/self_tuning_reveals_signal.py).
        """
        tau_min = self.params.get('tau_min', max(1.0, 0.1 * tau_base))
        z_thresh = self.params.get('z_thresh', 3.0)
        caught_thresh = self.params.get('caught_thresh', 0.2)  # output-space "caught up"
        # Per-step EMA rates (1 / window-in-steps): dt-independent and intuitive.
        r_fast = 1.0 / max(1.0, self.params.get('fast_steps', 8))
        r_slow = 1.0 / max(1.0, self.params.get('slow_steps', 40))
        r_var = 1.0 / max(1.0, self.params.get('sigma_steps', 60))

        # --- ONSET DETECTOR: difference-of-EMAs band-pass on the RAW drive,
        # z-scored by an online (event-frozen) variance.
        drive = gain * inputs
        m_fast = getattr(self, '_m_fast', drive)
        m_fast = m_fast + (drive - m_fast) * r_fast
        self._m_fast = m_fast
        m_slow = getattr(self, '_m_slow', drive)
        m_slow = m_slow + (drive - m_slow) * r_slow
        self._m_slow = m_slow
        bp = m_fast - m_slow                                  # band-pass (signed) change signal
        # Online variance of the band-pass. Updated every step but with a SLOW
        # window (sigma_steps), so a brief event barely inflates it (the noise
        # floor stays meaningful) yet z still recovers once the change is
        # absorbed (bp -> 0), which is what lets the chase terminate.
        var = getattr(self, '_bp_var', bp * bp)
        var = var + (bp * bp - var) * r_var
        self._bp_var = var
        sig = var ** 0.5                                      # online std of the band-pass
        z = abs(bp) / (sig + 1e-6)                            # significance, sigma units

        # --- CLOSED-LOOP FEEDBACK: a significant change LATCHES a 'chase'; the
        # neuron's own tracking lag SUSTAINS it and TERMINATES it once the
        # neuron has caught up. The lag is measured in OUTPUT space (tanh),
        # where saturation denoises it: near-steady input keeps the output
        # pinned (lag ~ 0) so noise cannot keep the chase alive, while a real
        # shift moves the output a lot (large lag) until it settles. Onset is
        # detected on the RAW drive above (no saturation, so a step is visible);
        # termination uses the output lag here. So a noise spike may trip z for
        # an instant but produces no real output lag -> the chase ends at once.
        drive_out = np.tanh(v_rest + tau_base * gain * inputs)
        out_lag = abs(drive_out - self.state.output)          # output-space lag (denoised)
        z_exit = self.params.get('z_exit', 1.5)               # hysteresis below z_thresh
        chasing = getattr(self, '_chasing', False)
        self._n_steps = getattr(self, '_n_steps', 0) + 1
        warm = self._n_steps >= self.params.get('sigma_steps', 60)  # let sigma settle first
        if warm and z > z_thresh:
            chasing = True                                    # onset detected -> chase fast
        elif chasing and z < z_exit and out_lag < caught_thresh:
            # SUSTAIN until the change is fully accommodated: stop only when the
            # input deviation has been absorbed into the baseline (z low) AND the
            # output has caught up (out_lag low). The two cover complementary
            # cases — z sees same-sign steps the saturated output cannot, while
            # out_lag sees a sign-crossing the baseline absorbs quickly.
            chasing = False
        self._chasing = chasing

        self.zscore, self.noise_scale, self.out_lag = z, sig, out_lag
        self.surprise = 1.0 if chasing else 0.0
        return tau_min if chasing else tau_base

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
        # clear self-tuning estimators so a fresh stream recalibrates
        for attr in ('_m_fast', '_m_slow', '_bp_var', '_n_steps', '_chasing',
                     'surprise', 'zscore', 'noise_scale', 'out_lag', 'tau_eff'):
            if hasattr(self, attr):
                delattr(self, attr)

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
