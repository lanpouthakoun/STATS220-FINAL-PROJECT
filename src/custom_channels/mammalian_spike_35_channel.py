import jaxley as jx
from jaxley.channels import Channel
import jax.numpy as jnp
from typing import Dict, Tuple, Optional


class MammalianSpike35(Channel):
    """
    HH style channels for spiking retinal ganglion cells.
    Based on Fohlmeister et al, 2010, J Neurophysiology 103, 1357-1354
    Temperature: 35°C
    """
    
    def __init__(self, name: str = "mammalian_spike_35"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name=name)
    
    def init_state(self, v: float, params: Dict[str, float]) -> Dict[str, float]:
        """
        Initialize channel states.
        Initial values determined at resting value of -65.02 mV
        """
        
        return {
            "m": 0.0353,  # Na activation
            "h": 0.9054,  # Na inactivation
            "n": 0.0677,  # K activation
            "c": 0.0019,  # Ca activation
        }
    
    @staticmethod
    def alpha_beta_m(v: float) -> Tuple[float, float]:
        """Alpha and beta for Na activation gate m"""
        alpha = (-2.725 * (v + 35)) / (jnp.exp(-0.1 * (v + 35)) - 1)
        beta = 90.83 * jnp.exp(-(v + 60) / 20)
        return alpha, beta
    
    @staticmethod
    def alpha_beta_h(v: float) -> Tuple[float, float]:
        """Alpha and beta for Na inactivation gate h"""
        alpha = 1.817 * jnp.exp(-(v + 52) / 20)
        beta = 27.25 / (1 + jnp.exp(-0.1 * (v + 22)))
        return alpha, beta
    
    @staticmethod
    def alpha_beta_n(v: float) -> Tuple[float, float]:
        """Alpha and beta for K activation gate n"""
        alpha = (-0.09575 * (v + 37)) / (jnp.exp(-0.1 * (v + 37)) - 1)
        beta = 1.915 * jnp.exp(-(v + 47) / 80)
        return alpha, beta
    
    @staticmethod
    def alpha_beta_c(v: float) -> Tuple[float, float]:
        """Alpha and beta for Ca activation gate c"""
        alpha = (-1.362 * (v + 13)) / (jnp.exp(-0.1 * (v + 13)) - 1)
        beta = 45.41 * jnp.exp(-(v + 38) / 18)
        return alpha, beta
    
    @staticmethod
    def compute_steady_state(v: float, params: Dict[str, float]) -> Dict[str, float]:
        """Compute steady-state values and time constants for all gates"""
        # Na activation (m)
        alpha_m, beta_m = MammalianSpike35.alpha_beta_m(v)
        tau_m = 1 / (alpha_m + beta_m)
        m_inf = alpha_m * tau_m
        
        # Na inactivation (h)
        alpha_h, beta_h = MammalianSpike35.alpha_beta_h(v)
        tau_h = 1 / (alpha_h + beta_h)
        h_inf = alpha_h * tau_h
        
        # K activation (n)
        alpha_n, beta_n = MammalianSpike35.alpha_beta_n(v)
        tau_n = 1 / (alpha_n + beta_n)
        n_inf = alpha_n * tau_n
        
        # Ca activation (c)
        alpha_c, beta_c = MammalianSpike35.alpha_beta_c(v)
        tau_c = 1 / (alpha_c + beta_c)
        c_inf = alpha_c * tau_c
        
        return {
            "m_inf": m_inf, "tau_m": tau_m,
            "h_inf": h_inf, "tau_h": tau_h,
            "n_inf": n_inf, "tau_n": tau_n,
            "c_inf": c_inf, "tau_c": tau_c,
        }
    
    @staticmethod
    def update_states(
        states: Dict[str, float],
        dt: float,
        v: float,
        params: Dict[str, float]
    ) -> Dict[str, float]:
        """Update channel states using exponential integration"""
        # Get steady states and time constants
        steady = MammalianSpike35.compute_steady_state(v, params)
        
        # Update states using exact exponential solution
        # state' = (state_inf - state) / tau
        # Solution: state(t+dt) = state_inf + (state(t) - state_inf) * exp(-dt/tau)
        new_states = {
            "m": steady["m_inf"] + (states["m"] - steady["m_inf"]) * jnp.exp(-dt / steady["tau_m"]),
            "h": steady["h_inf"] + (states["h"] - steady["h_inf"]) * jnp.exp(-dt / steady["tau_h"]),
            "n": steady["n_inf"] + (states["n"] - steady["n_inf"]) * jnp.exp(-dt / steady["tau_n"]),
            "c": steady["c_inf"] + (states["c"] - steady["c_inf"]) * jnp.exp(-dt / steady["tau_c"]),
        }
        
        return new_states
    
    @staticmethod
    def compute_current(
        states: Dict[str, float],
        v: float,
        params: Dict[str, float]
    ) -> float:
        """
        Compute total channel current.
        Note: The K(Ca) current component requires calcium concentration,
        which should be provided by a separate calcium dynamics mechanism.
        """
        # Extract parameters
        gnabar = params.get("gnabar_mammalian_spike_35", 0.1)  # S/cm2
        gkbar = params.get("gkbar_mammalian_spike_35", 0.05)   # S/cm2
        gcabar = params.get("gcabar_mammalian_spike_35", 0.00075)  # S/cm2
        gkcbar = params.get("gkcbar_mammalian_spike_35", 0.0002)   # S/cm2
        
        # Reversal potentials
        ena = params.get("ena", 61.02)    # mV
        ek = params.get("ek", -102.03)    # mV
        eca = params.get("eca", 120.0)    # mV (you may need to calculate this)
        
        # Get calcium concentration (from calcium dynamics)
        # In the full implementation, this would come from a calcium accumulation mechanism
        cai = states.get("cai", 0.0001)  # mM, default resting concentration
        
        # Compute individual currents
        ina = gnabar * states["m"]**3 * states["h"] * (v - ena)
        
        # Delayed rectifier K current
        idrk = gkbar * states["n"]**4 * (v - ek)
        
        # Calcium-activated K current
        # From MOD file: ((cai / 0.001) / (1 + (cai / 0.001)))
        cai_norm = cai / 0.001
        k_ca_activation = cai_norm / (1 + cai_norm)
        icak = gkcbar * k_ca_activation * (v - ek)
        
        # Total K current
        ik = idrk + icak
        
        # Calcium current
        ica = gcabar * states["c"]**3 * (v - eca)
        
        # Return total current (negative for Jaxley convention)
        return -(ina + ik + ica)
    
    @property
    def channel_params(self) -> Dict[str, float]:
        """Define channel parameters with default values from MOD file"""
        return {
            "gnabar_mammalian_spike_35": 0.1,      # S/cm2
            "gkbar_mammalian_spike_35": 0.05,      # S/cm2
            "gcabar_mammalian_spike_35": 0.00075,  # S/cm2
            "gkcbar_mammalian_spike_35": 0.0002,   # S/cm2
            "ena": 61.02,    # mV
            "ek": -102.03,   # mV
            "eca": 120.0,    # mV (may need to be calculated)
        }
    
    @property
    def channel_states(self) -> Dict[str, float]:
        """Define channel state variables"""
        return {
            "m": 0.0353,  # Na activation
            "h": 0.9054,  # Na inactivation
            "n": 0.0677,  # K activation  
            "c": 0.0019,  # Ca activation
        }



