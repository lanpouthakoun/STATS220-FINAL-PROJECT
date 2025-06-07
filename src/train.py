"""
Training Loop
"""

import jax.numpy as jnp
import numpy as np
import jaxley as jx
from jaxley.channels import HH
from jaxley.synapses import IonotropicSynapse
import matplotlib.pyplot as plt
from helperFuncs import generate_electrode_map

class JaxleyTrainer():
    def __init__(self, net, data_loc, dt, tmax,ncomp):
        # load through data
        self.network = net
        self.dataset = data_loc
        self.extracellular_current = self.stim_setup
        self.dt = dt
        self.tmax = tmax
        self.v_init = -64
        self.i_delay = 1.0  # ms
        self.i_dur = 0.15  # ms
        self.i_amp = 3.9  # nA
        self.electrode = generate_electrode_map(519)
        self.ncomp = ncomp

    def stim_vars(self):
        Stim = {}

        # temporal features of the sitmulus 
        Stim['pulseShape'] = 'triphasic'
        Stim['dt'] = .005 # units: [ms]
        Stim['delay'] = 10 # units: [ms]
        Stim['dur'] = 0.05 # units: [ms]
        Stim['stop'] = 80 # units: [ms]
        Stim['amp'] = 2.5 # units: [uA]

        # calibration phase of the stimulus 
        Stim['initDur'] = 50 # units: [ms]
        Stim['initDt'] = 1 # units: [ms]
        Stim['vInit'] = -70 # units: [mV] #TODO: Old is -70

        # spatial features of the stimulus 
        # --> specifically defining the electrode geometry 
        Stim['electrodes'] = [[-10,0,40]] # units: [um] 
        Stim['electrode_type'] = 'disk'
        Stim['rhoExt'] = 1000 # Ohm*cm
        Stim['elecDiam'] = 15 # um
        Stim['polarity'] = -1 # anodic == 1; cathodic == -1 

        Stim['pulseRatioTri'] = [-2/3,1,-1/3]
        Stim['pulseRatioBi'] = [1,-1]
        Stim['frequency'] = 10 # units: [Hz]

        return Stim

    def stim_setup(self):
        """Create a pulse train based on stimulus parameters"""
        stim_params = self.stim_vars
        
        time_vec = np.arange(0, self.tmax + self.dt, self.dt)
        stim = np.zeros_like(time_vec)
        
        # Create pulses at specified frequency
        period = 1000.0 / self.frequency  # ms
        pulse_times = np.arange(self.delay, self.delay + self.dur, period)
        
        for pulse_start in pulse_times:
            pulse_end = pulse_start + self.dur
            mask = (time_vec >= pulse_start) & (time_vec < pulse_end)
            
            if stim_params['pulseShape'] == 'triphasic':
                # Implement triphasic pulse
                ratios = stim_params['pulseRatioTri']
                pulse_dur_third = self.dur / 3
                for i, ratio in enumerate(ratios):
                    phase_start = pulse_start + i * pulse_dur_third
                    phase_end = phase_start + pulse_dur_third
                    phase_mask = (time_vec >= phase_start) & (time_vec < phase_end)
                    stim[phase_mask] = self.amp * ratio
            elif stim_params['pulseShape'] == 'biphasic':
                # Implement biphasic pulse
                ratios = stim_params['pulseRatioBi']
                pulse_dur_half = self.dur / 2
                for i, ratio in enumerate(ratios):
                    phase_start = pulse_start + i * pulse_dur_half
                    phase_end = phase_start + pulse_dur_half
                    phase_mask = (time_vec >= phase_start) & (time_vec < phase_end)
                    stim[phase_mask] = self.amp * ratio
                    
        return stim
    


    def simulate(self):
        """
        
        """
          # mV (matching h.v_init = -64)
        self.network.set("v", self.v_init)  # Set initial voltage

        # Create step current for intracellular stimulation
        current_clamp = jx.step_current(
            i_delay=self.i_delay,
            i_dur=self.i_dur,
            i_amp=self.i_amp,
            delta_t=self.dt,
            t_max=self.tmax
        )

        self.network.original.stimulate(current_clamp)
        self.network.record("v")
        voltages = jx.integrate(self.network, delta_t=self.dt, t_max=self.tmax)
        return voltages
    
    def compute_eap(self, outputs):
        '''
        Input: Output of jx.integrate (of shape 3*T here). Last row is the HH_current.
        Output: (N_points, T) array of EAPs across time at each probe site.
        '''
        surf_areas_CM2, comp_positions = self.get_morphology_data()
        true_distances_CM = self.calculate_electrode_distances(comp_positions)

        outputs = outputs.reshape(4, self.ncomp, -1)
        rho = 1000 #(cf.ej chilchinisky)
        resistivity = rho/(4.*jnp.pi)
        surf_currents = outputs[1:]
        currents = surf_currents * surf_areas_CM2[None,:,None]
        extr_voltage = jnp.sum(currents.sum(axis=0) / true_distances_CM[:,:,None], axis=1) * resistivity

        return extr_voltage



#### NEED TO FIX
    def get_morphology_data(self):
        """Extract compartment surface areas and positions"""
        surf_areas_CM2 = np.zeros(self.ncomp)
        comp_positions = np.zeros((self.ncomp, 3))
        
        comp_idx = 0
        for cell_idx in range(len(self.network.cells)):
            cell = self.network.cell(cell_idx)
            for branch in cell.branches:
                length = branch.length  # μm
                radius = branch.radius  # μm
                
                # Calculate surface area per compartment
                comp_length = length / n_comp
                comp_area = 2 * np.pi * radius * comp_length  # μm²
                comp_area_cm2 = comp_area * 1e-8  # Convert to cm²
                
                for i in range(n_comp):
                    surf_areas_CM2[comp_idx] = comp_area_cm2
                    # You'll need to implement proper position calculation
                    # based on your cell morphology
                    comp_positions[comp_idx] = [0, 0, comp_idx * 10]  # Placeholder
                    comp_idx += 1
        
        return surf_areas_CM2, comp_positions

    def calculate_electrode_distances(self, comp_positions):
        """Calculate distances between electrodes and compartments"""
        n_electrodes = self.electrode.shape[0]
        n_compartments = comp_positions.shape[0]
        
        # Calculate all pairwise distances
        distances_um = np.sqrt(
            ((self.electrode[:, None, :] - comp_positions[None, :, :]) ** 2).sum(axis=2)
        )
        
        # Convert to cm and avoid division by zero
        true_distances_CM = np.maximum(distances_um * 1e-4, 1e-4)
        
        return true_distances_CM
        # def loss(self, params, inputs, labels):
    #     pass

    # def produce_ei(self, outputs):
    #     pass

    def train(self,
              params
    ):
        """
        We expect params to be a dictionary in the following structure:

        Key = Group to Edit
        Value = Parameter to adjust
        """

        # Step 1: Pick which parameters must be propagrated
        # for key in params:
        #     print(key)


        # Step 2: Set 


        return self.network
    



