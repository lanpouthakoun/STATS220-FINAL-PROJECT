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
import jaxley.optimize.transforms as jt
import functions_helper as fh
from constants import LITKE_519_ARRAY_MAP, LITKE_519_ARRAY_GRID
from jax import random
import tensorflow_probability.substrates.jax as tfp
import optax
from jax import jit, vmap, value_and_grad
import jax
import jax.lax as lax



class JaxleyTrainer:
    def __init__(self, cell, data, epochs, dt = 0.05, tmax = 5 ):
        # load through data
        self.cell = cell
        self.data_point = data
        
        self.v_init = -64
        self.i_delay = 1.0  # ms
        self.i_dur = 0.15  # ms
        self.i_amp = 3.9  # nA
        self.electrode = generate_electrode_map(519)
        self.ncomp = self.cell.shape[1]
        self.params = self.setup_params()
        transforms = [{k: self.create_transforms(k) for k in param} for param in self.params]

        self.tf = jt.ParamTransform(transforms)
        self.lengths = self.cell.nodes["length"] ### fix
        self.lengths.to_numpy()
        self.lengths = jnp.array(self.lengths)
        self.N_ELECTRODES = LITKE_519_ARRAY_GRID.shape[0]
        self.grid = jnp.array(LITKE_519_ARRAY_GRID)
        self.delta_t = 0.05 # ms
        self.t_max = 5
        

        self.opt_params = self.tf.inverse(self.params)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),            
            optax.adam(learning_rate=1e-5)   
        )
        self.opt_state = self.optimizer.init(self.opt_params)
        self.num_epochs = epochs
        self.jitted_grad = jit(value_and_grad(self.loss, argnums=0))

    def compute_comp_xyz(self, transformed_params):
        '''
        Returns (Ncomps, 3) array of xyz positions of all the compartments
        '''
        
        # 
        # Extract x, y, z from all compartments
        all_x = []
        all_y = []
        all_z = []
    
        for param_dict in transformed_params:
            if 'x' in param_dict:
                all_x.append(param_dict['x'])
            if 'y' in param_dict:
                all_y.append(param_dict['y'])
            if 'z' in param_dict:
                all_z.append(param_dict['z'])
        print(all_z)
        # Concatenate all compartment coordinates
        xs = jnp.concatenate(all_x) 
        ys = jnp.concatenate(all_y) 
        zs = jnp.concatenate(all_z) 
        
        comp_xyz = jnp.stack([xs, ys, zs], axis=1)
        return comp_xyz
    def get_surface_areas(self, transformed_params):
        '''
        Returns (Ncomps,) array of the surface areas in um2 of all the compartments
        JAX-compatible version
        '''
        # Use JAX arrays directly instead of converting to numpy
        all_radii = []
        
        for param_dict in transformed_params:
            if 'radius' in param_dict:
                all_radii.append(param_dict['radius'])

        print(all_radii)        
        radii = jnp.concatenate(all_radii) 

        
        surf_areas = 2 * jnp.pi * radii * self.lengths
        return surf_areas
    
    def create_transforms(self, name):
        """ transforms"""
        if name == 'z':
            return jt.AffineTransform(1.0, -1.0) 
        elif name == 'radius':

            return jt.ChainTransform([
                jt.AffineTransform(1.0, 1e-6), 
                jt.SoftplusTransform(0) 
            ])
        elif name == 'axial_resistivity':
            jt.ChainTransform([
                jt.AffineTransform(100.0, 0), 
                jt.SoftplusTransform(0) 
            ])
        else:  # x, y
            return jt.AffineTransform(1.0, 0.0)  
        
    def setup_params(self):
        self.cell.delete_trainables()
        self.cell.comp("all").make_trainable("x")
        self.cell.comp("all").make_trainable("y")
        self.cell.comp("all").make_trainable("z")
        self.cell.comp("all").make_trainable("radius")
        # self.cell.comp("all").make_trainable("HH_eNa")
        # self.cell.comp("all").make_trainable("HH_eK")
        # self.cell.comp("all").make_trainable('axial_resistivity')
        # self.cell.comp("all").make_trainable('HH_gNa')
        # self.cell.comp("all").make_trainable('HH_gLeak')
        # self.cell.comp("all").make_trainable('gCa')
        # self.cell.comp("all").make_trainable('HH_gK')
        # self.cell.comp("all").make_trainable('gKCa')

        return self.cell.get_parameters()


    def simulate(self, params):
        """
        
        """
        
        self.cell.set('v', -64)  # mV (matching h.v_init = -64)
        
        i_delay = 1.0 
        i_dur = 0.5
        i_amp = 3.9
        current = jx.step_current(i_delay=i_delay, i_dur=i_dur, i_amp=i_amp, 
                                 delta_t=self.delta_t, t_max=self.t_max)

        self.cell.delete_stimuli()
        data_stimuli = None
        data_stimuli = self.cell.branch(0).comp(0).data_stimulate(current)

        self.cell.delete_recordings()
        self.cell.record("v")
        self.cell.record("i_HH")
        self.cell.record("i_Ca")
        self.cell.record("i_KCa")

        sim_outputs = jx.integrate(self.cell, params=params, data_stimuli=data_stimuli).reshape(4, self.ncomp, -1)
        return sim_outputs
    
    def gen_ei(self, outputs, transformed_params):
        """
        Generate extracellular potential from simulation outputs
        """
        tfd = tfp.distributions
        seed = random.PRNGKey(0)

        time_vec = np.arange(0, self.t_max+2*self.delta_t, self.delta_t)
        current_compartment_positions = self.compute_comp_xyz(transformed_params)
        compartment_surface_areas = self.get_surface_areas(transformed_params) * 10E-8  # Calculate surface areas in cm^2
        current_distance = fh.distance(LITKE_519_ARRAY_GRID, current_compartment_positions) * 10E-4 
        
        sim_v_extra = fh.compute_eap(outputs, current_distance, compartment_surface_areas) \
             + tfd.Normal(0, 0.0001).sample((self.N_ELECTRODES, len(time_vec)), seed=seed)
        sim_EI = self.with_ttl(sim_v_extra).T
        return sim_EI
    
    def with_ttl(self, raw_data):
        shape_array = jnp.array(raw_data.shape)
        has_512 = jnp.any(shape_array == 512)
        has_519 = jnp.any(shape_array == 519)
  
        critical_axis_512 = jnp.where(shape_array == 512, size=1, fill_value=-1)[0]
        critical_axis_519 = jnp.where(shape_array == 519, size=1, fill_value=-1)[0]
        
        critical_axis = jnp.where(has_512, critical_axis_512, critical_axis_519)

        append_data_shape = list(raw_data.shape)
        
        # Find which axis has 512 or 519
        for i, dim_size in enumerate(raw_data.shape):
            if dim_size == 512 or dim_size == 519:
                critical_axis = i
                break
        
        append_data_shape[critical_axis] = 1
        append_data = jnp.zeros(append_data_shape)
        
        return jnp.concatenate((append_data, raw_data), axis=critical_axis)

    def loss(self, opt_params, true_ei):
        """
        Computes a loss that is robust to temporal shifts by using a
        differentiable soft-alignment mechanism.
        """
        # 1. Simulate to get the predicted electrical image (EI)
        transformed_params = self.tf.forward(opt_params)
        outputs = self.simulate(transformed_params)
        predicted_ei = self.gen_ei(outputs, transformed_params)

        # Make shapes and sizes dynamic and robust
        true_len = true_ei.shape[0]
        pred_len = predicted_ei.shape[0]
        n_electrodes = true_ei.shape[1]
        max_offset = pred_len - true_len

        # Define a function to compute loss at a single offset
        def compute_loss_at_offset(offset):
            # Slice the predicted EI to match the true EI's length
            pred_window = lax.dynamic_slice(
                predicted_ei,
                start_indices=[offset, 0],
                slice_sizes=[true_len, n_electrodes]
            )

            # --- Stability Fix ---
            # Normalize both signals to focus on shape, not amplitude.
            # Use a larger epsilon for stability, especially if a window is flat (std=0).
            epsilon = 1e-6
            pred_norm = (pred_window - jnp.mean(pred_window)) / (jnp.std(pred_window) + epsilon)
            true_norm = (true_ei - jnp.mean(true_ei)) / (jnp.std(true_ei) + epsilon)

            # The loss for this specific alignment is the Mean Squared Error
            mse = jnp.mean((pred_norm - true_norm)**2)
            return mse

        # 2. Calculate the loss for ALL possible offsets
        offsets = jnp.arange(max_offset + 1)
        all_mse_losses = vmap(compute_loss_at_offset)(offsets)


        temperature = 10.0
        weights = jax.nn.softmax(-all_mse_losses * temperature)

        final_loss = jnp.sum(weights * all_mse_losses)


        return final_loss

    
    def train(self):
        epoch_losses = []
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            
            loss_val, gradient = self.jitted_grad(self.opt_params, self.data_point)
            flat_grads, _ = jax.tree_util.tree_flatten(gradient)
            if flat_grads: # Ensure list is not empty
                grad_norm = jnp.linalg.norm(jnp.concatenate([g.flatten() for g in flat_grads]))
                print(f"Epoch {epoch}, Loss {loss_val}, Grad Norm: {grad_norm}")
            else:
                print(f"Epoch {epoch}, Loss {loss_val}, No gradients found.")
            updates, self.opt_state = self.optimizer.update(gradient, self.opt_state)
            self.opt_params = optax.apply_updates(self.opt_params, updates)
            epoch_loss += loss_val
            self.cell.compute_compartment_centers()
        
            print(f"epoch {epoch}, loss {epoch_loss}")
            epoch_losses.append(epoch_loss)
        sim_outputs = self.simulate(self.opt_params)
        sim_ei = self.gen_ei(sim_outputs, self.opt_params)
        return fh.animate_519_array(sim_ei, title="Example Electrical Image", cell=self.cell), fh.plot_default_simulation_output(sim_outputs, current, time_vec)
    
    
        
    
