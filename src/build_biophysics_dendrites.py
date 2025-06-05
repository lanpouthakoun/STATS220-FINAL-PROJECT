import jaxley as jx
from jaxley.channels import Channel
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
from mammalian_spike_35_channel import MammalianSpike35
from jaxley.channels import Leak



def build_dendrites_bp(cell, params):
    param = params.iloc[0]
    dendrites = cell.basal
    ### NEED TO FIGURE OUT NCOMP BETTER
    dendrites.set("capacitance", float(param['cap_membrane']))
    dendrites.set("axial_resistivity", float(param['axial_resistance'])) 


    leak = Leak()
    dendrites.insert(leak)
    dendrites.set("Leak_gLeak",float(param['leakage_conductance'])) 
    dendrites.set("Leak_eLeak", float(param['pass_mem_potential']))
    # First, let's see what parameters the Leak channel actually has





    spike_biophysics = MammalianSpike35()
    dendrites.insert(spike_biophysics)
    dendrites.set("ena", float(param['eNa']) )
    dendrites.set("ek", float(param['eK']))
    dendrites.set("gnabar_mammalian_spike_35",  float(param['gNa'])  )
    dendrites.set("gkbar_mammalian_spike_35", float(param['gK']))
    dendrites.set("gcabar_mammalian_spike_35", float(param['gCa']))
    dendrites.set("gkcbar_mammalian_spike_35",float(param['gKc']) )

    #Need to add combined calcium current, then done

    











    # for i in np.arange(int(param['nseg'])):
    #         self.cell.dend[i].nseg = int(np.round((self.cell.dend[i].L)/10)+1)
    #         self.cell.dend[i].cm = float(param['cap_membrane'])
    #         self.cell.dend[i].Ra = float(param['axial_resistance']) 
    #         self.cell.dend[i].insert('pas') 
    #         self.cell.dend[i].e_pas = float(param['pass_mem_potential'])
    #         self.cell.dend[i].g_pas = float(param['leakage_conductance']) 
    #         self.cell.dend[i].insert(self.spike_biophysics) 
    #         self.cell.dend[i].ena = float(param['eNa']) 
    #         self.cell.dend[i].ek = float(param['eK']) 
    #         if self.spike_biophysics == 'mammalian_spike_35':
    #             self.cell.dend[i].gnabar_mammalian_spike_35 = float(param['gNa']) 
    #             self.cell.dend[i].gkbar_mammalian_spike_35 = float(param['gK'])
    #             self.cell.dend[i].gcabar_mammalian_spike_35 = float(param['gCa'])
    #             self.cell.dend[i].gkcbar_mammalian_spike_35 = float(param['gKc'])
    #         else:
    #             self.cell.dend[i].gnabar_mammalian_spike = float(param['gNa']) 
    #             self.cell.dend[i].gkbar_mammalian_spike = float(param['gK'])
    #             self.cell.dend[i].gcabar_mammalian_spike = float(param['gCa'])
    #             self.cell.dend[i].gkcbar_mammalian_spike = float(param['gKc'])
    #         self.cell.dend[i].insert('cad') 
    #         self.cell.dend[i].depth_cad = float(param['depth_cad'])
    #         self.cell.dend[i].taur_cad = float(param['taur_cad'])
    #         self.cell.dend[i].insert('extracellular') 
    #         self.cell.dend[i].xg[0] = float(param['mem_conductance']) 
    #         self.cell.dend[i].e_extracellular = float(param['e_ext']) 
    #         self.cell.dend[i].insert('xtra')