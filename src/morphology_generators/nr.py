import numpy as np, jaxley as jx, jax.numpy as jnp
from io import StringIO
import pandas as pd
import tempfile, os

class NR_Build():
    def __init__(self, axon_coords, length, diameter, index):
        self.xpts_axon = axon_coords[:,0]*(1000)
        self.ypts_axon = axon_coords[:,1]*(1000)
        self.zpts_axon = axon_coords[:,2]*(1000)
        self.length = length
        self.radii = float(diameter)/ 2.0
        self.index = index
    def create_branch(self):
        coords  = np.stack([self.xpts_axon, self.ypts_axon, self.zpts_axon], axis=1)
        coords = coords[self.index: self.index + self.length + 1]
        swc_df = pd.DataFrame({
            "n"     : np.arange(len(coords)),
            "type"  : 2,                              # 2 = axon
            "x"     : coords[:,0],
            "y"     : coords[:,1],
            "z"     : coords[:,2],
            "r"     : self.radii,
            "parent": np.concatenate(([-1], np.arange(self.length))),
        })
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".swc", delete=False) as f:
            swc_df.to_csv(f, sep=" ", header=False, index=False)
            tmp_name = f.name               # remember the path

        nr = jx.read_swc(tmp_name, ncomp=20)
        os.remove(tmp_name)   
        return nr