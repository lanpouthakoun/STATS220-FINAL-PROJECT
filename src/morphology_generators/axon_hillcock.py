
import numpy as np, jaxley as jx, jax.numpy as jnp
from io import StringIO
import pandas as pd
import tempfile, os

class build_Axon_Hillcock():
    def __init__(self, axon_coords, length, diameter):
        self.xpts_axon = axon_coords[:,0]*(1000)
        self.ypts_axon = axon_coords[:,1]*(1000)
        self.zpts_axon = axon_coords[:,2]*(1000)
        self.length = length
        self.radius= float(diameter)/ 2.0
    def create_branch(self):
        coords  = np.stack([self.xpts_axon, self.ypts_axon, self.zpts_axon], axis=1)
        coords = coords[: self.length]
        swc_df = pd.DataFrame({
            "n"     : np.arange(len(coords)),
            "type"  : 2,                              # 2 = axon
            "x"     : coords[:,0],
            "y"     : coords[:,1],
            "z"     : coords[:,2],
            "r"     : self.radius,
            "parent": np.concatenate(([-1], np.arange(self.length - 1))),
        })
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".swc", delete=False) as f:
            swc_df.to_csv(f, sep=" ", header=False, index=False)
            tmp_name = f.name               # remember the path

        hillock = jx.read_swc(tmp_name, ncomp=20)
        os.remove(tmp_name)   
        return hillock