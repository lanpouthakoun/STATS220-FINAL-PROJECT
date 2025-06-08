import numpy as np
from jax import vmap
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from tempfile import NamedTemporaryFile
import imageio
import base64
from constants import LITKE_519_ARRAY_MAP

def compute_comp_xyz(cell):
    '''
    Returns (Ncomps, 3) array of xyz positions of all the compartments
    '''
    
    cell.compute_compartment_centers()
    xs = cell.nodes["x"].to_numpy()
    ys = cell.nodes["y"].to_numpy()
    zs = cell.nodes["z"].to_numpy()

    comp_xyz = jnp.stack([xs,ys,zs],axis=1)

    return comp_xyz

def distance(grid, cell_positions):

    '''
    grid: (Npoints, 3) array
    cell_positions: (Ncomps, 3) array
    Returns (Npoints, Ncomps) array of pairwise distances in um
    '''

    def _distance(grid_xyz, cell_xyz):

        return jnp.linalg.norm(grid_xyz - cell_xyz)

    return vmap(vmap(_distance, in_axes=(None, 0)),in_axes=(0, None))(grid, cell_positions)

def get_surface_areas(cell):

    '''
    Returns (Ncomps,) array of the surface areas in um2 of all the compartments
    '''
    radii = cell.nodes["radius"].to_numpy()
    lengths = cell.nodes["length"].to_numpy()
    surf_areas = 2*jnp.pi*radii*lengths
    return surf_areas

def compute_eap(sim_outputs, current_distances, compartment_surface_areas, rho=1000):
    '''
    Input: Output of jx.integrate (of shape 3*T here). Last row is the HH_current.
    Output: (N_points, T) array of EAPs across time at each probe site.
    '''
    resistivity = rho/(4*jnp.pi)
    surface_currents = sim_outputs[1:]
    currents = surface_currents * compartment_surface_areas[None,:,None]
    v_extra = jnp.sum(currents.sum(axis=0) / current_distances[:,:,None], axis=1) * resistivity
    return v_extra

def __plot_static_ei_helper(ei:np.ndarray, LITKE_ARRAY_MAP:np.ndarray, title:str, ax=None, special_elecs:np.ndarray=None, basecolor='r', special_colors=None, cell=None):
    """
    Takes in a 2D EI which is (time_sample, channels)
    Plots the electrode array as a scatter plot where the size of the dot is proportional to the absolute value of the most
    negative value of the channel over the time samples.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.set_title(title)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.axes.set_aspect('equal')
    ax.axis('off')
    
    channel_vals = np.min(ei, axis=0)
    channel_vals = np.abs(channel_vals)
    assert channel_vals.size == LITKE_ARRAY_MAP.shape[0], f'channel_vals must have {LITKE_ARRAY_MAP.shape[0]} elements, but has {channel_vals.size} elements.'
    #plot all channels as relatively small black dots first
   
    ax.scatter(LITKE_ARRAY_MAP[:, 0], LITKE_ARRAY_MAP[:, 1], s=0.5, c='k')
    
    if special_elecs is None:
        ax.scatter(LITKE_ARRAY_MAP[:, 0], LITKE_ARRAY_MAP[:, 1], s=channel_vals * 5, c=basecolor) 
    else:
        for elec in range(LITKE_ARRAY_MAP.shape[0]):
            if elec not in special_elecs:
                ax.scatter(LITKE_ARRAY_MAP[elec, 0], LITKE_ARRAY_MAP[elec, 1], s=channel_vals[elec] * 5, c=basecolor)
        for elec, color in zip(special_elecs, special_colors):
            ax.scatter(LITKE_ARRAY_MAP[elec, 0], LITKE_ARRAY_MAP[elec, 1], s=channel_vals[elec] * 5, c=color)
            
    if cell is not None:
        cell.vis(ax=ax)
        ax.set_aspect('equal')
        ax.set_xlim([-400,400])
        ax.set_ylim([-450,450])
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
            
def plot_static_ei_519(ei:np.ndarray, title:str, ax=None, special_color_elecs=None, basecolor='r', special_colors=None, cell=None):
    """
    Takes in a 2D EI which is (time_sample, channels)
    Plots the electrode array as a scatter plot where the size of the dot is proportional to the absolute value of the most
    negative value of the channel over the time samples.
    """
    assert ei.ndim == 2, f'ei must have 2 dimensions, but has {ei.ndim} dimensions.'
    assert ei.shape[1] == 520, f'ei must have 520 columns, but has {ei.shape[1]} columns.'
    adjusted_elecs = special_color_elecs
    if special_color_elecs is not None:
        adjusted_elecs = np.array(special_color_elecs) - 1
    __plot_static_ei_helper(ei[:, 1:], LITKE_519_ARRAY_MAP, title, ax=ax, special_elecs=adjusted_elecs, basecolor=basecolor, special_colors=special_colors, cell=cell)
    
def __animate_helper(raw_data: np.ndarray, LITKE_ARRAY_MAP:np.ndarray, num_elecs:int, scalar: int = 10, auto_mean_adjust: bool = False, stim_electrodes=[], save_path=None, fps=10, noise_thresh=0.0, title="", presentable=False, cell=None) -> FuncAnimation:
    """
    Creates an animation to visualize the activity of an electrode map.

    Args:
        raw_data (np.ndarray): Array of shape (num_frames, num_electrodes) containing the activity values for each electrode over time.
        scalar (int, optional): Scaling factor for the size of scatter points representing electrode activity. Default is 10.
        auto_mean_adjust (bool, optional): If True, consider the first sample of raw_data as a zero baseline to subtract 
                                           before animating each sample. Default is False.

    Returns:
        matplotlib.animation.FuncAnimation: Animation object representing the electrode activity over time.
    """
    
    coords = LITKE_ARRAY_MAP
    fig = plt.figure(figsize=(12, 6) if not presentable else (6, 6))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title(title)
    n = ax.scatter(coords[:, 0], coords[:, 1], c='k', s=0)
    if not presentable:
        for num in range(num_elecs):
            ax.text(coords[num, 0], coords[num, 1], f'{num+1}', fontsize=5)
    ax.axis('off')
    
    if not presentable:
        label = ax.text(1000, 500, '0', fontsize=10)

    for electrode in stim_electrodes:
        circle = plt.Circle((coords[electrode, 0], coords[electrode, 1]), 20, color='yellow', fill=True)
        ax.add_patch(circle)
        
    if auto_mean_adjust:
        baseline = raw_data[0, :]
    else:
        baseline = 0
        
    if cell is not None:
        cell.vis(ax=ax)
        ax.set_aspect('equal')
        ax.set_xlim([-400,400])
        ax.set_ylim([-450,450])
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')

    def animate(i):
        new_sizes = np.abs((raw_data[i, :] - baseline)) * scalar
        new_sizes[np.where(new_sizes < noise_thresh * scalar)] = 0
        n.set_sizes(new_sizes)
        n.set_color(np.where(raw_data[i, :] > baseline, 'blue', 'red'))
        if not presentable:
            label.set_text(f'{i+1}/{raw_data.shape[0]}')
        #if cell is not None:
        #    cell.vis(ax=ax)

    if save_path is None:
        anim = FuncAnimation(fig, animate, frames=raw_data.shape[0], interval=100)
        plt.close(fig)
        return anim
    
    else:
        frames = []
        for frame in tqdm(range(raw_data.shape[0]), desc='Generating animation frame'):
            animate(frame)
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(image)
        imageio.mimsave(save_path, frames, duration=int(1000/fps), format='GIF', loop=0)
        plt.close(fig)

def animate_519_array(raw_data: np.ndarray, scalar: int = 10, auto_mean_adjust: bool = False, stim_electrodes = [], save_path=None, fps=10, noise_thresh=0.0, title="", presentable=False, cell=None) -> FuncAnimation:
    """
    Creates an animation to visualize the activity of the 519 electrode array.
    """
    assert raw_data.ndim ==2, f'data must have 2 dimensions, time and electrodes, but has {raw_data.ndim} dimensions.'
    assert raw_data.shape[1] == 520, f'raw_data must have 520 columns, but has {raw_data.shape[1]} columns.'
    return __animate_helper(raw_data[:, 1:], LITKE_519_ARRAY_MAP, 519, scalar=scalar, auto_mean_adjust=auto_mean_adjust, stim_electrodes=[e-1 for e in stim_electrodes], save_path=save_path, fps=fps, noise_thresh=noise_thresh, title=title, presentable=presentable, cell=cell)

_VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def _anim_to_html(anim: animation.Animation,
                  fps: int = 20) -> str:
    """
    Converts a matplotlib.animation.Animation object to an HTML video tag string.

    Saves the animation as a temporary MP4 file, encodes the video data in
    base64, and embeds it into an HTML <video> tag. Caches the base64 encoded
    video as an attribute `_encoded_video` on the `anim` object to avoid
    re-processing on subsequent calls.

    Args:
        anim: The matplotlib.animation.Animation object to convert.
        fps: Frames per second for the output MP4 video.

    Returns:
        An HTML string containing the <video> tag with the embedded animation.
    """
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video)

    return _VIDEO_TAG.format(anim._encoded_video.decode('ascii'))

def with_ttl(raw_data:np.ndarray):
  """
  Asserts that exactly one dimension of the provided raw data has a size of 512 or 519.
  It then returns a tensor with that dimension expanded by +1, and the 0th index filled in with all 0s
  """
  is_512 = 512 in raw_data.shape
  if is_512:
      assert 519 not in raw_data.shape, f'raw_data must have a dimension of size 512 or 519, but has shape {raw_data.shape}'
      critical_axis = np.where(np.array(raw_data.shape) == 512)[0][0]
  else:
      assert 519 in raw_data.shape, f'raw_data must have a dimension of size 512 or 519, but has shape {raw_data.shape}'
      critical_axis = np.where(np.array(raw_data.shape) == 519)[0][0]
  append_data_shape = list(raw_data.shape)
  append_data_shape[critical_axis] = 1
  append_data = np.zeros(append_data_shape)
  return np.concatenate((append_data, raw_data), axis=critical_axis)

def plot_default_simulation_output(sim_outputs, current, time_vec):

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(time_vec, sim_outputs[0, :, :].T)
    axs[0].set_ylabel('voltage (mV)')
    axs[0].set_title('Membrane Voltage by Compartment')

    axs[1].plot(time_vec, sim_outputs[1, :, :].T)
    axs[1].set_ylabel('current (nA)')
    axs[1].set_title('HH Channels Current by Compartment')

    axs[2].plot(time_vec, sim_outputs[2, :, :].T)
    axs[2].set_ylabel('current (nA)')
    axs[2].set_title('Ca Channel Current by Compartment')

    axs[3].plot(time_vec, sim_outputs[3, :, :].T)
    axs[3].set_ylabel('current (nA)')
    axs[3].set_title('KCa Channel Current by Compartment')
    axs[3].set_xlabel('time (ms)')

    # Plot the stimulus current on the right axis of each plot
    for ax in axs:
        ax2 = ax.twinx()
        ax2.plot(time_vec[:-1], current, label='Stimulus Current (nA)', color='blue', linestyle='--')
        ax2.set_ylabel('Stimulus Current (nA)')
        ax2.tick_params(axis='y', labelcolor='blue')
    plt.tight_layout()
    plt.show()

def plot_cell_array_3D(current_compartment_positions, array_grid):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter the electrode array
    ax.scatter(array_grid[:, 0], array_grid[:, 1], array_grid[:, 2], c='black', s=10, label='Electrode Array')
    # Scatter the cell compartment locations stored in true_comp_positions
    ax.scatter(current_compartment_positions[:, 0], current_compartment_positions[:, 1], current_compartment_positions[:, 2], c='red', s=10, label='Cell Compartments')
    ax.set_xlim([-400,400])
    ax.set_ylim([-450,450])
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    ax.set_title('Electrode Array and Cell Compartments in 3D')
    # rotate the view to be most visible
    ax.view_init(elev=0, azim=270)
    plt.show()