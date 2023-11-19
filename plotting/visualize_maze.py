import numpy as np
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

from rlpd.data import Dataset
from visualize import *

env_name = 'antmaze-medium-diverse-v2'
viz_env, _ = get_env_and_dataset(env_name)

seed=0
buffer_paths = {
    'Online': f'../exp_data/online-s{seed}/buffers/buffer.npz',
    'Online + RND': f'../exp_data/online_rnd-s{seed}/buffers/buffer.npz',
    'Naive': f'../exp_data/naive-s{seed}/buffers/buffer.npz',
    'Ours': f'../exp_data/ours-s{seed}/buffers/buffer.npz',
}

buffers = {}
for name, path in buffer_paths.items():
    with open(path, 'rb') as f:
        buffers[name] = np.load(f)["observations"]
print('min_buffer_length:', min([len(buf) for name, buf in buffers.items()]))

cutoff = 60000

# https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
def plot_online_coverage(env, buffers, colorbar_scale=1000):
    n_pts = cutoff // 200

    fig = plt.figure(tight_layout=True)
    axs = ImageGrid(
        fig, 111, nrows_ncols=(1, len(buffers)), 
        cbar_location='right', cbar_mode='single', cbar_size='5%', cbar_pad=0.05)
    
    canvas = FigureCanvas(fig)
    
    for i, (name, buffer) in enumerate(buffers.items()):
        axs[i].set_title(name)
        axs[i].axis('off')
        axs[i].set_box_aspect(1)
        env.draw(axs[i])

        ## add buffer pts
        obs = buffer[:cutoff]
        idxs = np.arange(len(obs))
        idxs = np.sort(np.random.choice(idxs, size=n_pts, replace=False))
        x, y = obs[idxs, 0], obs[idxs, 1]
        scatter = axs[i].scatter(x, y, c=idxs // colorbar_scale, **dict(alpha=0.75, s=5, cmap='viridis', marker='o'))

    axs[-1].cax.colorbar(scatter, label="Env Steps $\\left(\\times 10^3\\right)$", 
                         ticks=range(0, cutoff // colorbar_scale, 15))
    axs[-1].cax.toggle_label(True)

    image = get_canvas_image(canvas)
    plt.savefig(f'../plotting/figures/antmaze-exploration-{cutoff}.pdf', bbox_inches="tight")
    # plt.savefig(f'../plotting/figures/antmaze-exploration-{cutoff}.png', bbox_inches="tight")
    plt.close(fig)
    return image

img = plot_online_coverage(viz_env, buffers)
