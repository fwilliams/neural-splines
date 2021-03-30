import torch
import numpy as np

from mayavi import mlab





vgrid = np.zeros([128, 126, 99])
for coord, vmin, vmax in voxel_chunks(torch.tensor([128, 126, 99]), 6):
    clr = -1.0
    if sum(coord) % 2 == 0:
        clr = 1.0
    sum_vgrid = vgrid[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]].sum()
    assert sum_vgrid == 0.0, f"vgrid not empty {sum_vgrid}"
    vgrid[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]] += clr

mlab.volume_slice(vgrid)
mlab.colorbar()
mlab.show()


