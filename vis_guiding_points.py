import os
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize = (30, 30))
ax = plt.axes(projection ="3d")
with open('/home/anvd2aic/Desktop/scene-synthesis/training/guiding_points/MPH112_00151_03.npy', 'rb') as f:
    xyz = np.load(f)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # Creating figure
    
    # Creating plot
    ax.scatter3D(x, y, z)

with open('/home/anvd2aic/Desktop/scene-synthesis/data/protext/objs/MPH112/chest_of_drawers_0.npy', 'rb') as f:
    xyz = np.load(f)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # Creating figure
    
    # Creating plot
    ax.scatter3D(x, y, z)

plt.axis('off')
plt.legend(fontsize=20, markerscale=3.0)
plt.show()

