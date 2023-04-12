import os
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--guiding_points", type=str)
parser.add_argument("--datatype", type=str, default="proxd")
args = parser.parse_args()

if args.datatype == "proxd":
    guiding_points_folder = 'training/guiding_points_1'
    objs_folder = 'data/protext/objs/'
    context_folder = 'data/protext/proxd_test/context'
    scene = args.guiding_points.split('_')[0]
else:
    guiding_points_folder = 'training/guiding_points_2'
    objs_folder = 'data/humanise/objs/'
    context_folder = 'data/humanise/valid/context'
    scene = args.guiding_points[:11] + '0'

path = os.path.join(guiding_points_folder, args.guiding_points + '.npy')
context_path = os.path.join(context_folder, args.guiding_points + '.txt')

with open(context_path, 'r') as f:
    obj = f.readlines()[-1]

fig = plt.figure(figsize = (3, 3))
ax = plt.axes(projection ="3d")

with open(os.path.join(objs_folder, scene, obj + '.npy'), 'rb') as f:
    xyz = np.load(f)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_center = x.mean()
    y_center = y.mean()
    z_center = z.mean()
    # Creating figure
    
    # Creating plot
    ax.scatter3D(x, y, z, color='white', s=70, edgecolors='blue', label='Target PCD')


with open(path, 'rb') as f:
    xyz = np.load(f)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # Creating figure
    
    # Creating plot
    ax.scatter3D(x, y, z, color='red', s=70, edgecolors='red', label='Guiding points')
# plt.axis('off')
plt.xticks([], [])
plt.yticks([], [])
ax.set_zticks([], [])
# plt.legend(fontsize=20, markerscale=3.0)
plt.show()

