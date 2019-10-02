import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def rodrigues(axis_angle, v):
    theta = np.linalg.norm(axis_angle)
    rot_vec = axis_angle/theta
    return v*np.cos(theta) + np.cross(rot_vec, v)*np.sin(theta) + rot_vec*np.dot(rot_vec, v)*(1-np.cos(theta))

data = pd.read_csv('matched_pos_data.csv')
rot = data[['rx', 'ry', 'rz']].values
rot_mag = data.rot_mag.values

tmp = np.zeros(rot.shape)
for i in range(rot.shape[0]):
    tmp[i] = rodrigues(rot[i]*rot_mag[i], np.array([0,0,1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tmp[::10,0], tmp[::10,1], tmp[::10,2])
ax.set_aspect(1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
pos = data[['x', 'y', 'z']].values
ax.scatter(pos[::10,0], pos[::10,1], pos[::10,2])

ax = fig.add_subplot(122, projection='3d')
comp = pos + 0.05*tmp
ax.scatter(comp[::10,0], comp[::10,1], comp[::10,2])
plt.show()
