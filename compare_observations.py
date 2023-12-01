#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:34:44 2023.

@author: Judith Angel
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation
from ProblemRelatedFiles.read_left_bc import data, leftbc

T_N = 100
H = 0.3
pos = [1.5, 3.5, 5.5, 7.5]
t_array = np.arange(0, T_N+0.01, 0.01)
allObsBathy = []
allObs = []
path = "ProblemRelatedFiles/WaterchannelData/Comparison"

for i in range(20):

    postfix = f"try={i+1}"
    fileBathy = 'ProblemRelatedFiles/WaterchannelData/MitBathymetrie/' \
        + 'Tiefe=0,3_A=40_F=0,35_' + postfix + ".txt"
    dataObjectBathy = data(fileBathy)
    bcBathy = leftbc(fileBathy).f
    H_obsBathy = [H+bcBathy(t_array)]
    file = 'ProblemRelatedFiles/WaterchannelData/OhneBathymetrie/' \
        + 'Tiefe=0,3_A=40_F=0,35_' + postfix + ".txt"
    dataObject = data(file)
    bc = leftbc(file).f
    H_obs = [H+bc(t_array)]

    for p in range(len(pos)-1):

        H_obsBathy.append(H + dataObjectBathy.f[p](t_array))
        H_obs.append(H + dataObject.f[p](t_array))

    allObsBathy.append(H_obsBathy)
    allObs.append(H_obs)

allObsBathyArray = np.array(allObsBathy)
allObsArray = np.array(allObs)
meanBathy = np.mean(allObsBathyArray, axis=0)
stdBathy = np.std(allObsBathyArray, axis=0)
mean = np.mean(allObsArray, axis=0)
std = np.std(allObsArray, axis=0)

# meanmax = np.max(meanBathy-mean)
# start = np.argmin(abs(t_array-32))
# fig = plt.figure()


# def update_plot(i):

#     plt.clf()
#     plt.plot(pos, meanBathy[:, i]-mean[:, i], "ko")
#     plt.ylim([0, meanmax+0.001])
#     plt.xlabel('x [m]')
#     plt.ylabel('H(b)-H(0) [m]')
#     plt.title(f"Time t={round(t_array[i], 1)}")


# anim = FuncAnimation(fig, update_plot, frames=np.arange(
#     start, t_array[start:].size, max(1, (t_array[start:].size-1)//50)))
# anim.save(path + "/H_bathy_nobathy.gif", dpi=150, fps=1)

# for i in range(len(pos)):

#     plt.figure()
#     plt.plot(t_array[start:], meanBathy[i, start:], "k", label="with bathy")
#     plt.plot(t_array[start:], mean[i, start:], "b--", label="no bathy")
#     plt.legend()
#     plt.title(f"Sensor {i+1} at {pos[i]}m")
#     plt.savefig(path + f"/Sensor{i+1}.pdf")

#     plt.figure()
#     plt.plot(t_array[start:], meanBathy[i, start:]-mean[i, start:], "k")
#     plt.title(f"Difference surface elevation with and without b, Sensor {i+1}")
#     plt.savefig(path + f"/DiffSensor{i+1}.pdf")

# fig, axs = plt.subplots(len(pos))
# fig.tight_layout(pad=2.5)

# for i in range(len(pos)):

#     axs[i].plot(t_array[start:], meanBathy[i, start:], "k")
#     axs[i].plot(t_array[start:], mean[i, start:], "b--")
#     axs[i].set_xlabel('Time [s]')
#     axs[i].set_ylabel('H [m]')
#     axs[i].set_title(f"Sensor {i+1} at {pos[i]}m")

# plt.savefig(path + "/H_bathy_nobathy.pdf", bbox_inches='tight')
# plt.show()
# fig, axs = plt.subplots(len(pos))
# fig.tight_layout(pad=2.5)

# for i in range(len(pos)):

#     axs[i].plot(t_array[start:], meanBathy[i, start:]-mean[i, start:])
#     axs[i].set_xlabel('Time [s]')
#     axs[i].set_ylabel('diff [m]')
#     axs[i].set_title(f"Sensor {i+1} at {pos[i]}m")

# plt.savefig(path + "/diff.pdf", bbox_inches='tight')
# plt.show()

lines = np.swapaxes(meanBathy, 0, 1)

with open("ProblemRelatedFiles/WaterchannelData/MitBathymetrie/meanBathy.txt",
          "w") as f:

    for line in lines:

        f.write(np.array2string(line).replace('[', '').replace(']', ''))
        f.write("\n")

lines = np.swapaxes(mean, 0, 1)

with open("ProblemRelatedFiles/WaterchannelData/OhneBathymetrie/mean.txt",
          "w") as f:

    for line in lines:

        f.write(np.array2string(line).replace('[', '').replace(']', ''))
        f.write("\n")

# with h5py.File(path + "/meanBathy.hdf5", "w") as f:

#     f.create_dataset("mean", data=meanBathy)
#     f.create_dataset("st_dev", data=stdBathy)
#     f.create_dataset("t_array", data=t_array)

# with h5py.File(path + "/meanNoBathy.hdf5", "w") as f:

#     f.create_dataset("mean", data=mean)
#     f.create_dataset("st_dev", data=std)
#     f.create_dataset("t_array", data=t_array)
