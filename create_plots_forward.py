#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:41:09 2024.

@author: Judith Angel
Create plots for forward solution at sensor positions 2, 3, 4.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from ProblemRelatedFiles.read_left_bc import leftbc, data

#####################################################################
# --------------------- Insert file name here. -------------------- #
# ------- It is the file name created from the script ------------- #
# ------- "ProblemRelatedFiles/Compute_channel_forward.py". ------- #
file = "nobathyTiefe=0,3_A=40_F=0,35_mean_kappa2e-01_T=13"
#####################################################################

path = "ProblemRelatedFiles/WaterchannelData/" + file

if "sim_data_" in file:
    prefix = 'ProblemRelatedFiles/WaterchannelData/MitBathymetrie/'
    postfix = "meanBathy"
else:
    prefix = 'ProblemRelatedFiles/WaterchannelData/OhneBathymetrie/'
    postfix = "mean"

filename = 'Tiefe=0,3_A=40_F=0,35_' + postfix

lbc = leftbc(prefix+filename+".txt")
dataObject = data(prefix+filename+".txt")

#####################################################################
# ------- Set to True if you want to save the plots. -------------- #
save = False
#####################################################################

with h5py.File(path+".hdf5", "r") as sol:

    H_sensor = np.array(sol["H_sensor"][:])
    u_sensor = np.array(sol["u_sensor"][:])
    h = np.array(sol["h"][:])
    u = np.array(sol["u"][:])
    b_exact = np.array(sol["b_array"])
    xgrid = np.array(sol["xgrid"][:])
    t_array = np.array(sol["t_array"][:])
    pos = np.array(sol["pos"][:])
    start = sol.attrs.get("start")
    T_N = sol.attrs.get("T_N")
    xmin = sol.attrs.get("xmin")
    xmax = sol.attrs.get("xmax")
    timestep = sol.attrs.get("dt")
    dx = sol.attrs.get("dx")
    g = sol.attrs.get("g")
    H = sol.attrs.get("H")
    kappa = sol.attrs.get("kappa")
    M = sol.attrs.get("M")

fig, axs = plt.subplots(3)
fig.tight_layout(pad=2.0)
axs[0].plot(t_array, H_sensor[:, 0], "y",
            label="simulation")
axs[0].plot(t_array, H+dataObject.f[0](t_array+start), "k--",
            label="measurement")
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('H [m]')
axs[0].set_title(f"Sensor 2 at {pos[0]}m")

axs[1].plot(t_array, H_sensor[:, 1], "y",
            label="simulation")
axs[1].plot(t_array, H+dataObject.f[1](t_array+start), "k--",
            label="measurement")
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('H [m]')
axs[1].set_title(f"Sensor 3 at {pos[1]}m")

axs[2].plot(t_array, H_sensor[:, 2], "y",
            label="simulation")
axs[2].plot(t_array, H+dataObject.f[2](t_array+start), "k--",
            label="measurement")
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('H [m]')
axs[2].legend(loc="upper left")
axs[2].set_title(f"Sensor 4 at {pos[2]}m")
plt.tight_layout()
plt.savefig(f"{path}"+".pdf")
plt.show()
