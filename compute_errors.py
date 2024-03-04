#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:08:17 2024.

@author: Judith Angel
Compute discrete L2 error, error in maximum norm, mean squared error, root mean
 squared error and normalised root mean squared error and write it into a text
 file.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from subprocess import call
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import importlib

folder = "2024_02_19_09_21_AM_obs_everywhere"
path = "ProblemRelatedFiles/" + folder

params = importlib.import_module("ProblemRelatedFiles." + folder + ".params")
pa = params.params()
if pa.data != "sim_everywhere":
    pos = pa.pos
save = True

with h5py.File(path+"/opt_data.hdf5", "r") as sol:

    b_exact = np.array(sol["b_exact"][:])
    bs = np.array(sol["bs"][:])
    x = np.array(sol["x"][:])
    j = sol.attrs.get("jmax")

ximax = np.argmin(abs(x-10))  # Only consider bathymetry until length of 10m.
b = bs[j]
bmin = np.min(b_exact[:ximax])
bmax = np.max(b_exact[:ximax])
dx = x[1]-x[0]
l2_err = np.sqrt(dx)*np.linalg.norm(b[:ximax]-b_exact[:ximax])
rel_l2_err = l2_err/(np.sqrt(dx)*np.linalg.norm(b_exact[:ximax]))
max_err = np.amax(abs(b[:ximax]-b_exact[:ximax]))
rel_max_err = max_err/np.amax(abs(b_exact[:ximax]))
MSE = np.sum((b[:ximax]-b_exact[:ximax])**2)/b_exact[:ximax].size
RMSE = np.sqrt(MSE)
NRMSE = RMSE/(bmax-bmin)*100
print(f"rel_l2_err = {rel_l2_err}")
print(f"rel_max_err = {rel_max_err}")
print(f"MSE = {MSE}")
print(f"RMSE = {RMSE}")
print(f"NRMSE = {NRMSE}%")

if save:
    with open(path + "/errors.txt", "w") as f2:
        f2.write(f"Relative L2 error = {rel_l2_err}"
                 + f"\nRelative error in max.-norm = {rel_max_err}"
                 + f"\nMSE = {MSE}"
                 + f"\nRMSE = {RMSE}"
                 + f"\nNRMSE = {NRMSE}%")
