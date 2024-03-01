#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:08:17 2024.

@author: Judith Angel
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from subprocess import call
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import importlib

folder = "2024_02_26_10_13_AM_sim_sensor3"
path = "ProblemRelatedFiles/" + folder

params = importlib.import_module("ProblemRelatedFiles." + folder + ".params")
pa = params.params()
if pa.data != "sim_everywhere":
    pos = pa.pos
save = True

with h5py.File(path+"/opt_data.hdf5", "r") as sol:

    b_exact = np.array(sol["b_exact"][:])
    bs = np.array(sol["bs"][:])
    j = sol.attrs.get("jmax")

b = bs[j]
bmin = np.min(b_exact)
bmax = np.max(b_exact)
RMSE = np.sqrt(np.sum((b-b_exact)**2)/b_exact.size)
NRMSE = RMSE/(bmax-bmin)*100
print(f"NRMSE = {NRMSE}%")
with open(path + "/NRMSE.txt", "w") as f2:
    f2.write(f"NRMSE = {NRMSE}%")
