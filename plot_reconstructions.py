#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:41:53 2024.
Put two reconstructions in one plot.

@author: Judith Angel
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dedalus.extras.plot_tools import quad_mesh, pad_limits
import importlib

folder1 = "2024_02_19_09_22_AM_obs_everywhere_noise"
folder2 = "2024_02_19_09_26_AM_sim_sensor_noise"
path1 = "ProblemRelatedFiles/" + folder1
path2 = "ProblemRelatedFiles/" + folder2

params1 = importlib.import_module("ProblemRelatedFiles." + folder1 + ".params")
pa1 = params1.params()
if pa1.data != "sim_everywhere":
    raise ValueError("folder1 should contain reconstruction for observation on"
                     +"whole domain")
params2 = importlib.import_module("ProblemRelatedFiles." + folder2 + ".params")
pa2 = params2.params()
if pa2.data != "sim_sensor_pos":
    raise ValueError("folder2 should contain reconstruction for observation at"
                     +"sensor positions")
pos2 = pa2.pos
save = False

with h5py.File(path1+"/opt_data.hdf5", "r") as sol:

    f_vals1 = np.array(sol["f_vals"][:])
    f_err1s1 = np.array(sol["f_err1s"][:])
    f_err2s1 = np.array(sol["f_err2s"][:])
    f_regs1 = np.array(sol["f_regs"][:])
    f_b_exct1 = np.array(sol["f_b_exct"])
    alpha_js1 = np.array(sol["alpha_js"][:])
    v_norms1 = np.array(sol["v_norms"][:])
    b_exact1 = np.array(sol["b_exact"][:])
    bs1 = np.array(sol["bs"][:])
    vs1 = np.array(sol["vs"][:])
    saveall = sol.attrs.get("saveall")
    if saveall:
        qs1 = np.array(sol["qs"][:])
        ps1 = np.array(sol["ps"][:])
    else:
        q1 = np.array(sol["q"][:])
    obs1 = np.array(sol["obs"][:])
    x1 = np.array(sol["x"][:])
    t1 = np.array(sol["t"][:])
    j1 = sol.attrs.get("jmax")

with h5py.File(path2+"/opt_data.hdf5", "r") as sol:

    f_vals2 = np.array(sol["f_vals"][:])
    f_err1s2 = np.array(sol["f_err1s"][:])
    f_err2s2 = np.array(sol["f_err2s"][:])
    f_regs2 = np.array(sol["f_regs"][:])
    f_b_exct2 = np.array(sol["f_b_exct"])
    alpha_js2 = np.array(sol["alpha_js"][:])
    v_norms2 = np.array(sol["v_norms"][:])
    b_exact2 = np.array(sol["b_exact"][:])
    bs2 = np.array(sol["bs"][:])
    vs2 = np.array(sol["vs"][:])
    saveall = sol.attrs.get("saveall")
    if saveall:
        qs2 = np.array(sol["qs"][:])
        ps2 = np.array(sol["ps"][:])
    else:
        q2 = np.array(sol["q"][:])
    obs2 = np.array(sol["obs"][:])
    x2 = np.array(sol["x"][:])
    t2 = np.array(sol["t"][:])
    j2 = sol.attrs.get("jmax")

j = max(j1, j2)

# Error to exact bathymetry.
b_errs1 = np.zeros(j1)

for i in range(j1):
    if np.linalg.norm(b_exact1) > 1e-16:
        b_errs1[i] = np.linalg.norm(bs1[i]-b_exact1)/np.linalg.norm(b_exact1)
    else:
        b_errs1[i] = np.linalg.norm(bs1[i]-b_exact1)

b_errs2 = np.zeros(j2)

for i in range(j2):
    if np.linalg.norm(b_exact2) > 1e-16:
        b_errs2[i] = np.linalg.norm(bs2[i]-b_exact2)/np.linalg.norm(b_exact2)
    else:
        b_errs2[i] = np.linalg.norm(bs2[i]-b_exact2)

# Range for plots.
bmin = min(np.min(bs1), np.min(bs2), np.min(b_exact1))
bmax = max(np.max(bs1), np.max(bs2), np.max(b_exact1))
ymin = min(np.min(obs1), np.min(obs2))
ymax = max(np.max(obs1), np.max(obs2))

plt.figure()
plt.semilogy(range(j1), b_errs1[0:j1], '--k', label="obs. everywhere")
plt.semilogy(range(j2), b_errs2[0:j2], ':k', label="obs. at sensor pos.")
plt.xlabel(r'Optimisation iteration $j$')
plt.ylabel(r"$||b_j-b_{ex}||_2 \ / \ ||b_{ex}||_2$")
plt.legend()
# plt.title("Relative error to exact bathymetry")
if save:
    plt.savefig(path1 + "/errors_rel_twice.pdf", bbox_inches='tight')

#######################
# ----- Bathymetries ----- #
fig1 = plt.figure()
plt.figure()
plt.plot(x1, b_exact1, "-k", label="exact")
plt.plot(x1, bs1[j1], "--k", label="obs. everywhere")
plt.plot(x2, bs2[j2], ":k", label="obs. at sensor pos.")
plt.plot(pos2, np.zeros(len(pos2)), "k*", label="sensor position",
         markersize=10)
plt.ylim([bmin, bmax])
plt.xlabel('x [m]')
plt.ylabel('b [m]')
plt.legend()
if save:
    plt.savefig(path1 + "/bathymetries" + ".pdf", bbox_inches='tight')

fig1 = plt.figure()
