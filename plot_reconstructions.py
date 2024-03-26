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
from subprocess import call
import importlib

#####################################################################
# ---------------- Insert folder names here. ---------------------- #
# E.g. folder1 = "2024_02_19_09_21_AM_obs_everywhere"
folder1 = "2024_02_19_09_25_AM_sim_sensor234"  # Without noise.
folder2 = "2024_02_19_09_26_AM_sim_sensor234_noise"  # With noise.
#####################################################################

path1 = "ProblemRelatedFiles/" + folder1
path2 = "ProblemRelatedFiles/" + folder2

params1 = importlib.import_module("ProblemRelatedFiles." + folder1 + ".params")
pa1 = params1.params()
if pa1.noise != 0:
    raise ValueError("folder1 should contain reconstruction for observation"
                     + " without noise")
if pa1.data != "sim_everywhere":
    pos1 = pa1.pos
params2 = importlib.import_module("ProblemRelatedFiles." + folder2 + ".params")
pa2 = params2.params()
if pa2.noise != 1:
    raise ValueError("folder2 should contain reconstruction for observation "
                     + "with noise")
if pa1.data != "sim_everywhere":
    pos2 = pa2.pos
    if pa1.pos != pa2.pos:
        raise ValueError("Sensor positions not the same")
if pa1.data != pa2.data:
    raise ValueError("Observation type is not the same")

#####################################################################
# ------- Set to True if you want to save the plots. -------------- #
save = True
#####################################################################

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
    x1 = np.array(sol["x"][:])
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

fs = 5
lw = 0.8
fh = 1.5
x_10 = np.argmin(abs(x1-10))  # Only plot until x=10m.
# ----- Values of objective functional ----- #
fig, ax = plt.subplots(figsize=[1.95, fh])  # Size for paper.
ax.semilogy(range(j1), b_errs1[0:j1], '--k', label="without noise",
            linewidth=lw)
ax.semilogy(range(j2), b_errs2[0:j2], ':k', label="with noise", linewidth=lw)
ax.set_xlabel(r'Optimisation iteration $j$', fontsize=fs, labelpad=0.25)
ax.set_ylabel(r"$||b_j-b_{ex}||_2 \ / \ ||b_{ex}||_2$", fontsize=fs,
              labelpad=0.5)
ax.tick_params(axis='both', which='major', labelsize=fs, pad=2)
ax.tick_params(axis='both', which='minor', labelsize=fs, pad=2)
plt.legend(loc="upper right", fontsize=fs)
plt.tight_layout()
if save:
    filename = path1 + "/errors_rel_twice.pdf"
    plt.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

# ----- Bathymetries ----- #
# Black and white.
fig, ax = plt.subplots(figsize=[3.79, fh])  # Size for paper.
ax.plot(x1[:x_10], b_exact1[:x_10], "-k", label="exact", linewidth=lw)
ax.plot(x1[:x_10], bs1[j1][:x_10], "--k", label="without noise", linewidth=lw)
ax.plot(x2[:x_10], bs2[j2][:x_10], ":k", label="with noise", linewidth=lw)
if pa2.data != "sim_everywhere":
    ax.plot(pos2, np.zeros(len(pos2)), "k*", label="sensor position",
            markersize=fs)
plt.ylim([bmin-0.01, bmax+0.01])
ax.set_xlabel(r'$x [m]$', fontsize=fs, labelpad=0.25)
ax.set_ylabel(r'$b [m]$', fontsize=fs, labelpad=0.5)
ax.tick_params(axis='both', which='major', labelsize=fs, pad=3)
plt.legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=2, fontsize=fs)
plt.tight_layout()
ax.spines[['right', 'top']].set_visible(False)
if save:
    filename = path1 + "/bathymetries" + ".pdf"
    plt.savefig(filename, bbox_inches='tight')
    call(["pdfcrop", filename, filename])

# Same plots in colour.
plt.figure(figsize=[6.4, 1.5])
plt.plot(x1[:x_10], b_exact1[:x_10], "-k", label="exact")
plt.plot(x1[:x_10], bs1[j1][:x_10], "--b", label="without noise")
plt.plot(x2[:x_10], bs2[j2][:x_10], ":r", label="with noise")
if pa2.data != "sim_everywhere":
    plt.plot(pos2, np.zeros(len(pos2)), "k*", label="sensor position",
             markersize=10)
plt.ylim([bmin-0.01, bmax+0.01])
plt.xlabel('x [m]')
plt.ylabel('b [m]')
plt.legend(loc="upper right", bbox_to_anchor=(1, 1.4), ncol=5)
if save:
    plt.savefig(path1 + "/bathymetries_col" + ".pdf", bbox_inches='tight')
