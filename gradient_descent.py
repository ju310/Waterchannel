#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:27:32 2023.

@author: Judith Angel

Reconstruct the bathymetry in a water channel using the gradient descent
method.
"""
import numpy as np
import h5py
import os
from datetime import datetime
from timeit import default_timer as timer
import logging
import importlib

from ProblemRelatedFiles.grad_descent_aux import OptProblem

for system in ['subsystems', 'solvers']:
    logging.getLogger(system).setLevel(logging.WARNING)  # Suppress output.

#####################################################################
# Set 'useOtherParams' to True if you want use an existing parameter file.
useOtherParams = False
folder = "sim_obs_whole_domain"  # Folder with parameter file.

# Set 'save' to True if you want to save the optimisation data in a hdf5 file.
# A folder named in the format "Year_month_day_hour_minute_AM/PM" will be
# created automatically.
save = False
# Set 'saveall' to True if you want to save all forward and adjoint solutions
# additionally.
saveall = False
#####################################################################

params = importlib.import_module("ProblemRelatedFiles."
                                 + useOtherParams*(folder + ".") + "params")
pa = params.params()  # Object containing all parameters.
P = OptProblem(pa, save)

# Create folder for plots and data.
if save is True:

    newfolder = datetime.now().strftime(
        "%Y_%m_%d_%I_%M_%p")
    os.mkdir("ProblemRelatedFiles/" + newfolder)

    # Save parameter file.
    with open("ProblemRelatedFiles/" + useOtherParams*(folder + "/")
              + "params.py", "r") as f:
        parameters = f.read()

    with open(f"ProblemRelatedFiles/{newfolder}/params.py", "w") as file:
        file.write(parameters)

b = pa.b_start
y_d = P.y_d

# --- Exact bathymetry. ---
b_exact = pa.b_exact
f_b_exct = P.f(b_exact)
print("f(b_exact) =", f_b_exct)
print("Regularisation term: ", P.f_reg)

start = timer()
j = 0
jmax = pa.jmax

# --- Initialisations ---
f_vals = np.zeros(jmax+1)
alpha_j = pa.alpha
gradientchecks = []
if pa.test is True:
    f_vals[0] = f_b_exct
else:
    f_vals[0] = P.f(b)
breaker = False
min_found = False
x_coord = P.PDE.dist.local_grid(P.PDE.xbasis)

if save:

    f = h5py.File("ProblemRelatedFiles/" + newfolder
                  + "/opt_data.hdf5", "w")

    f_err1s = f.create_dataset("f_err1s", shape=(jmax+1,), dtype=np.float64)
    f_err2s = f.create_dataset("f_err2s", shape=(jmax+1,), dtype=np.float64)
    f_regs = f.create_dataset("f_regs", shape=(jmax+1,), dtype=np.float64)
    f.create_dataset("f_b_exct", data=f_b_exct)
    alpha_js = f.create_dataset("alpha_js", shape=(jmax,), dtype=np.float64)
    v_norms = f.create_dataset("v_norms", shape=(jmax,), dtype=np.float64)
    f.create_dataset("b_exact", data=b_exact)
    bs = f.create_dataset("bs", shape=(jmax+1, b.size), dtype=np.float64)
    vs = f.create_dataset("vs", shape=(jmax, b.size), dtype=np.float64)
    if saveall:
        qs = f.create_dataset("qs", shape=(jmax, pa.t_array.size, b.size, 2),
                              dtype=np.float64)
        ps = f.create_dataset("ps", shape=(jmax, pa.t_array.size, b.size, 2),
                              dtype=np.float64)
    f.create_dataset("obs", data=P.y_d)
    f.create_dataset("x", data=x_coord)
    f.create_dataset("t", data=P.PDE.t_array)
    f.attrs["saveall"] = saveall

if save:
    bs[0] = b
    f_err1s[0] = P.f_err1
    f_err2s[0] = P.f_err2
    f_regs[0] = P.f_reg

# -----------------------------------------------------------------------------
# ------------------------- GRADIENT DESCENT METHOD ---------------------------
while j < jmax:

    v = P.compute_gradient(b)

    if save:

        vs[j] = v

        if saveall:

            qs[j] = P.q
            ps[j] = P.p

    if np.amax(abs(v)) < pa.tol:

        min_found = True
        if pa.test:
            print("Test passed")
        else:
            print(f"Found a minimum after {j} iterations.")
        break

    # --- Value of functional for initial step size ---
    if alpha_j < pa.alpha:
        alpha_j *= 2
    f_new = P.f(b - alpha_j*v)

    # --- Backtracking line search with Armijo rule ---

    while f_new > f_vals[j] + 1e-5*alpha_j*P.dx*P.dt*np.linalg.norm(v)**2 \
            or np.isnan(f_new):

        alpha_j *= pa.beta

        if alpha_j < 1e-10:

            breaker = True
            print("Bad search direction")

            if breaker:
                break

        f_new = P.f(b - alpha_j*v)

    if save:

        alpha_js[j] = alpha_j
        v_norms[j] = np.amax(abs(v))

    if breaker:
        break

    j += 1
    if j % 20 == 0:
        print(f"Iteration {j}, f(b)={f_new}\n")
    P.PDE.j = j
    f_vals[j] = f_new
    b = b - alpha_j*v

    if save:
        f_err1s[j] = P.f_err1
        f_err2s[j] = P.f_err2
        f_regs[j] = P.f_reg
        bs[j] = b

    if min_found is True:
        break

end = timer()

print("Elapsed time:", end-start, "s")
if pa.test is True and min_found is False:
    print("Test failed")

if np.linalg.norm(b_exact.flatten()) > 1e-16:
    print("Relative error (2-norm) to exact optimal control:",
          np.linalg.norm((b-b_exact).flatten())
          / np.linalg.norm(b_exact.flatten()))
else:
    print("Absolute error (max-norm) to exact optimal control:",
          np.amax(b-b_exact))

# -----------------------------------------------------------------------------
# ---------------------------- SAVE DATA --------------------------------------

if save is True:

    f.create_dataset("f_vals", data=f_vals)
    f.attrs["jmax"] = j

    if not saveall:
        f.create_dataset("q", data=P.q)

    f.close()

if save is True:

    with open("ProblemRelatedFiles/" + newfolder + "/info.txt", "w") as f2:

        f2.write("Running time is " + str(end-start)
                 + ", relative error to exact control is "
                 + str(np.linalg.norm((b-b_exact).flatten())
                       / np.linalg.norm(b_exact.flatten()))
                 + ", norm of exact control is "
                 + str(np.linalg.norm(b_exact.flatten())) + f", j = {j}"
                 + f"\nf(b_exact): {f_b_exct}"
                 + f"\nf(b): {f_vals[j]}"
                 + f"\nValue mismatch [0,T]: {P.f_err1}"
                 + f"\nValue mismatch at T: {P.f_err2}"
                 + f"\nValue regularisation term: {P.f_reg}"
                 + f"\nOutput gradient check: {gradientchecks}"
                 + f"\ntime step = {P.dt}"
                 + f"\ngrid points in space = {P.M}"
                 + useOtherParams
                 * f"Reconstruction with parameters from {folder}"
                 + min_found*f"\nFound a minimum after {j} iterations."
                 + "Used optimisation algorithm is gradient descent.")
