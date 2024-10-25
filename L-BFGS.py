#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:25:16 2024.

@author: Judith Angel
"""

import numpy as np
import h5py
import os
from datetime import datetime
from mpi4py import MPI
import logging
import importlib
from timeit import default_timer as timer

from ProblemRelatedFiles.grad_descent_aux import OptProblem
from ProblemRelatedFiles.wolfe_line_search import strongWolfe
from ProblemRelatedFiles.armijo_line_search import Armijo

for system in ['subsystems', 'solvers']:
    logging.getLogger(system).setLevel(logging.WARNING)  # Suppress output.

oldOptAgain = False  # Set to True if you want to run an old optimisation again
# with exactly the same parameters.
folder = "nonlinSWE_2023_10_12_02_56_PM"  # Folder with old optimisation data.
params = importlib.import_module("ProblemRelatedFiles."
                                 + oldOptAgain*(folder + ".") + "params")
# Adapted from
# https://medium.com/@tru11631/l-bfgs-two-loop-recursion-6976b6298f6c


def two_loop_recursion(gradient, s_stored, y_stored, m):

    q = gradient.copy()
    alpha = np.zeros(m)
    rho = np.array(
        [1/np.dot(y_stored[j, :], s_stored[j, :]) for j in range(m)])

    for i in range(m):

        alpha[m-1-i] = rho[m-1-i] * np.dot(s_stored[m-1-i, :], q)
        q = q - alpha[m-1-i]*y_stored[m-1-i, :]

    H_k0 = np.dot(s_stored[m-1], y_stored[m-1]) \
        / np.dot(y_stored[m-1], y_stored[m-1])
    r = H_k0*q

    for i in range(m):

        beta = rho[i]*np.dot(y_stored[i, :], r)
        r = r + (alpha[i]-beta)*s_stored[i]

    return r


comm = MPI.COMM_WORLD
save = True
saveall = False
if not save:
    saveall = False
pa = params.params()  # Object containing all parameters.
bathy = []

m = 5  # Number of previous gradients to save.

P = OptProblem(pa, save)
L = strongWolfe(pa.c1, pa.c2, pa.alpha_max, pa.i_max)
# L = Armijo(pa.alpha, pa.beta, comm)

# Create folder for plots and data.
if save is True:

    newfolder = datetime.now().strftime("%Y_%m_%d_%I_%M_%p")
    os.mkdir("ProblemRelatedFiles/" + newfolder)

    # Save parameter file.
    with open("ProblemRelatedFiles/" + oldOptAgain*(folder + "/")
              + "params.py", "r") as f:
        parameters = f.read()

    with open(f"ProblemRelatedFiles/{newfolder}/params.py", "w") as file:
        file.write(parameters)

b = pa.b_start
y_d = P.y_d

# --- Exact solution, if available. ---
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
s_stored = []
y_stored = []
if pa.test is True:
    f_vals[0] = f_b_exct
else:
    f_vals[0] = P.f(b)
breaker = False
min_found = False
x_coord = P.PDE.dist.local_grid(P.PDE.xbasis)

if saveall:

    qs = []
    ps = []

if save:

    f = h5py.File("ProblemRelatedFiles/" + newfolder
                  + "/opt_data.hdf5", "w")

    if saveall:

        qs = f.create_dataset(
            "qs", shape=(jmax, pa.t_array.size, b.size, 2),
            dtype=np.float64)
        ps = f.create_dataset(
            "ps", shape=(jmax, pa.t_array.size, b.size, 2),
            dtype=np.float64)

    f_err1s = f.create_dataset("f_err1s", shape=(jmax+1,), dtype=np.float64)
    f_err2s = f.create_dataset("f_err2s", shape=(jmax+1,), dtype=np.float64)
    f_regs = f.create_dataset("f_regs", shape=(jmax+1,), dtype=np.float64)
    f.create_dataset("f_b_exct", data=f_b_exct)
    alpha_js = f.create_dataset("alpha_js", shape=(jmax,), dtype=np.float64)
    v_norms = f.create_dataset("v_norms", shape=(jmax,), dtype=np.float64)
    vs = f.create_dataset("vs", shape=(jmax, b.size), dtype=np.float64)
    dirs = f.create_dataset("dirs", shape=(jmax, b.size), dtype=np.float64)
    f.create_dataset("b_exact", data=b_exact)
    bs = f.create_dataset("bs", shape=(jmax+1, b.size), dtype=np.float64)

    f.create_dataset("obs", data=P.y_d)
    f.create_dataset("x", data=x_coord)
    f.create_dataset("t", data=P.PDE.t_array)
    f.attrs["saveall"] = saveall

    bs[0] = b
    f_err1s[0] = P.f_err1
    f_err2s[0] = P.f_err2
    f_regs[0] = P.f_reg

# -----------------------------------------------------------------------------
# ---------------------- L-BFGS WITH ARMIJO LINE SEARCH -----------------------
while j < jmax:

    if j == 0:
        v = P.compute_gradient(b)

        if saveall:
            qs[j] = P.q
            ps[j] = P.p
        d = v.copy()
    else:
        m_ = min(j, m)
        d = two_loop_recursion(
            v, np.array(s_stored), np.array(y_stored), m_)

    if save:
        vs[j] = v
        dirs[j] = d

    if np.amax(abs(v)) < pa.tol:
        min_found = True
        print(f"Found a minimum after {j} iterations.")
        break

    grad_old = v.copy()

    alpha_j, f_vals[j+1], v, breaker = L.line_search(P, b, d, f_vals[j], v)

    s_stored.append(-alpha_j*d)

    if j > m:
        s_stored.pop(0)
        y_stored.pop(0)

    if save:
        alpha_js[j] = alpha_j
        v_norms[j] = np.amax(abs(v))

    if breaker:
        break

    j += 1
    if j % 20 == 0:
        print(f"Iteration {j}, f(b)={f_vals[j]}\n")
    P.PDE.j = j
    b = b - alpha_j*d
    y_stored.append(v-grad_old)

    if saveall:
        qs.append(P.q)
        ps.append(P.p)

    if save:
        f_err1s[j] = P.f_err1
        f_err2s[j] = P.f_err2
        f_regs[j] = P.f_reg
        bs[j] = b

end = timer()

print("Elapsed time:", end-start, "s")

if np.linalg.norm(b_exact.flatten()) > 1e-16:
    print("Relative error (2-norm) to exact optimal control:",
          np.linalg.norm((b-b_exact).flatten())
          / np.linalg.norm(b_exact.flatten()))
    if save:
        with open(f"ProblemRelatedFiles/{newfolder}/info.txt", "w") as ft:
            ft.write("Relative error (2-norm) to exact optimal control:" +
                     str(np.linalg.norm((b-b_exact).flatten())
                         / np.linalg.norm(b_exact.flatten())))
else:
    print("Absolute error (max-norm) to exact optimal control:",
          np.amax(b-b_exact))

# -----------------------------------------------------------------------------
# -------------------- CREATE PLOTS AND SAVE DATA -----------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --------------------------------- DATA --------------------------------------

if save is True:

    if not saveall:
        f.create_dataset("q", data=P.q)
        f.create_dataset("p", data=P.p)

    f.create_dataset("f_vals", data=f_vals)
    f.attrs["jmax"] = j
    f.close()

if save:

    with open("ProblemRelatedFiles/" + newfolder + "/info.txt", "w") as f2:

        f2.write("Running time is " + str(end-start)
                 + ", relative error to exact control is "
                 + str(np.linalg.norm((b-b_exact).flatten())
                       / np.linalg.norm(b_exact.flatten()))
                 + ", norm of exact control is "
                 + str(np.linalg.norm(b_exact.flatten())) + f", j = {j}"
                 + f"\nValue of objective functional: {f_vals[j]}"
                 + f"\nValue mismatch [0,T]: {P.f_err1}"
                 + f"\nValue mismatch at T: {P.f_err2}"
                 + f"\nValue regularisation term: {P.f_reg}"
                 + f"\nBottom friction coefficient kappa = {P.PDE.kappa}"
                 + f"\ntime step = {P.dt}"
                 + f"\ngrid points in space = {P.M}"
                 + oldOptAgain*"\n Reconstruction with parameters from"
                 + f" {folder}"
                 + min_found*f"\nFound a minimum after {j} iterations."
                 + "Used optimisation algorithm is L-BFGS.")
