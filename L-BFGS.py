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

from ProblemRelatedFiles.grad_descent_aux import OptProblem

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
rank = comm.Get_rank()
size = comm.Get_size()
save = True
saveall = False
if not save:
    saveall = False
pa = params.params()  # Object containing all parameters.
bathy = []

m = 5  # Number of previous gradients to save.

P = OptProblem(pa, save)

# Create folder for plots and data.
if save is True:

    if rank == 0:

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
if rank == 0:
    print("f(b_exact) =", f_b_exct)
    print("Regularisation term: ", P.f_reg)

comm.Barrier()
start = MPI.Wtime()
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

if size > 1 and saveall:

    qs = []
    ps = []

if save and rank == 0:

    f = h5py.File("ProblemRelatedFiles/" + newfolder
                  + "/opt_data.hdf5", "w")

    if saveall and size == 1:

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
            if size == 1:
                qs[j] = P.q
                ps[j] = P.p
            elif size > 1 and rank == 0:
                qs.append(P.q)
                ps.append(P.p)
        d = v.copy()
    else:
        m_ = min(j, m)
        d = two_loop_recursion(
            v, np.array(s_stored), np.array(y_stored), m_)

    if np.amax(abs(d)) < pa.tol:
        min_found = True
        if rank == 0:
            print(f"Found a minimum after {j} iterations.")
            min_found = comm.bcast(min_found, root=0)
            break

    # --- Value of functional for initial step size ---
    if alpha_j < pa.alpha:
        alpha_j = pa.alpha  # ???

    f_new = P.f(b - alpha_j*d)  # TODO: Would be nice to use this for
    # the next gradient. The forward problem is being solved for this control
    # twice.

    # --- Backtracking line search with Armijo rule ---

    while f_new > f_vals[j] + 1e-5*alpha_j*P.dx*P.dt*np.linalg.norm(d)**2 \
            or np.isnan(f_new):

        alpha_j *= pa.beta

        if alpha_j < 1e-10:

            breaker = True
            if rank == 0:
                print("Bad search direction")
                breaker = comm.bcast(breaker, root=0)

            if breaker:
                break

        f_new = P.f(b - alpha_j*d)

    s_stored.append(-alpha_j*d)
    grad_old = v.copy()

    if j > m:
        s_stored.pop(0)
        y_stored.pop(0)

    if save and rank == 0:
        alpha_js[j] = alpha_j
        v_norms[j] = np.amax(abs(v))

    if breaker:
        break

    j += 1
    if j % 20 == 0 and rank == 0:
        print(f"Iteration {j}, f(b)={f_new}\n")
    P.PDE.j = j
    f_vals[j] = f_new
    b = b - alpha_j*d
    v = P.compute_gradient(b)
    y_stored.append(v-grad_old)

    if save and rank == 0:
        f_err1s[j] = P.f_err1
        f_err2s[j] = P.f_err2
        f_regs[j] = P.f_reg
        bs[j] = b

    if min_found is True:
        break

end = MPI.Wtime()

if rank == 0:

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

    if size == 1:

        if not saveall:
            f.create_dataset("q", data=P.q)
            f.create_dataset("p", data=P.p)

    else:

        if saveall:

            allqs = []
            allps = []
            for i in range(j):

                qBuf = np.zeros((P.t_array.size, pa.M, 2))
                comm.Gather(qs[i][1:], qBuf[1:], root=0)
                qBuf[0, :, 0] = P.PDE.ic
                allqs.append(qBuf)

                pBuf = np.zeros((P.t_array.size, pa.M, 2))
                if rank == size-1:
                    comm.Send(ps[i][-1], dest=0)
                if rank == 0:
                    comm.Recv(pBuf[-1], source=size-1)
                pTimeSlice = ps[i][:-1]
                comm.Gather(pTimeSlice, pBuf[:-1], root=0)
                allps.append(pBuf)

            if rank == 0:

                qs = allqs.copy()
                ps = allps.copy()

        else:

            qBuf = np.zeros((P.t_array.size, pa.M, 2))
            comm.Gather(P.q[1:], qBuf[1:], root=0)
            qBuf[0, :, 0] = P.PDE.ic
            pBuf = np.zeros((P.t_array.size, pa.M, 2))
            if rank == size-1:
                comm.Send(P.p[-1], dest=0)
            if rank == 0:
                comm.Recv(pBuf[-1], source=size-1)
            pTimeSlice = P.p[:-1]
            comm.Gather(pTimeSlice, pBuf[:-1], root=0)

            if rank == 0:
                f.create_dataset("q", data=qBuf)
                f.create_dataset("p", data=pBuf)
                f.attrs["c"] = pa.c

        comm.Barrier()

    if rank == 0:

        if size > 1 and saveall:

            f.create_dataset("qs", data=qs)
            f.create_dataset("ps", data=ps)

        f.create_dataset("f_vals", data=f_vals)
        f.attrs["jmax"] = j
        f.close()

# if size == 1 and P.PDE.bc == "waterchannel":
#     # Somehow the gradient test does not work in parallel.

#     for i in range(len(vs)):

#         cg = CheckGradient(P.f, P.dx*vs[i], bs[i])
#         # Need gradient*dx because of discrete scalar product
#         # in gradienttest.py.
#         gradientchecks.append(cg.check_order_2())
#         print(f"Iteration {i}: {cg.check_order_2()}")

if save and rank == 0:

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
                 + f" {folder}")
