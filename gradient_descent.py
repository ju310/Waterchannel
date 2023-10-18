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
from taylortest import CheckGradient
from mpi4py import MPI
import logging

from ProblemRelatedFiles.grad_descent_aux import OptProblem
from ProblemRelatedFiles.params import params

for system in ['subsystems', 'solvers']:
    logging.getLogger(system).setLevel(logging.WARNING)  # Suppress output.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
save = False
saveall = False
pa = params()  # Object containing all parameters.
P = OptProblem(pa, comm, save)

# Create folder for plots and data.
if save is True:

    if rank == 0:

        newfolder = f"{P.problem}" + "_" + datetime.now().strftime(
            "%Y_%m_%d_%I_%M_%p")
        os.mkdir("ProblemRelatedFiles/" + newfolder)

        # Save parameter file in a new folder.
        with open("ProblemRelatedFiles/params.py", "r") as f:
            parameters = f.read()

        with open(f"ProblemRelatedFiles/{newfolder}/params.py", "w") as file:
            file.write(parameters)

b = pa.b_start
y_d = P.y_d

# --- Exact bathymetry. ---
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
gradientchecks = []
if pa.test is True:
    f_vals[0] = f_b_exct
else:
    f_vals[0] = P.f(b)
breaker = False
min_found = False
x_coord = P.PDE.dist.local_grid(P.PDE.xbasis)

if save and size == 1:

    f = h5py.File("ProblemRelatedFiles/" + newfolder
                  + "/gradient_data.hdf5", "w")

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
        if rank == 0:
            print(f"Found a minimum after {j} iterations.")
            min_found = comm.bcast(min_found, root=0)
            break

    # --- Value of functional for initial step size ---
    alpha_j = pa.alpha
    f_new = P.f(b - alpha_j*v)

    # --- Backtracking line search with Armijo rule ---

    while f_new > f_vals[j] + 1e-5*alpha_j*P.dx*P.dt*np.linalg.norm(v)**2 \
            or np.isnan(f_new):

        alpha_j *= pa.beta

        if alpha_j < 1e-13:

            breaker = True
            if rank == 0:
                print("Bad search direction")
                breaker = comm.bcast(breaker, root=0)

            if breaker:
                break

        f_new = P.f(b - alpha_j*v)

    if save:

        alpha_js[j] = alpha_j
        v_norms[j] = np.amax(abs(v))

    if breaker:
        break

    j += 1
    P.PDE.j = j
    f_vals[j] = f_new
    b = b - alpha_j*v

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
        if save is True:
            with open(f"ProblemRelatedFiles/{newfolder}/data.txt", "w") as ft:
                ft.write("Relative error (2-norm) to exact optimal control:" +
                         str(np.linalg.norm((b-b_exact).flatten())
                             / np.linalg.norm(b_exact.flatten())))
    else:
        print("Absolute error (max-norm)to exact optimal control:",
              np.amax(b-b_exact))

# -----------------------------------------------------------------------------
# -------------------- CREATE PLOTS AND SAVE DATA -----------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --------------------------------- DATA --------------------------------------

if save is True:

    if size == 1:

        f.create_dataset("f_vals", data=f_vals)
        f.attrs["jmax"] = j

        if not saveall:
            f.create_dataset("q", data=P.q)

        f.close()

    if P.withParareal:

        allqs = []
        allps = []
        for i in range(j):

            qBuf = np.zeros((P.t_array.size, pa.M, 2))
            comm.Gather(qs[i][1:], qBuf[1:], root=0)
            qBuf[0, :, 0] = pa.ic
            allqs.append(qBuf)

            pBuf = np.zeros((P.t_array.size, pa.M, 2))
            if rank == size-1:
                comm.Send(ps[i][-1], dest=0)
            if rank == 0:
                comm.Recv(pBuf[-1], source=size-1)
            pTimeSlice = ps[i][:-1]
            comm.Gather(pTimeSlice, pBuf[:-1], root=0)
            allps.append(pBuf)

        qs = allqs.copy()
        ps = allps.copy()

        if rank == 0:

            with h5py.File("ProblemRelatedFiles/" + newfolder
                           + "/gradient_data.hdf5", "w") as f:

                f.create_dataset("f_vals", data=f_vals)
                f.create_dataset("f_err1s", data=f_err1s)
                f.create_dataset("f_err2s", data=f_err2s)
                f.create_dataset("f_regs", data=f_regs)
                f.create_dataset("f_b_exct", data=f_b_exct)
                f.create_dataset("alpha_js", data=alpha_js)
                f.create_dataset("v_norms", data=v_norms)
                f.create_dataset("b_exact", data=b_exact)
                f.create_dataset("bs", data=bs)
                f.create_dataset("vs", data=vs)
                f.create_dataset("qs", data=qs)
                f.create_dataset("ps", data=ps)
                f.create_dataset("obs", data=P.y_d)
                f.create_dataset("x", data=x_coord)
                f.create_dataset("t", data=P.PDE.t_array)
                f.attrs["jmax"] = j

# if size == 1 and P.PDE.bc == "waterchannel":

#     for i in range(len(vs)):

#         cg = CheckGradient(P.f, P.dx*vs[i], bs[i])
#         # Need gradient*dx because of discrete scalar product
#         # in gradienttest.py.
#         gradientchecks.append(cg.check_order_2())
#         print(f"Iteration {i}: {cg.check_order_2()}")


if save is True:

    if rank == 0:

        with open("ProblemRelatedFiles/" + newfolder + "/data.txt", "w") as f2:

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
                     + f"\nOutput gradient check: {gradientchecks}"
                     + f"\nHyperdiffusion coefficient D = {pa.D}"
                     + f"\nBottom friction coefficient kappa = {P.PDE.kappa}"
                     + f"\ntime step = {P.dt}"
                     + f"\ngrid points in space = {P.M}")
