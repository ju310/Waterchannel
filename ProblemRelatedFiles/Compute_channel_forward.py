#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:47:22 2023
Sensor 1 at 1.5m.
Sensor 2 at 3.5m.
Sensor 3 at 5.5m.
Sensor 4 at 7.5m.
Compute the solution of the forward problem with boundary condition taken from
the measurements at the wave flume. The solution will be saved in a hdf5 file.

@author: Judith Angel
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings
from scipy import interpolate

import dedalus.public as d3

from read_left_bc import leftbc, data

import logging
logger = logging.getLogger(__name__)

#####################################################################
# Set to True if you want to compute the solution for SWE with bathymetry.
bathy = True
#####################################################################

if bathy:
    prefix = 'WaterchannelData/MitBathymetrie/'
    postfix = "_meanBathy"
    # postfix = "_try=1"
else:
    prefix = 'WaterchannelData/OhneBathymetrie/'
    postfix = "_mean"

filename = 'Tiefe=0,3_A=40_F=0,35' + postfix

lbc = leftbc(prefix+filename+".txt")
dataObject = data(prefix+filename+".txt")


def eval_at_sensor_positions(sol, pos):

    temp = []

    for i in range(len(pos)):
        sol_int = sol(x=pos[i])
        temp.append(float(sol_int.evaluate()['g']))

    return temp


pos = [3.5, 5.5, 7.5]
xmin = 1.5  # First sensor is located at 1.5m.
xmax = 15  # Set right boundary to 15m to simulate the 'beach' in the real
# water channel.

#####################################################################
# --------- Choose variables for the discretisation here. --------- #
Nx = 64
# Nx = 128
# Nx = 110
# T = 13
T = 10
start = 30  # Number of seconds to cut off from beginning of experimental data.
# start = 0
# timestep = 5e-5
# timestep = 1e-4
timestep = 1e-3
# ----------------------------------------------------------------- #
#####################################################################

N = int(T/abs(timestep))+1
g = 9.81
H = 0.3
kappa = 0.2  # Makes amplitudes smaller and waves a bit smoother.
dealias = 3/2

#####################################################################
# ------- Set to True if you want to save the solution. ----------- #
save = False
#####################################################################

if save:

    if bathy:
        path = "WaterchannelData/sim_data_" + filename\
            + f"_ExactRamp_T={T}_M={Nx}"
    else:
        path = "WaterchannelData/nobathy" + filename\
            + f"_kappa{kappa:.0e}_T={T}_M={Nx}"

    with open(__file__, "r") as thisfile:
        filetext = thisfile.read()

    with open(path + ".txt", "w") as newfile:
        newfile.write(filetext)

# Bases and domain
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(xmin, xmax), dealias=dealias)
x = dist.local_grid(xbasis)

# Fields
h = dist.Field(name='h', bases=xbasis)
u = dist.Field(name='u', bases=xbasis)
b = dist.Field(name='b', bases=xbasis)
tau1 = dist.Field(name='tau1')
tau2 = dist.Field(name='tau2')
temp = dist.Field()
t = dist.Field()


def hl_function(*args):

    t = args[0].data
    htemp = lbc.f(t+start)
    temp["g"] = H + htemp - b_array[0]

    return temp["g"]


def hl(*args, domain=temp.domain, F=hl_function):

    return d3.GeneralFunction(dist, domain, layout='g', tensorsig=(),
                              dtype=np.float64, func=F, args=args)


# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)
lift_basis = xbasis.derivative_basis(1)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# Problem
problem = d3.IVP([h, u, tau1, tau2], time=t, namespace=locals())
problem.add_equation("dt(h) + lift(tau1, -1) + dx(u)"
                     + " =  - dx((h-1)*u)")
problem.add_equation("dt(u) + lift(tau2, -1)"
                     + " + g*dx(h) + kappa*u = - u*dx(u) - g*dx(b)")

problem.add_equation("h(x='left') = hl(t)")
problem.add_equation("u(x='right') = 0")

# Build solver
solver = problem.build_solver(d3.RK443)
solver.stop_wall_time = 15000
solver.stop_iteration = N-1
solver.stop_sim_time = T - 1e-13

# Measured points of the ramp.
b_points = np.concatenate(
    (np.zeros(4),
     np.array([0, 0.024, 0.053, 0.0905, 0.133, 0.182, 0.2, 0.182, 0.133,
               0.0905, 0.053, 0.024, 0]),
     np.zeros(21)))
x_points = np.concatenate(
    (np.arange(1.5, 3.5, 0.5),
     np.array([
         3.4125, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.5875]),
     np.arange(5, 15.5, 0.5)))
rampFunc = interpolate.PchipInterpolator(x_points, b_points)
b_array = rampFunc(x)

# b_array = source(x)
b['g'] = b_array
h['g'] = H*np.ones(Nx) - b_array
u['g'] = 0

# Store data for final plot
temph = np.array(eval_at_sensor_positions(h, pos))\
    + np.array(eval_at_sensor_positions(b, pos))
h_list = [temph]
h.change_scales(1)
h_list_full = [np.copy(h["g"])]
tempu = eval_at_sensor_positions(u, pos)
u_list = [tempu]
u.change_scales(1)
u_list_full = [np.copy(u["g"])]
t_list = [solver.sim_time]

# Main loop
try:
    while solver.proceed:
        solver.step(timestep)
        temph = np.array(eval_at_sensor_positions(h, pos)) \
            + np.array(eval_at_sensor_positions(b, pos))
        h_list.append(temph)
        h.change_scales(1)
        h_list_full.append(np.copy(h["g"]))
        tempu = eval_at_sensor_positions(u, pos)
        u_list.append(tempu)
        u.change_scales(1)
        u_list_full.append(np.copy(u["g"]))
        t_list.append(solver.sim_time)
        if np.max(h['g']) > 10 or np.any(np.isnan(h['g'])):
            warnings.warn("Solution instable")
            break
        if solver.iteration % 1000 == 0:
            logger.info(
                'Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise

H_array = np.array(h_list)  # h + b
u_array = np.array(u_list)
h_array_full = np.array(h_list_full)
u_array_full = np.array(u_list_full)
t_array = np.array(t_list)
dx = (xmax-xmin)/Nx
Hmax = np.amax(H_array)
Hmin = np.amin(H_array)
H_meas = np.array([H+lbc.f(t_array+start),
                   H+dataObject.f[0](t_array+start),
                   H+dataObject.f[1](t_array+start),
                   H+dataObject.f[2](t_array+start)])

plt.figure()
plt.plot(t_array, H_meas[0], "k")
plt.ylim([Hmin-0.001, Hmax+0.001])
plt.title("Sensor 1, boundary condition")

plt.figure()
plt.plot(t_array, H_array[:, 0], "y",
         label="simulation")
plt.plot(t_array, H_meas[1], "k",
         label="measurement")
plt.legend()
plt.ylim([Hmin-0.001, Hmax+0.001])
plt.title("Sensor 2")

plt.figure()
plt.plot(t_array, H_array[:, 1], "y",
         label="simulation")
plt.plot(t_array, H_meas[2], "k",
         label="measurement")
plt.legend()
plt.ylim([Hmin-0.001, Hmax+0.001])
plt.title("Sensor 3")

plt.figure()
plt.plot(t_array, H_array[:, 2], "y",
         label="simulation")
plt.plot(t_array, H_meas[3], "k",
         label="measurement")
plt.legend()
plt.ylim([Hmin-0.001, Hmax+0.001])
plt.title("Sensor 4")

fig, axs = plt.subplots(3)
fig.tight_layout(pad=2.0)
axs[0].plot(t_array, H_array[:, 0], "y",
            label="simulation")
axs[0].plot(t_array, H_meas[1], "k--",
            label="measurement")
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('H [m]')
axs[0].set_title(f"Sensor 2 at {pos[0]}m")

axs[1].plot(t_array, H_array[:, 1], "y",
            label="simulation")
axs[1].plot(t_array, H_meas[2], "k--",
            label="measurement")
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('H [m]')
axs[1].set_title(f"Sensor 3 at {pos[1]}m")

axs[2].plot(t_array, H_array[:, 2], "y",
            label="simulation")
axs[2].plot(t_array, H_meas[3], "k--",
            label="measurement")
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('H [m]')
axs[2].legend(loc="upper left")
axs[2].set_title(f"Sensor 4 at {pos[2]}m")
plt.tight_layout()
plt.show()

error = [np.linalg.norm(H_array[:, j]-H_meas[j+1]) for j in range(3)]
print(error)
print(np.mean(error))
rel_error = [error[j]/np.linalg.norm(H_meas[j+1]) for j in range(3)]
print(rel_error)
print(np.mean(rel_error))

# Save all data.
if save:
    parameters = f"T_N={T}, xmin={xmin}, xmax={xmax}, dt={timestep}, \
        dx={dx}, g={g}, kappa={kappa}, points={Nx}"

    with h5py.File(path + ".hdf5", "w") as f:

        f.create_dataset("H_sensor", data=H_array)
        f.create_dataset("u_sensor", data=u_array)
        f.create_dataset("h", data=h_array_full)
        f.create_dataset("u", data=u_array_full)
        f.create_dataset("b_array", data=b_array)
        f.create_dataset("b_points", data=b_points)
        f.create_dataset("x_points", data=x_points)
        f.create_dataset("xgrid", data=x)
        f.create_dataset("t_array", data=t_array)
        f.create_dataset("pos", data=pos)
        f.attrs["start"] = start
        f.attrs["T_N"] = T
        f.attrs["xmin"] = xmin
        f.attrs["xmax"] = xmax
        f.attrs["dt"] = timestep
        f.attrs["dx"] = dx
        f.attrs["g"] = g
        f.attrs["H"] = H
        f.attrs["kappa"] = kappa
        f.attrs["M"] = Nx

    with open(path + "errors.txt", "w") as newfile:

        newfile.write(f"Relative l2-error at sensor positions: {rel_error}, "
                      + f"mean of the errors: {np.mean(rel_error)}")
