#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:25:37 2023.

@author: Judith Angel
Parameter file for gradient descent. The initial condition, observation and
the exact control are being read from a hdf5 file.
"""
import numpy as np
import warnings
import h5py
import dedalus.public as d3
from mpi4py import MPI
from scipy import interpolate
from ProblemRelatedFiles.read_left_bc import leftbc, data


class params:
    """Class for all parameters for gradient descent."""

    def __init__(self):

        # Turn on/off test for gradient descent method. Start with exact b.
        self.test = False

        # Use either measurement data, simulated data everywhere or
        # simulated data at sensor positions.
        # self.data = "measurements"
        # self.data = "sim_everywhere"
        self.data = "sim_sensor_pos"

        # Use mean of measurements.
        self.mean = True

        # Put noise on observation.
        # self.noise = 0
        self.noise = 1

        # Set bottom friction coefficient.
        self.kappa = 0.2

        # Gravitational acceleration.
        self.g = 9.81

        # Set final time.
        self.T_N = 10

        if self.data == "measurements":
            path = "ProblemRelatedFiles/WaterchannelData/" \
                + "sim_data_Tiefe=0,3_A=40_F=0,35_meanBathy_ExactRamp_T=13.hdf5"
        else:
            path = "ProblemRelatedFiles/WaterchannelData/" \
                + "sim_data_Tiefe=0,3_A=40_F=0,35_ExactRamp_T=16.hdf5"
        if self.data != "measurements":
            # pathbc = "ProblemRelatedFiles/WaterchannelData/" \
            #     + "MitBathymetrie/Tiefe=0,3_A=40_F=0,35_meanBathy.txt"
            pathbc = "ProblemRelatedFiles/WaterchannelData/" \
                + "Tiefe=0,3_A=40_F=0,35.txt"
        else:
            if self.mean:
                pathbc = "ProblemRelatedFiles/WaterchannelData/" \
                    + "MitBathymetrie/Tiefe=0,3_A=40_F=0,35_meanBathy.txt"
            else:
                pathbc = "ProblemRelatedFiles/WaterchannelData/" \
                    + "MitBathymetrie/Tiefe=0,3_A=40_F=0,35_try=1.txt"

        # Tolerance for stopping criterion in gradient descent.
        # self.tol = 7e-8
        self.tol = 1e-7
        # self.tol = 1e-6

        # Set factor for regularisation term.
        if self.test:
            self.lambd = 0
            self.lambda_b = 0
        else:
            # self.lambd = 0.001
            # self.lambd = 1e-4
            self.lambd = 1e-5
            # self.lambd = 1e-6
            # self.lambd = 0

            L2reg = True
            if L2reg:
                # self.lambda_b = 1e-6
                self.lambda_b = 1e-7

        # Parameters for Armijo rule/Wolfe conditions.
        self.alpha = 128
        # self.alpha = 1
        self.beta = 0.5

        # Parameters for cost functional (observation over [0,T]).
        self.gamma = 0.5
        # self.gamma = 0.25
        # self.gamma = 0
        # self.gamma = 1
        self.delta = 1 - self.gamma

        # Maximum number of iterations in gradient descent.
        if self.test:
            self.jmax = 2
        else:
            self.jmax = 2000

        # Time step.
        self.dt = 1e-3

        # Number of points in time.
        N = int(self.T_N/self.dt) + 1

        # Number of grid points in space.
        self.M = 17*4
        # self.M = 70
        # self.M = 100

        # Left and right boundary.
        self.xmin = 1.5
        self.xmax = 15  # Matches better with measurements.
        # self.xmax = 12
        if self.data == "measurements":
            positions = [3.5, 5.5, 7.5]  # Sensor positions
            self.sensors = [2]  # Indices of sensors to use.
            self.pos = []
            for i in self.sensors:
                self.pos.append(positions[i])
        elif self.data == "sim_sensor_pos":
            self.pos = [3.5, 5.5, 7.5]
            # self.pos = [3.5, 6, 8.5]
            # self.pos = [3.5, 6]
            # self.pos = [2]
            # self.pos = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
            self.start = 0
        self.lbc = leftbc(pathbc).f  # CubicSpline

        # Load observation, exact bathymetry and parameters from hdf5 file.
        if self.data != "measurements":

            with h5py.File(path, "r") as f:

                b_points = np.array(f["b_points"])
                x_points = np.array(f["x_points"])
                b_exact_fine = np.array(f["b_array"])
                self.h_array_full = np.array(f["h"])  # Computed h.
                if self.data == "sim_everywhere":
                    h_array_full = np.array(f["h"])  # Computed h (fine grid).
                    y_d_fine = h_array_full.copy() + b_exact_fine
                pos_p = np.array(f["pos"])
                dt_p = f.attrs.get("dt")
                M_p = f.attrs.get("M")
                T_N_p = f.attrs.get("T_N")  # Final time.
                t_array_fine = np.array(f["t_array"])
                kappa_p = f.attrs.get("kappa")
                g_p = f.attrs.get("g")
                xmin_p = f.attrs.get("xmin")
                xmax_p = f.attrs.get("xmax")
                self.start = f.attrs.get("start")  # Number of seconds to
                # cut off from beginning of experimental data in lbc.
                self.H = f.attrs.get("H")

            # Check if parameters match the ones from the hdf5 file.
            if self.T_N > T_N_p:
                raise ValueError(
                    "Final time does not match the observation from the hdf5"
                    + f" file ({T_N_p})")
            if self.g != g_p:
                warnings.warn(
                    f"Parameter g={self.g} does not match the one from the"
                    + f" hdf5 file ({g_p})")
            if self.kappa != kappa_p:
                warnings.warn(
                    f"Parameter kappa={self.kappa} does not match the one"
                    + f" from the hdf5 file ({kappa_p})")
            if self.xmin != xmin_p:
                raise ValueError(
                    "Left boundary does not match the one from the hdf5 file"
                    + f" ({xmin_p})")
            if self.xmax != xmax_p:
                raise ValueError(
                    "Right boundary does not match the one from the hdf5 file"
                    + f" ({xmax_p})")
            if self.dt != dt_p or self.M != M_p:
                print(
                    "Note that you are not using the same discretisation as "
                    + "used for the observation in the hdf5 file.")

            if self.data != "sim_everywhere":
                for i in range(len(self.pos)):
                    if self.pos[i] not in pos_p:
                        warnings.warn(
                            f"Parameter pos={self.pos} does not match the one"
                            + f" from the hdf5 file ({pos_p})")
                        break

        else:

            self.H = 0.3  # Water level at rest.
            # Measured points of the ramp.
            b_points = np.concatenate(
                (np.zeros(4),
                 np.array([0, 0.024, 0.053, 0.0905, 0.133, 0.182, 0.2, 0.182,
                           0.133, 0.0905, 0.053, 0.024, 0]),
                 np.zeros(21)))
            x_points = np.concatenate(
                (np.arange(1.5, 3.5, 0.5),
                 np.array([
                     3.4125, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4,
                     4.5, 4.5875]),
                 np.arange(5, 15.5, 0.5)))

        rampFunc = interpolate.PchipInterpolator(x_points, b_points)

        # ---------------------------------------------------------------------
        # Scale initial condition, observation and exact control.
        xcoord = d3.Coordinate('x')
        dist = d3.Distributor(xcoord, comm=MPI.COMM_SELF, dtype=np.float64)
        xbasis = d3.Chebyshev(
            xcoord, size=self.M, bounds=(self.xmin, self.xmax), dealias=3/2)
        N = int(self.T_N/self.dt) + 1  # Number of points in time.
        self.t_array = np.linspace(0, self.T_N, N)
        self.y_d = np.zeros((N, self.M))
        x = dist.local_grid(xbasis)
        ramp = rampFunc(x)
        self.b_exact = ramp

        if self.data == "measurements":

            # Load observation data from text file.
            dataObject = data(pathbc)

            for p in self.sensors:

                H_sensor = self.H + dataObject.f[p](
                    self.t_array + self.start)
                i = np.argmin(abs(x-positions[p]))
                self.y_d[:, i] = H_sensor

        elif self.data == "sim_sensor_pos":

            # Put simulation data from hdf5 file on the coarser grid.

            def eval_at_pos(sol, pos):

                temp = []

                for i in range(len(pos)):
                    sol_int = sol(x=pos[i])
                    temp.append(float(sol_int.evaluate()['g']))

                return temp

            H_field = dist.Field(bases=xbasis)

            for n in range(N):

                H_field.change_scales(1)
                # Find corresponding index in fine time array.
                # This is the best way to do it as we need this loop
                # over time anyways.
                n_fine = np.argmin(abs(t_array_fine-self.t_array[n]))
                H_field.change_scales(M_p/self.M)
                H_field["g"] = np.copy(
                    self.h_array_full[n_fine] + b_exact_fine)
                H_pos = eval_at_pos(H_field, self.pos)

                for p in range(len(self.pos)):

                    i = np.argmin(abs(x-self.pos[p]))
                    self.y_d[n, i] = H_pos[p]

        elif self.data == "sim_everywhere":

            self.y_d_func = interpolate.CubicSpline(
                t_array_fine, y_d_fine, axis=0)
            y_d = self.y_d_func(self.t_array)

            for n in range(N):

                y_d_field = dist.Field(bases=xbasis)
                y_d_field.change_scales(M_p/self.M)
                y_d_field['g'] = np.copy(y_d[n])
                y_d_field.change_scales(1)
                self.y_d[n] = np.copy(y_d_field['g'])

        # Add noise to the observation.
        if self.data == "sim_sensor_pos":

            # The noise depends on the local wave height.
            for p in range(len(self.pos)):

                i = np.argmin(abs(x-self.pos[p]))
                self.y_d[2:, i] += self.noise \
                    * np.random.normal(
                        0, .05*(np.max(self.y_d[:, i]-self.H)), N-2)

        elif self.data == "sim_everywhere":

            # This noise depends on the local wave height.
            for m in range(self.M):

                self.y_d[2:, m] += self.noise \
                    * np.random.normal(
                        0, .05*(np.max(self.y_d[:, m]-self.H)), N-2)

        b_field = dist.Field(bases=xbasis)

        # ---------------------------------------------------------------------
        # SET INITIAL GUESS FOR BATHYMETRY.

        # b_field['g'] = 0.1*np.exp(-((dist.local_grid(xbasis)-5)/0.7)**2)
        # b_field['g'] = b_exct_field["g"] + np.random.normal(0, .001, self.M)
        # b_field['g'] = b_exct_field["g"] \
        #     + 0.005*np.sin(0.5*dist.local_grid(xbasis))
        # b_field['g'] = b_exct_field["g"]*0.99# + 0.1
        # b_field['g'] = 0.05*np.sin(np.pi*dist.local_grid(xbasis))
        b_field['g'] = np.zeros(self.M)
        if self.test:
            self.b_start = np.copy(self.b_exact)
        else:
            self.b_start = np.copy(b_field['g'])
