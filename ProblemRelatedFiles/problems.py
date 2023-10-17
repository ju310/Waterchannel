#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:40:16 2023.

@author: Judith Angel
"""
import numpy as np
import dedalus.public as d3
import warnings
from scipy import interpolate
import logging
logger = logging.getLogger(__name__)
for system in ['subsystems', 'solvers']:
    logging.getLogger(system).setLevel(logging.WARNING)  # Suppress output.


class nonlinSWE:
    """Class for nonlinear shallow water equations."""

    def __init__(self, pa, t_array, comm, y_d, save=False):

        self.j = 0  # Gradient descent iteration counter.
        self.save = save
        self.coarse_in_serial = pa.coarse_in_serial
        self.T_N = pa.T_N
        self.N_fine = pa.t_array.size
        self.dt = pa.dt
        self.M = pa.M
        self.g = pa.g
        self.kappa = pa.kappa
        self.gamma = pa.gamma
        self.delta = pa.delta
        self.H = pa.H
        self.lbc = pa.lbc
        self.pos = pa.pos
        self.current_b = pa.b_start.copy()
        self.b_exact = pa.b_exact
        self.t_array = t_array
        self.comm = comm
        self.xcoord = d3.Coordinate('x')
        self.dist = d3.Distributor(self.xcoord, dtype=np.float64)
        self.xbasis = d3.Chebyshev(
            self.xcoord, size=self.M, bounds=(pa.xmin, pa.xmax), dealias=3/2)
        self.domain = d3.Domain(self.dist, bases=[self.xbasis])

        # Fields
        self.h_field = self.dist.Field(name='h', bases=self.xbasis)
        self.u_field = self.dist.Field(name='u', bases=self.xbasis)
        self.p1_field = self.dist.Field(name='p1', bases=self.xbasis)
        self.p2_field = self.dist.Field(name='p2', bases=self.xbasis)
        self.t_field = self.dist.Field()
        self.temp = self.dist.Field(bases=self.xbasis)
        self.tau1 = self.dist.Field(name='tau1')
        self.tau2 = self.dist.Field(name='tau2')
        self.taup1 = self.dist.Field(name='taup1')
        self.taup2 = self.dist.Field(name='taup2')
        self.bcfield = self.dist.Field()
        self.temp_ul = self.dist.Field()
        self.y_d = y_d  # Here observation on finer spatial grid.
        self.ic = np.max(self.y_d[0])*np.ones(self.y_d.shape[1])
        self.b = self.dist.Field(bases=self.xbasis)

        # Substitutions
        self.dx = lambda A: d3.Differentiate(A, self.xcoord)
        lift_basis = self.xbasis.derivative_basis(1)
        self.lift = lambda A, n: d3.Lift(A, lift_basis, n)

        # Forward problem"waterchannel":

        def hl_function(*args):

            t = args[0].data
            htemp = self.lbc(t)
            self.bcfield["g"] = self.H + htemp - self.current_b[0]

            return self.bcfield["g"]

        def hl(*args, domain=self.bcfield.domain, F=hl_function):

            return d3.GeneralFunction(
                self.dist, domain, layout='g', tensorsig=(),
                dtype=np.float64, func=F, args=args)

        self.problem = d3.IVP(
            [self.h_field, self.u_field, self.tau1, self.tau2],
            time=self.t_field,
            namespace={"g": self.g, "kappa": self.kappa, "h": self.h_field,
                       "u": self.u_field, "dx": self.dx, "b": self.b,
                       "t": self.t_field, "tau1": self.tau1,
                       "tau2": self.tau2, "lift": self.lift, "hl": hl})
        self.problem.add_equation("dt(h) + lift(tau1, -1) + dx(u)"
                                  + " =  - dx((h-1)*u)")
        self.problem.add_equation(
            "dt(u) + lift(tau2, -1) + g*dx(h) + kappa*u = "
            + "- 2*u*dx(u) - g*dx(b)")
        self.problem.add_equation("h(x='left') =  hl(t)")
        self.problem.add_equation("u(x='right') = 0")

    def solvepde(self, inpt, opt):
        """
        Solve the forward or adjoint problem.

        Parameters
        ----------
        inpt : numpy array
            Forward problem: Control.
            Adjoint problem: Solution of forward problem.
        opt : string
            Decide whether to solve the forward ("primal") or the adjoint
            problem.

        Returns
        -------
        numpy array
            Numerical solution of the given problem.

        """
        if opt == "primal":

            self.current_b = np.copy(inpt)
            h_0 = self.ic - self.current_b

            # Build solver
            solver = self.problem.build_solver(d3.RK443)

            solver.stop_iteration = self.y_d.shape[0] - 1
            solver.stop_sim_time = self.T_N
            solver.sim_time = 0  # Reset sim_time.

            # Initial conditions
            self.b.change_scales(1)
            self.b['g'] = np.copy(self.current_b)
            self.h_field.change_scales(1)
            self.u_field.change_scales(1)
            self.h_field['g'] = h_0
            self.u_field['g'] = np.zeros(h_0.size)  # Here: u(0)
            tempH = self.eval_at_sensor_positions(
                self.h_field, self.pos) \
                + self.eval_at_sensor_positions(
                    self.b, self.pos)
            H_list_pos = [tempH]
            self.h_field.change_scales(1)
            h_list = [np.copy(self.h_field['g'])]
            self.u_field.change_scales(1)
            u_list = [np.copy(self.u_field['g'])]
            t_list = [solver.sim_time]

            # Main loop
            try:
                while solver.proceed:
                    solver.step(self.dt)

                    # Save solution for h.
                    tempH = self.eval_at_sensor_positions(
                        self.h_field, self.pos) \
                        + self.eval_at_sensor_positions(
                            self.b, self.pos)
                    H_list_pos.append(tempH)
                    self.h_field.change_scales(1)
                    h_list.append(np.copy(self.h_field['g']))

                    # Save solution for u.
                    self.u_field.change_scales(1)
                    u_list.append(np.copy(self.u_field['g']))
                    t_list.append(solver.sim_time)
                    if np.max(self.h_field['g']) > 100:
                        warnings.warn("Solution instable")
                        break
            except:
                logger.error(
                    'Exception raised, triggering end of main loop.')
                raise

            h_array = np.array(h_list)
            u_array = np.array(u_list)

            # H_array is H at the sensor positions and zero elsewhere.
            H_array_pos = np.array(H_list_pos)
            self.H_array = np.zeros(h_array.shape)
            x = self.dist.local_grid(self.xbasis)
            for p in range(len(self.pos)):
                i = np.argmin(abs(x-self.pos[p]))
                self.H_array[:, i] = H_array_pos[:, p]

            sol = np.concatenate(
                (h_array[:, :, None], u_array[:, :, None]),
                axis=2)

        elif opt == "adjoint":

            t_interval = self.t_array
            mismatch = self.gauss_peak(self.H_array-self.y_d)
            h_array = inpt[:, :, 0]
            u_array = inpt[:, :, 1]

            y_d = self.y_d

            # Define function h. Extrapolation is needed due to possible
            # rounding errors at the boundary of a time slice.
            h_func = interpolate.interp1d(t_interval, h_array, axis=0,
                                          fill_value="extrapolate")

            def h_function(*args):

                tau = args[0].data
                t = self.T_N - tau
                self.temp.change_scales(1)
                self.temp['g'] = h_func(t)
                self.temp.change_scales(self.domain.dealias)

                return self.temp['g']

            def h_fwd(*args, domain=self.domain, F=h_function):

                return d3.GeneralFunction(self.dist, self.domain, layout='g',
                                          tensorsig=(), dtype=np.float64,
                                          func=F, args=args)

            def mismatch_function(*args):

                tau = args[0].data
                t = self.T_N - tau
                self.temp.change_scales(1)
                i = np.argmin(abs(self.t_array-t))
                self.temp['g'] = mismatch[i]
                self.temp.change_scales(self.domain.dealias)

                return self.temp['g']

            def H_H_obs(*args, domain=self.domain, F=mismatch_function):

                return d3.GeneralFunction(self.dist, self.domain, layout='g',
                                          tensorsig=(), dtype=np.float64,
                                          func=F, args=args)

            u_func = interpolate.interp1d(t_interval, u_array, axis=0,
                                          fill_value="extrapolate")

            def U_function(*args):

                tau = args[0].data
                t = self.T_N - tau
                self.temp.change_scales(1)
                self.temp['g'] = u_func(t)
                self.temp.change_scales(self.domain.dealias)

                return self.temp['g']

            def U(*args, domain=self.domain, F=U_function):

                return d3.GeneralFunction(self.dist, self.domain, layout='g',
                                          tensorsig=(), dtype=np.float64,
                                          func=F, args=args)

            def y_d_function(*args):

                tau = args[0].data
                t = self.T_N - tau
                self.temp.change_scales(1)
                i = np.argmin(abs(self.t_array-t))
                self.temp['g'] = y_d[i]
                self.temp.change_scales(self.domain.dealias)

                return self.temp['g']

            def Y_d(*args, domain=self.domain, F=y_d_function):

                return d3.GeneralFunction(self.dist, self.domain, layout='g',
                                          tensorsig=(), dtype=np.float64,
                                          func=F, args=args)

            def udivhl_function(*args):

                tau = args[0].data
                t = self.T_N - tau
                i = np.argmin(t-self.t_array)
                hl = self.H + self.lbc(t) - self.current_b[0]
                udivh = u_array[i, 0] / hl

                return udivh

            def udivhl(
                    *args, domain=self.temp_ul.domain, F=udivhl_function):

                return d3.GeneralFunction(
                    self.dist, domain, layout='g', tensorsig=(),
                    dtype=np.float64, func=F, args=args)

            # Initialise adjoint problem.
            # For the adjoint problem, "t" is actually \tau.

            self.adjproblem = d3.IVP(
                [self.p1_field, self.p2_field, self.taup1, self.taup2],
                time=self.t_field,
                namespace={"g": self.g, "kappa": self.kappa,
                           "p1": self.p1_field, "p2": self.p2_field,
                           "b": self.b, "dx": self.dx, "h_fwd": h_fwd,
                           "U": U, "Y_d": Y_d, "t": self.t_field,
                           "gamma": self.gamma, "lift": self.lift,
                           "taup1": self.taup1, "taup2": self.taup2,
                           "udivhl": udivhl, "H_H_obs": H_H_obs})
            self.adjproblem.add_equation(
                "dt(p1) - g*dx(p2) + lift(taup1, -1)"
                + " = U(t)*dx(p1) + gamma*H_H_obs(t)")
            self.adjproblem.add_equation(
                "dt(p2) - dx(p1) + lift(taup2, -1)"
                + " = (h_fwd(t)-1)*dx(p1) + 2*U(t)*dx(p2) - kappa*p2")
            self.adjproblem.add_equation(
                "p1(x='left') = - 2*p2(x='left')*udivhl(t)")
            self.adjproblem.add_equation("p2(x='right') = 0")

            # Build solver
            solver = self.adjproblem.build_solver(d3.RK443)

            solver.stop_wall_time = 6000
            solver.stop_iteration = self.y_d.shape[0] - 1
            solver.stop_sim_time = self.T_N
            solver.sim_time = 0  # Reset sim_time.

            # Initial conditions
            self.p1_field['g'] = self.delta*(self.gauss_peak(
                self.H_array - self.y_d)[-1])

            self.p2_field['g'] = np.zeros(self.current_b.size)

            self.p1_field.change_scales(1)
            p1_list = [np.copy(self.p1_field['g'])]
            p2_list = [np.copy(self.p2_field['g'])]
            t_list = [solver.sim_time]

            # Main loop
            try:
                while solver.proceed:
                    solver.step(self.dt)
                    if solver.iteration % 1 == 0:
                        self.p1_field.change_scales(1)
                        p1_list.append(np.copy(self.p1_field['g']))
                        self.p2_field.change_scales(1)
                        p2_list.append(np.copy(self.p2_field['g']))
                        t_list.append(solver.sim_time)
                        if np.max(self.p1_field['g']) > 100:
                            warnings.warn("Solution instable")
                            break
            except:
                logger.error(
                    'Exception raised, triggering end of main loop.')
                raise

            p1_array = np.array(p1_list)
            p2_array = np.array(p2_list)
            sol = np.concatenate(
                (p1_array[:, :, None], p2_array[:, :, None]),
                axis=2)

        return sol

    def eval_at_sensor_positions(self, sol, pos):

        temp = []

        for i in range(len(pos)):
            sol_int = sol(x=pos[i])
            temp.append(float(sol_int.evaluate()['g']))

        return np.array(temp)

    def gauss_peak(self, H_pos):
        """
        Generates Gauss peaks at sensor positions.

        Parameters
        ----------
        H_pos : numpy array
            Water height at sensor positions, shape (self.N, self.M).

        Returns
        -------
        H_gauss : numpy array
            Gauss peaks at sensor positions. The height of the peaks is the
            previous value at the positions.

        """

        x = self.dist.local_grid(self.xbasis)
        H_gauss = np.zeros(H_pos.shape)
        for p in range(len(self.pos)):
            i = np.argmin(abs(x-self.pos[p]))
            for n in range(H_gauss.shape[0]):
                H_gauss[n, i-2:i+3] += H_pos[n, i] \
                    * np.exp(-((x[i-2:i+3]-x[i])/0.3)**2)

        return H_gauss
