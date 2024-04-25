#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:36:28 2023.
Auxiliary file for gradient_descent.py. Defines the optimisation problem,
objective functional and gradient.

@author: Judith Angel
"""
import numpy as np
import dedalus.public as d3
from scipy import integrate as intgr
from ProblemRelatedFiles.problems import nonlinSWE


class OptProblem:

    def __init__(self, pa, save=False):

        self.T_N = pa.T_N
        self.N = int(pa.T_N/pa.dt) + 1
        self.lambd = pa.lambd
        self.gamma = pa.gamma
        self.delta = pa.delta
        self.t_array = pa.t_array
        self.dt = self.t_array[1]-self.t_array[0]
        self.y_d = pa.y_d
        self.xmin = pa.xmin
        self.xmax = pa.xmax
        self.M = pa.M
        self.dx = (self.xmax-self.xmin)/(self.M-1)
        self.g = pa.g
        self.PDE = nonlinSWE(pa, self.t_array, self.y_d, save)
        self.b_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        self.bx_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        self.bxx_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        self.p2_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        self.p2x_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        self.h_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        self.u_field = self.PDE.dist.Field(bases=self.PDE.xbasis)
        if hasattr(pa, "lambda_b"):
            self.lambda_b = pa.lambda_b

    def f(self, control):
        """
        Objective functional.

        Parameters
        ----------
        control : numpy array
            Control variable.

        Returns
        -------
        float
            Value of objective functional for given control.

        """
        self.y = self.PDE.solvepde(control, "primal")[:, :, 0]
        self.y += np.tile(control, (self.y.shape[0], 1))

        self.b_field.change_scales(1)
        self.b_field['g'] = np.copy(control)
        self.bx_field.change_scales(3/2)
        self.bx_field["g"] = d3.Differentiate(
            self.b_field, self.PDE.xcoord).evaluate()['g']
        self.bx_field.change_scales(1)
        b_x = np.copy(self.bx_field["g"])

        if self.PDE.data != "sim_everywhere":
            self.mismatch = self.PDE.gauss_peak(self.PDE.H_array-self.y_d)
        else:
            self.mismatch = self.y-self.y_d

        self.f_err1 = self.gamma*0.5*self.dx*self.dt \
            * np.linalg.norm(self.mismatch)**2
        self.f_err2 = self.delta*0.5*self.dx \
            * np.linalg.norm(self.mismatch[-1])**2
        self.f_reg = 0.5*self.dx*self.lambd*np.linalg.norm(b_x)**2

        if hasattr(self, "lambda_b"):
            self.f_reg += 0.5*self.dx*self.lambda_b*np.linalg.norm(control)**2

        val = self.f_err1 + self.f_err2 + self.f_reg

        return val

    def compute_gradient(self, control):
        """
        Compute the gradient determined by the adjoint and the control.

        Parameters
        ----------
        control : numpy array
            Current control.

        Returns
        -------
        numpy array
            Reduced gradient of the cost functional.

        """
        self.q = self.PDE.solvepde(control, "primal")
        p = np.flipud(self.PDE.solvepde(self.q, "adjoint"))
        self.p = p.copy()

        if self.PDE.data != "sim_everywhere":
            self.mismatch = self.PDE.gauss_peak(
                self.PDE.H_array-self.y_d)
        else:
            self.y = np.copy(self.q[:, :, 0])
            self.y += np.tile(control, (self.y.shape[0], 1))
            self.mismatch = self.y-self.y_d

        p2_x = np.zeros((self.p.shape[0], self.p.shape[1]))
        h_x = np.zeros((self.q.shape[0], self.q.shape[1]))

        for n in range(self.p.shape[0]):

            self.p2_field.change_scales(1)
            self.p2_field['g'] = self.p[n, :, 1]
            self.p2x_field.change_scales(3/2)
            self.p2x_field["g"] = d3.Differentiate(
                self.p2_field, self.PDE.xcoord).evaluate()["g"]
            self.p2x_field.change_scales(1)
            p2_x[n, :] = self.p2x_field["g"]
            self.h_field.change_scales(1)
            self.h_field['g'] = self.q[n, :, 0]
            self.h_field["g"] = d3.Differentiate(
                self.h_field, self.PDE.xcoord).evaluate()['g']
            self.h_field.change_scales(1)
            h_x[n] = self.h_field["g"]
            self.u_field.change_scales(1)
            self.u_field['g'] = self.q[n, :, 1]

        self.b_field.change_scales(1)
        self.b_field['g'] = control
        self.bx_field.change_scales(3/2)
        self.bx_field["g"] = d3.Differentiate(
            self.b_field, self.PDE.xcoord).evaluate()['g']
        self.bx_field.change_scales(1)
        self.bxx_field.change_scales(3/2)
        self.bxx_field["g"] = d3.Differentiate(
            self.bx_field, self.PDE.xcoord).evaluate()['g']
        self.bxx_field.change_scales(1)
        b_xx = np.copy(self.bxx_field["g"])

        # Use mismatch from evaluation of objective functional.
        self.v_e = self.g*intgr.simpson(p2_x, dx=self.dt, axis=0) \
            - self.p[0, :, 0]
        self.v_J = self.gamma*intgr.simpson(
            self.mismatch, dx=self.dt, axis=0) \
            + self.mismatch[-1]*self.delta \
            - self.lambd*b_xx

        if hasattr(self, "lambda_b"):
            self.v_J += self.lambda_b*self.PDE.current_b

        self.L2grad = self.v_e + self.v_J

        # Fields
        v = self.PDE.dist.Field(name='v', bases=self.PDE.xbasis)
        vTilde = self.PDE.dist.Field(bases=self.PDE.xbasis)
        tauv = self.PDE.dist.Field()
        tauv2 = self.PDE.dist.Field()

        # Substitutions
        dx = lambda A: d3.Differentiate(A, self.PDE.xcoord)

        vTilde.change_scales(1)
        vTilde['g'] = self.L2grad
        vx = dx(v) + self.PDE.lift(tauv, -1)
        vxx = dx(vx) + self.PDE.lift(tauv2, -1)

        # Problem
        problem = d3.LBVP(
            [v, tauv, tauv2],
            namespace={
                "v": v, "dx": dx, "vTilde": vTilde,  "tauv": tauv,
                "tauv2": tauv2, "lift": self.PDE.lift, "vx": vx, "vxx": vxx})
        problem.add_equation("-vxx + v = vTilde")
        problem.add_equation("v(x='left') =  0")
        problem.add_equation("v(x='right') = 0")

        # Solver
        solver = problem.build_solver()
        solver.solve()

        # Gather global data
        grad = v.allgather_data('g')

        return grad
