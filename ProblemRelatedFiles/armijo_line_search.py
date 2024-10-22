#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:43:23 2024

@author: cja8741
"""
import numpy as np


class Armijo:

    def __init__(self, alpha, beta, comm):

        self.alpha = alpha
        self.alpha_j = alpha
        self.beta = beta
        self.rank = comm.Get_rank()

    def line_search(self, P, b, d, f_0, v):

        breaker = False

        if self.alpha_j < self.alpha:
            self.alpha_j *= 2

        f_new = P.f(b - self.alpha_j*d)
        grad_new = P.compute_gradient(b - self.alpha_j*d)

        # --- Backtracking line search with Armijo rule ---

        while f_new > f_0 + 1e-5*self.alpha_j*P.dx*P.dt*v.T@d \
                or np.isnan(f_new):

            self.alpha_j *= self.beta
            if self.alpha_j < 1e-10:
                breaker = True
                break
            f_new = P.f(b - self.alpha_j*d)
            grad_new = P.compute_gradient(b - self.alpha_j*d)

        return self.alpha_j, f_new, grad_new, breaker
