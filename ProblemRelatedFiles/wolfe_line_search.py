#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:09:39 2024

@author: judith
"""


class strongWolfe:

    def __init__(self, c1, c2, alpha_max, i_max):

        self.c1 = c1
        self.c2 = c2
        self.alpha_max = alpha_max
        self.i_max = i_max

    def line_search(self, P, b, d, f_0, grad_0):

        self.P = P
        self.b = b.copy()
        self.d = d.copy()
        self.f_0 = f_0
        self.grad_0 = grad_0
        alpha_old = 0
        f_old = f_0
        alpha_new = 1
        self.breaker = False

        for i in range(1, self.i_max):

            f_new = P.f(b - alpha_new*d)

            if f_new > f_0 + self.c1*alpha_new*P.dx*P.dt*grad_0.T@d  \
                    or (f_new >= f_old and i > 1):

                alpha, f, grad = self.zoom(alpha_old, alpha_new, f_old)
                break

            grad_new = P.compute_gradient(b-alpha_new*d)

            if abs(grad_new.T@d) <= abs(self.c2*grad_0.T@d):

                alpha = alpha_new
                f = f_new
                grad = grad_new
                break

            if -grad_new.T@d >= 0:

                alpha, f, grad = self.zoom(alpha_new, alpha_old, f_new)
                break

            alpha_old = alpha_new
            alpha_new *= 2
            f_old = f_new

        if i == self.i_max-1:
            self.breaker = True

        return alpha, f, grad, self.breaker

    def zoom(self, alpha_lo, alpha_hi, f_lo):

        for j in range(self.i_max):

            alpha_j = 0.5*alpha_hi+0.5*alpha_lo
            f_new = self.P.f(self.b-alpha_j*self.d)

            if f_new > self.f_0  \
                + self.c1*alpha_j*self.P.dx*self.P.dt*self.grad_0.T@self.d \
                    or f_new >= f_lo:

                alpha_hi = alpha_j

            else:

                grad_j = self.P.compute_gradient(self.b-alpha_j*self.d)

                if abs(grad_j.T@self.d) <= abs(self.c2*self.grad_0.T@self.d):

                    alpha = alpha_j
                    break

                if -grad_j.T@self.d*(alpha_hi-alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha_j
                f_lo = f_new

            if j == self.i_max-1:
                self.breaker = True
                alpha = 0
                grad_j = np.zeros(self.b.shape)

        return alpha, f_new, grad_j
