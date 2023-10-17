#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:30:26 2023

@author: cja8741
"""
import numpy as np
import math
import warnings


class CheckGradient:
    """Class for checking the gradient of some functional."""

    def __init__(self, f, grad, b):

        self.f = f
        self.grad = grad
        self.b = b
        self.delta_b = 0.01*np.linalg.norm(b)*np.ones(self.b.shape) \
            + 0.01*np.linalg.norm(grad)

    def check_order_2(self):
        r"""
        Check the gradient with Taylor remainder test.

        Check whether

        .. math::
            |J(b + h \delta b) - J(b) - h \nabla J(b) \delta b|

        is
        :math:`O(h^2)`.
        """
        it = 6
        err_log = np.zeros(it)
        self.order = np.zeros(it-1)

        for i in range(it):

            h = 0.01*2**(-i)
            fP = self.f(self.b+h*self.delta_b)
            fb = self.f(self.b)
            gradDeltab = np.inner(self.grad.flatten(order="F"),
                                  self.delta_b.flatten(order="F"))
            gradient_test = abs(fP - fb - h*gradDeltab)

            if gradient_test < 1e-12:
                warnings.warn("The taylor remainder is close to machine"
                              + " precision.")

            err_log[i] = math.log(gradient_test, 2)

        for i in range(it-1):

            self.order[i] = err_log[i] - err_log[i+1]

        if all(self.order > 2 - 0.5):

            return True

        else:

            return False


class CheckGradientWaterChannel:
    """Class for checking the gradient of some functional."""

    def __init__(self, f, grad, b):

        self.f = f
        self.grad = grad
        self.b = b
        self.delta_b = 0.01*np.linalg.norm(b)*np.ones(self.b.shape) \
            + 0.01*np.linalg.norm(grad)
        mask = b == 0
        self.delta_b[mask] = 0

    def check_order_2(self):
        r"""
        Check the gradient with Taylor remainder test.

        Check whether

        .. math::
            |J(b + h \delta b) - J(b) - h \nabla J(b) \delta b|

        is
        :math:`O(h^2)`.
        """
        it = 6
        err_log = np.zeros(it)
        self.order = np.zeros(it-1)

        for i in range(it):

            h = 0.01*2**(-i)
            fP = self.f(self.b+h*self.delta_b)
            fb = self.f(self.b)
            gradDeltab = np.inner(self.grad.flatten(order="F"),
                                  self.delta_b.flatten(order="F"))
            gradient_test = abs(fP - fb - h*gradDeltab)

            if gradient_test < 1e-12:
                warnings.warn("The taylor remainder is close to machine"
                              + " precision.")

            err_log[i] = math.log(gradient_test, 2)

        for i in range(it-1):

            self.order[i] = err_log[i] - err_log[i+1]

        if all(self.order > 2 - 0.5):

            return True

        else:

            return False
