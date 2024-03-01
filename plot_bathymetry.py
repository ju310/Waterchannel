#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:46:44 2024.

@author: Judith Angel
Plot bathymetry with Dedalus.
"""
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from subprocess import call
import dedalus.public as d3

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

xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=np.float64)
xbasis = d3.Chebyshev(xcoord, size=80, bounds=(1.5, 15), dealias=3/2)
x = dist.local_grid(xbasis)
x_mm = x*1000 - 3412.5  # Put to millimeters.
y = rampFunc(x)*1000  # Put to millimeters.

start = np.argmin(abs(x_mm-0))
stop = np.argmin(abs(x_mm-1175))

plt.figure()
plt.plot(x_mm[start:stop+1], y[start:stop+1], "k")
plt.xlabel("x [mm]")
plt.ylabel("b(x) [mm]")
plt.ylim([0, 205])
ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
filename = "bathymetry_dedalus.pdf"
# plt.savefig(filename)
# call(["pdfcrop", filename, filename])
