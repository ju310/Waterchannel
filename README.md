# Bathymetry Reconstruction in a Water Channel

## Description
This code reconstructs the bathymetry in a water channel using either simulated or experimental observation data. We use gradient descent to minimise an objective functional, where we compute the numerical solution of the continuous adjoint problem in order to determine the gradient. The forward problem is modelled by the nonlinear nonrotating shallow water equations and discretised with the spectral methods framework [Dedalus](https://dedalus-project.org/).

-------------
## Requirements
Installation of Dedalus version 3.0.0 is required.
