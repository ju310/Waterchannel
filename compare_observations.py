#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:34:44 2023.
Compare measurements of water heights in the water channel and plot those with
confidence intervals.

@author: Judith Angel
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation
from ProblemRelatedFiles.read_left_bc import data, leftbc

T_N = 42
H = 0.3
pos = [1.5, 3.5, 5.5, 7.5]
t_array = np.arange(0, T_N+0.01, 0.01)
allObsBathy = []
allObs = []

# Load measurement data.
path = "ProblemRelatedFiles/WaterchannelData/Comparison"

for i in range(20):

    postfix = f"try={i+1}"
    fileBathy = 'ProblemRelatedFiles/WaterchannelData/MitBathymetrie/' \
        + 'Tiefe=0,3_A=40_F=0,35_' + postfix + ".txt"
    dataObjectBathy = data(fileBathy)
    bcBathy = leftbc(fileBathy).f
    H_obsBathy = [H+bcBathy(t_array)]
    file = 'ProblemRelatedFiles/WaterchannelData/OhneBathymetrie/' \
        + 'Tiefe=0,3_A=40_F=0,35_' + postfix + ".txt"
    dataObject = data(file)
    bc = leftbc(file).f
    H_obs = [H+bc(t_array)]

    for p in range(len(pos)-1):

        H_obsBathy.append(H + dataObjectBathy.f[p](t_array))
        H_obs.append(H + dataObject.f[p](t_array))

    allObsBathy.append(H_obsBathy)
    allObs.append(H_obs)

allObsBathyArray = np.array(allObsBathy)
allObsArray = np.array(allObs)

# Load simulation of SWE with bathymetry.
sim_file = "ProblemRelatedFiles/WaterchannelData/" \
    + "sim_data_Tiefe=0,3_A=40_F=0,35_meanBathy_ExactRamp_T=13.hdf5"

with h5py.File(sim_file, "r") as sol:

    H_sensor = np.array(sol["H_sensor"][:])
    t_start_sim = sol.attrs.get("start")
    t_array_sim = np.array(sol["t_array"][:])

# Compute mean and standard deviation.
meanBathy = np.mean(allObsBathyArray, axis=0)
stdBathy = np.std(allObsBathyArray, axis=0)
mean = np.mean(allObsArray, axis=0)
std = np.std(allObsArray, axis=0)
stdDiff = np.sqrt(1/38*(19*std**2 + 19*stdBathy**2))
diff = mean - meanBathy
tValue19 = 1.729  # One-sided t value for 19 dof.
tValue38 = 1.686  # One-sided t value for 38 dof.
# https://www.statistik.tu-dortmund.de/fileadmin/user_upload/Lehrstuehle/Oekonometrie/Lehre/WiSoOekoSS17/tabelletV.pdf

# Confidence intervals
ciBathyLeft = meanBathy - stdBathy*tValue19/np.sqrt(20)
ciBathyRight = meanBathy + stdBathy*tValue19/np.sqrt(20)
ciLeft = mean - std*tValue19/np.sqrt(20)
ciRight = mean + std*tValue19/np.sqrt(20)
ciLeftDiff = mean - meanBathy - 1/np.sqrt(10)*tValue38*stdDiff
ciRightDiff = mean - meanBathy + 1/np.sqrt(10)*tValue38*stdDiff

# meanmax = np.max(meanBathy-mean)
start = np.argmin(abs(t_array-32))
end = np.argmin(abs(t_array-T_N))
start_sim = np.argmin(abs(t_array_sim-32+t_start_sim))
end_sim = np.argmin(abs(t_array_sim-T_N+t_start_sim))

# Plot simulation and mean of measurements with confidence interval.
for i in range(1, len(pos)):

    plt.figure()
    plt.plot(
        t_array_sim[start_sim:end_sim]+t_start_sim,
        H_sensor[start_sim:end_sim, i-1],
        "k:", label="simulation", linewidth=1)
    plt.plot(t_array[start:end], meanBathy[i, start:end], "k--",
              label="measurement", linewidth=1)
    ax = plt.gca()
    ax.fill_between(t_array[start:end], ciBathyLeft[i, start:end],
                    ciBathyRight[i, start:end], alpha=0.3, facecolor="k",
                    label="confidence interval")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('H [m]')
    plt.legend()
    plt.title(f"Sensor {i+1} at {pos[i]}m")
    plt.savefig(path + f"/sim_mean_bathy{i+1}.pdf")

plt.show()

# Plot difference of means with and without bathymetry with confidence interval.
for i in range(1, len(pos)):

    plt.figure()
    plt.plot(t_array[start:end], diff[i, start:end], "k--", linewidth=0.5,
             label=r"$H(b_{ex})-H(0)$")
    ax = plt.gca()
    ax.fill_between(t_array[start:end], ciLeftDiff[i, start:end],
                    ciRightDiff[i, start:end], alpha=0.3, facecolor="k",
                    label="confidence interval")
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('H [m]')
    plt.legend()
    plt.title(f"Sensor {i+1} at {pos[i]}m")
    plt.savefig(path + f"/confidence_diff{i+1}.pdf", bbox_inches='tight')

plt.show()

# --------------------- Write means into text files. --------------------------
# lines = np.swapaxes((meanBathy-H)*100, 0, 1)  # Subtract H and convert to cm.

# with open("ProblemRelatedFiles/WaterchannelData/MitBathymetrie/"
#           + "Tiefe=0,3_A=40_F=0,35_meanBathy.txt", "w") as f:

#     for line in lines:

#         f.write(np.array2string(line).replace('[', '').replace(']', ''))
#         f.write("\n")

# lines = np.swapaxes((mean-H)*100, 0, 1)  # Subtract H and convert to cm.

# with open("ProblemRelatedFiles/WaterchannelData/OhneBathymetrie/"
#           + "Tiefe=0,3_A=40_F=0,35_mean.txt", "w") as f:

#     for line in lines:

#         f.write(np.array2string(line).replace('[', '').replace(']', ''))
#         f.write("\n")

# with h5py.File(path + "/meanBathy.hdf5", "w") as f:

#     f.create_dataset("mean", data=meanBathy)
#     f.create_dataset("st_dev", data=stdBathy)
#     f.create_dataset("t_array", data=t_array)

# with h5py.File(path + "/meanNoBathy.hdf5", "w") as f:

#     f.create_dataset("mean", data=mean)
#     f.create_dataset("st_dev", data=std)
#     f.create_dataset("t_array", data=t_array)
