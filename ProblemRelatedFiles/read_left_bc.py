import numpy as np
from scipy.interpolate import CubicSpline


class leftbc(object):

    def __init__(self, filename):
        heights = []
        time    = 0
        with open(filename, 'r') as f:
            for line in f.readlines():
                fields = line.split()
                heights.append(float(fields[0])/100) # convert from cm to m
                time += 0.01 # sampling rate in seconds
        nt =  np.shape(heights)[0]
        taxis = np.linspace(0.0, time, nt)
        self.final_time = time
        self.f = CubicSpline(taxis, heights)


class data(object):

    def __init__(self, filename):
        heights1 = []
        heights2 = []
        heights3 = []
        time    = 0
        with open(filename, 'r') as f:
            for line in f.readlines():
                fields = line.split()
                heights1.append(float(fields[1])/100) # convert from cm to m
                heights2.append(float(fields[2])/100) # convert from cm to m
                heights3.append(float(fields[3])/100) # convert from cm to m
                time += 0.01 # sampling rate in seconds

        nt =  np.shape(heights1)[0]
        heights = np.zeros((nt,3))
        heights[:,0] = heights1
        heights[:,1] = heights2
        heights[:,2] = heights3
        taxis = np.linspace(0.0, time, nt)
        self.final_time = time
        self.f = []
        for j in range(3):
            self.f.append(CubicSpline(taxis, heights[:,j]))
