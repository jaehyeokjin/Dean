import numpy as np
import math as ma
from numpy import *
from pylab import *
from scipy import *
import os.path
from matplotlib import rc, rcParams
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import fftpack as ftp

from scipy import *
from scipy.optimize import curve_fit, leastsq

from scipy import interpolate as interplt
from scipy.interpolate import interp1d
import sys
from datetime import datetime
import os.path
import matplotlib.ticker as ticker
import imp
import meshio

plt.ion()
#plt.ioff() 

sysL=20
L_phys = 2*pi
elementary_area = (2*pi/sysL)**2
rho_phys = 1/L_phys**2
Density_Profiles = []
num_files = 3999
folder =  'fine-crystal2' 
for k in range(num_files):
    print(k, '/', num_files)
    mh = meshio.read('./80_{0:d}.vtu'.format(k+1))
    pts = mh.points
    vls = mh.point_data['function_7']

    X = []
    Y = []
    for i in range(len(pts)):
        if pts[i,0] not in X:
            X.append(pts[i,0])
        if pts[i,1] not in Y:
            Y.append(pts[i,1])
    X.sort()
    Y.sort()
    X = np.array(X)
    Y = np.array(Y)

    # Xp=[X[0]]
    # Yp=[Y[0]]
    # for i in range(len(X)-1):
    #     if abs(X[i+1]-X[i])>0.00001:
    #         Xp.append(X[i+1])
    # for i in range(len(Y)-1):
    #     if abs(Y[i+1]-Y[i])>0.00001:
    #         Yp.append(Y[i+1])

    dX = np.diff(X)
    dY = np.diff(Y)

    dx = np.max(dX)
    dy = np.max(dY)

    # fild=np.zeros([sysL+1, sysL+1])
    # Box = np.ones([81, 81, 6])*(-0.5)
    d_field = np.zeros([sysL+1, sysL+1])
    for i in range(len(pts)):
        x = pts[i,0]
        y = pts[i,1]
        v = vls[i]
        ix = int(round(x/dx))
        iy = int(round(y/dy))
        # Box[ix, iy, int(fild[ix, iy])] = v
        # fild[ix, iy] = fild[ix, iy] + 1
        d_field[ix, iy] = v #*elementary_area
    
    Density_Profiles.append(d_field[:-1, :-1]) #-np.mean(d_field[:-1, :-1]))
    # print(np.mean(d_field[:-1, :-1])*elementary_area, ' vs ', rho_phys)

fluc = []
for i in range(len(Density_Profiles)):
    rho_avg = np.mean(Density_Profiles[i])
    rho_sq = np.mean(Density_Profiles[i]**2)
    fluctuation = rho_sq - rho_avg**2
    fluc.append(fluctuation)
np.savetxt('./fluc.txt',np.c_[fluc])
#plt.figure()
#for i in range(len(Density_Profiles)):
##     plt.title('density evolution t.{:d}'.format(i))
#     plt.imshow(Density_Profiles[i])
#     plt.colorbar()
#     name_fig = './image/'+str(i)+'.png'
#     plt.savefig(name_fig)
#     plt.pause(0.0001)
#     plt.clf()
