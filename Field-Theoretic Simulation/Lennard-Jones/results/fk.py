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
num_files = 1999
folder =  'fine-crystal2' 
for k in range(999,num_files):
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

# plt.imshow(Density_Profiles[10], cmap=cm.coolwarm, interpolation='quadric') 

density_pectrum = np.zeros([sysL, sysL])
Density_Profiles_np = np.array(Density_Profiles)
Density_Profiles_tf = np.fft.fftn(Density_Profiles_np)
Power_spec = np.abs(Density_Profiles_tf)**2.0/(len(Density_Profiles)*sysL**2)

f_kt = np.zeros([sysL,sysL,len(Density_Profiles)])

for i in range(sysL):
    for j in range(sysL):
        f_kt[i,j] = np.fft.ifftn(Power_spec[:,i,j])

linear_bins = 20
max_R = linear_bins-1-int(linear_bins/2) + 0.5
min_R = -0.5
num_rpts = 10
R_bounds = np.linspace(min_R, max_R, num_rpts+1)
abs_k = (R_bounds[:-1]+R_bounds[1:])*0.5

dR = np.diff(R_bounds)
color_map = np.ones([linear_bins, linear_bins])*(-10)
for i in range(linear_bins):
    for j in range(linear_bins):
        new_i = i*(i<=linear_bins/2) + (i-linear_bins)*(i>linear_bins/2)
        new_j = j*(j<=linear_bins/2) + (j-linear_bins)*(j>linear_bins/2)
        dst = (new_i**2+new_j**2)**0.5
        color = int(round(np.sum(1.0*(dst>R_bounds))-1))
        color_map[i,j] = color
plt.imshow(color_map)

# sys.exit()

F_abs_kt = []
total_steps = int(len(Density_Profiles)/2)
for i in range(num_rpts):
    ft= []
    for j in range(total_steps):
        fk_fild = f_kt[:,:,j]
        ft.append( np.sum((color_map==i)*fk_fild)/np.sum(color_map==i) )
    F_abs_kt.append(ft)
F_abs_kt=np.array(F_abs_kt)

clrs = cm.rainbow(np.linspace(0,1,len(abs_k)))
time = np.arange(total_steps)
plt.figure()
for i in range(len(abs_k)):
    plt.plot(time*abs_k[i], F_abs_kt[i]/F_abs_kt[i,0], lw=1, color=clrs[i], label='k={:.1f}'.format(abs_k[i]))
    outname1 = './fk/abs_md_'+str(i)+'.out'
    outname2 = './fk/md_'+str(i)+'.out'
    np.savetxt(outname1,np.c_[F_abs_kt[i]/F_abs_kt[i,0]])
    np.savetxt(outname2,np.c_[F_abs_kt[i]])
