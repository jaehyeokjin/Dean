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
num_files = 3000
folder =  'fine-crystal2' 
for k in range(2500,num_files):
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

for i in range(len(Density_Profiles)):
    d_tf = np.fft.fft2(Density_Profiles[i])
    density_pectrum = density_pectrum + np.abs(d_tf)**2

S_k = density_pectrum / len(Density_Profiles)
center_f = np.zeros([sysL, sysL])
center_i,  center_j = int(sysL/2), int(sysL/2)
center_f[center_i, center_j] = 1.0
center_tf = np.fft.fft2(center_f)
dnsty_dnsty_crltn = np.fft.ifft2(S_k*center_tf)/sysL**2
S_k_center = np.fft.ifft2(np.fft.fft2(S_k)*center_tf)

g_vector_r = np.fft.ifft2(((S_k - 1.0))*center_tf)

# radial_correlation 
max_R = sysL-1-int(sysL/2) + 0.5
min_R = -0.5
num_rpts = 10
R_bounds = np.linspace(min_R, max_R, num_rpts+1)
Rs = (R_bounds[:-1]+R_bounds[1:])*0.5
Rs = Rs * L_phys/sysL
dR = np.diff(R_bounds)
color_map = np.ones([sysL, sysL])*(-10)
for i in range(sysL):
    for j in range(sysL):
        dst = sqrt((i-center_i)**2+(j-center_j)**2)
        color = int(round(np.sum(1.0*(dst>R_bounds))-1))
        color_map[i,j] = color
# plt.imshow(color_map)

R_correlation = []
R2_correlation = []
R3=[]
rho_zero = 1/L_phys**2
for i in range(num_rpts):
    R_correlation.append(np.sum((color_map==i)*dnsty_dnsty_crltn.real)/dR[i])
    R3.append(np.sum((color_map==i)*dnsty_dnsty_crltn.real)/np.sum((color_map==i))/rho_zero**2)
    R2_correlation.append(np.sum((color_map==i)*S_k_center.real)/np.sum((color_map==i)))
R_correlation = np.array(R_correlation)
R2_correlation = np.array(R2_correlation)
R2_correlation_final=np.insert(R2_correlation,0,0.0)
Rs2 = (R_bounds[:-1]+R_bounds[1:])*0.5
Rs2_final=np.insert(Rs2,0,0.0)

#plt.figure()
#plt.title(folder + r': Radia Correlation $r\int d\theta C_{\rho}(r\hat{e}_\theta)$')
#plt.plot(Rs, R_correlation, 'bo--', ms=10, mew=2, mfc='none')
#plt.savefig('correl/Radial.png')
#np.savetxt('correl/radial.txt',np.c_[Rs,R_correlation])
#radial = np.load('correl/radial.npz')
#print radial

plt.figure()
plt.title(folder + r': Averaged Correlation $r\int d\theta C_{\rho}(r\hat{e}_\theta)$')
plt.plot(Rs, R3, 'bo--', ms=10, mew=2, mfc='none')
plt.savefig('correl/rdf.png')
np.savetxt('correl/rdf.txt',np.c_[Rs,R3])


#plt.figure()
#plt.title(folder + r': Sk norm $r\int d\theta C_{\rho}(r\hat{e}_\theta)$')
#plt.plot(Rs2_final, R2_correlation_final, 'bo--', ms=10, mew=2, mfc='none')
#plt.savefig('correl/Sk.png')
#np.savetxt('correl/sk.txt',np.c_[Rs2_final,R2_correlation_final])
#print(Rs2)
#print(R2_correlation)

# fig = plt.figure(figsize=(10,5))
# ax1 = fig.add_axes([0.03, 0.05, 0.45, 0.9])
# ax2 = fig.add_axes([0.03+0.45+0.03, 0.05, 0.45, 0.9])
# # ax1.imshow(density_pectrum)
# ax2.imshow(dnsty_dnsty_crltn.real)
# ax1.imshow(dnsty_dnsty_crltn.imag)

#plt.figure()
#plt.title('{} Sk'.format(folder))
#plt.imshow(S_k)
#plt.savefig('correl/2dSk.png')
#plt.colorbar()
#
#plt.figure()
#plt.title('{} Sk_center'.format(folder))
#plt.imshow(S_k_center.real)
#plt.savefig('correl/2dSk_center.png')
#plt.colorbar()
#
#plt.figure()
#plt.title(folder + r': $C_{\rho}(x,y)$ (real)')
#plt.imshow(dnsty_dnsty_crltn.real)
#plt.savefig('correl/RealCR.png')
#plt.colorbar()
# plt.figure()
# plt.title('H imag')
# plt.imshow(dnsty_dnsty_crltn.imag)
# plt.colorbar()

# plt.figure()
# plt.title('g_vector real')
# plt.imshow(g_vector_r.real)
# plt.colorbar()
# plt.figure()
# plt.title('g_vector imag')
# plt.imshow(g_vector_r.imag)
# plt.colorbar()

# plt.figure()
#for i in range(len(Density_Profiles)):
#     plt.title('density evolution t.{:d}'.format(i))
#     plt.imshow(Density_Profiles[i])
#     name_fig = './image/'+str(i)+'.png'
#     plt.savefig(name_fig)
#     plt.pause(0.001)
#     plt.clf()
