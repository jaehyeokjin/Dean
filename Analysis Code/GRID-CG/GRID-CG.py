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
from scipy.optimize import curve_fit
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
#import meshio

plt.ion()

Deal_L = 2*np.pi
Rescale = 0

info_line_num = 9
N_particle = 400
line_num_shapshot = info_line_num + N_particle

file_name = 'HighT-GCM_Liquid.lammpstrj' #'Hightemp-trim.lammpstrj' # 'mscg_nvt.lammpstrj'
data_path = '/Users/chenliu/Research_Projects/DeanDynamics/DATA/'
file_path = os.path.join(data_path, file_name)


Box_len = 30
linear_bins = int(sys.argv[1]) # 20 before 10 original

Vo=3.17799765
sigma_md = 1.87500
md_cutoff = 15.0
Gaussian_Energy = lambda dist: np.exp(-0.5*dist**2/sigma_md**2)*Vo
x_edges = np.linspace(0, Box_len, linear_bins+1)
y_edges = np.linspace(0, Box_len, linear_bins+1)
ds = np.diff(x_edges)[0]**2
dx = np.diff(x_edges)[0]

print('loading ....')
md_trj_file = open("../trim.lammpstrj",'r')
file_lines = md_trj_file.readlines()
md_trj_file.close()


print('constructing ....')
# H, xedg, yedg = np.histogram2d(X_cor, Y_cor, bins=linear_bins)
Num_Snapshots = int(len(file_lines)/(line_num_shapshot))
Num_UseSnap = 10000
starting_snap = Num_Snapshots - Num_UseSnap
Snapshot_Times = []
Snapshot_Configs = []
Density_Profiles = []
Coefficient_k = []
for i in range(Num_UseSnap):
    print('i.{0:d} / {1:d}'.format(i, Num_Snapshots))
    data_snap = file_lines[(i+starting_snap)*line_num_shapshot: (i+starting_snap+1)*line_num_shapshot]
    X_cor = []
    Y_cor = []
    for j in range(line_num_shapshot):
        if j < info_line_num:
            if j == 1:
                Snapshot_Times.append(float(data_snap[j]))
        else:
            particle_id = j - info_line_num+1
            the_line = data_snap[j].split()
            atom_id = int(the_line[0])
            atom_type = int(the_line[1])
            # x_cor = float(the_line[2])
            # y_cor = float(the_line[3])
            X_cor.append(float(the_line[2]))
            Y_cor.append(float(the_line[3]))
            if particle_id!= atom_id:
                print(particle_id, ' vs ', atom_id)
    X_cor = np.array(X_cor)
    Y_cor = np.array(Y_cor)
    H, xedg, yedg = np.histogram2d(X_cor, Y_cor, bins=[x_edges, y_edges])
    rho = H/N_particle**0/ds
    Snapshot_Configs.append({'x':X_cor , 'y': Y_cor})
    Density_Profiles.append(rho)
    Coefficient_k.append(np.abs(np.fft.fft2(rho))**2)

Coefficient_k = np.array(Coefficient_k)

D=[]
# Define the r-space distance metric
for i in range(0,linear_bins):
    for j in range(0,linear_bins):
        d_value = (i*i+j*j)**0.5 * dx
        D.append(d_value)
        #if (d_value <= Box_len/2.0):
        #    D.append(d_value)
D = np.array(D)
D = np.unique(D)

finD = D
finE = []
del file_lines

colors=cm.rainbow(np.linspace(0,1,Num_UseSnap))
plt.figure()

print('energy ... ')
Energy_Mirco=[]
for i in range(Num_UseSnap):
    if i % 50 == 0:
        D_count = np.zeros(len(D))
        E_count = np.zeros(len(D))
        Config = Snapshot_Configs[i]
        E_total = 0
        E_step = []
        print(i, '/', Num_UseSnap)
        for j in range(N_particle):
            for k in range(N_particle):
                if j!=k:
                    x_1 = Config['x'][j]
                    y_1 = Config['y'][j]
                    x_2 = Config['x'][k]
                    y_2 = Config['y'][k]
                    dx_12 = abs(x_1 - x_2)*(abs(x_1 - x_2) <= Box_len/2.0)+(Box_len - abs(x_1 - x_2))*(abs(x_1 - x_2) > Box_len/2.0)
                    dy_12 = abs(y_1 - y_2)*(abs(y_1 - y_2) <= Box_len/2.0)+(Box_len - abs(y_1 - y_2))*(abs(y_1 - y_2) > Box_len/2.0)
                    d12 = (dx_12**2.0+dy_12**2.0)**0.5
                    #d12 = sqrt((x_1-x_2)**2+(y_1-y_2)**2)
                    #d12 = d12*(d12<Box_len/2) + (Box_len-d12)*(d12>Box_len/2)
                    if d12<=md_cutoff:
                        E_total = E_total + 0.5*Gaussian_Energy(d12)
                        i_1 = np.floor(x_1/dx)
                        i_2 = np.floor(x_2/dx)
                        j_1 = np.floor(y_1/dx)
                        j_2 = np.floor(y_2/dx)
                        di_12 = abs(i_1 - i_2)*(abs(i_1 - i_2) <= linear_bins/2.0)+(linear_bins - abs(i_1 - i_2))*(abs(i_1 - i_2) > linear_bins/2.0)
                        dj_12 = abs(j_1 - j_2)*(abs(j_1 - j_2) <= linear_bins/2.0)+(linear_bins - abs(j_1 - j_2))*(abs(j_1 - j_2) > linear_bins/2.0)
                        discrete_distance = dx*(di_12**2.0+dj_12**2.0)**0.5                  
                        #print(di_12,dj_12,dx_12,dy_12)
                        #discrete_distance_x = dx*abs(np.floor(x_1/dx)-np.floor(x_2/dx))*(dx*abs(np.floor(x_1/dx)-np.floor(x_2/dx))<=Box_len/2)
                        #discrete_distance_x += (Box_len-dx*abs(np.floor(x_1/dx)-np.floor(x_2/dx)))*(dx*abs(np.floor(x_1/dx)-np.floor(x_2/dx))>Box_len/2)
                        #discrete_distance_y = dx*abs(np.floor(y_1/dx)-np.floor(y_2/dx))*(dx*abs(np.floor(y_1/dx)-np.floor(y_2/dx))<=Box_len/2)
                        #discrete_distance_y += (Box_len-dx*abs(np.floor(y_1/dx)-np.floor(y_2/dx)))*(dx*abs(np.floor(y_1/dx)-np.floor(y_2/dx))>Box_len/2)
                        #discrete_distance = (discrete_distance_x**2+discrete_distance_y**2)*0.5
                        #print(discrete_distance, d12)
                        index = np.where(np.abs(D - discrete_distance)<1e-4)[0][0]
                        #index = np.where(D == discrete_distance)[0][0]
                        D_count[index] += 1.0
                        E_count[index] += 0.5*Gaussian_Energy(d12)
        for j in range(len(E_count)):
            if D_count[j] != 0: 
                finvalue = 2.0*E_count[j]/D_count[j]
            else:
                finvalue = 0.0
            E_step.append(finvalue)
        plt.plot(D,2.0*E_count/D_count,color=colors[i])
        finE.append(E_step)
        #Energy_Mirco.append(E_total)

#Average for each realization 
avgE = np.zeros(len(D_count))
for i in range(len(avgE)):
    cnt = 0.0
    val = 0.0
    for j in range(len(finE)):
        if finE[j][i] != 0.0:
            val += finE[j][i]
            cnt += 1.0
    if cnt != 0.0:
        avgE[i] = val/cnt
    else:
        avgE[i] = 0.0
plt.plot(D,avgE,'ro--')
plt.plot(D,Gaussian_Energy(D),'b')
#plt.figure()

#Curve fit
#def func(x,av,a,bv,b):
#    return av* np.exp(-(x/a)**2) + bv*np.exp(-(x/b)**2)
#popt, pconv = curve_fit(func,D,avgE)
#plt.plot(D,func(D,*popt),'r-')
#plt.plot(D,avgE,'b')


Energy_Mirco = np.array(Energy_Mirco)
print('energy done ! ')
plt.plot(D, Gaussian_Energy(D),'ro--')

outname = "./avg_e_"+str(linear_bins)+".out"
np.savetxt(outname, np.c_[D,avgE])


#plt.figure()
#plt.plot(D,E_count/D_count)
#plt.plot(D, Gaussian_Energy(D),'ro--')


'''
print('construct M ...')
M = np.zeros([linear_bins, linear_bins, linear_bins, linear_bins])
for i in range(linear_bins):
    for j in range(linear_bins):
        for k in range(linear_bins):
            for l in range(linear_bins):
                M[i,j,k,l] = np.sum( Coefficient_k[:, i,j]*Coefficient_k[:, k,l]*0.5)/Num_UseSnap
print('construct M done !')

print('construct B ...')
B = np.zeros([linear_bins, linear_bins])
for i in range(linear_bins):
    for j in range(linear_bins):
        B[i,j] = np.sum(Coefficient_k[:,i,j]*Energy_Mirco)/Num_UseSnap
print('construct B done !')

Vk = np.linalg.tensorsolve(M, B)

delta=np.zeros([linear_bins, linear_bins])
delta[int(linear_bins/2), int(linear_bins/2)] = 1
delta_tf = np.fft.fft2(delta)

Vr = np.fft.ifft2(delta_tf*Vk)

plt.figure()
plt.imshow(Vr.real)
plt.colorbar()


# radial_correlation
sysL = linear_bins
center_i,  center_j = int(sysL/2), int(sysL/2)
max_R = sysL-1-int(sysL/2) + 0.5
min_R = -0.5
num_rpts = 15
R_bounds = np.linspace(min_R, max_R, num_rpts+1)
Rs = ((R_bounds[:-1]+R_bounds[1:])*0.5)
Rs = Rs/np.max(Rs)*Box_len/sigma_md
dR = np.diff(R_bounds)
color_map = np.ones([sysL, sysL])*(-10)
for i in range(sysL):
    for j in range(sysL):
        dst = sqrt((i-center_i)**2+(j-center_j)**2)
        color = int(round(np.sum(1.0*(dst>R_bounds))-1))
        color_map[i,j] = color
# plt.imshow(color_map)

R_correlation = []
for i in range(num_rpts):
    R_correlation.append(np.sum((color_map==i)*Vr.real)/np.sum((color_map==i))/dR[i])
R_correlation = np.array(R_correlation)

plt.figure()
# plt.title(folder + r': Radia Correlation $r\int d\theta C_{\rho}(r\hat{e}_\theta)$')
plt.plot(Rs, R_correlation, 'bo--', ms=10, mew=2, mfc='none')
'''


