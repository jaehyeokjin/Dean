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
import meshio

plt.ion()

Deal_L = 2*np.pi
Rescale = 0

info_line_num = 9
N_particle = 400
line_num_shapshot = info_line_num + N_particle

Box_len = 30
interaction_grid = int(sys.argv[1])
linear_bins = interaction_grid

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
    #print('i.{0:d} / {1:d}'.format(i, Num_Snapshots))
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
    rho = H/N_particle**1/ds
    rho2= H/ds *(Box_len/(2*pi))**2
    Snapshot_Configs.append({'x':X_cor , 'y': Y_cor})
    Density_Profiles.append(rho)
    Coefficient_k.append(np.abs(np.fft.fft2(rho))**2)

# Settings for parameters
# Perparticle interaction parameters
Vo=3.17799765
sigma_md = 1.87500
md_cutoff = 15.0
Gaussian_Energy = lambda dist: np.exp(-0.5*dist**2/sigma_md**2)*Vo
#Vo=39.7249706
#sigma_md = 3.00
#md_cutoff = Box_len/2.0
#Gaussian_Energy = lambda dist: np.exp(-0.5*dist**2/sigma_md**2)*Vo 

# CG interaction from the first code
#sigma_scaled = 2.12132034/Box_len*2*pi
scale_factor = 2*pi/Box_len

#General double exponential form
v0_a = 3.17799765
length_a = 1.87500
u_r_function = lambda r: v0_a*exp(-0.5*(r/length_a)**2)

# CG (microscopic -> Field) details
gaussian_factor = linear_bins/(2*interaction_grid)
width = Deal_L/linear_bins*gaussian_factor #Original was 2
Ggaussian = lambda x, y: 1/(2*pi*width**2)*np.exp(-0.5*(x**2+y**2)/width**2)
#gauss_bins = int(linear_bins/gaussian_factor)
#gauss_x_edges = np.linspace(0, Box_len, gauss_bins+1)
#gauss_y_edges = np.linspace(0, Box_len, gauss_bins+1)

field_energy=[]
particle_energy=[]

for t in range(Num_UseSnap):
    if t % 50 == 0:
        str_in = "Timestep: %d/%d started" %(t, Num_UseSnap)
        print(str_in)
        field = np.zeros([linear_bins, linear_bins])
        x_coordinate = np.linspace(0, Deal_L, linear_bins+1)
        y_coordinate = np.linspace(0, Deal_L, linear_bins+1)
        x_coordinate = (x_coordinate[0:-1]+x_coordinate[1:])*0.5
        y_coordinate = (y_coordinate[0:-1]+y_coordinate[1:])*0.5
        x_mesh, y_mesh = np.meshgrid(x_coordinate,y_coordinate)
        E_total = 0 # Per-particle MD Energy
        print("Calculate")
        for i in range(N_particle):
            x_value = Snapshot_Configs[t]['x'][i]*Deal_L/Box_len
            y_value = Snapshot_Configs[t]['y'][i]*Deal_L/Box_len
            # Generate CG field
            ddx = (x_mesh - x_value)
            ddx = ddx*(abs(ddx)<Deal_L/2)-np.sign(ddx)*(Deal_L-abs(ddx))*(abs(ddx)>=Deal_L/2)
            ddy = (y_mesh - y_value)
            ddy = ddy*(abs(ddy)<Deal_L/2)-np.sign(ddy)*(Deal_L-abs(ddy))*(abs(ddy)>=Deal_L/2)
            field += Ggaussian(ddx,ddy)
            # Now calculate per particle energy
            for j in range(N_particle):
                if i != j:
                    x_1 = Snapshot_Configs[t]['x'][i]
                    y_1 = Snapshot_Configs[t]['y'][i]
                    x_2 = Snapshot_Configs[t]['x'][j]
                    y_2 = Snapshot_Configs[t]['y'][j]
                    dx_12 = abs(x_1 - x_2)*(abs(x_1 - x_2) <= Box_len/2.0)+(Box_len - abs(x_1 - x_2))*(abs(x_1 - x_2) > Box_len/2.0)
                    dy_12 = abs(y_1 - y_2)*(abs(y_1 - y_2) <= Box_len/2.0)+(Box_len - abs(y_1 - y_2))*(abs(y_1 - y_2) > Box_len/2.0)
                    d12 = (dx_12**2.0+dy_12**2.0)**0.5
                    if d12 <= md_cutoff:
                        E_total = E_total + 0.5*Gaussian_Energy(d12)

        # Field Energy 
        x_mesh_new = x_mesh*(abs(x_mesh)<Deal_L/2)-np.sign(x_mesh)*(Deal_L-abs(x_mesh))*(abs(x_mesh)>=Deal_L/2)
        y_mesh_new = y_mesh*(abs(y_mesh)<Deal_L/2)-np.sign(y_mesh)*(Deal_L-abs(y_mesh))*(abs(y_mesh)>=Deal_L/2)
        u_r = u_r_function((x_mesh_new**2+y_mesh_new**2)**0.5)
        u_k = np.fft.fft2(u_r)
        rho_k = np.fft.fft2(rho2) #*ds*(2*pi/Box_len)**2 Rescale ds to match the total number of particle
        u = 0.5*np.sum(u_k*np.abs(rho_k)**2)

        # Scale the Fourier energy by the factor of 1/N_l^2 and consider ds^2 
        u = u.real * (ds*(2*pi/Box_len)**2)**2/(linear_bins)**2
        # Deduct the self energy
        self_energy = N_particle/2 * u_r_function(0) # For particle description
        #gauss_H, gauss_xedg, gauss_yedg = np.histogram2d(Snapshot_Configs[t]['x'], Snapshot_Configs[t]['y'], bins=[gauss_x_edges, gauss_y_edges])
        #field_zeros = (gauss_H !=0).sum()
        #self_energy = field_zeros/2 * u_r_function(0) # For particle description
        u = u - self_energy
        # Print out the energy
        field_energy.append(u)
        particle_energy.append(E_total)
        Str_out = "Particle E: %.6f\nField E: %.6f %.6f %.6f" %(E_total, u.real, self_energy, u.real+self_energy)
        print(Str_out)
field_energy = np.array(field_energy)
particle_energy = np.array(particle_energy)

print("Done! Now computing the average")
field_average = np.mean(field_energy)
particle_average = np.average(particle_energy)
str_out_2 = "Particle E: %6f Field E: %6f" %(particle_average, field_average)
outname = "./grid-energy-results/energy_"+str(interaction_grid)+".txt"
np.savetxt(outname,c_[field_energy, particle_energy])
