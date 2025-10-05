import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

linear_bins = 50
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
    plt.plot(time*abs_k[i], F_abs_kt[i], lw=1, color=clrs[i], label='k={:.f}'.format(abs_k[i]))
