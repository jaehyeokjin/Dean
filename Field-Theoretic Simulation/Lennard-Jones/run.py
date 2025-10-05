# Firedrake RIDK 2d
# ====================================
#
# RIDK equation
# on a periodic domain in 2d
#
# rho_t=-div j
# j_t = -gamma j - c grad rho -rho (grad V_pair * rho) - grad V_ext rho+ (sigma/root(N)) sqrt(rho) xi
#
# for white-in-time and spatially correlated noise xi with length scale epsilon and c=sigma^2/2 gamma,
# with parameters (gamma,sigma,N,epsilon)
# V_pair is a radial pair potential
# V_ext is an external potentl
#
import importlib
import os
import sys
import gc
sys.setrecursionlimit(10000)

#
if len(sys.argv)>1:
  run_case=int(sys.argv[1])
  print("run_case", run_case)
else:
  run_case=0
  print("Default params")
#############################
import matplotlib as mpl
from math import pi as pi
from math import sqrt as sqrt
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'lmodern'
mpl.rcParams['font.sans-serif'] = 'cm'
import matplotlib.pyplot as plt
import tikzplotlib
###########################
#
#export OMP_NUM_THREADS=1
import firedrake
#
import RIDK3d
from RIDK3d import *
importlib.reload(RIDK3d)

#
linear_solver={}
#linear_solver = {"ksp_type": "preonly", "pc_type": "lu"}
#linear_solver         = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
#
params_m = {  # model parameters
    "gamma": 1.00,
    "N": 400,
    "mass":1,
    "sigma": 100.0, #1.0 for old
    "reaction_rate":0.0,
    "Lx": 2.0 * pi,
    "Ly": 2.0 * pi,
    "epsilon": 0.15,
    "final_time": 1.0, # Original value is 1
    "initial_rho": lj,
    "initial_j": zero_j_2d,
    "pair": "6030.95239*exp(-0.5*(r/ 0.231095556)**2) - 8.55454239*exp(-0.5*((r-0.446608812)/0.00728849496)**2)", 
    #"pair": "15*sin(r*2*pi/50.0)/(r*2*pi/40)*(0.5*(1-tanh((2*pi/40*r-1.25*pi)/0.8)))",
    #"pair": "r**(-2)",
    #"pair": '-cos(pi*r)',
    #"c_to_one": True,
    #"external": "0.5*(cos(x)**2+cos(y)**2)",
    #"external": "0.0",
    "filename":"fig_2d"
}
#
params_n = {  # numerical-discretisation parameters
    "degree": 1,  # of elements
    "noise_degree": 1,  # of elements
    "element_rho": "CG", #Original DG
    "element_j": "CG",#Raviart-Thomas",
    "element_noise": "DG", # CG
    "delta":0,
    "SIPG_D":0, # extra diffusion term we used 0.5
    "SIPG_reg": 1,
    "weak_form": weak_form_em,
    "solver_parameters": linear_solver,
    "no_x": 20,
    "no_y": 20,
    "no_t": 3001, #original 101
    "screen_output_steps": 10,
    "convolution_radius": 3.141592, #pi
    "noise_truncation": 40,
    "rho_thresh":0, 
    "tau":0, # phi regularisation
}
kbT=params_m["sigma"]**2/(2*params_m["gamma"]); 
gdx=params_m["Lx"]/params_n["no_x"]; gdt=params_m["final_time"]/params_n["no_t"]; wave_speed=sqrt(kbT)
cfl=wave_speed*gdt/gdx
print("CFL constant:", cfl,"\n\n")
if (cfl>1):
  exit()

if run_case==1:
  tau_in=0.025
  filename="fig_tau_large"
  print("tau:", tau_in," (time-scale reg) filename:", filename)
  params_n["tau"]=tau_in
  params_m["filename"]=filename
# Set-up mesh etc
get_test_trial_space(params_m, params_n)
# Set-up initial data
rho0, j0 = set_up_2d(params_m, params_n)
# define weak form and time-stepper
# all the work is done here
one_step, dt = params_n["weak_form"](params_m, params_n, rho0, j0)
#
t = 0.0
step = 0
coeffs = [0.0, 0.0]
#
height=0.1
levels = np.linspace(0, height, 11)
cmap = "hsv"
total_mass0 = assemble(rho0 * dx)
print("Initial mass", total_mass0)
#
print("Starting time-stepping..")
rt = time.perf_counter()
# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes_flat=axes.flat
print("Time step:",dt)
#
outfile=File("./results/80.pvd")
while t < params_m["final_time"] - dt:
    gc.collect()
    one_step(rho0)
    #
    step += 1
    t += dt
    # regularly print to screen and save to file
    if step % params_n["screen_output_steps"] == 0:
     #   outfile.write(rho0)
        total_mass = assemble(rho0 * dx)
        mass_deviation = total_mass - total_mass0
        print(
            "t={:.3f}".format(t),
            "; run_time {:.1f}".format(time.perf_counter() - rt), 
            "secs; total_mass={:.3f}".format(total_mass),
            ", deviation={:.3e}".format(mass_deviation),
            end=" \r")
        if abs(mass_deviation )> 1:
            print("\n")
            exit()
    axnum=(step//10)-1
    print("axnum",axnum)
    # outfile.write(rho0)
    print("save:",t,"\n")
    ##print(rho0.dat)

    ## Wiwi plot
    #fig, ax= plt.subplots()
    print(rho0)
    #contours = firedrake.tricontourf(rho0, levels=levels, axes=ax, cmap=cmap)
    #ax.set_aspect("equal")
    #ax.axis('off')
    #fig.colorbar(contours, location='right', shrink=0.5)
    #plt.savefig('figures/filename_axnum{0}.pdf'.format(axnum))
    outfile.write(rho0)

    #plt.show()

    # contours = firedrake.tricontourf(rho0, levels=levels, axes=axes[axnum], cmap=cmap)
    # axes[axnum].set_aspect("equal")
    # axes.set_xlabel(r'$x$')
    # axes.set_ylabel(r'$y$')
    # axes[axnum].axis('off')
    gc.collect()

#fig.colorbar(contours, ax=axes[:], location='right', shrink=0.5)
#fig.savefig("fig1.pdf", bbox_inches="tight",dpi=50)
#os.system('cp fig1.pdf ./figs/'+params_m["filename"]+'_.pdf')

print("\n")

wait = input("Press Enter to continue.")
