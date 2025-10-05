import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
plt.ion()
def generate_random_field_2D(field_example, lx, ly, fluctuation_strength, plot=1):
    Lx, Ly = np.shape(field_example)
    xc = np.zeros([Lx, Ly])
    yc = np.zeros([Lx, Ly])
    kx = np.zeros([Lx, Ly])
    ky = np.zeros([Lx, Ly])
    random_modes = np.zeros([Lx, Ly])
    ds = (lx/Lx)*(ly/Ly)
    mod_0 = 1.0 /Lx/Ly/ds
    for i in range(Lx):
        for j in range(Ly):
            xc[i,j] = i*lx/Lx
            yc[i,j] = j*ly/Ly
            kx[i,j] = 2*np.pi/lx*(i*(i<=Lx/2) + (i-Lx)*(i>Lx/2))
            ky[i,j] = 2*np.pi/ly*(j*(j<=Ly/2) + (j-Ly)*(j>Ly/2))
            if i!=0 or j!=0:
                random_modes[i,j] = np.random.normal(0,1)*mod_0*fluctuation_strength*np.exp(-(kx[i,j]**2+ky[i,j]**2)/(0.3*np.pi/lx*Lx)**2)
            else:
                random_modes[i,j] = mod_0

    f_com = np.zeros([Lx,Ly])
    f_expression = ''
    num_term = 0
    for i in range(Lx):
        for j in range(Ly):
            if np.sqrt(kx[i,j]**2 + ky[i,j]**2)/(2*np.pi/lx) < 10:
                print(num_term)
                num_term=+num_term+1
                # f_com = f_com + norm_f_tf[i,j]*(np.cos(kx[i,j]*xc+ky[i,j]*yc) + 1j*np.sin(kx[i,j]*xc+ky[i,j]*yc))
                f_com = f_com + ( random_modes[i,j].real*np.cos(kx[i,j]*xc+ky[i,j]*yc) - random_modes[i,j].imag*np.sin(kx[i,j]*xc+ky[i,j]*yc) )
                # f_expression = f_expression + '+(({0:.6E})*cos({2:.1f}*x+{3:.1f}*y) - ({1:.6E})*sin({2:.1f}*x+{3:.1f}*y))'.format(random_modes[i,j].real, random_modes[i,j].imag, kx[i,j], ky[i,j])
                f_expression = f_expression + '+ ({0:.6E})*cos({2:.1f}*x+{3:.1f}*y)'.format(random_modes[i,j].real, random_modes[i,j].imag, kx[i,j], ky[i,j])*(random_modes[i,j].real>1e-6) + ' - ({1:.6E})*sin({2:.1f}*x+{3:.1f}*y)'.format(random_modes[i,j].real, random_modes[i,j].imag, kx[i,j], ky[i,j])*(random_modes[i,j].imag>1e-6)

    return f_com, f_expression[1:], ds


l_phys = 2*np.pi #float(l)
grid_size = int(25)
fluctuation_strength = 0.01

fd = np.zeros([grid_size, grid_size])
f_com, expression, ds= generate_random_field_2D(fd, lx=l_phys, ly=l_phys, fluctuation_strength=fluctuation_strength, plot=0)

print('np.sum(f_com)*ds = {:.6E}, should be 1'.format(np.sum(f_com)*ds))

exp_file = open('./expression_random.txt', 'w')
exp_file.write(expression)
exp_file.close()

plt.figure()
#plt.title('analytically expressed')
plt.imshow(f_com)
