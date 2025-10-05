import math
f=open("lj.cut",'w')
x=[0.10 + 0.02 * i for i in range(0,496)]
y=[]
a=  2029.9
b=0.3783
c=-0.0217783
d=1.3439
Energy = lambda dd: a*math.exp(-0.5*(dd/b)**2)+c*math.exp(-0.5*(dd/d)**2)
force = lambda dd: a*dd/(b**2)*math.exp(-0.5*(dd/b)**2)+c*dd/(d**2)*math.exp(-0.5*(dd/d)**2)

ee=[]
ff=[]
ff1=[]

str_out = ""
for i in range(0,len(x)):
        y_val = Energy(x[i])
        ee.append(y_val)
        f_val = force(x[i])
        ff.append(f_val)

for i in range(0,len(x)-1):
        f_manual = -1.0*(ee[i+1]-ee[i])/(x[i+1]-x[i])
        ff1.append(f_manual)
ff1.append(0.0000)
for i in range(0,len(ff)):
        #str_out += "%d %.9f %.9f %.9f %.9f\n" %(i+1,x[i],ee[i],ff[i],ff1[i])
        str_out += "%d %.9f %.9f %.9f\n" %(i+1,x[i],ee[i],ff[i])

f.write(str_out)
f.close()
