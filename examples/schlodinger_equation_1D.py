from pyeqsim.boundary import Constant,Continuous
from pyeqsim import Variable,System
import matplotlib.pyplot as plt
import numpy as np


system = System()

Y = Variable(size=[50],initial_value=0,boundary_condition=Constant(0),args_name=["x"])
Y.initial_condition[20] = 0.5
Y.initial_condition[30] = 0.5

E = Variable(size=Y.size,initial_value=0,boundary_condition=Constant(0),args_name=["x"])
E.initial_condition[40:45] = 500

U = Variable(size=Y.size,boundary_condition=Constant(0),args_name=["x"])
#Y.initial_condition[-5] = 0.5


#dt = 0.000001
#dx = 0.000001
dt = 0.00000001
dx = dt
h = 0.1
m = 0.00001


system <= "(-h**2/2*m)*(d**2Y/dx**2)+U(x)[+0]+Y(x)[+0] = E(x)[+0]*Y(x)[+0]"

#print(system.)

while True:
    system()

    plt.plot(np.abs(Y.data[0]))
    #plt.title(f'(-ħ**2/2*m)*(d**2Ψ(x,t)/dx**2)+U(x)+Ψ(x,t) = E(x)*Ψ(x,t)\nt={system.t_sim*dt}\n{(np.abs(Y.data[0])**2).sum()}')
    #print(Y.data[-1])
    plt.pause(0.01)
    plt.cla()

    Y.opti()
    
    #if system.t_sim > 22:
    #    break
plt.show()