from pyeqsim.boundary import Constant,Continuous
from pyeqsim import Variable,System
import matplotlib.pyplot as plt
import numpy as np



system = System()

U = Variable(size=[10],initial_value=0.01,boundary_condition=Constant(0),args_name=["x"])
#U.initial_condition = np.sin(np.arange(10))
U.initial_condition = np.sin(np.linspace(0,10,10))
#Y.initial_condition[-5] = 0.5


dt = 0.00001
dx = dt
c = 0.001

system <= "d**2U/dt**2 = c *  d**2U/dx**2"

print(system.eqs[0].transformed_eq)

for i in range(1000000):
    #st = time.perf_counter()
    system()
    if len(U.data) > 5:
        pass
        #U.opti()
    #print(time.perf_counter()-st)
    #print(Y.data)
    if i % 10 == 0:
        plt.plot(U.data[-1])
        plt.title(i)
        #print(Y.data[-1])
        plt.pause(0.01)
        plt.cla()
    
plt.show()