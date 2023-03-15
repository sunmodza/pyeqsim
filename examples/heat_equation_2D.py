from pyeqsim.boundary import Constant,Continuous
from pyeqsim import Variable,System
import matplotlib.pyplot as plt

# create system
system = System()

# define system's variables
T = Variable((60, 60),boundary_condition=Continuous(),args_name=["x","y"],initial_value=10)
Q = Variable((60, 60),boundary_condition=Continuous(),args_name=["x","y"],initial_value=1)

# define initial condition
T.initial_condition[45:50, 45:50] = 100
Q.initial_condition[:, 45:55] = 100000
Q.initial_condition[45:55, :] = 0

# define constant
dx = 0.00001
k = 1
q = 1
dy = 0.00001

# Insert Equation Into system
system <= "d**2T/dx**2 + d**2T/dy**2 - Q/k = 0"

# simulate
for i in range(10000):
    system.simulate()
    T.opti()
    T.viz()
    plt.pause(0.001)
    plt.cla()
plt.show()