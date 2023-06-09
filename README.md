# pyeqsim

## This Library will make simulating equation system super easy
## Offer super simple interface and highly customizable

### 1. Import dependencies
```
from pyeqsim.boundary import Constant,Continuous
from pyeqsim import Variable,System
import matplotlib.pyplot as plt
```

### 2. Define System, Variables, Constants, Boundary Conditions, Initial Values
```
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
```

### 4. Define Relationships with system <= ....
  Differiential Equation Or the equation represent system transformation
  
  you can also define multiple relation ships if want to
```
system <= "d**2T/dx**2 + d**2T/dy**2 - Q/k = 0"
```

### 5. Run The Simulation
```
for i in range(10000):
    system.simulate()
    T.opti()
    T.viz()
    plt.pause(0.001)
    plt.cla()
plt.show()
```
### 6. Enjoy the result
![image](https://user-images.githubusercontent.com/62195081/225412376-0189f100-b10b-4c88-9146-c8fa2c2fe67a.png)


## Write your own solution if possible. this library definitely not the fastest solution
