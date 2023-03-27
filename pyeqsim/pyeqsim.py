from __future__ import annotations
from copy import copy
from typing import List, Union
import numpy as np
import sympy as sp
import itertools
import math
import re
import matplotlib.pyplot as plt
import inspect
import time
from numpy import cos
#from boundary import Boundary,Constant,Continuous
try:
    from pyeqsim.boundary import Boundary,Constant,Continuous
except ModuleNotFoundError:
    from boundary import Boundary,Constant,Continuous
#from numba import njit,int32,float32

x = "x"
y = "y"
z = "z"

def first_ode_formatter(vname,wrt_arg,has_dt=False):
    #(f(x)-f(x-1))/dx
    stack = inspect.stack()
    caller_frame = stack[-1].frame
    obj_v : Variable = caller_frame.f_locals.get(vname)
    if obj_v.is_singular and wrt_arg == "t": # case Dt
        # dH/dt = (dH(+1)-dH(0))/dt
        new_arg = get_arg("", obj_v)
    else:
        new_arg = get_arg(wrt_arg, obj_v)

    next_time = "(+1)" if has_dt and wrt_arg == "t" else "(+0)" 

     # case not Dt
    #new_arg = get_arg(wrt_arg, obj_v)

    is_1 = obj_v.my_arg

    v1 = f'{vname}{is_1}{next_time}'.replace(" ","")
    v2 = f'{vname}{tuple(new_arg)}(+0)'.replace(" ","")
        
    eq = f'({v1}-{v2})/d{wrt_arg}'
    #print(eq)
    return eq,f'{v1}',[v1,v2]

def get_arg(wrt_arg, obj_v, prefix="+1"):
    new_arg = []
    if obj_v.args_name is not None:
        for arg in obj_v.args_name:
            if arg == wrt_arg:
                new_arg.append(f'{arg}{prefix}')
            else:
                new_arg.append(arg)
    return tuple(new_arg)

def one_plus_ode_formatter(vname,wrt_arg,order,has_dt=False):
    stack = inspect.stack()
    caller_frame = stack[-1].frame
    obj_v : Variable = caller_frame.f_locals.get(vname)
    if int(order) == 2:
        arg_names = obj_v.my_arg#tuple(obj_v.args_name)
        a = get_arg(wrt_arg,obj_v,prefix="+1")
        b = get_arg(wrt_arg,obj_v,prefix="-1")

        next_time = "(+1)" if has_dt and wrt_arg == "t" else "(+0)" 

        v1 = f'{vname}{a}(+0)'.replace(" ","")
        v2 = f'{vname}{arg_names}{next_time}'.replace(" ","")
        v3 = f'{vname}{b}(+0)'.replace(" ","")
        eq = f'({v1}-(2*{v2})+{v3})/(d{wrt_arg}**2)'.replace(" ","")
        return eq,f'{v2}',[v1,v2,v3]

def transformation(eq:str):
    eq = eq.replace(" ","")
    # d**2T/dx**2
    req1 = "d[A-Za-z]+/d[A-Za-z]+"
    req1p = "d\*\*\d[A-Za-z]+/d[A-Za-z]+\*\*\d"

    has_dt = "dt" in eq
    first_odes = re.findall(req1,eq)
    solve_fields = []
    variables = []
    for ode in first_odes:
        #print(ode[1:].split("/d"))
        ode_transformed,field_name,variable = first_ode_formatter(*ode[1:].split("/d"),has_dt=has_dt)
        variables.extend(variable)
        solve_fields.append(field_name)
        eq = eq.replace(ode,ode_transformed)

    one_plus_odes : List[str] = re.findall(req1p,eq)
    for ode in one_plus_odes:
        backup = copy(ode)
        last_ast = ode.rfind("*")
        td = ode.rfind("/d")
        order = int(ode[last_ast+1:])
        arg = ode[:last_ast-1][td+2:]
        
        field = ode[:td].replace(f"d**{order}","")


        ode_transformed,field_name,variable = one_plus_ode_formatter(field,arg,order,has_dt=has_dt)
        variables.extend(variable)
        solve_fields.append(field_name)
        eq = eq.replace(ode,ode_transformed)
        
        
        
    solve_fields = list(set(solve_fields))
    variables = list(set(variables))

    return eq,solve_fields,variables

def find_add_one_solve(eq):
    if "+1" in eq:
        return True
    return False

def return_solved_equation(eq):
    #print("ASDDDDDDDDDDDDDDDDDDDD")
    eq,solve_fields,variables = transformation(eq)
    #print(solve_fields,variables)
    #raise NotImplementedError
    pfsolve = None
    pf = "HJIK"
    to_same = {}
    saewae = True in ["+1" in i for i in solve_fields]
    #print("ASDASJIODASDAS(D AS)", saewae)
    for i,v in enumerate(variables):
        vnew = pf+str(range(len(variables))[i])
        to_same[vnew] = v
        eq = eq.replace(v,vnew)
        #print(solve_fields,v)
        if v in solve_fields:
            if saewae and find_add_one_solve(v):
                pfsolve = vnew
            else:
                pfsolve = vnew
        psb = vnew
        exec(f'{vnew} = sp.Symbol(vnew)')
    
    if pfsolve is None:
        pfsolve = psb

    for s in Equation.get_matched_variable_representation_string(eq):
        vnew = pf+str(len(to_same))
        eq = eq.replace(s,vnew)
        to_same[vnew] = s
        exec(f'{vnew} = sp.Symbol(vnew)')

    to_solve = eq.split("=")

    cse = eval(f'({to_solve[0]})-({to_solve[1]})')
    #print(cse)
    #print(cse)
    #print(cse)
    ans = str(sp.solve(cse,sp.Symbol(pfsolve))[0]).replace(" ","")
    #print("ASDSAD ")
    #ans = str(sp.solve(f'({to_solve[0]})-({to_solve[1]})',sp.Symbol(pfsolve))[0]).replace(" ","")
    ans = f'{pfsolve}={ans}'
    

    for v in to_same:
        ans = ans.replace(v,to_same[v])
    print(ans,pfsolve,"SADASDq3",to_same)
    return ans


def is_boundary_check(cord, shape):
    for v, s in zip(cord, shape):
        if v < 0 or v > s-1:
            return True
    return False


def select_data_with_multi_index(data, indexs, modifier, viewer, boundary:Boundary = None, boundary_checking=False, mt_c=None):
    l = np.zeros_like(data)
    shape = l.shape
    unvalid = []
    for i,cord in enumerate(indexs):
        c = tuple(np.array(cord)+np.array(modifier))
        if boundary.boundary_check(viewer,c):
            value = boundary.boundary_value(viewer, c)
            l[cord] = value
            if boundary_checking:
                unvalid.append(cord)
        else:
            l[cord] = data[c]
        """
        if is_boundary_check(c, shape):
            if boundary_checking:
                unvalid.append(mt_c[i])
            else:
                bv = boundary.boundary_value(viewer, c)
                l[cord] = bv
        """
        
    if not boundary_checking:
        return l.reshape(-1)
    else:
        return unvalid


class Calculable:
    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)
    
    def __neg__(self):
        return Mul(self, -1)


class Viewer(Calculable):
    def __init__(self, data, step, variable_obj: Variable, args, inside_args, args_name, modifier) -> None:
        self.modifier = modifier
        if step <= 0:
            step -= 1

        self._data = data
        self.step = step
        self.variable_obj = variable_obj
        self.args = args
        self.inside_args = inside_args
        self.args_name = args_name

    @property
    def data(self):
        try:
            return self._data[self.step]
        except:
            return self._data[-1]

    def set(self, args: Union[Operation, Viewer]):
        self.variable_obj.being_set = True
        if isinstance(args, (Operation, Viewer)):
            val = args.eval(self.inside_args, self.args_name)
        else:
            val = (self.variable_obj.create_data()+args).reshape((-1))
        phd = self.variable_obj.create_data()
        # print(len(val))

        self.set_to_value(val, phd)

        for i in range(self.step-1):
            self.variable_obj.data.append(self.variable_obj.create_data())

        self.variable_obj.data.append(phd)

    def set_to_value(self, val, phd):
        for pos, v in zip(self.args, val.reshape(-1)):
            phd[pos] = v

    def __le__(self,value):
        self.set(value)

    def boundary_checking(self, args: Union[Operation, Viewer]):
        if isinstance(args, (Operation, Viewer)):
            v = args.bs(self.inside_args, self.args_name, self.args)
            return np.array(tuple(v))
        return np.array(tuple([]))

    def bs(self, args, args_name, outer_arg):
        if self.variable_obj.is_singular:
            return []

        if args_name == self.args_name:
            v = select_data_with_multi_index(self.data, args, modifier=self.modifier, viewer=self, boundary_checking=True, mt_c=outer_arg)
            return v

    def eval(self, args, args_name):
        if self.variable_obj.is_singular:
            return self.data[0]

        if args_name == self.args_name:
            return select_data_with_multi_index(self.data, args, modifier=self.modifier, viewer=self, boundary=self.variable_obj.boundary)


class Operation(Calculable):
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    def calculate(self, a, b):
        raise NotImplementedError

    def eval(self, args, args_name):
        if isinstance(self.a, (Operation, Viewer)):
            a = self.a.eval(args, args_name)
        else:
            a = self.a

        if isinstance(self.b, (Operation, Viewer)):
            b = self.b.eval(args, args_name)
        else:
            b = self.b

        return self.calculate(a, b)

    def bs(self, args, args_name, outer_arg):
        if isinstance(self.a, (Operation, Viewer)):
            a = self.a.bs(args, args_name, outer_arg)
        else:
            a = []

        if isinstance(self.b, (Operation, Viewer)):
            b = self.b.bs(args, args_name, outer_arg)
        else:
            b = []

        v = a+b
        return v


class SOperation(Operation):
    def __init__(self, a) -> None:
        super().__init__(a, None)

    def calculate(self, a, b=None):
        raise NotImplementedError


class Add(Operation):
    def calculate(self, a, b):
        return a+b


class Sub(Operation):
    def calculate(self, a, b):
        return a-b


class Mul(Operation):
    def calculate(self, a, b):
        return a*b


class Div(Operation):
    def calculate(Self, a, b):
        return a/b


class Sin(SOperation):
    def calculate(self, a, b=None):
        return np.sin(a)


class Cos(SOperation):
    def calculate(self, a, b=None):
        return np.cos(a)


class Tan(SOperation):
    def calculate(self, a, b=None):
        return np.tan(a)


class Variable:
    def __init__(self, size=1, boundary_condition=Constant(0), initial_value=0, args_name=None) -> None:
        self.boundary = boundary_condition
        self.args_name = args_name
        self.initial_value = initial_value
        self.boundary.set_variable(self)
        self.is_singular = False
        if size == 1:
            size = [1]
            self.is_singular = True
        self.size = tuple(size)
        self.data = [self.create_data()+self.initial_value]
        self.addition_step = 0
        self.being_set = False

    def opti(self):
        self.data.pop(0)

    def __repr__(self) -> str:
        return f'{self.data[-1]}'

    def create_data(self):
        return (np.zeros(self.size))

    def set_at(self, step, val):
        while step > self.addition_step:
            self.data.append(self.create_data())
            self.addition_step += 1

        step -= self.addition_step

        if step <= 0:
            step -= 1

    def __call__(self, *arg_name):
        self_apply_args = []
        apply_inside_args = []
        names = []
        modifier = []
        for i in range(len(arg_name)):
            name = arg_name[i]
            val = 0
            if "+" in name:
                name, val = name.split("+")
                val = int(val)*-1
            elif "-" in name:
                name, val = name.split("-")
                val = int(val)
            modifier.append(val)
            names.append(name)
            apply_inside_args.append(
                [cord+val for cord in range(self.size[i])])
            self_apply_args.append([cord for cord in range(self.size[i])])

        self_apply_args = [i for i in itertools.product(*self_apply_args)]
        apply_inside_args = [i for i in itertools.product(*apply_inside_args)]

        if self.args_name is not None:
            names = self.args_name

        def at_step(step):
            if step <= 0:
                step = len(self.data)-1-step
            return Viewer(self.data, step, self, self_apply_args, apply_inside_args, names, modifier)

        return at_step
    
    def value(self):
        return self.data[-1]
    
    @property
    def initial_condition(self):
        return self.data[0]
    
    @initial_condition.setter
    def initial_condition(self,v):
        self.data[0] = v
    
    def viz(self):
        plt.imshow(self.data[-1], cmap='hot', interpolation='nearest')
        plt.title(f'Frame = {len(self.data)}')

    def my_name_is(self):
        for name, value in globals().items():
            if value is self:
                return name
    
    @property
    def my_arg(self):
        if self.args_name is None:
            return "()"
        return tuple(self.args_name)

    def __truediv__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return sp.Symbol(name)/other
    
    def __truerdiv__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return other/sp.Symbol(name)
    
    def __add__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return sp.Symbol(name)+other
    
    def __sub__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return sp.Symbol(name)-other
    
    def __rsub__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return other-sp.Symbol(name)
    
    def __mul__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return sp.Symbol(name)*other
    
    def __rmul__(self,other):
        name = (f"{self.my_name_is()}{self.my_arg}(+0)")
        return other*sp.Symbol(name)

    

class Equation:
    def __init__(self,eq:str) -> None:
        # T(x,y)[+1] = T(x-1,y)[+0] + T(x,y-1)[+0]
        self.eq = eq
        self.transformed_eq = copy(self.eq)
        self.transform()

    def transform(self):
        self.transformed_eq = self.transformed_eq.replace("=","<=")
        res = self.get_matched_variable_representation_string(self.transformed_eq)
        for r in res:
            name = r.split("(")[0]

            cord_names = self.get_cordinate_representation_string(r)

            step_i1 = r.rfind("[")
            step_i2 = r.rfind("]")
            step = r[step_i1+1:step_i2]

            represent = f'{name}{cord_names}({step})'
            self.transformed_eq = self.transformed_eq.replace(r,represent)
        return self.transformed_eq

    @staticmethod
    def get_matched_variable_representation_string(str):
        res : List[str] = re.findall("[A-Za-z]+\([^)]*\)\[[^\]]*\]",str)
        return res

    def get_cordinate_representation_string(self, r):
        cord_i1 = r.find("(")
        cord_i2 = r.rfind(")")
        cord = r[cord_i1+1:cord_i2]
        if cord == "":
            return "()"
        cord_names = []
        for point in cord.split(","):
            cord_names.append(point.replace(" ",""))
        cord_names = tuple(cord_names)
        return cord_names
    

class System:
    def __init__(self) -> None:
        self.eqs = []
        self.t_sim = 0
        self.op_flag = False
        #self.reinit()

    def __call__(self):
        #self.reinit()
        for eq in self.eqs:
            eval(eq.transformed_eq)
        self.t_sim += 1
    
    def __le__(self,arg:str):
        self.reinit()
        arg = return_solved_equation(arg)
        self.eqs.append(Equation(arg))

    def reinit(self):
        stack = inspect.stack()
        caller_frame = stack[-1].frame
        for v_siew in caller_frame.f_locals:
            obj_v_siew = caller_frame.f_locals.get(v_siew)
            exec(f'global {v_siew}\n{v_siew} = obj_v_siew')

    def optimize(self):
        #print(self.t_sim)
        if self.op_flag or self.t_sim > 2:
            #print("SADSAD")
            self.op_flag = True

            stack = inspect.stack()
            caller_frame = stack[-1].frame
            for v_siew in caller_frame.f_locals:
                obj_v_siew = caller_frame.f_locals.get(v_siew)
                if isinstance(obj_v_siew,Variable):
                    asdasd = eval(f"{v_siew}.being_set")
                    #print(asdasd)
                    if asdasd:
                        #print(f"{v_siew}.opti()")
                        eval(f"{v_siew}.opti()")
                    #if len(obj_v_siew.data) > 5:
                        #print("HELLO",v_siew)
                    #    obj_v_siew.opti()
                    #print(v_siew)
            #print("ASDSADew")
        #print("sad")
            


    def simulate(self):
        self.__call__()
        

if __name__ == "__main__2":
    system = System()

    H = Variable(initial_value=1000000)
    I = Variable(initial_value=0)
    V = Variable(initial_value=100)

    kr1,kr2,kr3,kr4,kr5,kr6 = 1e5,0.1,2e-7,0.5,5,100
    dt = 0.001

    system <= "dH/dt = kr1-kr2*H-kr3*V*H"
    system <= "dI/dt = kr3*H*V-kr4*I"
    system <= "dV/dt = -kr3*H*V-kr5*V+kr6*I"

    while True:
        #st = time.perf_counter()
        system()
        #print(time.perf_counter()-st)
        if system.t_sim % 100 == 0:
            plt.semilogy(np.arange(len(H.data))*dt,H.data,label="H")
            plt.semilogy(np.arange(len(I.data))*dt,I.data,label="I")
            plt.semilogy(np.arange(len(V.data))*dt,V.data,label="V")
            plt.legend()
            plt.pause(0.001)
            plt.cla()
        
        if system.t_sim > 15000:
            break
    plt.show()


if __name__ == "__main__":

    system = System()

    T = Variable((100, 100),boundary_condition=Continuous(),args_name=["x","y"],initial_value=10)
    T.initial_condition[45:50, 45:50] = 100

    Q = Variable((100, 100),boundary_condition=Continuous(),args_name=["x","y"],initial_value=100)
    #Q.initial_condition[:, 45:55] = 0
    #Q.initial_condition[45:55, :] = 0
    
    dx = 0.00001
    k = 0.001
    q = 1
    dy = 0.00001

    #T(m, n)[+1] = (T(m+1, n)[+0]+T(m-1, n)[+0] + T(m, n+1)[+0]+T(m, n-1)[+0] + (Q(m,n)[+0]*(dx**2)/k))/4
    #print(eval("d**2T/dx**2 + d**2T/dy**2 - Q/k = 0"))

    system <= "d**2T/dx**2 + d**2T/dy**2 - Q/k = 0"
    print(system.eqs[0].transformed_eq)
    for _ in range(10000):
        system.simulate()
        T.viz()
        T.opti()
        plt.pause(0.1)
        plt.cla()
    plt.show()


