from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyeqsim import Variable,Viewer


class Boundary:
    def __init__(self, variable=None, cords=None) -> None:
        self.variable : Variable = variable
        self.cords = cords

    def set_variable(self, variable):
        self.variable = variable

    def boundary_check(self, viewer, cord):
        for v, s in zip(cord, viewer.variable_obj.size):
            if v < 0 or v > s-1:
                return True
        return False

    def boundary_value(self, viewer, cord):
        raise NotImplementedError


class Constant(Boundary):
    def __init__(self, value) -> None:
        self.b_value = value
        super().__init__(None, None)

    def boundary_value(self, viewer, cord):
        return self.b_value
    

class Continuous(Boundary):
    def __init__(self) -> None:
        super().__init__(variable=None, cords=None)


    def boundary_value(self, viewer, cord):
        shape = list(self.variable.size)
        d = viewer.data
        for i in range(len(cord)):
            
            if cord[i] >= shape[i]:
                shape[i] = cord[i]-shape[i]
            elif cord[i] < 0:
                shape[i] = shape[i]+cord[i]
            else:
                shape[i] = cord[i]

        return d[tuple(shape)]