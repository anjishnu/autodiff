import numpy as np
from typing import List, NamedTuple, Callable


class Dependency(NamedTuple):
    tensor : 'Tensor'
    gradient_function : Callable[[np.ndarray], np.ndarray]



def ensure_array(arrayable_data):
    if type(arrayable_data) != np.ndarray:
        arrayable_data = np.array(arrayable_data, dtype=np.float64)
    return arrayable_data


def ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    array = ensure_array(data)
    return Tensor(array)


class Tensor(object):

    def __init__(self, 
                 data : np.ndarray,
                 requires_grad : bool = True,
                 depends_on = []):

        self.data = ensure_array(data)
        self.requires_grad: boolean = requires_grad
        self.depends_on = depends_on
        self.grad : Tensor = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)


    def backward(self, grad: 'Tensor' = None) -> 'Tensor':

        if grad is None:
            if self.shape == (): 
                grad = Tensor(1, requires_grad=False)
            else: 
                raise RuntimeError('grad must be specified for non-0 tensor')
        else:
            grad = ensure_tensor(grad)

        assert self.requires_grad            
        self.grad.data += grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.gradient_function(grad)
            dependency.tensor.backward(backward_grad)
            

    def sum(self):
        return Sum()(self)


    def add(self, b):
        return Add()(self, b)


    def mult(self, b):
        return Multiply()(self, b)


    def neg(self):
        return Neg()(self)


    def sub(self, b):
        return Add()(self, b.neg())


class Operator(object):
    
    def _get_grad_fn(self) -> Callable:
        raise NotImplementedError('Implement grad function')

    @classmethod
    def subclasses(self):
        all_subclasses = {cls.__name__:cls for cls in self.__subclasses__()}

    def _operate(self, data : np.ndarray):
        raise NotImplementedError('Implement operation!')
    

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


    def execute(self, tensor: Tensor):
        raise NotImplementedError


class UnaryOperator(Operator):    

    def execute(self, tensor: Tensor) -> Tensor:
        requires_grad = tensor.requires_grad
        data = self._operate(tensor)
        dependencies = []
        if requires_grad:
            dependency = Dependency(tensor, self._get_grad_fn(tensor))
            dependencies.append(dependency)
            
        return Tensor(data, requires_grad, dependencies)


class BinaryOperator(Operator):    

    def execute(self, t1: Tensor, t2: Tensor) -> Tensor:
        requires_grad = t1.requires_grad or t2.requires_grad
        data = self._operate(t1, t2)
        dependencies = []

        if t1.requires_grad:
            dependency = Dependency(t1, self._get_grad_fn(t1, t2))
            dependencies.append(dependency)

        if t2.requires_grad:
            dependency = Dependency(t2, self._get_grad_fn(t2, t1))
            dependencies.append(dependency)
            
        return Tensor(data, requires_grad, dependencies)



class Sum(UnaryOperator):

    def _operate(self, tensor : Tensor):
        data = tensor.data.sum()
        return data

    def _get_grad_fn(self, tensor: Tensor):

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            ''' grad is a 0-tensor '''
            return grad.data * np.ones_like(tensor.data)
        return grad_fn


class Neg(UnaryOperator):

    def _operate(self, tensor):
        return -tensor.data

    def _get_grad_fn(self, tensor):
        return lambda t : -t.data


class Add(BinaryOperator):
    
    def _operate(self, t1: Tensor, t2: Tensor):
        assert(t1.data.shape == t2.data.shape)
        data = t1.data + t2.data
        return data

    def _get_grad_fn(self, t1 : Tensor, t2 : Tensor):
        def grad_fn(grad : np.ndarray) -> np.ndarray:
            return grad
        return grad_fn


class Multiply(BinaryOperator):
    
    def _operate(self, t1: Tensor, t2: Tensor):
        assert(t1.data.shape == t2.data.shape)
        data = t1.data * t2.data
        return data

    def _get_grad_fn(self, t1 : Tensor, t2 : Tensor):
        def grad_fn(grad : np.ndarray) -> np.ndarray:
            return grad.data * t2.data 
        return grad_fn
    
