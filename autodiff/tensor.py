import numpy as np
from typing import List, NamedTuple, Callable
import copy


class Dependency(NamedTuple):
    tensor : 'Tensor'
    gradient_function : Callable[[np.ndarray], np.ndarray]


def ensure_array(arrayable_data):
    if type(arrayable_data) != np.ndarray: arrayable_data = np.array(arrayable_data, dtype=np.float64)
    return arrayable_data


def ensure_tensor(data):
    if isinstance(data, Tensor): return data
    array = ensure_array(data)
    return Tensor(array)


class Tensor(object):

    def __init__(self, data : np.ndarray, requires_grad : bool = True, depends_on = []):
        self.data = ensure_array(data)
        self.requires_grad: boolean = requires_grad
        self.depends_on = depends_on
        self.grad : Tensor = None
        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self): return self.data.shape

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64), requires_grad=False)

    def backward(self, grad: 'Tensor' = None) -> 'Tensor':
        if grad is None:
            if self.shape == (): grad = Tensor(1, requires_grad=False)
            else: raise RuntimeError('grad must be specified for non-0 tensor')
        else: grad = ensure_tensor(grad)
        assert self.requires_grad 
        self.grad.data += grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.gradient_function(copy.deepcopy(grad))
            dependency.tensor.backward(backward_grad)
            
    def sum(self): return Sum().execute(self)
    
    @property
    def data(self) -> np.ndarray: return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = value
        # Setting the data manually means the gradients are invalidated
        self.grad = None

    def add(self, b): return Add().execute(self, ensure_tensor(b))

    def mult(self, b): return Multiply().execute(self, ensure_tensor(b))

    def matmul(self, other): return MatrixMultiply().execute(self, ensure_tensor(other))

    def neg(self): return Neg().execute(self)

    def sub(self, b): return Add().execute(self, ensure_tensor(b).neg())
    
    def __repr__(self) -> str: return "Tensor with shape={}".format(str(self.data.shape))
    
    def __add__(self, other): return self.add(ensure_tensor(other))

    def __iadd__(self, other): # x += 2
        self.data += ensure_tensor(other).data
        return self

    def __isub__(self, other): # x-= 2
        self.data -= ensure_tensor(other).data
        return self

    def __imul__(self, other): # x *= 2
        self.data *= ensure_tensor(other).data
        return self

    def __radd(self, other): return self.__add__(other)

    def __mul__(self, other): return self.mult(ensure_tensor(other))

    def __rmul__(self, other): return self.__mul__(other)

    def __neg__(self): return self.neg()

    def __sub__(self, other): return self.sub(other)

    def __rsub__(self, other): return Add().execute(ensure_tensor(other), -self)

    def __matmul__(self, other): return self.matmul(other)


class Operator(object):
    
    def _get_grad_fn(self) -> Callable:
        raise NotImplementedError('Implement grad function')

    @classmethod
    def subclasses(self):
        all_subclasses = {cls.__name__:cls for cls in self.__subclasses__()}

    def _operate(self, data : np.ndarray):
        raise NotImplementedError('Implement operate function!')
    
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

    def _compress_grad(self, grad, other_tensor): # Compress grad w.r.t a tensor
        ndims_added = grad.data.ndim - other_tensor.data.ndim
        for _ in range(ndims_added): grad.data = grad.data.sum(axis=0)         
        for i, dim in enumerate(other_tensor.data.shape):
            if dim == 1: grad.data = grad.data.sum(axis=i, keepdims=True) 
        return grad

    def _get_grad_fn_1(self, *args, **kwargs):
        return self._get_grad_fn(*args, **kwargs)

    def _get_grad_fn_2(self, *args, **kwargs):
        return self._get_grad_fn(*args, **kwargs)
    
    def execute(self, t1: Tensor, t2: Tensor) -> Tensor:
        requires_grad = t1.requires_grad or t2.requires_grad
        data = self._operate(t1, t2)
        dependencies = []
        if t1.requires_grad: 
            dependencies.append(Dependency(t1, self._get_grad_fn_1(t1, t2)))
        if t2.requires_grad: 
            dependencies.append(Dependency(t2, self._get_grad_fn_2(t1, t2)))            
        return Tensor(data, requires_grad, dependencies)


class Sum(UnaryOperator):

    def _operate(self, tensor : Tensor):
        return tensor.data.sum()

    def _get_grad_fn(self, tensor: Tensor):
        return lambda grad : grad.data * np.ones_like(tensor.data)


class Neg(UnaryOperator):

    def _operate(self, tensor):
        return -tensor.data

    def _get_grad_fn(self, tensor):
        return lambda t : -t.data


class Add(BinaryOperator):

    def _operate(self, t1: Tensor, t2: Tensor):
        return t1.data + t2.data

    def _get_grad_fn_2(self, t1, t2):
        return self._get_grad_fn(t2, t1)

    def _get_grad_fn(self, t1 : Tensor, t2 : Tensor):
        return lambda grad : self._compress_grad(grad, t1)


class Multiply(BinaryOperator):
    
    def _operate(self, t1: Tensor, t2: Tensor):
        return t1.data * t2.data

    def _get_grad_fn_2(self, t1, t2):
        return self._get_grad_fn(t2, t1)


    def _get_grad_fn(self, t1 : Tensor, t2 : Tensor):
        return lambda grad : self._compress_grad(grad, t1).data * t2.data
    

class MatrixMultiply(BinaryOperator):

    def _operate(self, t1: Tensor, t2:Tensor):
        return t1.data @ t2.data

    def _get_grad_fn_1(self, t1, t2):
        return lambda grad : grad.data @ t2.data.T
 
    def _get_grad_fn_2(self, t1, t2):
        return lambda grad : t1.data.T @ grad.data
