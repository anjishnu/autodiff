import numpy as np
from typing import List, NamedTuple, Callable


class Dependency(NamedTuple):
    tensor : 'Tensor'
    gradient_function : Callable[[np.ndarray], np.ndarray]



def ensure_array(arrayable_data):
    if type(arrayable_data) != np.ndarray:
        arrayable_data = np.array(arrayable_data)
    return arrayable_data

def ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    array = ensure_array(data)
    return Tensor(array)


class Tensor(object):

    def __init__(self, 
                 data : np.ndarray,
                 requires_grad : bool = False,
                 depends_on = []):

        self.data = ensure_array(data)
        self.requires_grad : boolean = requires_grad
        self.depends_on = depends_on
        self.grad : Tensor = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))


    def backward(self, grad: 'Tensor' = None) -> 'Tensor':
        assert self.requires_grad
        #grad = ensure_tensor(grad) if grad else grad
        if grad is None:
            if self.shape == (): 
                grad = Tensor(1)
            else: 
                raise RuntimeError('grad must be specified for non-0 tensor')
            
        self.grad.data += grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.gradient_function(grad)
            dependency.tensor.backward(backward_grad)
            

    def sum(self):
        return Sum()(self) #return _sum(self)



class Operator(object):
    
    def _get_grad_fn(self) -> Callable:
        raise NotImplementedError('Implement grad function')


    def _operate(self, data : np.ndarray):
        raise NotImplementedError('Implement operation!')
    

    def __call__(self, tensor: Tensor) -> Tensor:
        requires_grad = tensor.requires_grad
        data = self._operate(tensor)
        dependencies = []
        if requires_grad:
            dependency = Dependency(tensor, self._get_grad_fn(tensor))
            dependencies.append(dependency)
            
        return Tensor(data, requires_grad, dependencies)


class Sum(Operator):

    def _operate(self, tensor : Tensor):
        data = tensor.data.sum()
        return data

    def _get_grad_fn(self, tensor: Tensor):

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            ''' grad is a 0-tensor '''
            return grad.data * np.ones_like(tensor.data)

        return grad_fn


class GradFunction(object):
    def __init__(self, fn):
        self.fn = fn

    def execute(T: Tensor) -> Tensor:
        return self.fn(T)


def _sum(tensor : Tensor) -> Tensor:
    data = tensor.data.sum()
    requires_grad = tensor.requires_grad
    dependencies = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            ''' grad is a 0-tensor '''
            return grad.data * np.ones_like(tensor.data)        

        dependency = Dependency(tensor, grad_fn)        
        dependencies.append(dependency)

    return Tensor(data, requires_grad, depends_on=dependencies)
