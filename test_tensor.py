from autodiff.tensor import Tensor

test_registry = {}
def register_test(func):
    test_registry[func.__name__] = func
    return func

@register_test
def test_simple_sum():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = t1.sum()
    t2.backward()
    print(t1.grad.data)
    assert t1.grad.data.tolist() == [1, 1, 1]


@register_test
def test_simple_sum():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = t1.sum()
    t2.backward(Tensor(3))
    print(t1.grad.data)
    assert t1.grad.data.tolist() == [3, 3, 3]


for name, test in test_registry.items():
    print('Running test', name)
    test()
