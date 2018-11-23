from autodiff.tensor import Tensor

test_registry = {}
def register_test(func):
    test_registry[func.__name__] = func
    return func

@register_test
def test_simple_sum():
    t1 = Tensor([1, 2, 3])
    t2 = t1.sum()
    t2.backward(Tensor(3))
    print(t1.grad.data)
    assert t1.grad.data.tolist() == [3, 3, 3]


@register_test
def test_simple_add():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([3, 4, 5])
    t3 = t1 + t2
    t3.backward([-1, -2, -3])
    print('t1.grad.data', t1.grad.data)
    print('t2.grad.data', t2.grad.data)
    assert t1.grad.data.tolist() == [-1, -2, -3]


@register_test
def test_simple_sub():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    t3 = t1 - t2
    t3.backward([-1, -2, -3])
    print('t1.grad.data', t1.grad.data)
    print('t2.grad.data', t2.grad.data)
    assert t1.grad.data.tolist() == [-1, -2, -3]
    assert t2.grad.data.tolist() == [1, 2, 3]


@register_test
def test_simple_mul():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    t3 = t1 * t2
    t3.backward([-1., -2., -3.])
    print('t1.grad.data', t1.grad.data)
    print('t2.grad.data', t2.grad.data)
    assert t1.grad.data.tolist() == [-4., -10., -18.]
    assert t2.grad.data.tolist() == [-1., -4., -9.]

@register_test
def minimize_a_function():
    x = Tensor([10, -10, 10, -5, 6, 3, 1])
    # we want to minimize the sum of squares

    for i in range(100):
        sum_of_squares = (x*x).sum() # 0-Tensor
        sum_of_squares.backward()
        delta_x = x.grad.data * 0.1 
        x = Tensor(x.data - delta_x)
        if i%20 == 0:
            print('iteration', i, 'sum_of_squares.data', sum_of_squares.data)

    assert sum_of_squares.data < 1e-12


for name, test in test_registry.items():
    print('Running test', name)
    test()
