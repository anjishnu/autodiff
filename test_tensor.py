from autodiff.tensor import Tensor
import numpy as np

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
def test_simple_add_2():

    t1 = Tensor([[1, 2, 3], [4,5,6]]) # shape=(2, 3)
    t2 = Tensor([[7, 8, 9]]) # shape=(1, 3)

    t3 = t1 + t2 # shape=(2, 3)
    t3.backward([[1,1,1], [1,1,1]])

    print('t1.grad.data', t1.grad.data)
    print('t2.grad.data', t2.grad.data)

    assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert t2.grad.data.tolist() == [[2, 2, 2]]


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
            print('iteration', i, 'sum_of_squares.data', 
                  sum_of_squares.data)
    assert sum_of_squares.data < 1e-12

@register_test
def test_matmul():
    # (3, 2)
    t1 = Tensor([[1, 2], [3, 4], [5, 6]])
    # (2, 1)
    t2 = Tensor([[10], [20]])
    t3 = t1 @ t2
    assert t3.data.tolist() == [[50], [110], [170]]
    grad = Tensor([[-1], [-2], [-3]])
    t3.backward(grad)
    np.testing.assert_array_equal(t1.grad.data, 
                                  grad.data @ t2.data.T)
        
@register_test
def test_learned_function():
    x_raw = np.random.randn(100, 3)
    coef = np.array([-1, + 3, -2], dtype=np.float64)
    y_raw = x_raw @ coef + 5 + np.random.randint(-2, 2, size=(100,))
    x_data = Tensor(x_raw, requires_grad=False)
    y_data = Tensor(y_raw, requires_grad=False)    
    w = Tensor(np.random.randn(3))
    b = Tensor(np.random.randn())


    print ('x_raw', x_raw)
    print ('y_raw', y_raw)

    for epoch in range(100):

        w.zero_grad()
        b.zero_grad()

        predicted = (x_data @ w) + b
        errors = predicted - y_data

        loss = (errors * errors).sum()
        loss.backward()

        w -= w.grad * 0.001
        b -= b.grad * 0.001

        if not epoch % 20:
            print(epoch, 'loss', loss.data)


for name, test in test_registry.items():
    print('Running test', name)
    test()
