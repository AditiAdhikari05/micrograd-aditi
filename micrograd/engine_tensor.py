import numpy as np


class Tensor:
    """NumPy-backed tensor with reverse-mode automatic differentiation.

    Mirrors the scalar Value API but operates on N-d arrays, with correct
    gradient accumulation through broadcasting and matmul.
    """

    def __init__(self, data, _children=(), _op=''):
        if isinstance(data, (int, float)):
            data = np.array(float(data))
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # forward ops
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = _wrap(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _wrap(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Matrix multiply; handles batched (... x m x n) @ (... x n x p)."""
        other = _wrap(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
            dA = out.grad @ other.data.swapaxes(-1, -2)
            dB = self.data.swapaxes(-1, -2) @ out.grad
            self.grad  += _unbroadcast(dA, self.data.shape)
            other.grad += _unbroadcast(dB, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, exp):
        assert isinstance(exp, (int, float)), "exponent must be scalar"
        out = Tensor(self.data ** exp, (self,), f'**{exp}')

        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            # axis=None case: grad is 0-d; broadcast_to expands it correctly
            self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            grad = out.grad
            if not keepdims and axis is not None:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape) / n
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self.data.T, (self,), 'T')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # backward pass
    # ------------------------------------------------------------------

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # ------------------------------------------------------------------
    # sugar
    # ------------------------------------------------------------------

    def __neg__(self):          return self * -1.0
    def __radd__(self, other):  return self + other
    def __sub__(self, other):   return self + (-_wrap(other))
    def __rsub__(self, other):  return _wrap(other) + (-self)
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other):   return self * _wrap(other) ** -1
    def __rtruediv__(self, other):  return _wrap(other) * self ** -1
    def __rmatmul__(self, other):   return _wrap(other) @ self

    @property
    def shape(self): return self.data.shape

    @property
    def ndim(self): return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')\ndata={self.data}\ngrad={self.grad}"


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad, shape):
    """Sum `grad` over axes that were broadcast to reach its current shape.

    Handles two cases:
    - Leading dimensions: grad has more dims than shape (e.g. batch was added)
    - Size-1 dims: grad was expanded from size-1 axes in the original tensor
    """
    if shape == ():
        return grad.sum()
    # Remove extra leading dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum over axes that were size-1 in the original
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)
