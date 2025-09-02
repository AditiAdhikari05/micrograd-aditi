import numpy as np
from micrograd.engine_tensor import Tensor


class Module:
    def parameters(self): return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad[:] = 0.0


class Linear(Module):
    """Fully-connected layer: y = x @ W + b.

    Accepts batched input (batch, nin) or unbatched (nin,).
    He initialisation (scale = sqrt(2/nin)) suits ReLU networks.
    """

    def __init__(self, nin, nout, bias=True):
        scale = np.sqrt(2.0 / nin)
        self.W = Tensor(np.random.randn(nin, nout) * scale)
        self.b = Tensor(np.zeros(nout)) if bias else None

    def __call__(self, x):
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])

    def __repr__(self):
        return f"Linear({self.W.shape[0]} -> {self.W.shape[1]})"


class MLP(Module):
    """Multi-layer perceptron built from Linear layers + ReLU.

    Args:
        nin:   input dimension
        nouts: list of layer widths; last entry = output dimension
               (no ReLU after the final layer)
    """

    def __init__(self, nin, nouts):
        dims = [nin] + list(nouts)
        self.layers = [Linear(dims[i], dims[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP([{', '.join(str(l) for l in self.layers)}])"
