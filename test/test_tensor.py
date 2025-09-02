"""Tensor autograd correctness tests — compared against PyTorch as oracle."""
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from micrograd.engine_tensor import Tensor

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required as oracle")

RNG = np.random.default_rng(42)
ATOL = 1e-6


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _t(arr):
    """numpy array -> torch float64 tensor with grad tracking."""
    return torch.tensor(arr, dtype=torch.float64, requires_grad=True)


def close(a: np.ndarray, b: torch.Tensor, atol=ATOL):
    return np.allclose(a, b.detach().numpy(), atol=atol)


# ------------------------------------------------------------------
# individual op tests
# ------------------------------------------------------------------

def test_add_broadcast():
    x_np = RNG.standard_normal((4, 3))
    b_np = RNG.standard_normal((3,))

    x, b = Tensor(x_np.copy()), Tensor(b_np.copy())
    ((x + b) ** 2).sum().backward()

    xt, bt = _t(x_np), _t(b_np)
    ((xt + bt) ** 2).sum().backward()

    assert close(x.grad, xt.grad), "x.grad"
    assert close(b.grad, bt.grad), "b.grad"


def test_mul_broadcast():
    x_np = RNG.standard_normal((5, 1, 3))
    y_np = RNG.standard_normal((1, 4, 3))

    x, y = Tensor(x_np.copy()), Tensor(y_np.copy())
    (x * y).sum().backward()

    xt, yt = _t(x_np), _t(y_np)
    (xt * yt).sum().backward()

    assert close(x.grad, xt.grad), "x.grad"
    assert close(y.grad, yt.grad), "y.grad"


def test_matmul_2d():
    A_np = RNG.standard_normal((3, 4))
    B_np = RNG.standard_normal((4, 5))

    A, B = Tensor(A_np.copy()), Tensor(B_np.copy())
    (A @ B).sum().backward()

    At, Bt = _t(A_np), _t(B_np)
    (At @ Bt).sum().backward()

    assert close(A.grad, At.grad), "A.grad"
    assert close(B.grad, Bt.grad), "B.grad"


def test_matmul_batched():
    """Batched input (batch x nin) @ weight (nin x nout) — typical Linear layer."""
    X_np = RNG.standard_normal((8, 4))
    W_np = RNG.standard_normal((4, 6))

    X, W = Tensor(X_np.copy()), Tensor(W_np.copy())
    (X @ W).sum().backward()

    Xt, Wt = _t(X_np), _t(W_np)
    (Xt @ Wt).sum().backward()

    assert close(X.grad, Xt.grad), "X.grad"
    assert close(W.grad, Wt.grad), "W.grad"


def test_pow():
    x_np = RNG.standard_normal((4, 4))

    x = Tensor(x_np.copy())
    (x ** 3).sum().backward()

    xt = _t(x_np)
    (xt ** 3).sum().backward()

    assert close(x.grad, xt.grad)


def test_relu():
    x_np = RNG.standard_normal((6, 5))

    x = Tensor(x_np.copy())
    x.relu().sum().backward()

    xt = _t(x_np)
    torch.relu(xt).sum().backward()

    assert close(x.grad, xt.grad)


def test_tanh():
    x_np = RNG.standard_normal((4, 4)) * 0.5

    x = Tensor(x_np.copy())
    x.tanh().sum().backward()

    xt = _t(x_np)
    torch.tanh(xt).sum().backward()

    assert close(x.grad, xt.grad)


def test_exp_log():
    x_np = np.abs(RNG.standard_normal((3, 5))) + 0.1  # positive for log

    x = Tensor(x_np.copy())
    (x.exp() + x.log()).sum().backward()

    xt = _t(x_np)
    (torch.exp(xt) + torch.log(xt)).sum().backward()

    assert close(x.grad, xt.grad)


def test_sum_axis():
    x_np = RNG.standard_normal((3, 4, 5))

    x = Tensor(x_np.copy())
    x.sum(axis=1).sum().backward()

    xt = _t(x_np)
    xt.sum(dim=1).sum().backward()

    assert close(x.grad, xt.grad)


def test_mean():
    x_np = RNG.standard_normal((6, 4))

    x = Tensor(x_np.copy())
    x.mean().backward()

    xt = _t(x_np)
    xt.mean().backward()

    assert close(x.grad, xt.grad)


def test_reshape():
    x_np = RNG.standard_normal((3, 4))

    x = Tensor(x_np.copy())
    x.reshape(2, 6).sum().backward()

    xt = _t(x_np)
    xt.reshape(2, 6).sum().backward()

    assert close(x.grad, xt.grad)


def test_transpose():
    x_np = RNG.standard_normal((3, 5))

    x = Tensor(x_np.copy())
    x.T.sum().backward()

    xt = _t(x_np)
    xt.T.sum().backward()

    assert close(x.grad, xt.grad)


# ------------------------------------------------------------------
# composite / end-to-end
# ------------------------------------------------------------------

def test_linear_layer_forward_backward():
    """y = relu(X @ W + b) — single vectorized layer matches PyTorch Linear."""
    X_np = RNG.standard_normal((16, 8))
    W_np = RNG.standard_normal((8, 4))
    b_np = RNG.standard_normal((4,))

    X, W, b = Tensor(X_np.copy()), Tensor(W_np.copy()), Tensor(b_np.copy())
    loss = (X @ W + b).relu().mean()
    loss.backward()

    Xt = _t(X_np); Wt = _t(W_np); bt = _t(b_np)
    lt = (Xt @ Wt + bt).relu().mean()
    lt.backward()

    assert close(X.grad, Xt.grad), "X.grad"
    assert close(W.grad, Wt.grad), "W.grad"
    assert close(b.grad, bt.grad), "b.grad"
    assert np.isclose(loss.data, lt.item(), atol=ATOL), "forward value"


def test_mlp_gradient_flow():
    """MLP gradients reach all parameters (no dead paths)."""
    from micrograd.nn_tensor import MLP

    np.random.seed(0)
    X = Tensor(RNG.standard_normal((32, 4)))
    y = Tensor(RNG.standard_normal((32, 1)))

    mlp = MLP(4, [16, 8, 1])
    pred = mlp(X)
    loss = ((pred - y) ** 2).mean()
    loss.backward()

    for p in mlp.parameters():
        assert not np.allclose(p.grad, 0.0), f"dead gradient in param shape={p.shape}"


def test_mlp_sgd_step_reduces_loss():
    """One gradient-descent step must reduce training loss."""
    from micrograd.nn_tensor import MLP

    np.random.seed(1)
    X_np = RNG.standard_normal((64, 8))
    y_np = RNG.standard_normal((64, 1))
    X, y = Tensor(X_np), Tensor(y_np)

    mlp = MLP(8, [32, 16, 1])

    def fwd():
        pred = mlp(X)
        return ((pred - y) ** 2).mean()

    loss0 = fwd()
    loss0.backward()

    lr = 0.01
    for p in mlp.parameters():
        p.data -= lr * p.grad

    mlp.zero_grad()
    loss1 = fwd()

    assert loss1.data < loss0.data, (
        f"loss did not decrease: {loss0.data:.6f} -> {loss1.data:.6f}"
    )
