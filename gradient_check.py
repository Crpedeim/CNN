"""
gradient_check.py — verify the from-scratch backprop with finite differences.

For a correct implementation, the analytical gradient (from backward()) and the
numerical gradient (from perturbing a parameter and measuring the loss change)
should agree to within a tiny relative error (< 1e-5 with float64).

We use a small network to keep the pure-Python loops fast; it exercises the same
layer classes (Conv, MaxPool, ReLU, Flatten, FC, Softmax + cross-entropy) used
in the full model.
"""
import numpy as np
from cnn import (CNNLayer, Relu, MaxPool, Flatten, FC, Softmax,
                      cross_entropy_loss)

EPS = 1e-5
np.random.seed(0)


def build_net():
    return {
        "conv1": CNNLayer(2, 1, 3),   # 2 filters, depth 1, 3x3
        "relu1": Relu(),
        "pool1": MaxPool(2, 2),
        "flatten": Flatten(),
        "fc1": FC(6, 32),          # 2*4*4 = 32  -> 6
        "relu2": Relu(),
        "fc2": FC(4, 6),             # 6 -> 4 classes
        "softmax": Softmax(),
    }


def forward(net, x):
    o = net["conv1"].forward(x)
    o = net["relu1"].forward(o)
    o = net["pool1"].forward(o)          # (2, 2, 2)
    o = net["flatten"].forward(o)        # (8,)
    o = o.reshape(-1, 1)
    o = net["fc1"].forward(o)
    o = net["relu2"].forward(o)
    o = net["fc2"].forward(o)
    o = o.flatten()
    return net["softmax"].forward(o)


def loss_of(net, x, y):
    return cross_entropy_loss(forward(net, x), y)


def analytical_grads(net, x, y):
    out = forward(net, x)
    dout = (out.reshape(-1, 1) - y.reshape(-1, 1))   # softmax+CE combined grad
    dout = net["fc2"].backward(dout, 0.0)
    dout = net["relu2"].backward(dout)
    dout = net["fc1"].backward(dout, 0.0)
    dout = net["flatten"].backward(dout.flatten())
    dout = net["pool1"].backward(dout)
    dout = net["relu1"].backward(dout)
    net["conv1"].backward(dout, 0.0)


def rel_err(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-8)


def check_param(net, x, y, arr, name, n=8):
    """Numerically check n random entries of a parameter array."""
    flat = arr.reshape(-1)
    idxs = np.random.choice(flat.size, min(n, flat.size), replace=False)
    worst = 0.0
    for k in idxs:
        orig = flat[k]
        flat[k] = orig + EPS; lp = loss_of(net, x, y)
        flat[k] = orig - EPS; lm = loss_of(net, x, y)
        flat[k] = orig
        num = (lp - lm) / (2 * EPS)
        ana = _stored_grad(net, name).reshape(-1)[k]
        worst = max(worst, rel_err(ana, num))
    return worst


def _stored_grad(net, name):
    if name == "conv1_k0": return net["conv1"].kernels[0].dkernel
    if name == "conv1_k1": return net["conv1"].kernels[1].dkernel
    if name == "fc1_W":    return net["fc1"].dW
    if name == "fc2_W":    return net["fc2"].dW


def main():
    net = build_net()
    x = np.random.randn(1, 8, 8) * 0.5
    y = np.zeros(4); y[np.random.randint(4)] = 1.0

    analytical_grads(net, x, y)   # populates stored grads

    print("Finite-difference gradient check (relative error, want < 1e-5):\n")
    checks = {
        "conv1 filter[0] kernel": ("conv1_k0", net["conv1"].kernels[0].kernel),
        "conv1 filter[1] kernel": ("conv1_k1", net["conv1"].kernels[1].kernel),
        "fc1 weights":            ("fc1_W",    net["fc1"].weights),
        "fc2 weights":            ("fc2_W",    net["fc2"].weights),
    }
    all_pass = True
    for label, (name, arr) in checks.items():
        # grads must be re-computed fresh each check (weights unchanged, alpha=0)
        analytical_grads(net, x, y)
        err = check_param(net, x, y, arr, name)
        status = "PASS" if err < 1e-5 else "FAIL"
        if err >= 1e-5: all_pass = False
        print(f"  {label:24s}  max rel error = {err:.2e}   [{status}]")

    print("\n" + ("All gradients verified correct." if all_pass
                  else "Some gradients FAILED — investigate."))


if __name__ == "__main__":
    main()
