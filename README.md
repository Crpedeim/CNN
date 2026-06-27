# CNN from Scratch in NumPy

A convolutional neural network built entirely from first principles in **pure NumPy** — every forward and backward pass derived and implemented by hand, with **no deep-learning frameworks** (no PyTorch, no TensorFlow, no autograd). Trained and evaluated on MNIST.

The goal wasn't to beat a benchmark — it was to understand exactly what `loss.backward()` does by deriving and coding every gradient myself.

---

## What's implemented from scratch

Every layer implements its own `forward` and `backward`, with gradients derived by hand:

- **Convolution** — sliding-window conv with padding; manual gradients for kernel weights (`dkernel`), bias (`dbias`), and input (`dinput`, scattered back through the kernel for the upstream layer).
- **Max pooling** — argmax indices cached on the forward pass so gradients route *only* to the positions that were the max.
- **ReLU** — gradient gated on the original pre-activation input (`input > 0`), not the incoming gradient.
- **Fully-connected** — `dW`, `db`, and `dinput` computed with the pre-update weights before the SGD step.
- **Softmax + cross-entropy** — the combined gradient simplifies to `(prediction − label)`, derived and used directly.

Supporting pieces: **He initialization** (`√(2/fan_in)`) for the ReLU layers, **5-fold cross-validation** split, **gradient clipping** for stability, and per-sample SGD.

---

## Architecture

```
Input  (1, 28, 28)
   │
   ├─ Conv  8 filters, 3×3, pad 1   → ReLU → MaxPool 2×2   →  (8, 14, 14)
   │
   ├─ Conv 16 filters, 3×3, pad 1   → ReLU → MaxPool 2×2   →  (16, 7, 7)
   │
   ├─ Flatten                                              →  (784,)
   │
   ├─ FC 784 → 64   → ReLU
   │
   └─ FC  64 → 10   → Softmax                              →  (10,)

Loss: cross-entropy   •   Optimizer: SGD (lr = 5e-4)   •   Init: He
```

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **~87%** |
| Trained on | 1,000 images (subset) |
| Validated on | 200 images |
| Epochs | 5 |

> **Honest scope note:** because every operation runs in pure-Python loops (no vectorization / im2col), a full pass over 60k images is impractically slow, so training uses a subset to prove the implementation *learns correctly*. The point of this project is a verified-correct from-scratch implementation, not a leaderboard number. With an im2col rewrite (see below) the same network trains on full MNIST.

---

## Run it

```bash
pip install numpy scikit-learn
python cnn.py
```

The script downloads MNIST via `sklearn.datasets.fetch_openml`, builds the 5-fold split, and trains on the first fold, printing per-epoch loss, train accuracy, and validation accuracy.

---

## What I learned

- Why the **softmax + cross-entropy** gradient collapses to such a clean form — and how much numerical pain that saves.
- That **max-pool backprop** is really just bookkeeping: cache where the max came from, route the gradient straight back there.
- Convolution backprop as the mirror of the forward pass — accumulate weight gradients from the patches, scatter input gradients back through the kernel.
- Subtle correctness traps: gating ReLU on the input not the gradient, and using a layer's pre-update weights to compute its input gradient before stepping.

## What I'd improve next

- **Vectorize with im2col** to replace the Python loops — the single biggest speedup, and what makes full-MNIST training feasible.
- **Numerical gradient checking** (finite differences) to formally verify every analytical gradient.
- Mini-batching + a momentum/Adam optimizer; loss/accuracy curve plots and a confusion matrix.
