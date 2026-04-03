import numpy as np
from sklearn.datasets import fetch_openml


def load_preprocess():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    X = X.reshape(-1, 1, 28, 28)  # (N, depth, H, W)
    y_onehot = np.zeros((y.size, 10))
    y_onehot[np.arange(y.size), y] = 1
    return X, y_onehot


def k_fold_split(X, y, k=5):
    N = X.shape[0]
    indices = np.random.permutation(N)
    fold_size = N // k
    folds = []
    for i in range(k):
        val_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.concatenate([
            indices[:i * fold_size],
            indices[(i + 1) * fold_size:]
        ])
        folds.append({
            'X_train': X[train_idx], 'y_train': y[train_idx],
            'X_val': X[val_idx], 'y_val': y[val_idx]
        })
    return folds


# =============================================================================
# LAYERS
# =============================================================================

class CNNFilter():

    def __init__(self, size, depth):
        rng = np.random.default_rng()
        shape = (depth, size, size)
        fan_in = depth * size * size
        self.kernel = rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)
        self.bias = 0.0

    def forward(self, input, padding=1):
        self.input = input
        self.padding = padding

        H, W = input.shape[1], input.shape[2]
        kH, kW = self.kernel.shape[1], self.kernel.shape[2]

        padded = np.pad(input, ((0, 0), (padding, padding), (padding, padding)))

        output = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                patch = padded[:, i:i+kH, j:j+kW]
                output[i, j] = np.sum(patch * self.kernel) + self.bias
        return output

    def backward(self, dout, alpha):
        """
        dout: (H, W) — gradient of loss w.r.t. this filter's output
        
        Three gradients needed:
        1. dkernel: how much to change each weight
           - Same logic as ANN: "error × input"
           - For each output position (i,j), the patch that contributed gets 
             multiplied by dout[i,j] and accumulated into dkernel
        
        2. dbias: just sum of all dout (same as ANN bias gradient)
        
        3. dinput: gradient to pass to previous layer
           - During forward, each patch contributed to one output position
           - During backward, scatter each output's gradient back to the 
             patch positions, weighted by kernel values
        """
        padding = self.padding
        depth, H, W = self.input.shape
        kH, kW = self.kernel.shape[1], self.kernel.shape[2]

        padded = np.pad(self.input, ((0, 0), (padding, padding), (padding, padding)))

        # 1. Gradient w.r.t. kernel weights
        dkernel = np.zeros_like(self.kernel)
        for i in range(H):
            for j in range(W):
                patch = padded[:, i:i+kH, j:j+kW]
                dkernel += patch * dout[i, j] ## wokring vertically
                # ^ same as ANN's "error × input" but spatially local

        # 2. Gradient w.r.t. bias
        dbias = np.sum(dout)

        # 3. Gradient w.r.t. input (to pass backward)
        # For each output position, the gradient flows back through the kernel
        # to the input positions that were in that patch
        dinput_padded = np.zeros_like(padded)
        for i in range(H):
            for j in range(W):
                dinput_padded[:, i:i+kH, j:j+kW] += self.kernel * dout[i, j]  ##???

        # Remove the padding to get gradient w.r.t. original input
        dinput = dinput_padded[:, padding:padding+H, padding:padding+W]

        # Update weights
        self.kernel -= alpha * dkernel
        self.bias -= alpha * dbias

        return dinput


class CNNLayer():

    def __init__(self, num_filters, kernel_depth, size):
        self.kernels = []
        for i in range(num_filters):
            self.kernels.append(CNNFilter(size, kernel_depth))

    def forward(self, input):
        self.input = input
        activation_maps = []
        for kernel in self.kernels:
            activation_maps.append(kernel.forward(input))
        return np.array(activation_maps)  # (num_filters, H, W)

    def backward(self, dout, alpha):
        """
        dout: (num_filters, H, W)
        
        Each filter produced one feature map during forward.
        Each filter's backward gives a dinput contribution.
        We SUM all contributions because the same input fed into all filters.
        """
        dinput = np.zeros_like(self.input)
        for i, kernel in enumerate(self.kernels):
            dinput += kernel.backward(dout[i], alpha)
        return dinput


class Relu():

    def forward(self, input):
        self.input = input  # save for backward
        return np.maximum(0, input)

    def backward(self, dout):
        """
        ReLU derivative: 1 where input > 0, else 0
        
        YOUR BUG: you checked (inputGrad > 0) — that checks if the GRADIENT 
        is positive. We need to check if the ORIGINAL INPUT was positive.
        Then multiply by incoming gradient to chain rule it through.
        """
        return dout * (self.input > 0).astype(float)


class MaxPool():

    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def forward(self, input):
        self.input_shape = input.shape  # save full shape for backward
        depth, H, W = input.shape
        outH = H // self.stride
        outW = W // self.stride

        output = np.zeros((depth, outH, outW))
        self.max_indices = []  # reset each forward pass

        for d in range(depth):
            for i in range(outH):
                for j in range(outW):
                    si = i * self.stride
                    sj = j * self.stride
                    patch = input[d, si:si+self.size, sj:sj+self.size]
                    max_val = np.max(patch)
                    output[d, i, j] = max_val

                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                    self.max_indices.append((d, si + max_idx[0], sj + max_idx[1]))

        return output

    def backward(self, dout):
        """
        Gradient only flows to the position that was the max.
        
        YOUR BUG: you looped over ALL input positions and checked 
        'if target_idx in self.Maxindices' — O(n²) and wrong mapping.
        
        Instead: iterate through stored indices in order (same order as output).
        Each output position maps to exactly one input position.
        """
        dinput = np.zeros(self.input_shape)
        depth, outH, outW = dout.shape

        idx = 0
        for d in range(depth):
            for i in range(outH):
                for j in range(outW):
                    d_orig, i_orig, j_orig = self.max_indices[idx]
                    dinput[d_orig, i_orig, j_orig] = dout[d, i, j]
                    idx += 1

        return dinput


class Flatten():

    def __init__(self):
        self.shape = None

    def forward(self, input):
        self.shape = input.shape  # FIX: store ALL 3 dims, not just 2
        return input.flatten()

    def backward(self, dout):
        return dout.reshape(self.shape)


class FC():
    """Fully Connected layer — same as your ANN but with corrected backward."""

    def __init__(self, output_size, input_size):
        rng = np.random.default_rng()
        self.weights = rng.standard_normal((output_size, input_size)) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input  # save for backward
        return self.weights @ input + self.bias

    def backward(self, dout, alpha):
        dW = dout @ self.input.T
        db = dout
        dinput = self.weights.T @ dout  # use old weights FIRST

        self.weights -= alpha * dW     # update AFTER
        self.bias -= alpha * db

        return dinput


class Softmax():

    def forward(self, input):
        e_x = np.exp(input - np.max(input))
        return e_x / e_x.sum()


def cross_entropy_loss(output, label):
    return -np.sum(label * np.log(output + 1e-8))  # 1e-8 prevents log(0)


# =============================================================================
# TRAINING
# =============================================================================

def main():

    ## Load and preprocess
    print("Loading MNIST...")
    X, y = load_preprocess()
    folds = k_fold_split(X, y, k=5)

    fold = folds[0]
    X_train, y_train = fold['X_train'], fold['y_train']
    X_val, y_val = fold['X_val'], fold['y_val']

    ## Initialize layers
    ## CRITICAL FIX: separate instances for each usage!
    ## You were reusing Relu1 and maxPool1 — self.input gets overwritten
    ## on the second forward call, so backprop through the first layer
    ## would use the wrong saved input.

    conv1 = CNNLayer(8, 1, 3)
    relu1 = Relu()            # separate relu for conv block 1
    pool1 = MaxPool(2, 2)     # separate pool for conv block 1

    conv2 = CNNLayer(16, 8, 3)
    relu2 = Relu()            # separate relu for conv block 2
    pool2 = MaxPool(2, 2)     # separate pool for conv block 2

    flatten = Flatten()
    fc1 = FC(64, 784)         # 16*7*7 = 784
    relu3 = Relu()            # separate relu for FC
    fc2 = FC(10, 64)
    softmax = Softmax()

    ## Training parameters
    alpha = 0.0005
    epochs = 5

    ## IMPORTANT: Pure NumPy CNN is SLOW — ~1-2 seconds per image
    ## Train on a small subset first to verify it works
    train_size = 1000  # start small, increase once you verify it learns
    val_size = 200

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        ## Shuffle training data each epoch
        shuffle_idx = np.random.permutation(train_size)

        for n in range(train_size):
            idx = shuffle_idx[n]
            image = X_train[idx]              # (1, 28, 28)
            label = y_train[idx]              # (10,)

            # =========== FORWARD PASS ===========

            # Conv block 1
            out = conv1.forward(image)        # (8, 28, 28)
            out = relu1.forward(out)          # (8, 28, 28)
            out = pool1.forward(out)          # (8, 14, 14)

            # Conv block 2
            out = conv2.forward(out)          # (16, 14, 14)
            out = relu2.forward(out)          # (16, 14, 14)
            out = pool2.forward(out)          # (16, 7, 7)

            # FC layers
            out = flatten.forward(out)        # (784,)
            out = out.reshape(-1, 1)          # (784, 1)
            out = fc1.forward(out)            # (64, 1)
            out = relu3.forward(out)          # (64, 1)
            out = fc2.forward(out)            # (10, 1)
            out = out.flatten()               # (10,)
            output = softmax.forward(out)     # (10,)

            # =========== LOSS ===========

            loss = cross_entropy_loss(output, label)
            total_loss += loss

            if np.argmax(output) == np.argmax(label):
                correct += 1

            # =========== BACKWARD PASS ===========

            # Softmax + Cross-Entropy combined gradient
         
            output_col = output.reshape(-1, 1)  # (10, 1)
            label_col = label.reshape(-1, 1)    # (10, 1)
            dout = output_col - label_col       # (10, 1)
            np.clip(dout, -1.0, 1.0, out=dout)

            # FC layers backward
            dout = fc2.backward(dout, alpha)    # (64, 1)
            dout = relu3.backward(dout)         # (64, 1)
            dout = fc1.backward(dout, alpha)    # (784, 1)

            # Reshape back to conv dimensions
            dout = dout.flatten()               # (784,)
            dout = flatten.backward(dout)       # (16, 7, 7)

            # Conv block 2 backward
            dout = pool2.backward(dout)         # (16, 14, 14)
            dout = relu2.backward(dout)         # (16, 14, 14)
            dout = conv2.backward(dout, alpha)  # (8, 14, 14)

            # Conv block 1 backward
            dout = pool1.backward(dout)         # (8, 28, 28)
            dout = relu1.backward(dout)         # (8, 28, 28)
            dout = conv1.backward(dout, alpha)  # (1, 28, 28) — we don't need this

            # =========== LOGGING ===========

            if n % 100 == 0:
                print(f"  Epoch {epoch+1}, Image {n}/{train_size}, Loss: {loss:.4f}")

        avg_loss = total_loss / train_size
        train_acc = correct / train_size * 100
        print(f"Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # =========== VALIDATION ===========

        val_correct = 0
        for idx in range(val_size):
            image = X_val[idx]
            label = y_val[idx]

            out = conv1.forward(image)
            out = relu1.forward(out)
            out = pool1.forward(out)
            out = conv2.forward(out)
            out = relu2.forward(out)
            out = pool2.forward(out)
            out = flatten.forward(out)
            out = out.reshape(-1, 1)
            out = fc1.forward(out)
            out = relu3.forward(out)
            out = fc2.forward(out)
            out = out.flatten()
            output = softmax.forward(out)

            if np.argmax(output) == np.argmax(label):
                val_correct += 1

        val_acc = val_correct / val_size * 100
        print(f"         Validation Accuracy: {val_acc:.2f}%")
        print()


if __name__ == "__main__":
    main()







     









    
        








    
        



        


        
    
        

        






        



        

