"""
neural_network.py
------------------
A small feed-forward neural network implemented from scratch in NumPy:

    Input(5) -> Hidden(H, sigmoid) -> Output(1, sigmoid)

The same architecture/weight-vector representation is reused by three
different trainers so they can be fairly compared:

    1. Backpropagation (gradient descent)   -> see train_backprop()
    2. Genetic Algorithm                    -> see genetic_algorithm.py
    3. Particle Swarm Optimization           -> see pso.py

Because GA/PSO are population/swarm based *weight optimizers*, the network
exposes get_weights()/set_weights() to flatten/unflatten all parameters into
a single vector, which is exactly what GA and PSO operate on.
"""

import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class SimpleNN:
    def __init__(self, input_dim=5, hidden_dim=8, output_dim=1, seed=42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = rng.normal(0, 0.5, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.5, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    # -- weight <-> vector helpers (used by GA / PSO) ---------------------
    @property
    def num_weights(self):
        return (self.input_dim * self.hidden_dim + self.hidden_dim +
                self.hidden_dim * self.output_dim + self.output_dim)

    def get_weights(self):
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()
        ])

    def set_weights(self, flat):
        i = 0
        n = self.input_dim * self.hidden_dim
        self.W1 = flat[i:i + n].reshape(self.input_dim, self.hidden_dim); i += n
        n = self.hidden_dim
        self.b1 = flat[i:i + n]; i += n
        n = self.hidden_dim * self.output_dim
        self.W2 = flat[i:i + n].reshape(self.hidden_dim, self.output_dim); i += n
        n = self.output_dim
        self.b2 = flat[i:i + n]; i += n

    # -- forward pass -------------------------------------------------------
    def forward(self, X, cache=False):
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        if cache:
            return a2, (X, z1, a1, z2, a2)
        return a2

    def predict(self, X):
        return self.forward(X).ravel()

    def mse(self, X, y):
        y = y.reshape(-1, 1)
        pred = self.forward(X)
        return float(np.mean((pred - y) ** 2))

    # -- training method 1: backpropagation ---------------------------------
    def train_backprop(self, X, y, epochs=500, lr=0.5):
        y = y.reshape(-1, 1)
        n = X.shape[0]
        history = []
        for epoch in range(epochs):
            pred, (Xc, z1, a1, z2, a2) = self.forward(X, cache=True)

            d_out = (a2 - y) * a2 * (1 - a2) * 2 / n         # dL/dz2
            dW2 = a1.T @ d_out
            db2 = d_out.sum(axis=0)

            d_hidden = (d_out @ self.W2.T) * a1 * (1 - a1)   # dL/dz1
            dW1 = Xc.T @ d_hidden
            db1 = d_hidden.sum(axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if epoch % 5 == 0 or epoch == epochs - 1:
                history.append({"epoch": epoch, "mse": self.mse(X, y.ravel())})
        return history


if __name__ == "__main__":
    nn = SimpleNN()
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    print("weights:", nn.num_weights)
    hist = nn.train_backprop(X, y, epochs=50)
    print(hist[-1])
