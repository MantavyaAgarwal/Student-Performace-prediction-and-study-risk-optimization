"""
pso.py
------
Particle Swarm Optimization (PSO) used to optimize the weight vector of
`SimpleNN`, giving a second (swarm-intelligence) alternative to
backpropagation and the Genetic Algorithm.

Each particle = one candidate weight vector.
Each particle keeps a velocity, a personal best position, and is attracted
towards both its own personal best and the swarm's global best.
"""

import numpy as np
from .neural_network import SimpleNN


def run_pso(X, y, hidden_dim=8, n_particles=30, iterations=60,
            w=0.7, c1=1.5, c2=1.5, seed=11):
    rng = np.random.default_rng(seed)
    nn = SimpleNN(input_dim=X.shape[1], hidden_dim=hidden_dim, seed=seed)
    n_weights = nn.num_weights

    positions = rng.normal(0, 0.6, (n_particles, n_weights))
    velocities = rng.normal(0, 0.1, (n_particles, n_weights))

    def mse_of(pos):
        nn.set_weights(pos)
        return nn.mse(X, y)

    personal_best_pos = positions.copy()
    personal_best_mse = np.array([mse_of(p) for p in positions])

    global_best_idx = np.argmin(personal_best_mse)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_mse = personal_best_mse[global_best_idx]

    history = []

    for it in range(iterations):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)

        velocities = (w * velocities
                      + c1 * r1 * (personal_best_pos - positions)
                      + c2 * r2 * (global_best_pos - positions))
        positions = positions + velocities

        mses = np.array([mse_of(p) for p in positions])

        improved = mses < personal_best_mse
        personal_best_pos[improved] = positions[improved]
        personal_best_mse[improved] = mses[improved]

        it_best_idx = np.argmin(personal_best_mse)
        if personal_best_mse[it_best_idx] < global_best_mse:
            global_best_mse = personal_best_mse[it_best_idx]
            global_best_pos = personal_best_pos[it_best_idx].copy()

        history.append({
            "iteration": it,
            "best_mse": round(float(global_best_mse), 5),
            "avg_mse": round(float(mses.mean()), 5),
        })

    nn.set_weights(global_best_pos)
    return nn, history, float(global_best_mse)


if __name__ == "__main__":
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    nn, hist, mse = run_pso(X, y, iterations=20)
    print("final MSE:", mse)
    print(hist[-1])
