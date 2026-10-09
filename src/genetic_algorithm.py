"""
genetic_algorithm.py
---------------------
A Genetic Algorithm (GA) used to optimize the weight vector of `SimpleNN`
as an alternative to backpropagation, demonstrating:

    - Chromosome encoding (flattened NN weight vector, real-valued genes)
    - Fitness function (negative MSE on the training set)
    - Selection (tournament selection)
    - Crossover (uniform crossover)
    - Mutation (gaussian mutation with decaying rate)
    - Elitism
    - Generational convergence history (for the frontend chart)
"""

import numpy as np
from .neural_network import SimpleNN


def _fitness(weights, nn: SimpleNN, X, y):
    nn.set_weights(weights)
    return -nn.mse(X, y)  # higher is better


def _tournament_select(pop, fitnesses, k=3, rng=None):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmax(fitnesses[idx])]
    return pop[best]


def _uniform_crossover(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child


def _mutate(child, rate, strength, rng):
    mask = rng.random(len(child)) < rate
    child = child.copy()
    child[mask] += rng.normal(0, strength, size=mask.sum())
    return child


def run_ga(X, y, hidden_dim=8, pop_size=40, generations=60,
           mutation_rate=0.15, mutation_strength=0.6, elitism=2, seed=7):
    rng = np.random.default_rng(seed)
    nn = SimpleNN(input_dim=X.shape[1], hidden_dim=hidden_dim, seed=seed)
    n_weights = nn.num_weights

    population = rng.normal(0, 0.6, (pop_size, n_weights))
    history = []
    best_weights = population[0]
    best_fitness = -np.inf

    for gen in range(generations):
        fitnesses = np.array([_fitness(ind, nn, X, y) for ind in population])

        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_weights = population[gen_best_idx].copy()

        history.append({
            "generation": gen,
            "best_mse": round(float(-best_fitness), 5),
            "avg_mse": round(float(-fitnesses.mean()), 5),
        })

        # elitism
        elite_idx = np.argsort(fitnesses)[-elitism:]
        new_population = [population[i].copy() for i in elite_idx]

        # decaying mutation strength for exploitation over time
        cur_strength = mutation_strength * (1 - gen / generations) + 0.05

        while len(new_population) < pop_size:
            parent1 = _tournament_select(population, fitnesses, rng=rng)
            parent2 = _tournament_select(population, fitnesses, rng=rng)
            child = _uniform_crossover(parent1, parent2, rng)
            child = _mutate(child, mutation_rate, cur_strength, rng)
            new_population.append(child)

        population = np.array(new_population)

    nn.set_weights(best_weights)
    return nn, history, float(-best_fitness)


if __name__ == "__main__":
    X = np.random.rand(50, 5)
    y = np.random.rand(50)
    nn, hist, mse = run_ga(X, y, generations=20)
    print("final MSE:", mse)
    print(hist[-1])
