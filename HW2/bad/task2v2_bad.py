import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Load and preprocess the data
data = pd.read_csv('CreditCard.csv')
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['CarOwner'] = data['CarOwner'].map({'Y': 1, 'N': 0})
data['PropertyOwner'] = data['PropertyOwner'].map({'Y': 1, 'N': 0})
X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values
y = data['CreditApprove'].values

def f(x: np.ndarray, w: np.ndarray) -> float:
    return np.dot(x, w)

def er(w: np.ndarray) -> float:
    predictions = np.array([f(x, w) for x in X])
    return np.mean((predictions - y) ** 2)

def create_chromosome() -> np.ndarray:
    return np.random.choice([-1, 1], size=6)

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    return np.concatenate((parent1[:3], parent2[3:]))

def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
    return np.array([gene if np.random.random() > mutation_rate else -gene for gene in chromosome])

def select_parents(population: List[np.ndarray], errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Rank-based selection
    ranks = np.argsort(np.argsort(errors))
    probabilities = (len(population) - ranks) / np.sum(len(population) - ranks)
    parents = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    return population[parents[0]], population[parents[1]]

def genetic_algorithm(population_size: int, generations: int, mutation_rate: float) -> Tuple[np.ndarray, List[float]]:
    population = [create_chromosome() for _ in range(population_size)]
    best_er_history = []

    for _ in range(generations):
        errors = np.array([er(w) for w in population])
        best_w = population[np.argmin(errors)]
        best_er_history.append(er(best_w))

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, errors)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    errors = np.array([er(w) for w in population])
    best_w = population[np.argmin(errors)]
    return best_w, best_er_history

# Run the genetic algorithm
population_size = 20
generations = 100
mutation_rate = 0.01

optimal_w, er_history = genetic_algorithm(population_size, generations, mutation_rate)

# Plot er(w) vs. generations
plt.figure(figsize=(10, 6))
plt.plot(range(generations), er_history)
plt.title('Error Rate vs. Generations')
plt.xlabel('Generation')
plt.ylabel('Error Rate')
plt.grid(True)
plt.savefig('error_rate_vs_generations.png')
plt.close()

# Print the optimal w and er(w)
print(f"Optimal w: {optimal_w}")
print(f"er(w): {er(optimal_w)}")