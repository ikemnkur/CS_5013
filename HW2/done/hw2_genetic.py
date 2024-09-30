import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Load and preprocess the CreditCard.csv data
data = pd.read_csv('CreditCard.csv')

# Drop the rows with missing values
data = data.dropna()

# Encode categorical variables
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['CarOwner'] = data['CarOwner'].map({'Y': 1, 'N': 0})
data['PropertyOwner'] = data['PropertyOwner'].map({'Y': 1, 'N': 0})

# Extract features and target variable
X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values
y = data['CreditApprove'].values

def f(x: np.ndarray, w: np.ndarray) -> float:
    # Basic dot product: x * w
    return np.dot(x, w)

def er(w: np.ndarray) -> float:
    predictions = np.dot(X, w)
    # MSE calculation
    return np.mean((predictions - y) ** 2)

def fitness(w: np.ndarray) -> float:
    error = er(w)
    # Use the exponential to match e^{-er(w)} as the fitness
    return np.exp(-error)  

def create_chromosome() -> np.ndarray:
    # swap the bit at random
    return np.random.choice([-1, 1], size=6)

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    # merge the first half of parent1 and the last half of parent2 to form a new child
    return np.concatenate((parent1[:3], parent2[3:]))

def mutate(chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
    for i in range(len(chromosome)):
        # 1% of the time do this
        if np.random.random() < mutation_rate: 
            # Flip the gene
            chromosome[i] = -chromosome[i] 
    return chromosome

def select_parents(population: List[np.ndarray], fitnesses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total_fitness = np.sum(fitnesses)
    if total_fitness == 0 or np.isnan(total_fitness):
        # If all fitnesses are zero or NaN, select randomly
        parents = np.random.choice(len(population), size=2, replace=False)
    else:
        probabilities = fitnesses / total_fitness
        # Handle potential NaN in probabilities
        probabilities = np.nan_to_num(probabilities)
        parents = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    return population[parents[0]], population[parents[1]]

def genetic_algorithm(population_size: int, generations: int, mutation_rate: float) -> Tuple[np.ndarray, List[float]]:
    population = [create_chromosome() for blank in range(population_size)]
    best_er_history = []
    best_w = None
    best_error = float('inf')

    for blank in range(generations):
        fitnesses = np.array([fitness(w) for w in population])
        errors = np.array([er(w) for w in population])
        min_error_index = np.argmin(errors)
        current_best_w = population[min_error_index]
        current_best_error = errors[min_error_index]

        if current_best_error < best_error:
            best_error = current_best_error
            best_w = current_best_w

        best_er_history.append(best_error)

        new_population = []
        for blank in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    return best_w, best_er_history

# Run the genetic algorithm
population_size = 10
generations = 100
mutation_rate = 0.01

optimal_w, er_history = genetic_algorithm(population_size, generations, mutation_rate)

# Plot er(w) vs. generations
plt.figure(figsize=(10, 6))
plt.plot(range(generations), er_history)
plt.title('er(w) vs. Generations')
plt.xlabel('Generation')
plt.ylabel('er(w)')
plt.grid(True)
plt.show()

# Print the optimal w and er(w)
print(f"Optimal w vector: {optimal_w}")
print(f"Minimum er(w): {er(optimal_w)}")
