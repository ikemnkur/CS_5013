import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('CreditCard.csv')

# Drop rows with missing values
data = data.dropna()

# Encode categorical variables
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['CarOwner'] = data['CarOwner'].map({'Y': 1, 'N': 0})
data['PropertyOwner'] = data['PropertyOwner'].map({'Y': 1, 'N': 0})

# Extract features and target variable
X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values
y = data['CreditApprove'].values
n_samples = len(y)

def compute_er(w, X, y):
    """
    Compute the MSE (mean squared error) er(w) for a given weight vector w.
    """
    f_x = np.dot(X, w)
    er = (1 / n_samples) * np.sum((f_x - y) ** 2)
    return er

def compute_fitness(w, X, y):
    """
    Compute the fitness of a weight vector, vector w = e^{-er(w)}.
    """
    er = compute_er(w, X, y)
    fitness = np.exp(-er)
    return fitness

def initialize_population(pop_size, chrom_length):
    """
    Initialize the population with some random chromosomes.
    """
    population = []
    for _ in range(pop_size):
        w = np.random.choice([-1, 1], size=chrom_length)
        population.append(w)
    return population

def crossover(parent1, parent2):
    """
    Perform crossover operation between two parents to produce a child object.
    """
    crossover_point = len(parent1) // 2
    child = np.empty(len(parent1), dtype=int)
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]
    return child

def mutate(chromosome, mutation_rate):
    """
    Mutate a chromosome by flipping bits with a probability given by a mutation rate.
    """
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = -chromosome[i]  # Flip the gene
    return chromosome

# Parameters  for the Genetic Algorithm 
population_size = 20
chromosome_length = 6
number_of_generations = 100
mutation_rate = 0.1

# Initialize population
population = initialize_population(population_size, chromosome_length)

# Lists to store the best error at each generation
best_er_list = []
best_w = None
best_er = float('inf')

# Genetic Algorithm loop
for generation in range(number_of_generations):
    # Compute fitness and error for each chromosome
    fitnesses = np.array([compute_fitness(w, X, y) for w in population])
    er_list = np.array([compute_er(w, X, y) for w in population])
    
    # Record the best chromosome
    min_er = np.min(er_list)
    min_er_index = np.argmin(er_list)
    if min_er < best_er:
        best_er = min_er
        best_w = population[min_er_index]
    
    best_er_list.append(best_er)
    
    # Normalize fitness to get selection probabilities
    total_fitness = np.sum(fitnesses)
    probabilities = fitnesses / total_fitness
    
    # Create new population
    new_population = []
    for _ in range(population_size // 2):
        # Select two parents based on probabilities
        parents_indices = np.random.choice(population_size, size=2, p=probabilities)
        parent1 = population[parents_indices[0]]
        parent2 = population[parents_indices[1]]
        
        # Perform crossover
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)
        
        # Apply mutation
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        
        # Add offspring to new population
        new_population.extend([child1, child2])
    
    population = new_population

# Plot er(w) versus generation
plt.plot(range(number_of_generations), best_er_list)
plt.xlabel('Generation')
plt.ylabel('er(w)')
plt.title('er(w) vs Generation')
plt.show()

# Present the optimal w and er(w)
print("Optimal w:", best_w)
print("er(w):", best_er)
