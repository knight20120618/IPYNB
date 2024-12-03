# %% [markdown]
# 輪盤法

# %%
import random

def create_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

def fitness(individual):
    return sum(individual)

def roulette_wheel_selection(population):
    total_fitness = sum(fitness(individual) for individual in population)
    selection_probs = [fitness(individual) / total_fitness for individual in population]
    selected = random.choices(population, weights=selection_probs, k=1)
    return selected[0]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm(population_size, individual_length, generations, mutation_rate=0.01):
    population = [create_individual(individual_length) for _ in range(population_size)]
    
    for generation in range(generations):
        new_population = []
        
        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)
            
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.append(child1)
            new_population.append(child2)
        
        population = new_population[:population_size]

        best_individual = max(population, key=fitness)
        print(f'世代 {generation +1}: 分數 = {fitness(best_individual)}')
    
    return max(population, key=fitness)

population_size = 10 # 族群大小
individual_length = 10 # 個體基因長度
generations = 4 # 迭代代數

best_individual = genetic_algorithm(population_size, individual_length, generations)
print('基因:', best_individual)
print('分數:', fitness(best_individual))

# %% [markdown]
# 競賽法

# %%
import random

def create_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

def fitness(individual):
    return sum(individual)

def tournament_selection(population, k=3):
    tournament = random.sample(population, k)
    best_individual = max(tournament, key=fitness)
    return best_individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm(population_size, individual_length, generations, mutation_rate=0.01):
    population = [create_individual(individual_length) for _ in range(population_size)]
    
    for generation in range(generations):
        new_population = []
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.append(child1)
            new_population.append(child2)

        population = new_population[:population_size]

        best_individual = max(population, key=fitness)
        print(f'世代 {generation + 1}: 分數 = {fitness(best_individual)}')
    
    return max(population, key=fitness)

population_size = 10 # 群體大小
individual_length = 10 # 個體基因長度
generations = 4 # 迭代代數

best_individual = genetic_algorithm(population_size, individual_length, generations)
print('基因:', best_individual)
print('分數:', fitness(best_individual))


