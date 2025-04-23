import random as rn
import numpy as np
import matplotlib.pyplot as plt
import math

#_____________________________________________________________________________
def random_population(nv, n, lb, ub):
    pop = np.zeros((n, nv))
    for i in range(n):
        pop[i, :] = np.random.uniform(lb, ub)
    return pop

#_____________________________________________________________________________
def crossover(pop, crossover_prob=0.6):
    n, nv = pop.shape
    offspring = []
    for i in range(n // 2):
        if rn.random() < crossover_prob:
            r1 = rn.randint(0, n - 1)
            r2 = rn.randint(0, n - 1)
            while r1 == r2:
                r2 = rn.randint(0, n - 1)
            cutting_point = rn.randint(1, nv - 1)
            child1 = np.concatenate((pop[r1, 0:cutting_point], pop[r2, cutting_point:]))
            child2 = np.concatenate((pop[r2, 0:cutting_point], pop[r1, cutting_point:]))
            offspring.append(child1)
            offspring.append(child2)
    return np.array(offspring) if len(offspring) > 0 else np.zeros((0, nv))

#_____________________________________________________________________________
def mutation(pop, mutation_prob=0.05):
    n, nv = pop.shape
    offspring = []
    for i in range(n):
        if rn.random() < mutation_prob:
            chromosome = pop[i].copy()
            mutation_point = rn.randint(0, nv - 1)
            chromosome[mutation_point] = rn.uniform(lb[mutation_point], ub[mutation_point])
            offspring.append(chromosome)
    return np.array(offspring) if len(offspring) > 0 else np.zeros((0, nv))

#_____________________________________________________________________________
def local_search(pop, n, step_size):
    offspring = np.zeros((n, pop.shape[1]))
    for i in range(n):
        r1 = rn.randint(0, pop.shape[0] - 1)
        chromosome = pop[r1, :].copy()
        r2 = rn.randint(0, pop.shape[1] - 1)
        chromosome[r2] += rn.uniform(-step_size, step_size)
        chromosome[r2] = max(lb[r2], min(ub[r2], chromosome[r2]))
        offspring[i, :] = chromosome
    return offspring

#_____________________________________________________________________________
def evaluation(pop):
    fitness_values = np.zeros((pop.shape[0], 2))
    n = 30  
    for i, chromosome in enumerate(pop):
        f1 = chromosome[0]
        fitness_values[i, 0] = f1
        
        g = 1 + (9/29) * sum(chromosome[1:])  
        
        h = 1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1)
        
        fitness_values[i, 1] = g * h
    return fitness_values

#_____________________________________________________________________________
def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
    normalize_fitness_values = (fitness_values - fitness_values.min(0)) / np.ptp(fitness_values, axis=0)
    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1
        crowding_results[pop_size - 1] = 1
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])
        crowding_results[1:pop_size - 1] = (sorting_normalize_fitness_values[2:pop_size] - sorting_normalize_fitness_values[:pop_size - 2])
        re_sorting = np.argsort(sorting_normalized_values_index)
        matrix_for_crowding[:, i] = crowding_results[re_sorting]
    crowding_distance = np.sum(matrix_for_crowding, axis=1)
    return crowding_distance

#_____________________________________________________________________________
def remove_using_crowding(fitness_values, number_solutions_needed):
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros((number_solutions_needed), dtype=int)
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            selected_pop_index[i] = pop_index[solution_1]
            pop_index = np.delete(pop_index, solution_1)
            fitness_values = np.delete(fitness_values, solution_1, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_1)
        else:
            selected_pop_index[i] = pop_index[solution_2]
            pop_index = np.delete(pop_index, solution_2)
            fitness_values = np.delete(fitness_values, solution_2, axis=0)
            crowding_distance = np.delete(crowding_distance, solution_2)
    return selected_pop_index

#_____________________________________________________________________________
def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0
                break
    return pop_index[pareto_front]

#_____________________________________________________________________________
def selection(pop, fitness_values, pop_size):
    pop_index_0 = np.arange(pop.shape[0])
    pop_index = np.arange(pop.shape[0])
    pareto_front_index = []
    while len(pareto_front_index) < pop_size:
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed)
            new_pareto_front = new_pareto_front[selected_solutions]
        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))
    selected_pop = pop[pareto_front_index.astype(int)]
    return selected_pop

#_____________________________________________________________________________
# Parameters
nv = 30  
lb = [0] * 30  
ub = [1] * 30  
pop_size = 200  
crossover_prob = 0.6
mutation_prob = 0.05
rate_local_search = 30
step_size = 0.02
pop = random_population(nv, pop_size, lb, ub)

#_____________________________________________________________________________
# Main loop of NSGA II
for i in range(300):  
    offspring_from_crossover = crossover(pop, crossover_prob)
    offspring_from_mutation = mutation(pop, mutation_prob)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)
    pop = np.vstack((pop, offspring_from_crossover, offspring_from_mutation, offspring_from_local_search))
    fitness_values = evaluation(pop)
    pop = selection(pop, fitness_values, pop_size)
    print(f"iteration {i}")

#_____________________________________________________________________________
# Pareto front visualization
fitness_values = evaluation(pop)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]

unique_indices = np.unique(pop, axis=0, return_index=True)[1]
pop = pop[unique_indices]
fitness_values = fitness_values[unique_indices]
print("_________________")
print("Optimal solutions:")
print("     x1           x2           ...           x30")
print(pop)
print("______________")
print("Fitness values:")
print("objective 1  objective 2")
print("      |          |")
print(fitness_values)
plt.scatter(fitness_values[:, 0], fitness_values[:, 1], s=10, color='red', marker='s')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.grid(True, color='black', linewidth=0.5)
plt.title('Pareto Front')
plt.xlim(0, 0.9)     
plt.ylim(-0.8, 1)    
plt.gca().set_facecolor('white')
plt.show()