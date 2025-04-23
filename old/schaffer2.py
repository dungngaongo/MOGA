import random as rn
import numpy as np
import matplotlib.pyplot as plt
 
#_____________________________________________________________________________
def random_population(nv,n,lb,ub):
    pop=np.zeros((n, nv)) 
    for i in range(n):
        pop[i,:] = np.random.uniform(lb,ub)
 
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
            child1 = (pop[r1] + pop[r2]) / 2
            child2 = (pop[r1] + pop[r2]) / 2
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
            chromosome[0] = rn.uniform(lb[0], ub[0])
            offspring.append(chromosome)
    
    return np.array(offspring) if len(offspring) > 0 else np.zeros((0, nv))
#_____________________________________________________________________________
def local_search(pop, n, step_size):
    offspring = np.zeros((n, pop.shape[1]))
    for i in range(n):
        r1=np.random.randint(0,pop.shape[0] - 1)
        chromosome = pop[r1,:].copy()
        chromosome[0] += rn.uniform(-step_size, step_size)
        chromosome[0] = max(lb[0], min(ub[0], chromosome[0]))
       
        offspring[i,:] = chromosome
    return offspring
#_____________________________________________________________________________
def evaluation(pop):
    fitness_values = np.zeros((pop.shape[0], 2))
    for i, chromosome in enumerate(pop):
        x = chromosome[0]
        if x <= 1:
            fitness_values[i, 0] = -x
        elif 1 < x <= 3:
            fitness_values[i, 0] = x - 2
        elif 3 < x <= 4:
            fitness_values[i, 0] = 4 - x
        else:
            fitness_values[i, 0] = x - 4

        fitness_values[i, 1] = (x - 5)**2
    return fitness_values
#_____________________________________________________________________________
def crowding_calculation(fitness_values):
     
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))
    normalize_fitness_values = (fitness_values - fitness_values.min(0))/np.ptp(fitness_values, axis=0) 

    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1  
        crowding_results[pop_size - 1] = 1  
        sorting_normalize_fitness_values = np.sort(normalize_fitness_values[:, i])
        sorting_normalized_values_index = np.argsort(normalize_fitness_values[:, i])
        
        crowding_results[1:pop_size - 1] = (sorting_normalize_fitness_values[2:pop_size] - sorting_normalize_fitness_values[:pop_size-2])
        re_sorting = np.argsort(sorting_normalized_values_index)  
        matrix_for_crowding[:, i] = crowding_results[re_sorting]
    
    crowding_distance = np.sum(matrix_for_crowding, axis=1) 
 
    return crowding_distance
#_____________________________________________________________________________
def remove_using_crowding(fitness_values, number_solutions_needed):
   
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros((number_solutions_needed))
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))
     
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
    
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0)
            fitness_values = np.delete(fitness_values, (solution_1), axis=0) 
            crowding_distance = np.delete(crowding_distance, (solution_1), axis=0) 
        else:
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int) 
     
    return (selected_pop_index)
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
            selected_solutions = (remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed))
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front)) 
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop = pop[pareto_front_index.astype(int)]

    return selected_pop
#_____________________________________________________________________________
# Parameters
nv = 1
lb = [-5]
ub = [10]
pop_size = 100
crossover_prob = 0.6
mutation_prob = 0.05
rate_local_search = 30
step_size = 0.02
pop = random_population(nv,pop_size,lb,ub)
#_____________________________________________________________________________
# Main loop of NSGA II
 
for i in range(200):
    offspring_from_crossover = crossover(pop,crossover_prob)
    offspring_from_mutation = mutation(pop,mutation_prob)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)
    # extend the population
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
print("_________________")
print("Optimal solutions:")
print("     x1           x2           x3")
print(pop)
fitness_values = fitness_values[pareto_front_index]
print("______________")
print("Fitness values:")
print("objective 1  objective 2")
print("      |          |")
print(fitness_values)
plt.scatter(fitness_values[:, 0],fitness_values[:, 1])
plt.xlabel('Objective function 1')
plt.ylabel('Objective function 2')
plt.show()