import random as rn
import numpy as np
import matplotlib.pyplot as plt
import math

#------------------ Khởi tạo --------------------
def random_population(nv,n,lb,ub):
    pop=np.zeros((n, nv)) 
    for i in range(n):
        pop[i,:] = np.random.uniform(lb,ub)
    return pop
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
def local_search(pop, n, step_size):
    offspring = np.zeros((n, pop.shape[1]))
    for i in range(n):
        r1=np.random.randint(0,pop.shape[0])
        chromosome = pop[r1,:]
        r2=np.random.randint(0, pop.shape[1])
        chromosome[r2] += np.random.uniform(-step_size, step_size)
        if chromosome[r2] < lb[r2]:
            chromosome[r2] = lb[r2]
        if chromosome[r2] > ub[r2]:
            chromosome[r2] = ub[r2]

        offspring[i,:] = chromosome
    return offspring
#-----------------------------------------------------------------------------
def evaluation(pop):
    fitness_values = np.zeros((pop.shape[0], 2))
    for i, chromosome in enumerate(pop):
        n = 3 
        for j in range(2):
            if j == 0:      
                fitness_values[i,j] = 1 - math.exp(-sum((chromosome - 1/math.sqrt(n))**2))
 
            elif j == 1:    
                fitness_values[i,j] = 1 - math.exp(-sum((chromosome + 1/math.sqrt(n))**2))
         
    return fitness_values
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool) 
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0  
                break  
    return pop_index[pareto_front]
#-----------------------------------------------------------------------------
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

#------------------ Tham số --------------------
nv = 3
lb = [-4, -4, -4]
ub = [4, 4, 4]
pop_size = 100
crossover_prob = 0.6
mutation_prob = 0.05
rate_local_search = 30
step_size = 0.02
pop = random_population(nv,pop_size,lb,ub)

#------------------ Lưu archive --------------------
archive = []

# Hàm lọc các điểm trong archive
def filter_pareto_front_using_crowding(archive, max_size=300):
    def euclidean_distance(point1, point2):
        return np.linalg.norm(point1 - point2)
    
    # Hàm tính crowding distance cho một front
    def crowding_distance(front):
        n = len(front)
        if n == 0:
            return np.array([])
        distances = np.zeros(n)
        for i in range(front.shape[1]):  
            sorted_idx = np.argsort(front[:, i])
            min_val = front[sorted_idx[0], i]
            max_val = front[sorted_idx[-1], i]
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf 
            for j in range(1, n - 1):
                distances[sorted_idx[j]] += (front[sorted_idx[j + 1], i] - front[sorted_idx[j - 1], i]) / (max_val - min_val)
        return distances

    # Lọc các điểm trong archive theo crowding distance
    filtered_archive = []
    for front in archive:
        if len(front) > max_size:
            dist = crowding_distance(front)
            sorted_idx = np.argsort(dist)[::-1]  
            front = front[sorted_idx[:max_size]] 
        filtered_archive.append(front)
    return filtered_archive

def limit_archive_size(archive, max_size=300):
    all_points = np.vstack(archive)
    fitness_values = evaluation(all_points)  

    crowding_distances = crowding_calculation(fitness_values)
    
    sorted_idx = np.argsort(crowding_distances)[::-1]
    selected_points = all_points[sorted_idx[:max_size]]

    filtered_archive = []
    current_start = 0
    for front in archive:
        front_size = len(front)
        selected_front = selected_points[current_start:current_start + front_size]
        filtered_archive.append(selected_front)
        current_start += front_size
    return filtered_archive


#------------------ Vòng lặp chính --------------------
for i in range(200):
    offspring_from_crossover = crossover(pop,crossover_prob)
    offspring_from_mutation = mutation(pop,mutation_prob)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)
    
    pop = np.vstack((pop, offspring_from_crossover, offspring_from_mutation, offspring_from_local_search))
    fitness_values = evaluation(pop)
    pop = selection(pop, fitness_values, pop_size)

    # Lưu Pareto front hiện tại vào archive
    fitness_values = evaluation(pop)
    index = np.arange(pop.shape[0]).astype(int)
    pareto_front_index = pareto_front_finding(fitness_values, index)
    pareto_front = fitness_values[pareto_front_index]
    archive.append(pareto_front)

    archive = limit_archive_size(archive, max_size=150)  

    print(f"iteration {i}")

#------------------ Kết quả cuối --------------------
fitness_values = evaluation(pop)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
fitness_values = fitness_values[pareto_front_index]

print("_________________")
print("Optimal solutions (x):")
print(pop) 
print("______________")
print("Fitness values:")
print("objective 1  objective 2")
print(fitness_values)

# ------------------ Kiểm tra số điểm được tô --------------------
total_points = sum([front.shape[0] for front in archive])
print(f"Tổng số điểm được tô trên đồ thị: {total_points}")

#------------------ Vẽ Pareto front hội tụ --------------------
plt.figure(figsize=(6, 6))
for front in archive:
    plt.scatter(front[:, 0], front[:, 1], s=10, color='red', marker='s')  

plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.grid(True, color='black', linewidth=0.5)
plt.title('Pareto Front')
plt.xlim(0, 1)     
plt.ylim(0, 1)    
plt.gca().set_facecolor('white')
plt.show()
