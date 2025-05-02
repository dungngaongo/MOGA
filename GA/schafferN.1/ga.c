#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

#define NV 1          // Number of variables
#define POP_SIZE 100  // Population size
#define MAX_ITER 200  // Maximum iterations
#define MAX_ARCHIVE 300 // Maximum archive size

// Structure for a solution
typedef struct {
    double x[NV];      // Decision variables
    double fitness[2]; // Fitness values (2 objectives)
} Solution;

// Structure for population
typedef struct {
    Solution solutions[POP_SIZE * 3]; // Population with extra space for offspring
    int size;
} Population;

// Structure for archive
typedef struct {
    Solution solutions[MAX_ARCHIVE];
    int size;
} Archive;

// Global variables
double lb[NV] = {-100.0};
double ub[NV] = {100.0};
double crossover_prob = 0.6;
double mutation_prob = 0.05;
int rate_local_search = 30;
double step_size = 0.02;

// Random number generator between 0 and 1
double rand01() {
    return (double)rand() / RAND_MAX;
}

// Random number between min and max
double rand_range(double min, double max) {
    return min + (max - min) * rand01();
}

// Initialize a random population
void random_population(Population *pop) {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < NV; j++) {
            pop->solutions[i].x[j] = rand_range(lb[j], ub[j]);
        }
    }
    pop->size = POP_SIZE;
}

// Evaluate a solution
void evaluate(Solution *sol) {
    double x = sol->x[0];
    sol->fitness[0] = x * x;
    sol->fitness[1] = (x - 2) * (x - 2);
}

// Crossover operator
void crossover(Population *pop, Population *offspring) {
    offspring->size = 0;
    for (int i = 0; i < pop->size / 2; i++) {
        if (rand01() < crossover_prob) {
            int r1 = rand() % pop->size;
            int r2 = rand() % pop->size;
            while (r1 == r2) {
                r2 = rand() % pop->size;
            }
            
            // Create two children
            for (int j = 0; j < NV; j++) {
                offspring->solutions[offspring->size].x[j] = 
                    (pop->solutions[r1].x[j] + pop->solutions[r2].x[j]) / 2.0;
                offspring->solutions[offspring->size + 1].x[j] = 
                    (pop->solutions[r1].x[j] + pop->solutions[r2].x[j]) / 2.0;
            }
            
            evaluate(&offspring->solutions[offspring->size]);
            evaluate(&offspring->solutions[offspring->size + 1]);
            
            offspring->size += 2;
        }
    }
}

// Mutation operator
void mutation(Population *pop, Population *offspring) {
    offspring->size = 0;
    for (int i = 0; i < pop->size; i++) {
        if (rand01() < mutation_prob) {
            offspring->solutions[offspring->size] = pop->solutions[i];
            offspring->solutions[offspring->size].x[0] = rand_range(lb[0], ub[0]);
            evaluate(&offspring->solutions[offspring->size]);
            offspring->size++;
        }
    }
}

// Local search operator
void local_search(Population *pop, Population *offspring) {
    offspring->size = rate_local_search;
    for (int i = 0; i < rate_local_search; i++) {
        int r1 = rand() % pop->size;
        offspring->solutions[i] = pop->solutions[r1];
        offspring->solutions[i].x[0] += rand_range(-step_size, step_size);
        
        // Ensure bounds
        if (offspring->solutions[i].x[0] < lb[0]) offspring->solutions[i].x[0] = lb[0];
        if (offspring->solutions[i].x[0] > ub[0]) offspring->solutions[i].x[0] = ub[0];
        
        evaluate(&offspring->solutions[i]);
    }
}

// Check if solution1 dominates solution2
bool dominates(Solution *sol1, Solution *sol2) {
    bool better = false;
    for (int i = 0; i < 2; i++) {
        if (sol1->fitness[i] > sol2->fitness[i]) {
            return false;
        }
        if (sol1->fitness[i] < sol2->fitness[i]) {
            better = true;
        }
    }
    return better;
}

// Find non-dominated solutions (Pareto front)
void find_pareto_front(Solution *solutions, int size, int *front_indices, int *front_size) {
    *front_size = 0;
    for (int i = 0; i < size; i++) {
        bool is_dominated = false;
        for (int j = 0; j < size; j++) {
            if (i == j) continue;
            if (dominates(&solutions[j], &solutions[i])) {
                is_dominated = true;
                break;
            }
        }
        if (!is_dominated) {
            front_indices[(*front_size)++] = i;
        }
    }
}

// Calculate crowding distance
void crowding_distance(Solution *front, int front_size, double *distances) {
    if (front_size == 0) return;
    
    // Initialize distances
    for (int i = 0; i < front_size; i++) {
        distances[i] = 0.0;
    }
    
    // For each objective
    for (int obj = 0; obj < 2; obj++) {
        // Create array of indices sorted by this objective
        int *indices = malloc(front_size * sizeof(int));
        for (int i = 0; i < front_size; i++) indices[i] = i;
        
        // Sort indices based on current objective
        for (int i = 0; i < front_size - 1; i++) {
            for (int j = i + 1; j < front_size; j++) {
                if (front[indices[i]].fitness[obj] > front[indices[j]].fitness[obj]) {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }
        
        // Set infinite distance for boundary solutions
        distances[indices[0]] = DBL_MAX;
        distances[indices[front_size - 1]] = DBL_MAX;
        
        // Calculate crowding distance for others
        double fmin = front[indices[0]].fitness[obj];
        double fmax = front[indices[front_size - 1]].fitness[obj];
        double range = fmax - fmin;
        
        if (range > 0) {
            for (int i = 1; i < front_size - 1; i++) {
                distances[indices[i]] += 
                    (front[indices[i + 1]].fitness[obj] - front[indices[i - 1]].fitness[obj]) / range;
            }
        }
        
        free(indices);
    }
}

// Select solutions using crowding distance
void select_by_crowding(Solution *solutions, int size, int num_to_select, Solution *selected) {
    double *distances = malloc(size * sizeof(double));
    crowding_distance(solutions, size, distances);
    
    for (int i = 0; i < num_to_select; i++) {
        // Find solution with maximum crowding distance
        int best = 0;
        for (int j = 1; j < size; j++) {
            if (distances[j] > distances[best]) {
                best = j;
            }
        }
        
        // Select this solution and set its distance to -1 so it's not selected again
        selected[i] = solutions[best];
        distances[best] = -1.0;
    }
    
    free(distances);
}

// Selection operator
void selection(Population *pop, Population *selected) {
    int remaining_indices[pop->size];
    int remaining_size = pop->size;
    for (int i = 0; i < pop->size; i++) remaining_indices[i] = i;
    
    selected->size = 0;
    
    while (selected->size < POP_SIZE) {
        // Find current Pareto front
        int front_indices[remaining_size];
        int front_size;
        find_pareto_front(pop->solutions, remaining_size, front_indices, &front_size);
        
        // If adding this front would exceed population size
        if (selected->size + front_size > POP_SIZE) {
            int needed = POP_SIZE - selected->size;
            Solution temp_front[front_size];
            for (int i = 0; i < front_size; i++) {
                temp_front[i] = pop->solutions[front_indices[i]];
            }
            
            Solution selected_from_front[needed];
            select_by_crowding(temp_front, front_size, needed, selected_from_front);
            
            for (int i = 0; i < needed; i++) {
                selected->solutions[selected->size++] = selected_from_front[i];
            }
            break;
        } else {
            // Add entire front to selected population
            for (int i = 0; i < front_size; i++) {
                selected->solutions[selected->size++] = pop->solutions[front_indices[i]];
            }
            
            // Remove these solutions from remaining indices
            int new_remaining_size = 0;
            for (int i = 0; i < remaining_size; i++) {
                bool in_front = false;
                for (int j = 0; j < front_size; j++) {
                    if (remaining_indices[i] == front_indices[j]) {
                        in_front = true;
                        break;
                    }
                }
                if (!in_front) {
                    remaining_indices[new_remaining_size++] = remaining_indices[i];
                }
            }
            remaining_size = new_remaining_size;
        }
    }
}

// Update archive with current Pareto front
void update_archive(Population *pop, Archive *archive) {
    int front_indices[pop->size];
    int front_size;
    find_pareto_front(pop->solutions, pop->size, front_indices, &front_size);
    
    // Add current front to archive
    for (int i = 0; i < front_size; i++) {
        if (archive->size < MAX_ARCHIVE) {
            archive->solutions[archive->size++] = pop->solutions[front_indices[i]];
        }
    }
    
    // If archive is full, perform crowding-based selection
    if (archive->size > MAX_ARCHIVE) {
        Solution temp[MAX_ARCHIVE];
        select_by_crowding(archive->solutions, archive->size, MAX_ARCHIVE, temp);
        for (int i = 0; i < MAX_ARCHIVE; i++) {
            archive->solutions[i] = temp[i];
        }
        archive->size = MAX_ARCHIVE;
    }
}

// Save results to file
void save_results(Population *pop, Archive *archive, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file!\n");
        return;
    }
    
    // Write header
    fprintf(f, "type,x,f1,f2\n");
    
    // Write final population
    for (int i = 0; i < pop->size; i++) {
        fprintf(f, "final,%f,%f,%f\n", 
                pop->solutions[i].x[0],
                pop->solutions[i].fitness[0],
                pop->solutions[i].fitness[1]);
    }
    
    // Write archive
    for (int i = 0; i < archive->size; i++) {
        fprintf(f, "archive,%f,%f,%f\n", 
                archive->solutions[i].x[0],
                archive->solutions[i].fitness[0],
                archive->solutions[i].fitness[1]);
    }
    
    fclose(f);
}

// Main function
int main() {
    srand(time(NULL));
    
    Population pop, offspring_cross, offspring_mut, offspring_ls, combined, selected;
    Archive archive = {0};
    
    // Initialize population
    random_population(&pop);
    for (int i = 0; i < pop.size; i++) {
        evaluate(&pop.solutions[i]);
    }
    
    // Main loop
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Generate offspring
        crossover(&pop, &offspring_cross);
        mutation(&pop, &offspring_mut);
        local_search(&pop, &offspring_ls);
        
        // Combine populations
        combined.size = 0;
        for (int i = 0; i < pop.size; i++) {
            combined.solutions[combined.size++] = pop.solutions[i];
        }
        for (int i = 0; i < offspring_cross.size; i++) {
            combined.solutions[combined.size++] = offspring_cross.solutions[i];
        }
        for (int i = 0; i < offspring_mut.size; i++) {
            combined.solutions[combined.size++] = offspring_mut.solutions[i];
        }
        for (int i = 0; i < offspring_ls.size; i++) {
            combined.solutions[combined.size++] = offspring_ls.solutions[i];
        }
        
        // Selection
        selection(&combined, &selected);
        pop = selected;
        
        // Update archive
        update_archive(&pop, &archive);
        
        printf("Iteration %d\n", iter);
    }

    // Save results to file
    save_results(&pop, &archive, "nsga2_results.csv");
    
    // Final results
    printf("_________________\n");
    printf("Optimal solutions (x):\n");
    for (int i = 0; i < pop.size; i++) {
        printf("%f\n", pop.solutions[i].x[0]);
    }
    printf("______________\n");
    printf("Fitness values:\n");
    printf("objective 1  objective 2\n");
    for (int i = 0; i < pop.size; i++) {
        printf("%f  %f\n", pop.solutions[i].fitness[0], pop.solutions[i].fitness[1]);
    }
    
    printf("Total points in archive: %d\n", archive.size);
    
    return 0;
}