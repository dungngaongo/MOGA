#include "ga.h"

#define VOLTAGE 5.0f      
#define CURRENT 0.5f

int main() {
    srand(time(NULL));
    
    Population pop, offspring_cross, offspring_mut, offspring_ls, combined, selected;

    clock_t start = clock();
    
    // Initialize population
    random_population(&pop);
    for (int i = 0; i < pop.size; i++) {
        evaluate(&pop.solutions[i]);
    }
    
    // Main optimization loop
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
        
        selection(&combined, &selected);
        pop = selected;
        
        printf("Iteration %d\n", iter);
    }

    clock_t end = clock();
    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;

    double energy_joules = VOLTAGE * CURRENT * elapsed_sec;
    
    // Extract Pareto front from final population
    int front_indices[pop.size];
    int front_size;
    find_pareto_front(pop.solutions, pop.size, front_indices, &front_size);
    
    Population pareto_front = {0};
    for (int i = 0; i < front_size; i++) {
        pareto_front.solutions[i] = pop.solutions[front_indices[i]];
    }
    pareto_front.size = front_size;
    
    save_results(&pareto_front, "nsga2_results.csv");
    
    printf("\nFinal Pareto Front Solutions:\n");
    printf("---------------------------\n");
    printf("     x1           x2           x3           f1           f2\n");
    for (int i = 0; i < pareto_front.size; i++) {
        printf("%10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n", 
               pareto_front.solutions[i].x[0],
               pareto_front.solutions[i].x[1],
               pareto_front.solutions[i].x[2],
               pareto_front.solutions[i].fitness[0],
               pareto_front.solutions[i].fitness[1]);
    }
    
    printf("\nResults saved to nsga2_results.csv\n");
    printf("Execution time: %.3f seconds\n", elapsed_sec);
    printf("Estimated energy consumed: %.4f J\n", energy_joules);
    
    return 0;
}