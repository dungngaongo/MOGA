#include "ga.h"

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
    printf("Results saved to nsga2_results.csv\n");
    
    return 0;
}