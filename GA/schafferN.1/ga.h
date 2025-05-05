#ifndef GA_H
#define GA_H

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

// Global variables (extern declarations)
extern double lb[NV];
extern double ub[NV];
extern double crossover_prob;
extern double mutation_prob;
extern int rate_local_search;
extern double step_size;

// Function declarations
double rand01();
double rand_range(double min, double max);
void random_population(Population *pop);
void evaluate(Solution *sol);
void crossover(Population *pop, Population *offspring);
void mutation(Population *pop, Population *offspring);
void local_search(Population *pop, Population *offspring);
bool dominates(Solution *sol1, Solution *sol2);
void find_pareto_front(Solution *solutions, int size, int *front_indices, int *front_size);
void crowding_distance(Solution *front, int front_size, double *distances);
void select_by_crowding(Solution *solutions, int size, int num_to_select, Solution *selected);
void selection(Population *pop, Population *selected);
void update_archive(Population *pop, Archive *archive);
void save_results(Population *pop, Archive *archive, const char *filename);

#endif // GA_H