#ifndef GA_H
#define GA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>

#define NV 30           
#define POP_SIZE 200   
#define MAX_ITER 300   
#define PI 3.141592653589793

typedef struct {
    double x[NV];      
    double fitness[2];  
} Solution;

typedef struct {
    Solution solutions[POP_SIZE * 3]; 
    int size;
} Population;

extern double lb[NV];
extern double ub[NV];
extern double crossover_prob;
extern double mutation_prob;
extern int rate_local_search;
extern double step_size;

double rand01();
double rand_range(double min, double max);
void initialize_bounds();

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
void save_results(Population *pop, const char *filename);

#endif 