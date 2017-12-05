# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:40 2017

@author: chandler
"""

from NN_classification import *;

if __name__ == "__main__":
    max_generation = 1000;
    local_search_iter = 1;
    population = 2;
    crossover_rate = 0.1;
    mutation_rate = 0.1;
    learning_rate = 0.01;

    hidden_layer_unit = [10,10,10];
    batch_size=4;
    validation_split=0.1;
    
    x,y = import_data('two_spiral.dat');
    model = build_network(hidden_layer_unit);
    
#    =====Genetic Algorithm=====
    candidates,fitness = initialization(model,population);
    
    for g in range(max_generation):
#    1.fitness calculation
        candidates,fitness = population_fitness(candidates,x,y,model,local_search=True,batch_size=batch_size,local_search_iter=local_search_iter);
                
        elite = candidates[np.argmin(fitness)];
        
        print("iter: {:d}, best loss: {:.20f}".format(g,np.min(fitness)))             
        
    #    2.selection
        next_generation, next_fitness = selection(candidates,fitness);
                
    #    3.crossover
        crossover_generation,crossover_fitness = crossover(next_generation,next_fitness,crossover_rate);
        
    #    4.mutation
        candidates,fitness = mutation(crossover_generation,crossover_fitness,mutation_rate,learning_rate)
        
        candidates[0]=elite;