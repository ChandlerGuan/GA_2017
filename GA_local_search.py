# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:40 2017

@author: chandler
"""

from NN_classification import *;

if __name__ == "__main__":
    max_generation = 10000;
    local_search_iter = 5;
    population = 64;
    crossover_rate = 0.3;
    mutation_rate = 0.3;
    learning_rate = 0.5;

    hidden_layer_unit = [64,64,64];
    batch_size=4;
    validation_split=0.1;
    
    x,y = import_data('two_spiral.dat');
    model = build_network(hidden_layer_unit);
    
#    =====Genetic Algorithm=====
    candidates,fitness = initialization(model,population);
    
    for g in range(max_generation):
#    1.fitness calculation
        for i in range(population):
            fitness[i] = value_function(x,y,model,candidates[i]);
                
        print("iter: {:d}, best loss: {:f}".format(g,np.min(fitness)))
        
        elite = candidates[np.argmin(fitness)]
    #    2.selection
        next_generation, next_fitness = selection(candidates,fitness);
    
        
    #    3.crossover
        next_generation, next_fitness = crossover(next_generation,next_fitness,crossover_rate);
        
    #    4.mutation
        candidates,fitness = mutation(candidates,fitness,mutation_rate,learning_rate)
        
        candidates[0]=elite;
        candidates[int(3/4*population)]=elite
        candidates[int(1/2*population)]=elite
        candidates[int(1/4*population)]=elite
#    print(value_function(x,y,model,candidates[0]));
    
    
#    for i in range(10):
#        result = model.fit(x,y,epochs=i+1,initial_epoch=i,batch_size=batch_size,validation_split=validation_split);
##    model.fit(x,y,epochs=1000);
#        plot_plain(model,model.get_weights());
    
    
    
	# for i in range(max_generation):
		# weights = model.get_weights();
		# for j in range(len(weights)):
			# print(weights[j].shape);
		# break;
		
		# model.set_weights(weights);
		
		# model.fit(x,y,epochs=local_search_iter,batch_size=batch_size,validation_split=validation_split);