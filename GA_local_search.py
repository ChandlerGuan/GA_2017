# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:24:40 2017

@author: chandler
"""

from NN_classification import *;
import pylab as pl;
import random

if __name__ == "__main__":
    max_generation = 1000;
    local_search_iter = 1;
    population = 2;
    crossover_rate = 0.1;
    mutation_rate = 0.1;
    learning_rate = 0.01;

    hidden_layer_unit = [8,8];
    batch_size=4;
    validation_split=0.1;
    
    
#    ======chanllenge======    
    x,y = import_data('two_spiral.dat');
    model_a = build_network([64,64,64]);
    model_a.fit(x,y,4,1000)
    model_b = build_network([8,8])
    x_teacher = generate_plain_vector(threshold=0.35,interval=0.01);
    y_teacher = model_a.predict(x_teacher,batch_size=4,verbose=0);
    state = np.random.RandomState(28)
    state.shuffle(x_teacher)
    state = np.random.RandomState(28)
    state.shuffle(y_teacher)
    model_b.fit(x_teacher,y_teacher,64,10000)
    model_b.save('challenge.h5')
    
    
    
#    x,y = import_data('two_spiral.dat');
#    model = build_network(hidden_layer_unit);
    
    

        
    
    
  
#    result = model.fit(x,y,batch_size,max_generation);  
    
##    ======Grandient Discent======
#    candidates,fitness = initialization(model,population);
#    weight = model.get_weights();
#    for i in range(len(weight)):
#        weight[i] = 8*weight[i]
#    model.set_weights(weight)
#    result = model.fit(x,y,batch_size,max_generation);
#    loss = result.history['loss']
##    pl.plot(range(1,len(loss)+1),loss)
##    pl.show()
#    np.save('gd_loss.npy',loss)
    
    
##    =====Genetic Algorithm=====
#    candidates,fitness = initialization(model,population);
#    
#    loss = [];
#    
#    
#    for g in range(max_generation):
##    1.fitness calculation
#        candidates,fitness = population_fitness(candidates,x,y,model,local_search=True,batch_size=batch_size,local_search_iter=local_search_iter);
#                
#        elite = candidates[np.argmin(fitness)];
#        loss.append(np.min(fitness))        
#        
#        
##        if (g%50==0):
##            plot_plain(model,elite,os.path.join('visulization','{:1d}_iter.png'.format(g)))
#        
#        print("iter: {:d}, best loss: {:.20f}".format(g,np.min(fitness)))             
#        
#    #    2.selection
#        next_generation, next_fitness = selection(candidates,fitness);
#                
#    #    3.crossover
#        crossover_generation,crossover_fitness = crossover(next_generation,next_fitness,crossover_rate);
#        
#    #    4.mutation
#        candidates,fitness = mutation(crossover_generation,crossover_fitness,mutation_rate,learning_rate)
#        
#        candidates[0]=elite;
#        
#    np.save('ga_loss_FPM.npy'.format(population),loss)
