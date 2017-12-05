import numpy as np
import tensorflow.contrib.keras as keras
import random;
#import matplotlib  
import matplotlib.pyplot as plt 

random.seed(28)
	
def plot_scatter(data):
#    call example
#    plot_scatter(np.concatenate((x,y.reshape((y.shape[0],1))),axis=1));
    plt.scatter(data[data[:,2]>0.5,0],data[data[:,2]>0.5,1],marker='x',color='r');
    plt.scatter(data[data[:,2]<0.5,0],data[data[:,2]<0.5,1],marker='o',color='b');
    plt.show();

def generate_plain_vector(threshold=0.5,interval=0.05):
    x_axis,y_axis = np.meshgrid(np.arange(-threshold,threshold,interval),
                                np.arange(-threshold,threshold,interval));
    x_axis = np.asarray(x_axis).reshape((-1,1));
    y_axis = np.asarray(y_axis).reshape((-1,1));
    x = np.concatenate((x_axis,y_axis),axis=1);
#    plt.scatter(x_axis,y_axis);
    return x;
    
    
def plot_plain(model,weight):
    x = generate_plain_vector(threshold=0.35,interval=0.01);
    y = model.predict(x,batch_size=4,verbose=0);
    plot_scatter(np.concatenate((x,y),axis=1))

def import_data(file_name):
    x = [];
    y = [];
    file = open(file_name,'r');
    while (True):
        line = file.readline();
        if (line==''):
            break;
        line = line.split(' ');
        for i in range(len(line)-1,-1,-1):
            if (line[i]==''):
                del line[i];
        x.append([float(line[0]),float(line[1]),float(line[2])]);
    file.close();

#    shuffle data
    random.shuffle(x);

    x = np.asarray(x);

    y = x[:,2];
    x = x[:,0:2];
    y[y>0.5]=1;
    y[y<0.5]=0;	
    
    return x,y;
	
def build_network(hidden_layer_unit):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_layer_unit[0], input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(hidden_layer_unit[1], activation='relu'))
    model.add(keras.layers.Dense(hidden_layer_unit[2], activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model;

#initialize the polulation with gaussian distribution
def initialization(model,population):
    weights = model.get_weights();
    weight_size = [];
    for i in range(len(weights)):
        weight_size.append(weights[i].shape);
        
    candidate = [];
    fitness = [];
    for i in range(population):
        candidate.append([]);
        fitness.append(0);
        for j in range(len(weight_size)):
#            initialize with gaussian distribution
            candidate[i].append(np.random.randn(*weight_size[j]));
#            candidate[i].append(np.ones(weight_size[j])*(i+1))
    
    return candidate,fitness;

#calculate the fitness value based on MSE
def value_function(x,y,model,weights):
    model.set_weights(weights);
    return model.evaluate(x,y,verbose=0)[0];
    
def population_fitness(candidates,x,y,model,local_search=True,batch_size=4,local_search_iter=1):
    fitness = [];
    next_generation = [];
    if (local_search):
        for i in range(len(candidates)):
            model.set_weights(candidates[i]);
            fitness.append(np.average(model.fit(x,y,epochs=local_search_iter,batch_size=batch_size,verbose=0).history['loss']));
            next_generation.append(model.get_weights());
    else:
        for i in range(len(candidates)):
            fitness.append(value_function(x,y,model,candidates[i]));
            next_generation.append(candidates[i]);
    return next_generation,fitness;
    
def select_with_probability(shape,probability):
    return np.random.choice(2,np.prod(shape),p=[1-probability,probability]).reshape(shape);
    
def selection(candidates,fitness):
    probability = np.exp(-np.asarray(fitness));
    probability = probability/np.sum(probability);
    choice = np.random.choice(len(fitness),len(fitness),p=probability);
    choice = choice.tolist();
    next_fitness = [];
    next_generation = [];
    for i in range(len(choice)):
        next_generation.append(candidates[choice[i]]);
        next_fitness.append(fitness[choice[i]]);
    return next_generation,next_fitness;
    
def crossover(candidates,fitness,crossover_rate):
    next_generation = [];
    next_fitness = [];
    for i in range(len(fitness)):
        alpha = fitness[i]/(fitness[i]+fitness[(i+1)%len(fitness)]);
        tmp_weight = [];
        for j in range(len(candidates[0])):
            crossover_position = select_with_probability(candidates[i][j].shape,crossover_rate);
            n = np.multiply(candidates[i][j],1-crossover_position);
            p = np.multiply(alpha*candidates[i][j]+(1-alpha)*candidates[(i+1)%len(fitness)][j],crossover_position);
            tmp_weight.append(n+p);
        next_generation.append(tmp_weight);
#        next_generation.append(map(lambda x,y:x+y,alpha*candidates[i],(1-alpha)*candidates[i]));
        next_fitness.append(0.5*(fitness[i]+fitness[(i+1)%len(fitness)]));
    return next_generation,next_fitness;
    
def mutation(candidates,fitness,mutation_rate,learning_rate):
    next_generation = [];
    next_fitness = [];
    for i in range(len(candidates)):
        probability = 1-np.exp(-fitness[i]);
        tmp_weight = [];
        for j in range(len(candidates[0])):
            mutation_position = select_with_probability(candidates[i][j].shape,mutation_rate);
            mutation_amount = np.multiply(np.random.randn(*candidates[i][j].shape),candidates[i][j])*learning_rate*probability;
#            mutation_amount = np.multiply(np.random.randn(*candidates[i][j].shape),candidates[i][j])*learning_rate;            
            mutation_amount = np.multiply(mutation_amount,mutation_position);
            tmp_weight.append(np.add(mutation_amount,candidates[i][j]));
        next_generation.append(tmp_weight);
        next_fitness.append(fitness[i])
    return next_generation,next_fitness;
    
        
	
if __name__ == "__main__":
    max_generation = 1000;
    local_search_iter = 1;
    population = 2;
    crossover_rate = 0.1;
    mutation_rate = 0.1;
    learning_rate = 0.01;

    hidden_layer_unit = [64,64,64];
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
#        print(fitness)        
        
        
    #    2.selection
        next_generation, next_fitness = selection(candidates,fitness);
        
#        print(next_fitness)
        
        
    #    3.crossover
        crossover_generation,crossover_fitness = crossover(next_generation,next_fitness,crossover_rate);
        
#        print(crossover_fitness)
        
    #    4.mutation
        candidates,fitness = mutation(crossover_generation,crossover_fitness,mutation_rate,learning_rate)
        
        candidates[0]=elite;
#        candidates[int(3/4*population)]=elite
#        candidates[int(1/2*population)]=elite
#        candidates[int(1/4*population)]=elite
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