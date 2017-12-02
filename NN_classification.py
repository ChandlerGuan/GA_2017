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
	
def build_network():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model;
	
def initialization(model,population):
    weights = model.get_weights();
    weight_size = [];
    for i in range(len(weights)):
        weight_size.append(weights[i].shape);
        
    candidate = [];
    for i in range(population):
        candidate.append([]);
        for j in range(len(weight_size)):
            candidate[i].append(np.random.randn(*weight_size[j]));
    
    return candidate;

def value_function(model,weights):
    model.set_weights(weights);
    return model.evaluate(x,y,batch_size=4);
	
if __name__ == "__main__":
    max_generation = 100;
    local_search_iter = 5;
    population = 4;

    batch_size=4;
    validation_split=0.1;
    
    x,y = import_data('two_spiral.dat');
    model = build_network();
    
	# candidates = initialization(model,population);
	# print(value_function(model,candidates[0]))[0];
    
    
#    for i in range(10):
#        result = model.fit(x,y,epochs=i+1,initial_epoch=i,batch_size=batch_size,validation_split=validation_split);
    model.fit(x,y,epochs=1000);
    plot_plain(model,model.get_weights());
    
    
    
	# for i in range(max_generation):
		# weights = model.get_weights();
		# for j in range(len(weights)):
			# print(weights[j].shape);
		# break;
		
		# model.set_weights(weights);
		
		# model.fit(x,y,epochs=local_search_iter,batch_size=batch_size,validation_split=validation_split);