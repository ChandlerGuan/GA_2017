import numpy as np
import tensorflow.contrib.keras as keras
import random;
import matplotlib  
import matplotlib.pyplot as plt 

# def generate_data(range=0.5):
	
def plot_plain(data):
	# plt.scatter(x[idx_1,1], x[idx_1,0], marker = 'x', color = 'm', label='1', s = 30)  
	plt.scatter(data[data[:,2]>0.5,0],data[data[:,2]>0.5,0],marker='x',color='r');
	plt.scatter(data[data[:,2]<0.5,0],data[data[:,2]<0.5,0],marker='o',color='b');
	plt.show();

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
		
	# model = build_network();
	
	# candidates = initialization(model,population);
	# print(value_function(model,candidates[0]))[0];
	print(x,shape,y.shape);
	plot_plain(np.concatenate((x,y),axis=1));
	# for i in range(max_generation):
		# weights = model.get_weights();
		# for j in range(len(weights)):
			# print(weights[j].shape);
		# break;
		
		# model.set_weights(weights);
		
		# model.fit(x,y,epochs=local_search_iter,batch_size=batch_size,validation_split=validation_split);