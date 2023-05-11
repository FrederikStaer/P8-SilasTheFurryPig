#Code modified from: https://github.com/wannabeOG/ExpertNet-Pytorch 

#!/usr/bin/env python
# coding: utf-8

"""
This module tests the "experts" that have been generated by the generate_models.py file. If the wrong autoencoder were to be 
selected, the performance would obviously suffer. A metric to determine how many times this occurs has been designed. Since in 
these tasks. 

"""
import torch
torch.backends.cudnn.benchmark=True

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import argparse 
import numpy as np
from random import shuffle
import os
from math import sqrt

import copy
from autoencoder import *

import sys
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from tqdm import tqdm
from collections import defaultdict

from encoder_train import *
from encoder_utils import *

from model_train import *
from model_utils import *

def test_models(args):
	use_gpu = args.use_gpu  and torch.cuda.is_available()
	batch_size = args.batch_size
	
	new_path = os.path.join(os.getcwd(), "models", "autoencoders")
	num_ae = len(next(os.walk(new_path))[1])
	task_number_list = [x for x in range(1,num_ae+1)]
	#shuffle(task_number_list)

	classes = []

	#transformations for the test data
	data_transforms_tin = {
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}

	#transforms for the mnist dataset. Applicable for the tasks 5-9
	data_transforms_mnist = {
		'test': transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize([0.1307,], [0.3081,])
		])
	}


	#get the paths to the data and model
	data_path = os.path.join(os.getcwd(), "Data")
	encoder_path = os.path.join(os.getcwd(), "models", "autoencoders")
	model_path = os.path.join(os.getcwd(), "models", "trained_models")


	#Get the number of classes in each of the given task folders
	for task_number in task_number_list:

		path_task = os.path.join(data_path, "Task_" + str(task_number))
		if(task_number <=4):
			#get the image folder
			image_folder = datasets.ImageFolder(os.path.join(path_task, 'test'), transform = data_transforms_tin['test'])
			classes.append(len(image_folder.classes))

		else:
			image_folder = datasets.ImageFolder(os.path.join(path_task, 'test'), transform = data_transforms_mnist['test'])
			classes.append(len(image_folder.classes))


	#shuffle the sequence of the tasks
	#print("Shuffling tasks")
	#shuffle(task_number_list)

	#set the device to be used and initialize the feature extractor to feed the data into the autoencoder
	device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
	feature_extractor = Alexnet_FE(models.alexnet(pretrained=True))
	feature_extractor.to(device)

	if args.approach == "export":
		#Loading relatedness_matrix for use in latent coordinate estimation
		relatedness_vectors = []
		with open(os.path.join(path, 'relatedness_matrix.txt')) as f:
			[relatedness_vectors.append(line.strip('][\n').split(',')) for line in f]
		f.close()	
		#Converting string values ([['0.0']])  to floats ([[0.0]])
		relatedness_matrix = [[float(j) for j in i] for i in relatedness_vectors]

		#Make matrix of connections between autoencoders with relatedness score > 0.85 
		n = len(relatedness_matrix)
		ae_graph_connections = np.zeros((n, n))

		# for i in range(len(relatedness_matrix)):
		# 	ae_graph_connections.append([])
		# 	for j in range(len(relatedness_matrix)):
		# 		ae_graph_connections[i].append(0)

		for i in range(len(relatedness_matrix)):
			for j in range(len(relatedness_matrix[i])):
				if j != 0 and relatedness_matrix[i][j] > 0.85:
					ae_graph_connections[i][j-1] = 1
					ae_graph_connections[j-1][i] = 1
		
		print(ae_graph_connections)

		#Make clusters of autoencoders
		clusters = find_autoencoder_clusters(ae_graph_connections)

		
				
					

	for task_number in task_number_list:
		print("Testing task " + str(task_number))

		#get the paths to the data and model
		path_task = os.path.join(data_path, "Task_" + str(task_number))

		if(task_number >=1 and task_number <=4):
			#get the image folder
			image_folder = datasets.ImageFolder(os.path.join(path_task, 'test'), transform = data_transforms_tin['test'])
			dset_size = len(image_folder)

		else:
			#get the image folder
			image_folder = datasets.ImageFolder(os.path.join(path_task, 'test'), transform = data_transforms_mnist['test'])
			dset_size = len(image_folder)

	
		dset_loaders = torch.utils.data.DataLoader(image_folder, batch_size = batch_size,
														shuffle=True, num_workers=2)

		best_loss = 99999999999
		model_number = 0

		if args.approach == "export":
			
			#Find best cluster
			cluster_number = []
			for i in range(len(clusters)):
				print()
				print("Cluster no. " + str(i))
				ae_path = os.path.join(encoder_path, "autoencoder_" + str(clusters[i][0]))
				
				#Load a trained autoencoder model
				model = Autoencoder()
				model.load_state_dict(torch.load(os.path.join(ae_path, 'best_performing_model.pth')))

				running_loss = 0
				model.to(device)

				#Test out the different auto encoder models and check their reconstruction error
				for data in tqdm(dset_loaders):
					input_data, labels = data



					if (use_gpu):
						input_data = input_data.to(device)
					
					else:
						input_data  = Variable(input_data)
					
					#get the input to the autoencoder from the conv backbone of the Alexnet
					input_to_ae = feature_extractor(input_data)
					input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)

					input_to_ae = F.sigmoid(input_to_ae)
				
					#get the outputs from the model
					preds = model(input_to_ae)
					loss = encoder_criterion(preds, input_to_ae)

					running_loss = running_loss + loss.item()

				model_loss = running_loss/dset_size

				if(model_loss < best_loss):
					best_loss = model_loss
					cluster_number = i

			task_number_list = clusters[cluster_number]
			
			best_loss = 99999999999

		#Load autoencoder models for tasks 1-9; need to select the best performing autoencoder model
		for ae_number in task_number_list:
			print()
			print("Autoencoder no. " + str(ae_number))
			ae_path = os.path.join(encoder_path, "autoencoder_" + str(ae_number))
			
			#Load a trained autoencoder model
			model = Autoencoder()
			model.load_state_dict(torch.load(os.path.join(ae_path, 'best_performing_model.pth')))

			running_loss = 0
			model.to(device)

			#Test out the different auto encoder models and check their reconstruction error
			for data in tqdm(dset_loaders):
				input_data, labels = data



				if (use_gpu):
					input_data = input_data.to(device)
				
				else:
					input_data  = Variable(input_data)


				#get the input to the autoencoder from the conv backbone of the Alexnet
				input_to_ae = feature_extractor(input_data)
				input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)

				input_to_ae = F.sigmoid(input_to_ae)
			
				#get the outputs from the model
				preds = model(input_to_ae)
				loss = encoder_criterion(preds, input_to_ae)
	
				running_loss = running_loss + loss.item()

			model_loss = running_loss/dset_size

			if(model_loss < best_loss):
				best_loss = model_loss
				model_number = ae_number
		



		if(model_number == task_number):
			print ("\nThe correct autoencoder has been found")

		else:
			print ("\nIncorrect routing, wrong model has been selected (selected model " + str(model_number) + ")")


		#Load the expert that has been found by this procedure into memory
		trained_model_path = os.path.join(model_path, "model_" + str(model_number))

		#Get the number of classes that this expert was exposed to
		file_name = os.path.join(trained_model_path, "classes.txt") 
		file_object = open(file_name, 'r')

		num_of_classes = file_object.read()
		file_object.close()

		num_of_classes = int(num_of_classes)

		model = GeneralModelClass(num_of_classes)
		model.load_state_dict(torch.load(os.path.join(trained_model_path, 'best_performing_model.pth')))

		#initialize the results statistics
		running_loss = 0
		running_corrects = 0

		#run the test loop over the model
		print("Model test")
		for data in tqdm(dset_loaders):
			input_data, labels = data


			if (use_gpu):
				input_data = Variable(input_data.to(device))
				labels = Variable(labels.to(device)) 
		
			else:
				input_data  = Variable(input_data)
				labels = Variable(labels)
		
			model.to(device)

			outputs = model(input_data)

		
			#(currently not functional) for a more robust analysis check over the entire output layer (similar to multi head setting)
			#_, preds = torch.max(outputs, 1
			#loss = model_criterion(outputs, labels, 'CE')

			#check over only the specific layer identified by the AE (similar to single head setting)
			_, preds = torch.max(outputs[:, -classes[model_number-1]:], 1)
			fitted_outputs = torch.zeros(outputs.shape[0], max(classes[task_number-1], outputs.shape[1]), dtype=outputs.dtype, device=outputs.device)
			fitted_outputs[:, -classes[model_number-1]:] = outputs[:, -classes[model_number-1]:]
			loss = model_criterion(fitted_outputs, labels, args, flag = 'CE')
			
			running_corrects += torch.sum(preds==labels.data)
			running_loss = running_loss + loss.item()


		model_loss = running_loss/dset_size
		model_accuracy = running_corrects.double()/dset_size
		print("\nModel accuracy: " + str(model_accuracy))

		#Store the results into a file
		with open("results.txt", "a") as myfile:
			myfile.write("\n{}: {}".format(task_number, model_accuracy*100))
			myfile.close()

def find_autoencoder_clusters(graph):
	n = len(graph)
	distance_matrix = np.ones((n, n))
	#for number of paths
	next_vertices = np.zeros((n, n))
	prev = [[0 for i in range(n)] for j in range(n)]
	
	#Make distance matrix and next vertices
	for i in range(n):
		for j in range(n):
			if graph[i][j] == 1:
				distance_matrix[i][j] = 1
			elif i == j:
				distance_matrix[i][j] = 0
			else:
				distance_matrix[i][j] = np.Inf
			next_vertices[i][j] = j
	
	
	#Calculate shortest paths (Floyd-Warshall)
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if distance_matrix[i][j] > (distance_matrix[i][k] + distance_matrix[k][j]):
					distance_matrix[i][j] = (distance_matrix[i][k] + distance_matrix[k][j])


	#Function for finding the paths to make a list of these
	def paths(graph, v):
		path = [v]
		seen = {v}
		def search():
			dead_end = True
			for neighbour in graph[path[-1]]:
				if neighbour not in seen:
					dead_end = False
					seen.add(neighbour)
					path.append(neighbour)
					yield from search()
					path.pop()
					seen.remove(neighbour)
			if dead_end:
				yield list(path)
		yield from search()


	def calc_betas(graph):
		#Make adjecency dictionary for establishing the possible paths	
		adjList = defaultdict(list)
		for i in range(n):
			for j in range(n):
				if graph[i][j] != 0:
					adjList[i].append(j)

		#Make a list of all possible paths in the graph	
		all_paths = []
		for i in range(n):
			all_paths.append(sorted(paths(adjList, i)))

		#Variables for calculating beta-values
		num_shortest_paths = np.zeros((n, n))
		edge_shortest_paths = np.zeros((n, n))
		explored_paths = []

		#Count the number of shortest paths connecting each node and number of shortest paths connecting nodes through each edge
		for start in range(len(all_paths)):
			for path in all_paths[start]:
				explored_paths.append([])
				for index in range(len(path)):
					node = path[index]
					min_distance = distance_matrix[start][node]
					path_distance = index
					current_path = explored_paths[-1] + [node]

					if min_distance > 0 and path_distance == min_distance and current_path not in explored_paths:
						num_shortest_paths[start][node] += 1.0
						
						for edge_id in range(index):
							edge_shortest_paths[path[edge_id]][path[edge_id + 1]] += 1.0
							edge_shortest_paths[path[edge_id + 1]][path[edge_id]] += 1.0
					
					explored_paths.append(current_path)

		beta_matrix = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if num_shortest_paths[i][j] != 0:
					beta_matrix[i][j] = edge_shortest_paths[i][j] / num_shortest_paths[i][j]

		return beta_matrix
	
	def get_clusters(graph):
		clusters = [[i] for i in range(graph.shape[0])]
		for i in range(n):
			for j in range(n):
				if graph[i][j] == 1:
					#Connect the clusters
					cluster_i_j = [x for x in clusters if i in x or j in x]
					cluster_i_j = [item for sublist in cluster_i_j for item in sublist]
					clusters = [x for x in clusters if not(i in x or j in x)]
					clusters.append(cluster_i_j)

		return clusters
					
	def newman_girvan(graph):
		clusters = get_clusters(graph)
	
		while(len(clusters) < sqrt(n)):
			beta_values = calc_betas(graph)
			(max_i, max_j) = np.unravel_index(np.argmax(beta_values, axis=None), beta_values.shape)
			graph[max_i][max_j] = 0
			new_clusters = get_clusters(graph)
			clusters = new_clusters

		return clusters

	clusters = newman_girvan(graph)
	final_clusters = [[x+1 for x in cluster] for cluster in clusters]

	return final_clusters











