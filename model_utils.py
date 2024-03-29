#Code modified from: https://github.com/wannabeOG/ExpertNet-Pytorch 

#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch.autograd import Variable

import copy
import os
import warnings
import time
import sys

from pathlib import Path
path = Path(os.getcwd())

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from encoder_utils import *

from autoencoder import Autoencoder, Alexnet_FE

from tqdm import tqdm



def task_metric(r_error_ref, r_error_comp):
	""" 
	Inputs: 
		1) r_error_comp = Reconstruction error for the new task
		2) r_error_ref = Reconstruction error for some prior task
		
	Outputs:
		1) task_metric = The task metric that was used in the paper
		   Max value = 1 indicates that the tasks are heavily related
		   Max value = 0 indicates that the tasks are not at all related 
		   (P8 group's note: the above "limits" are false, as can be seen by the new, equivalent function at the return)
		
	Function: This function returns the task metric

	"""
	return 2-(r_error_ref/r_error_comp) #equivalent to (1-((r_error_ref-r_error_comp)/r_error_comp))


def kaiming_initilaization(layer):
	nn.init.kaiming_normal_(layer.weight, nonlinearity='sigmoid')


def get_related_model(feature_extractor, dset_loaders, dataset_size, encoder_criterion, use_gpu, ae_idxs):
	""" 
	Inputs: 
		1) feature_extractor = A reference to the model which needs to be initialized
		2) dset_loaders = The number of classes in the new task for which we need to train a expert  
		3) dataset_size = The number of classes in the model that is used as a reference for
		   initializing the new model
	   	4) encoder_criterion = The loss function for the encoders
	   	5) use_gpu = Flag set to True if the GPU is to be used  
		6) ae_idxs = List of indexes of the autoencoders to be tested (1st idx is the one to test against)

	Outputs:
		1) model_number = The number for the model that is most closely related to the present task  
		2) best_relatedness = The relatedness this model task bears with the present task calculated as per
			section 3.3 of the paper

	Function: Returns the model number from the given idxs that is most related to the 1st idx and a metric that measures 
	how related these two given models are

	"""	
	path = os.getcwd()
	destination = os.path.join(path, "models", "autoencoders")
	best_relatedness = -999
	model_number = None
	device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
	feature_extractor = feature_extractor.to(device)
	rerror_comp = None

	for i in ae_idxs:
		running_loss = 0

		print("testing AE no. " + str(i))	
		
		model_path = os.path.join(destination, "autoencoder_"+str(i), "best_performing_model.pth")
		model = Autoencoder(13*13*256)
		
		#print("Loading the model")
		model.load_state_dict(torch.load(model_path))
		#print("Loaded the model")
		
		model.to(device)
		
		model.train(False)

		for data in tqdm(dset_loaders):
			input_data, labels = data
			input_data = input_data.to(device)
			


			
			input_to_ae = feature_extractor(input_data)
			input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)
			
			input_to_ae = F.sigmoid(input_to_ae)

			outputs = model(input_to_ae)
			loss = encoder_criterion(outputs, input_to_ae)
			
			


			
			
			running_loss = loss.item() + running_loss

		running_loss = running_loss/dataset_size
		


		if (i == ae_idxs[0]): 
			rerror_comp = running_loss
			print("\nae loss: " + str(running_loss))
		
		else:
			relatedness = task_metric(running_loss, rerror_comp)
			print("\nae loss: " + str(running_loss) + ", relatedness: " + str(relatedness))
			
			if (relatedness > best_relatedness):
				best_relatedness = relatedness
				model_number = (i)
	



	print("The Model number is ", model_number)
	print("The best relatedness is ", best_relatedness)

	return model_number, best_relatedness 



def initialize_new_model(model_init, num_classes, num_of_classes_old, args):
	""" 
	Inputs: 
		1) model_init = A reference to the model which needs to be initialized
		2) num_classes = The number of classes in the new task for which we need to train a expert  
		3) num_of_classes_old = The number of classes in the model that is used as a reference for
		   initializing the new model.
		4) flag = to indicate if best_relatedness is greater or less than 0.85     

	Outputs:
		1) autoencoder = A reference to the autoencoder object that is created 
		2) store_path = Path to the directory where the trained model and the checkpoints will be stored

	Function: This function takes in a reference model and initializes a new model with the reference model's
	weights (for the old task) and the weights for the new task are initialized using the kaiming initialization
	method

	"""	

	device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")

	weight_info = model_init.Tmodel.classifier[-1].weight.data.to(device)
	
	weight_info = weight_info.to(device)
	model_init.Tmodel.classifier[-1] = nn.Linear(model_init.Tmodel.classifier[-1].in_features, num_of_classes_old + num_classes)
	
	nn.init.kaiming_normal_(model_init.Tmodel.classifier[-1].weight, nonlinearity='relu')
	
	#kaiming_initilaization()
	model_init.Tmodel.classifier[-1].weight.data[:-num_classes, :] = weight_info
	model_init.to(device)
	model_init.train(True)
	
	return model_init 



def model_criterion(preds, labels, args, flag, T = 2):
	"""
		Temperature is used to produce softer values of probability and 
		this parameter is used only when the flag option is set with the "Distill"
		option
	"""
	device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")

	preds = preds.to(device)
	labels = labels.to(device)

	if(flag == "CE"):
		loss = nn.CrossEntropyLoss()
		return loss(preds, labels)
	
	elif(flag == "Distill"):
		
		""" 
		The labels are the teacher scores or the reference
		scores in this case
		
		"""	
		preds = preds.to(device)
		labels = labels.to(device)

		preds = F.softmax(preds, dim = 1)
		labels = F.softmax(labels, dim = 1)
		
		if preds[0][0] != preds[0][0]:
			preds = torch.ones(preds.shape, dtype=preds.dtype, device=preds.device)
			preds = F.softmax(preds, dim = 1)
		if  labels[0][0] != labels[0][0]:
			labels = torch.ones(labels.shape, dtype=labels.dtype, device=labels.device)
			labels = F.softmax(labels, dim = 1)

		preds = preds.pow(1/T)
		labels = labels.pow(1/T)

		sum_preds = torch.sum(preds, dim = 1)
		sum_labels = torch.sum(preds, dim = 1)

		sum_preds_ref = torch.transpose(sum_preds.repeat(preds.size(1), 1), 0, 1)
		sum_preds_ref = sum_preds_ref.to(device)
		
		sum_labels_ref = torch.transpose(sum_labels.repeat(labels.size(1), 1), 0, 1)
		sum_labels_ref = sum_labels_ref.to(device)
		
		preds = preds/sum_preds_ref
		labels = labels/sum_labels_ref
		


		loss = torch.sum(-1*preds*torch.log(labels), dim = 1)
		batch_size = loss.size()[0]

		loss = torch.sum(loss, dim = 0)/batch_size
		
		return loss
