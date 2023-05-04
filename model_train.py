#Code modified from: https://github.com/wannabeOG/ExpertNet-Pytorch 

#!/usr/bin/env python
# coding: utf-8

import torch 
import os
from torchvision import models
from autoencoder import GeneralModelClass

import copy

import sys
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from model_utils import *

from tqdm import tqdm

def train_model(num_classes, feature_extractor, encoder_criterion, dset_loaders, dset_size, num_epochs, use_gpu, task_number, relatedness_info, args, lr = 0.1, alpha = 0.01):
	""" 
	Inputs: 
		1) num_classes = The number of classes in the new task  
		2) feature_extractor = A reference to the feature extractor model  
		3) encoder_criterion = The loss criterion for training the Autoencoder
		4) dset_loaders = Dataset loaders for the model
		5) dset_size = Size of the dataset loaders
		6) num_of_epochs = Number of epochs for which the model needs to be trained
		7) use_gpu = A flag which would be set if the user has a CUDA enabled device
		8) task_number = A number which represents the task for which the model is being trained
		9) relatedness_info = the idx and relatedness for most related model
		10) lr = initial learning rate for the model
		11) alpha = Tradeoff factor for the loss   

	Function: Trains the model on the given task
		1) If the task relatedness is greater than 0.85, the function uses the Learning without Forgetting method
		2) If the task relatedness is lesser than 0.85, the function uses the normal finetuning procedure as outlined
			in the "Learning without Forgetting" paper ("https://arxiv.org/abs/1606.09282")

		Whilst implementing finetuning procedure, PyTorch does not provide the option to only partially freeze the 
		weights of a layer. In order to implement this idea, I manually zero the gradients from the older classes in
		order to ensure that these weights do not have a learning signal from the loss function. 

	"""	
	(model_number, best_relatedness) = relatedness_info
	
	device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

	path = os.getcwd()
	destination = os.path.join(path, "models", "autoencoders")
	
	# Load the most related model in the memory and finetune the model
	new_path = os.path.join(os.getcwd(), "models", "trained_models")
	path = os.path.join(os.getcwd(), "models", "trained_models", "model_")
	path_to_dir = path + str(model_number) 
	file_name = os.path.join(path_to_dir, "classes.txt")
	file_object = open(file_name, 'r')
	
	num_of_classes_old = file_object.read()
	file_object.close()
	num_of_classes_old = int(num_of_classes_old)

	#Create a variable to store the new number of classes that this model is exposed to
	new_classes = num_of_classes_old + num_classes
	
	#Check the number of models that already exist
	num_ae = len(next(os.walk(destination))[1])

	
	print("Checking if a prior training file exists")
	
	#mypath is the path where the model is going to be stored
	mypath = path + str(num_ae)


	#Will have to create a new directory since it does not exist at the moment
	print("Creating the directory for the new model")
	os.mkdir(mypath)


	# Store the number of classes in the file for future use
	with open(os.path.join(mypath, 'classes.txt'), 'w') as file1:
		input_to_txtfile = str(new_classes)
		file1.write(input_to_txtfile)
		file1.close()

	# Store the associated tasks in the file for future use
	with open(os.path.join(mypath, 'tasks.txt'), 'a') as file1:
		input_to_txtfile = str(task_number) + ","
		file1.write(input_to_txtfile)
		file1.close()

	# Load the most related model into memory
	
	print("Loading the most related model")
	model_init = GeneralModelClass(num_of_classes_old)
	model_init.load_state_dict(torch.load(os.path.join(path_to_dir, "best_performing_model.pth")))
	print("Model loaded")

	#Create (Recreate) the ref_model that has to be used
	ref_model = copy.deepcopy(model_init)
	ref_model.train(False)
	ref_model.to(device)

	#print(ref_model)

		


	print()
		

	# Reference model to compute the soft scores for the LwF(Learning without Forgetting) method
	
	print("Initializing an Adam optimizer")
	optimizer = optim.Adam(model_init.Tmodel.parameters(), lr = 0.003, weight_decay= 0.0001)

		
	#Actually makes the changes to the model_init, so slightly redundant
	print("Initializing the model to be trained")
	model_init = initialize_new_model(model_init, num_classes, num_of_classes_old, args)

	for param in model_init.Tmodel.classifier.parameters():
		param.requires_grad = True

	for param in model_init.Tmodel.features.parameters():
		param.requires_grad = False

	for param in model_init.Tmodel.features[8].parameters():
		param.requires_grad = True

	for param in model_init.Tmodel.features[10].parameters():
		param.requires_grad = True

	#print(model_init)
	model_init.to(device)
	start_epoch = 0

	#The training process format or LwF (Learning without Forgetting)
	# Add the start epoch code 
	
	if (best_relatedness > 0.85):

		model_init.to(device)
		ref_model.to(device)

		print("Using the LwF approach")
		for epoch in range(start_epoch, num_epochs):
			
			print("Epoch {}/{}".format(epoch+1, num_epochs))
			print("-"*20)
			
			running_loss = 0
			running_distill_loss = 0
			steps = 0
			
			#scales the optimizer every 10 epochs 
			optimizer = exp_lr_scheduler(optimizer, epoch, lr)
			#model_init = model_init.train(True)
			
			for data in tqdm(dset_loaders):
				input_data, labels = data



				if (use_gpu):
					input_data = Variable(input_data.to(device))
					labels = Variable(labels.to(device)) 
				
				else:
					input_data  = Variable(input_data)
					labels = Variable(labels)
			
				output = model_init(input_data)
				ref_output = ref_model(input_data)

				maybeNull = torch.sum(output)
				if maybeNull != maybeNull:
					#print every non-Null item
					print("\nNull found in output. All non-null items: ")
					for prediction in output:
						for item in prediction:
							if item == item:
								print(item)
				
				optimizer.zero_grad()
				model_init.zero_grad()

				# loss_1 only takes in the outputs from the nodes of the old classes 
				loss1_output = output[:, :-num_classes]
				loss2_output = output[:, -num_classes:]

				#print()

				loss_1 = model_criterion(loss1_output, ref_output, args, flag = "Distill")

				
				# loss_2 takes in the outputs from the nodes that were initialized for the new task
				loss_2 = model_criterion(loss2_output, labels, args, flag = "CE")


				total_loss = alpha*loss_1 + loss_2

				backup_optim = copy.deepcopy(optimizer)
				backup_model = copy.deepcopy(model_init)
				if total_loss == total_loss and total_loss != float("inf"):
					steps += 1
					total_loss.backward()
					optimizer.step()
					test_null_output = model_init(input_data)
					if test_null_output[0][0] != test_null_output[0][0]:
						steps -= 1
						model_init = copy.deepcopy(backup_model)
						optimizer = copy.deepcopy(backup_optim)
					running_loss += total_loss.item()
					running_distill_loss += alpha*loss_1.item()

				#if total_loss.item() != total_loss.item():
					#print("error: NaN loss")
					#output = model_init(input_data)
				
				
			epoch_loss = running_loss/dset_size
			epoch_distill_loss = running_distill_loss/dset_size


			print('\nEpoch Loss:{}'.format(epoch_loss))
			print('Epoch Distill Loss:{}'.format(epoch_distill_loss))
			print("Steps taken: " + str(steps))

			if(epoch != 0 and epoch != num_epochs-1 and (epoch+1) % 10 == 0):
				epoch_file_name = os.path.join(mypath, str(epoch+1)+'.pth.tar')
				torch.save({
				'epoch': epoch,
				'epoch_loss': epoch_loss, 
				'model_state_dict': model_init.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),

				}, epoch_file_name)


		torch.save(model_init.state_dict(), os.path.join(mypath, "best_performing_model.pth"))	
		



	
	#Process for finetuning the model
	else:
		
		model_init.to(device)
		print("Using the finetuning approach")
		
		for epoch in range(start_epoch, num_epochs):


			print("Epoch {}/{}".format(epoch+1, num_epochs))
			print("-"*20)

			optimizer = exp_lr_scheduler(optimizer, epoch, lr)
			model_init = model_init.train(True)
			
			running_loss = 0
			steps = 0
			
			for data in tqdm(dset_loaders):
				input_data, labels = data


				if (use_gpu):
					input_data = Variable(input_data.to(device))
					labels = Variable(labels.to(device)) 
				
				else:
					input_data  = Variable(input_data)
					labels = Variable(labels)

				output = model_init(input_data)
				


				
				optimizer.zero_grad()
				model_init.zero_grad()
				
				#loss for new classes
				loss = model_criterion(output[:, -num_classes:], labels, args, flag = 'CE')



				
				total_loss = loss

				backup_optim = copy.deepcopy(optimizer)
				backup_model = copy.deepcopy(model_init)
				if total_loss == total_loss and total_loss != float("inf"):
					steps += 1
					total_loss.backward()
					optimizer.step()
					test_null_output = model_init(input_data)
					if test_null_output[0][0] != test_null_output[0][0]:
						steps -= 1
						model_init = copy.deepcopy(backup_model)
						optimizer = copy.deepcopy(backup_optim)
					running_loss += total_loss.item()
					running_distill_loss += alpha*loss_1.item()
				
			epoch_loss = running_loss/dset_size

			print('\nEpoch Loss:{}'.format(epoch_loss))
			print("Steps taken: " + str(steps))

			if(epoch != 0 and (epoch+1) % 5 == 0 and epoch != num_epochs -1):
				epoch_file_name = os.path.join(mypath, str(epoch+1)+'.pth.tar')
				torch.save({
				'epoch': epoch,
				'epoch_loss': epoch_loss, 
				'model_state_dict': model_init.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),

				}, epoch_file_name)


		torch.save(model_init.state_dict(), os.path.join(mypath, "best_performing_model.pth"))


