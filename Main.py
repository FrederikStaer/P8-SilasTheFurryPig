#Code modified from: https://github.com/wannabeOG/ExpertNet-Pytorch 
#				and: https://github.com/rhythm-on-github/P7

if __name__ == "__main__":
	print("come here piggy piggy")

	# imports
	import torch
	import torch.nn as nn
	import argparse
	import os
	import pathlib
	import datetime 
	from datetime import datetime
	from tqdm import tqdm
	from multiprocessing import freeze_support
	import shutil

	import torchvision.datasets as datasets
	import torchvision.models as models
	import torchvision.transforms as transforms

	from autoencoder import Autoencoder, Alexnet_FE
	from encoder_train import *
	from initial_model_train import *
	from model_train import *
	from data_utils.data_prep_mnist import *
	from data_utils.data_prep_tin import *
	from test_models import *

	parser = argparse.ArgumentParser()
	# Learning options
	parser.add_argument("--lr",					type=float, default=0.0002, help="Learning rate")
	parser.add_argument("--batch_size",			type=int,   default=16,     help="Size of the batches")
	parser.add_argument("--code_dims",			type=int,   default=100,    help="Dimensionality of the latent space for autoencoders")
	parser.add_argument('--num_epochs_encoder', default=1,	type=int,		help='Number of epochs you want the encoder model to train on')
	parser.add_argument('--num_epochs_model',	default=1,	type=int,		help='Number of epochs you want  model to train on')
	parser.add_argument("--beta1",				type=float, default=0.5,    help="Beta1 hyperparameter for Adam optimizer")

	# Dataset options
	parser.add_argument("--no_of_tasks",		type=int,	default=2,		help="Number of tasks")
	parser.add_argument("--dataset_boundaries", type=list,	default=[4,9],  help="Final task index for each dataset")
	#parser.add_argument("--dataset",			type=str,	default="FB15K237",	help="Which dataset folder to use as input")
	parser.add_argument("--download_dataset",	type=str,	default="False",	help="Whether to (re-)download dataset")

	# General options
	parser.add_argument("--mode",				type=str,	default="run",	help="Which thing to do, overall ('train', 'test', or 'run' which does both)")
	parser.add_argument("--use_gpu",			type=str,	default="True",	help="Use GPU for training? (cuda)")
	parser.add_argument("--worker_threads",     type=int,	default=4,		help="Number of threads to use for loading data")

	# Output options 
	parser.add_argument("--sample_interval",	type=int,	default=5000,   help="Iters between image samples")
	parser.add_argument("--tqdm_columns",		type=int,	default=60,     help="Total text columns for tqdm loading bars")

	opt = parser.parse_args()

	#convert "Booleans" to actual bools (command line compatibility)
	if opt.download_dataset == "False":
		opt.download_dataset = False
	else:
		opt.download_dataset = True

	if opt.use_gpu == "False":
		opt.use_gpu = False
	else:
		opt.use_gpu = True

	print(opt)


	# --- setup ---

	# Dataset directory
	workDir  = pathlib.Path().resolve()
	dataDir  = os.path.join(workDir.parent.resolve(), 'datasets')
	#inDataDir = os.path.join(dataDir, opt.dataset)
	#loss_graphDir = os.path.join(dataDir, "_loss_graph")
	#if not os.path.exists(loss_graphDir):
	#	os.makedirs(loss_graphDir)

	# filepath for storing loss graph
	#graphDirAndName = os.path.join(loss_graphDir, "loss_graph.png")
	
	#Downloading datasets
	if opt.download_dataset:
		print("Downloading MNIST dataset")
		download_mnist(workDir)
		print("Downloading TIN dataset")
		download_tin(workDir)


	# Seed
	seed = torch.Generator().seed()
	print("Current seed: " + str(seed))

	# Computing device
	cuda = opt.use_gpu and torch.cuda.is_available()
	#device = 'cpu'
	#if cuda: device = 'cuda:0'
	print("cuda: " + str(cuda))


	#transforms for the tiny-imagenet dataset. Applicable for the tasks 1-4
	data_transforms_tin = {
		'train': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}


	#transforms for the mnist dataset. Applicable for the tasks 5-9
	data_transforms_mnist = {
		'train': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.1307,], [0.3081,])
		]),
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.1307,], [0.3081,])
		])
	}


	#Initial model 
	pretrained_alexnet = models.alexnet(pretrained = True)

	#Derives a feature extractor model from the Alexnet model
	feature_extractor = Alexnet_FE(pretrained_alexnet)

	if opt.mode == "train" or opt.mode == "run":
		#remove prior models
		ae_dir = os.path.join(workDir, "models", "autoencoders")
		expert_dir = os.path.join(workDir, "models", "trained_models")
		if(os.path.exists(ae_dir)):
			shutil.rmtree(ae_dir)
		if(os.path.exists(expert_dir)):
			shutil.rmtree(expert_dir)

		#start training
		for task_number in range(1, opt.no_of_tasks+1):
	
			print("Task Number {}".format(task_number))
			data_path = os.path.join(os.getcwd(), "Data")
			encoder_path = os.path.join(os.getcwd(), "models", "autoencoders")
			#model_path = os.path.join(os.getcwd(), "models", "trained_models")

			path_task = os.path.join(data_path, "Task_" + str(task_number))

			image_folder = None
			if (task_number <= opt.dataset_boundaries[0]):
				image_folder = datasets.ImageFolder(os.path.join(path_task, 'train'), transform = data_transforms_tin['train'])
			else:
				image_folder = datasets.ImageFolder(os.path.join(path_task, 'train'), transform = data_transforms_mnist['train'])	
	
			dset_size = len(image_folder)

			#device = torch.device(device)

			dset_loaders = torch.utils.data.DataLoader(image_folder, batch_size = opt.batch_size, shuffle=True, num_workers=opt.worker_threads)
			
			#current autoencoder path
			path_ae = os.path.join(path, "models", "autoencoders")
			num_ae = 1
			if task_number > 1:
				num_ae = len(next(os.walk(path_ae))[1])
			mypath = os.path.join(encoder_path, "autoencoder_" + str(num_ae))

			if os.path.isdir(mypath):
				############ check for the latest checkpoint file in the autoencoder ################
				onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
				max_train = -1
				flag = False

				model = Autoencoder(256*13*13)
		
				store_path = mypath
		
				for file in onlyfiles:
					if(file.endswith('pth.tr')):
						flag = True
						test_epoch = int(file[0])
						if(test_epoch > max_train): 
							max_epoch = test_epoch
							checkpoint_file_encoder = file
				#######################################################################################
		
				if (flag == False): 
					checkpoint_file_encoder = ""

			else:
				checkpoint_file_encoder = ""

			#get an autoencoder model and the path where the autoencoder model would be stored
			model, store_path = add_autoencoder(256*13*13, opt.code_dims, task_number)

			#Define an optimizer for this model 
			optimizer_encoder = optim.Adam(model.parameters(), lr = 0.003, weight_decay= 0.0001)

			print("Reached here for {}".format(task_number))
			print("")
			#Training the autoencoder
			autoencoder_train(model, feature_extractor, store_path, optimizer_encoder, encoder_criterion, dset_loaders, dset_size, opt.num_epochs_encoder, checkpoint_file_encoder, cuda)

			#Train the model
			if(task_number == 1):
				train_model_1(len(image_folder.classes), feature_extractor, encoder_criterion, dset_loaders, dset_size, opt.num_epochs_model, cuda, task_number,  lr = opt.lr)
			else: 
				print("Determining the most related model")
				path = os.getcwd()
				destination = os.path.join(path, "models", "autoencoders")
				num_ae = len(next(os.walk(destination))[1])
				ae_idxs = list(reversed(range(1, num_ae+1)))
				model_number, best_relatedness = get_related_model(feature_extractor, dset_loaders, dset_size, encoder_criterion, cuda, ae_idxs)
				relatedness_info = (model_number, best_relatedness)

				train_model(len(image_folder.classes), feature_extractor, encoder_criterion, dset_loaders, dset_size, opt.num_epochs_model, cuda, task_number, relatedness_info,  lr = opt.lr)

	if opt.mode == "test" or opt.mode == "run":
		test_models()