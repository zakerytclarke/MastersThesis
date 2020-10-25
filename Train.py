"""
Code By Hrituraj Singh
"""

from DataLoader import Loader
from Encoder import FOLEncoder
from Decoder import FOLDecoderBase
from Criterion import FOLCriterion
from tqdm import tqdm
from Generate import FOLGenerator
import torch
from utils import parseConfig
import sys
import os
import time
import random
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

def main(configFile):
	config = parseConfig(configFile)
	dataVars = config['Data']
	vocabVars = config['Vocab']
	encoderVars = config['Encoder']
	decoderVars = config['Decoder']
	trainingVars = config['Training']

	# Getting the data variables
	path = dataVars.get('path', 'data') # Path where the data is stored
	mode = dataVars.get('mode','Sample') # Mode of loading data
	batchSize = dataVars.get('batchsize', 4) # Batch size to be used for training
	shuffle = dataVars.get('shuffle', True) # Whether to shuffle the data before training
	dumpData = dataVars.get('dumpData', True) # Whether to dump the data

	# Getting the Vocabulary variables
	writePath = vocabVars.get('writpath', 'vocab') # Where to dump the data if dumping is reqd.
	vocabMaxSizes = vocabVars.get('vocabmaxsizes', [30000, 30000, 30000, 30000, 30000, 30000]) # Vocabulary sizes
	maxSeqLens = vocabVars.get('maxseqlens', [30, 30]) # Maximum source lengths for the source and target
	loadPretrained = vocabVars.get('loadpretrained', False) # Whether we will be loading pretrained embeddings for input sentence
	pretrainedEmbeddingsPath = vocabVars.get('embeddingspath', 'embedding') # Path to embeddings if we are laoding pretrained, only glove function yet
	pretrainedEmbeddingSize = trainingVars.get('pretrainedembeddingsize', 100) # Pretrained embedding size, Keep this same as your model embedding size

	# Getting the encoder variables
	numLayersEncoder = encoderVars.get('numlayers', 2) # No of layers of LSTM in encoder
	embeddingSizeInput = encoderVars.get('embeddingsize', 100) # Embedding size for the input
	encoderHiddenSize = encoderVars.get('hiddensize', 200) # Hidden layer dimensions for the encoder

	# Getting the decoder variables
	numLayersDecoder = decoderVars.get('numLayers', 1) # Num of layers  of LSTM in the decoder
	embeddingSizes = decoderVars.get('embeddingsizes', [100, 100]) # Embedding sizes for unary and binary predicates

	# Getting the training vars
	epochs = trainingVars.get('epochs', 25) #No of epochs
	modelSavePath = trainingVars.get('nodelsavepath', 'Models') # Path where the models are to be saved
	saveAfterEvery = trainingVars.get('saveafterevery', 5) # Save models after every x number of epochs
	learningRate = trainingVars.get('learningrate', 0.001) # Learning rate for training
	decayRate = trainingVars.get('decayrate',1e-4) # Decay rate for training
	pretrainedEpoch = trainingVars.get('pretrainedepoch', 0) # If start training for a pretrained model for a particular epoch
	pretrainedModelPath = trainingVars.get('pretrainedmodelpath', None) # If start training for a pretrained model, path to checkpoint of that model


	# Loading Data
	assert os.path.exists(path), "Please provide a valid path to the data!"
	dataloader = Loader(loadPretrained = loadPretrained, batchSize = batchSize, mode = mode, shuffle = shuffle, embeddingsPath = pretrainedEmbeddingsPath, pretrainedEmbeddingSize = pretrainedEmbeddingSize, maxSeqLens = maxSeqLens, vocabMaxSizes=vocabMaxSizes)
	DataIterators, metadata = dataloader.load(path)

	# Defining the encoder and decoder
	encoder = FOLEncoder(vocabSize = metadata['vocabSizes'][0], embeddingSize = embeddingSizeInput, hiddenSize = encoderHiddenSize, numLayers = numLayersEncoder, pretrained = loadPretrained, pretrain = metadata['pretrainedEmbeddings'])
	decoder = FOLDecoderBase(vocabSizes = metadata['vocabSizes'][1:], embeddingSizes = embeddingSizes, numLayers = numLayersDecoder, encoderHiddenSize = encoderHiddenSize, batchSize = batchSize, maxSeqLenOut = maxSeqLens[1])
	
	# Defining the optimizers
	encoderOptimizer = optim.Adam(filter(lambda x: x.requires_grad, encoder.parameters()), lr = learningRate, weight_decay = decayRate) 	
	decoderOptimizer = optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()), lr = learningRate, weight_decay = decayRate)

	# Defining the criterion
	criterion = FOLCriterion()

	# Defining the generator
	generator = FOLGenerator()
	logs = open('log.txt','w')


	print("Training Started!")
	timeStart = time.time()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	encoder.to(device)
	decoder.to(device)

	epoch = 0
	while epoch < epochs:
		"""
		This is where training for each epoch actually happens
		"""
		if pretrainedEpoch !=0 and epoch == 0: # Shift epoch in the first iteration itself and load models
			assert os.path.exists(pretrainedModelPath), "Path to pretrained model does not exist!"
			print("Loading the pretrained Mode!")
			encoder.load_state_dict(torch.load(pretrainedModelPath)[0])
			decoder.load_state_dict(torch.load(pretrainedModelPath)[1])
			print("Pretrained Model Loaded!")
			epoch = pretrainedEpoch + 1


		epochLoss = 0 # Only for printing, no direct optimization happens over it
		batches = tqdm(DataIterators, ncols = 100) # For decorating purposes
		for index, batch in enumerate(batches):
			inputs = batch['input'].to(device).long()
			decoderInput = batch['decoderInput'].to(device).long()
			masks =	batch['masks'].to(device).float()
			copyDecisions = batch['copyDecisions'].to(device).float()
			copyIndices = batch['copyIndices'].to(device).long()
			targets = batch['target'].to(device).long()



			# Setting the gradients to zero and putting the encoder and decoder in traning mode
			encoder.train(True)
			decoder.train(True)
			encoderOptimizer.zero_grad()
			decoderOptimizer.zero_grad()

			# Feed Forwarding the batch once
			encoderOut, encoderHidden = encoder(inputs)
			# Making bidirectional hidden state of encoder compatible with unidirection hidden state of decoder
			hidden = (torch.cat((encoderHidden[0][-2], encoderHidden[0][-1]), 1).unsqueeze(0),torch.cat((encoderHidden[1][-2], encoderHidden[1][-1]), 1).unsqueeze(0))
			loss = 0.0

			past = []
			for decoderState in range(decoderInput.shape[1]): # Basically the length of sequence of output
				output, hidden , copy= decoder(decoderInput[:,decoderState].unsqueeze(1), hidden, encoderOut, past)
				past.append(hidden[0].transpose(0,1))
				# Calculating the loss
				loss += criterion(output = output, copy = copy, target = targets[:, decoderState], masks = masks[:, decoderState], copyDecisions = copyDecisions[:,decoderState], copyIndices = copyIndices[:,decoderState]) # Adding the loss for the step

			batches.set_description('Epoch: ' + str(epoch) + ' Loss: ' + str(loss.item())) # For printing in tqdm Bar
			loss.backward() # Propagating the gradients backward
			encoderOptimizer.step() # Taking optimization step for encoder
			decoderOptimizer.step() # Taking optimization step for decoder
			epochLoss += loss


		epochLoss = epochLoss / len(batches)
		print("Epoch " + str(epoch) + "/" + str(epochs) + " Finished!")
		print("Loss for the last epoch: ", str(epochLoss.item()))
		logs.write("Epoch " + str(epoch) + "/" + str(epochs) + " Finished!\n")
		logs.write("Loss for the last epoch: " + str(epochLoss.item()) + '\n')

# 		if epoch > 5: # Initially the outputs might be incorrect spoiling the masks and throwing errors
# 			print("Printing some sentences for this epoch: ")

# 			# Selecting some random batch to generate output
# 			selectedBatch = None
# 			selectFlag = False
# 			while(not selectFlag):
# 				for batch in batches:
# 					if selectFlag:
# 						break
# 					else:
# 						if random.random() > .6:
# 							selectedBatch = batch
# 							selectFlag = True


# 			generator.generate(encoder, decoder, selectedBatch['input'][:5], selectedBatch['target'][:5], device = device)

		if ((epoch+1) % saveAfterEvery == 0):
			#Saving the model
			if not os.path.exists(modelSavePath):
				os.makedirs(modelSavePath)
			print("Saving Checkpoint!")
			if(epoch==epochs-1):
				torch.save([encoder.state_dict(), decoder.state_dict()], modelSavePath + '/Model_Checkpoint_' + 'Last' + '.pt')
			else:
				torch.save([encoder.state_dict(), decoder.state_dict()], modelSavePath + '/Model_Checkpoint_' + str(epoch) + '.pt')
			print('Checkpoint Saved!')
		print("**********************************************************")
		epoch += 1


	print("Training Finished in Time: ", str(float(time.time() - timeStart)), 'secs')





if __name__=="__main__":
	configFile = sys.argv[1]
	assert os.path.exists(configFile), "Configuration file does not exit!"
	main(configFile)