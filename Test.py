"""
Code By Hrituraj Singh
"""

from DataLoader import Loader
from Encoder import FOLEncoder
from Decoder import FOLDecoderBase
from Criterion import FOLCriterion
from tqdm import tqdm
from Generate import FOLGenerator
from Evaluate import FOLEvaluator
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



def main(configFile):
	config = parseConfig(configFile)
	dataVars = config['Data']
	vocabVars = config['Vocab']
	encoderVars = config['Encoder']
	decoderVars = config['Decoder']
	testingVars = config['Testing']

	# Getting the data variables
	path = dataVars.get('path', 'data') # Path where the data is stored
	batchSize = dataVars.get('batchsize', 4) # Batch size to be used for training
	shuffle = dataVars.get('shuffle', False) # Whether to shuffle the data before training

	# Getting the Vocabulary variables
	vocabMaxSizes = vocabVars.get('vocabmaxsizes', [30000, 30000, 30000, 30000, 30000, 30000]) # Vocabulary sizes
	maxSeqLens = vocabVars.get('maxseqlens', [30, 50]) # Maximum source lengths for the source and target
	loadPretrained = vocabVars.get('loadpretrained', False) # Whether we will be loading pretrained embeddings for input sentence
	pretrainedEmbeddingsPath = vocabVars.get('embeddingspath', 'embedding') # Path to embeddings if we are laoding pretrained, only glove function yet
	pretrainedEmbeddingSize = testingVars.get('pretrainedembeddingsize', 100) # Pretrained embedding size, Keep this same as your model embedding size

	# Getting the encoder variables
	numLayersEncoder = encoderVars.get('numlayers', 2) # No of layers of LSTM in encoder
	embeddingSizeInput = encoderVars.get('embeddingsize', 100) # Embedding size for the input
	encoderHiddenSize = encoderVars.get('hiddensize', 200) # Hidden layer dimensions for the encoder

	# Getting the decoder variables
	numLayersDecoder = decoderVars.get('numLayers', 1) # Num of layers  of LSTM in the decoder
	embeddingSizes = decoderVars.get('embeddingsizes', [100, 100]) # Embedding sizes for unary and binary predicates

	# Getting the testing vars
	modelSavePath = testingVars.get('modelsavepath', 'Models') # Path where the models are to be saved
	pretrainedModelPath = testingVars.get('pretrainedmodelpath', None) # If start training for a pretrained model, path to checkpoint of that model


	# Loading Data
	assert os.path.exists(path), "Please provide a valid path to the data!"
	dataloader = Loader(loadPretrained = loadPretrained, dumpData = False, batchSize = batchSize, shuffle = shuffle, embeddingsPath = pretrainedEmbeddingsPath, pretrainedEmbeddingSize = pretrainedEmbeddingSize, maxSeqLens = maxSeqLens, mode = 'test')
	DataIterators, metadata = dataloader.load(path)

	# Defining the encoder and decoder
	encoder = FOLEncoder(vocabSize = metadata['vocabSizes'][0], embeddingSize = embeddingSizeInput, hiddenSize = encoderHiddenSize, numLayers = numLayersEncoder, pretrained = loadPretrained, pretrain = metadata['pretrainedEmbeddings'])
	decoder = FOLDecoderBase(vocabSizes = metadata['vocabSizes'][1:], embeddingSizes = embeddingSizes, numLayers = numLayersDecoder, encoderHiddenSize = encoderHiddenSize)


	# Defining the generator
	generator = FOLGenerator(maxSeqLen = maxSeqLens[1])

	# Defining the evaluator - deprecated now
	# evaluator = FOLEvaluator(vocabs = generator.vocabs[1:], outputMaxLen = maxSeqLens[1], mode='alignment')

	print("Testing Started!")
	timeStart = time.time()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	encoder.to(device)
	decoder.to(device)
	assert os.path.exists(pretrainedModelPath), "Path to pretrained model does not exist!"
	print("Loading the pretrained Mode!")
	encoder.load_state_dict(torch.load(pretrainedModelPath, map_location=device)[0])
	decoder.load_state_dict(torch.load(pretrainedModelPath, map_location=device)[1])
	print("Pretrained Model Loaded!")





	batches = DataIterators
	for index, batch in enumerate(batches):
		inputs = batch['input'].to(device).long()
		targets = batch['target'].to(device).long()
		outputs = generator.generate(encoder, decoder, inputs, targets, device = device)

		# evaluator.evaluateBatch(outputs,targets.transpose(1,2))
	# evaluator.showScores()

	print("=======================================================================	")
	print("Testing Finished in Time: ", str(float(time.time() - timeStart)), 'secs')





if __name__=="__main__":
	configFile = sys.argv[1]
	assert os.path.exists(configFile), "Configuration file does not exit!"
	main(configFile)
