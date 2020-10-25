"""
Code By Hrituraj Singh
"""

from Preprocess import Preprocess
from Vocab import Vocab
from Tokenizer import Tokenizer
from Encoder import FOLEncoder
from Decoder import FOLDecoderBase
from Criterion import FOLCriterion
from Dataset import FOLDataset
from utils import loadGlove
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import os




class Instance:
	"""
	Training Instance to be used in the final dataset in Tensor forms


	1. InputSentence: This is the input sentence for the instance, a Tensor of shape : [srcSeqLen]
	2. DecoderInput: The Input for decoder in Tensor form. It is variablized final form where each element is in final concatenated form: [batchSize, seqLen, types]
	3. Masks: Relavant masks for different types
	4. Target: The target over which we need to train in Tensorized form
	"""

	def __init__(self, InputSentence, DecoderInput, Masks, Target, CopyDecisions, CopyIndices):
		"""
		Initializes the instance and converts all lists to torch tensors for compatibility

		[Inpit]
		InputSentence: Input to be given to encoder, a tensor of shape : (maxseqlen)
		DecoderInput: A list of lists where outer list equals seq len and inner equals types
		Masks: A list of lists where outer list equals seq len and inner equals types
		Target: A list of lists where outer list equals seq len and inner equals types
		CopyDecisions: A tensor of len maxSeqLenOut which has 1s at positions where copying should happen
		CopyIndices: A tensor of len maxSeqLenOut which has indices from where copy should happen
		"""
		self.InputSentence = InputSentence # Already a torch tensor
		self.DecoderInput = torch.stack([torch.stack(DecoderInput[idx]) for idx in range(len(DecoderInput))])
		self.Masks = torch.stack([torch.stack(Masks[idx]) for idx in range(len(Masks))])
		self.Target = torch.stack([torch.stack(Target[idx]) for idx in range(len(Target))])
		self.CopyDecisions = CopyDecisions
		self.copyIndices = CopyIndices


class Loader:
	"""
	Loader class to perform all the dirty work related to preprocessing and preparations.
	1. Preprocess and parses the .txt files containing raw FOL dataset
	2. Creates the vocabularies related to different predicates, variables etc
	3. Performs tokenization of the text to make it go training ready
	4. Creates supplementary features like masks etc.
	"""

	def __init__(self, maxSeqLens = [30, 30], vocabMaxSizes = [30000, 30000, 30000, 30000, 30000, 30000], batchSize = 4, shuffle = True, dumpData = True, loadPretrained = False, embeddingsPath = None, pretrainedEmbeddingSize = 100, mode = 'train'):
		"""
		Initializer for the DataLoader class

		[Input]
		maxSeqLens: list/tuple of maximum sequence lengths in order: source, target
		vocabMaxsizes: list/tuple of maximum sizes of vocabularies in order: unary, binary, variables, pointers, types, inputSentences
		batchSize: batch size to be used while training
		shuffle: whether to shuffle the data before putting to training
		dumpData: Whether to dump data like different vocabularies, and instances
		loadPretrained: If the pretrained Embeddings need to be loaded for input sentences, only gloves can be loaded for now
		embeddingsPath: Path to the embedding file
		mode: Whether we are loading the data in training or inference (testing) mode : train, test, sample
		"""

		# Getting maximum sequence lengths
		self.srcSeqLen = maxSeqLens[0] # Source length
		self.tgtSeqLen = maxSeqLens[1] # Target length

		# Getting Vocabulary sizes
		self.unaryVocabSize = vocabMaxSizes[0] # Unary Predicates
		self.binaryVocabSize = vocabMaxSizes[1] # Binary Predicates
		self.variablesVocabSize = vocabMaxSizes[2] # Variables - redundant in a way
		self.pointersVocabSize = vocabMaxSizes[3] # Pointers
		self.typesVocabSize = vocabMaxSizes[4] # Vocabulary for types - redundant in a way
		self.inputVocabSize = vocabMaxSizes[5] # Input sentences


		self.batchSize = batchSize
		self.shuffle = shuffle
		self.dumpData = dumpData
		self.loadPretrained = loadPretrained
		self.embeddingsPath = embeddingsPath
		self.pretrainedEmbeddingSize = pretrainedEmbeddingSize
		self.mode = mode

	def load(self, filepath):
		"""
		Loads the files at the filepath and finished all the prepreocessing part for us

		[Input]
		filepath: Data path where all the data is stored

		[Output]
		Dataloader: A Dataloader object from torch.utils.data applied over the dataset
		"""

		if os.path.exists('Dumps') and not self.mode=='test':
			print("Dump for dataset found. Loading...")
			with open('Dumps/train.pkl', 'rb') as file_ptr:
				instances = pickle.load(file_ptr)
			print("Loaded!")
		else:
			preprocessor = Preprocess(filepath + '/*' + 'test'+ '*', maxSeqlenOut = self.tgtSeqLen)
			instances = preprocessor.perform()

			if self.dumpData:
				os.makedirs('Dumps')
				with open('Dumps/'+self.mode+'.pkl','wb') as file_ptr:
					pickle.dump(instances, file_ptr)









		# Adding the interface between preprocessor and vocabulary
		# =============================================================================================================
		instancesInput = [(instance.inputSentence, instance.inputSentenceMask) for instance in instances]
		instancesUnary = [(instance.unaryPredicates, instance.unaryPredicatesMask) for instance in instances]
		instancesBinary = [(instance.binaryPredicates, instance.binaryPredicatesMask) for instance in instances]
		instancesvariables = [(instance.variables, instance.variablesMask) for instance in instances]
		instancesPointers = [(instance.pointers, instance.pointersMask) for instance in instances]
		instancestypes = [(instance.types, instance.typesMask) for instance in instances]

		# Initializing the vocabularies
		# =============================================================================================================
		vocabInput = Vocab(maxWords = self.inputVocabSize)
		vocabUnary = Vocab(maxWords = self.unaryVocabSize)
		vocabBinary = Vocab(maxWords = self.binaryVocabSize)
		vocabVariables = Vocab(maxWords = self.variablesVocabSize)
		vocabPointers = Vocab(maxWords = self.pointersVocabSize)
		vocabtypes = Vocab(maxWords = self.typesVocabSize)

		# Loading the vocabulary (during testing) or creating (during training)
		# =============================================================================================================
		if self.mode == 'test' or os.path.exists('vocab'):
			vocabInput.readFromText('vocab/vocabInput.txt')
			vocabUnary.readFromText('vocab/vocabUnary.txt')
			vocabBinary.readFromText('vocab/vocabBinary.txt')
			vocabVariables.readFromText('vocab/vocabVariables.txt')
			vocabPointers.readFromText('vocab/vocabPointers.txt')
			vocabtypes.readFromText('vocab/vocabtypes.txt')
		else:
			vocabInput.create(instancesInput)
			vocabUnary.create(instancesUnary)
			vocabBinary.create(instancesBinary)
			vocabVariables.create(instancesvariables)
			vocabPointers.create(instancesPointers)
			vocabtypes.create(instancestypes)

		# Writing the vocabularies to files
		# =============================================================================================================
		if self.dumpData and not os.path.exists('vocab'):
			if not os.path.exists('vocab'):
				os.makedirs('vocab')
			vocabInput.writeToText('vocab/vocabInput.txt')
			vocabUnary.writeToText('vocab/vocabUnary.txt')
			vocabBinary.writeToText('vocab/vocabBinary.txt')
			vocabVariables.writeToText('vocab/vocabVariables.txt')
			vocabPointers.writeToText('vocab/vocabPointers.txt')
			vocabtypes.writeToText('vocab/vocabtypes.txt')

		# Getting the masks
		# =============================================================================================================
		unaryMask = [instance.unaryPredicatesMask for instance in instances]
		binaryMask = [instance.binaryPredicatesMask for instance in instances]
		variablesMask = [instance.variablesMask for instance in instances]
		pointersMask = [instance.pointersMask for instance in instances]
		typesMask = [instance.typesMask for instance in instances]

		# Getting the copy mechanism data
		# =============================================================================================================
		copyDecisions = [instance.copyDecisions for instance in instances]
		copyIndices = [instance.copyIndices for instance in instances]


		# Adding the interface for tokenizer
		# =============================================================================================================

		instancesInput = [instance.inputSentence for instance in instances]
		instancesUnary = [instance.unaryPredicates for instance in instances]
		instancesBinary = [instance.binaryPredicates for instance in instances]
		instancesvariables = [instance.variables for instance in instances]
		instancesPointers = [instance.pointers for instance in instances]
		instancestypes = [instance.types for instance in instances]


		# Initializing the tokenizers
		# =============================================================================================================
		tokenizerInput = Tokenizer(vocabInput, maxSeqLen = self.srcSeqLen)
		tokenizerUnary = Tokenizer(vocabUnary, maxSeqLen = self.tgtSeqLen, mode = 'output')
		tokenizerBinary = Tokenizer(vocabBinary, maxSeqLen = self.tgtSeqLen, mode = 'output')
		tokenizerVariables = Tokenizer(vocabVariables, maxSeqLen = self.tgtSeqLen, mode = 'output')
		tokenizerPointers = Tokenizer(vocabPointers, maxSeqLen = self.tgtSeqLen, mode = 'output')
		tokenizertypes = Tokenizer(vocabtypes, maxSeqLen = self.tgtSeqLen, mode = 'output', vocabType = 'types')

		# Running the tokenizers
		# =============================================================================================================
		print("Tokenizing the inputs..")
		_ , tokenizedinstancesInput= tokenizerInput.tokenize(instancesInput)
		tokenizedinstancesUnary, targetUnary = tokenizerUnary.tokenize(instancesUnary)
		tokenizedinstancesBinary, targetBinary = tokenizerBinary.tokenize(instancesBinary)
		tokenizedinstancesvariables, targetvariables = tokenizerVariables.tokenize(instancesvariables)
		tokenizedinstancesPointers, targetPointers = tokenizerPointers.tokenize(instancesPointers)
		tokenizedinstancestypes, targettypes = tokenizertypes.tokenize(instancestypes)
		print("Tokenization over!")



	
		print("Getting the instances ready..")
		# Getting the instances ready
		instancesReady = []
		for instanceIdx in tqdm(range(len(instances))):
			inputVariables = tokenizedinstancesInput[instanceIdx]
			DecoderInput = [[tokenizedinstancesUnary[instanceIdx][idx], tokenizedinstancesBinary[instanceIdx][idx], tokenizedinstancesvariables[instanceIdx][idx], tokenizedinstancesPointers[instanceIdx][idx], tokenizedinstancestypes[instanceIdx][idx]] for idx in range(self.tgtSeqLen)]
			copyDec = copyDecisions[instanceIdx]
			copyInd = copyIndices[instanceIdx]
			masks = [[unaryMask[instanceIdx][idx], binaryMask[instanceIdx][idx], variablesMask[instanceIdx][idx], pointersMask[instanceIdx][idx], typesMask[instanceIdx][idx]] for idx in range(self.tgtSeqLen)]
			targets = [[targetUnary[instanceIdx][idx], targetBinary[instanceIdx][idx], targetvariables[instanceIdx][idx], targetPointers[instanceIdx][idx], targettypes[instanceIdx][idx]] for idx in range(self.tgtSeqLen)]
			instance = Instance(inputVariables, DecoderInput, masks, targets, copyDec, copyInd)
			instancesReady.append(instance)



		# Creating the Dataset
		Dataset = FOLDataset(instancesReady)
		dataloader = DataLoader(Dataset, batch_size = self.batchSize, shuffle = self.shuffle)

		# Storing metadata existing here which might be needed further
		vocabSizes = [vocabInput.numWords, vocabUnary.numWords, vocabBinary.numWords, vocabVariables.numWords, vocabPointers.numWords, vocabtypes.numWords]
		metadata = {'vocabSizes': vocabSizes, 'pretrainedEmbeddings': None}

		if self.loadPretrained: # If pretrained embedding to be used, load it (glove working only)
			metadata['pretrainedEmbeddings'] = loadGlove(self.embeddingsPath, vocabInput, self.pretrainedEmbeddingSize)

		return dataloader, metadata