"""
Code By Hrituraj Singh
"""



from tqdm import tqdm
from Vocab import Vocab
from Tokenizer import Tokenizer
import torch
import sys
import os
import time
import torch


# Default word tokens
UNK_token = 0  # Unknown or OOV word
PAD_token = 1  # Used for padding short sentences
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence

class FOLGenerator:
	"""
	FOLGenerator class for the object which generates the output
	for a given input
	"""
	def __init__(self, vocabs = None, loadVocabs = True, maxSeqLen = 30):
		"""
		Initializer for the FOLgenerator class

		[Input]
		encoder: FOLEncoder to be used while encoding the input sentence
		decoder: FOLDecoder to be used while decoding and generating the output
		loadVocabs: If vocabularies are to be loaded from .txt files
		maxSeqLen: maximum sequence length of the the output
		"""

		self.maxSeqLen = maxSeqLen

		# Getting all the vocabularies ready
		self.vocabs = vocabs

		if loadVocabs: # If vocabs are to be loaded from .txt files
			self.vocabs = self.loadVocabs()



		# Getting the tokenizer/detokenizer
		self.input = Tokenizer(vocab = self.vocabs[0])
		self.unary = self.vocabs[1]
		self.binary = self.vocabs[2]
		self.variables = self.vocabs[3]
		self.pointers = self.vocabs[4]
		self.types = self.vocabs[5]

		# Output File
		self.outputFile = open('Outputs.txt','w')


	def loadVocabs(self):
		"""
		Loads all the vocabularis from the vocab path and returns a list in order
		input, unary, binary, variablesunary, variable binary 1, variable binary2
		and types of different tokens.



		[Output]
		vocabs: A list of vocabularies
		"""
		vocabInput = Vocab()
		vocabUnary = Vocab()
		vocabBinary = Vocab()
		vocabVariables = Vocab()
		vocabPointers = Vocab()
		vocabtypes = Vocab()

		# Reading vocabularies from text
		vocabInput.readFromText('vocab/vocabInput.txt')
		vocabUnary.readFromText('vocab/vocabUnary.txt')
		vocabBinary.readFromText('vocab/vocabBinary.txt')
		vocabVariables.readFromText('vocab/vocabVariables.txt')
		vocabPointers.readFromText('vocab/vocabPointers.txt')
		vocabtypes.readFromText('vocab/vocabtypes.txt')

		return [vocabInput, vocabUnary, vocabBinary, vocabVariables, vocabPointers, vocabtypes]

	def generate(self, encoder, decoder, inputSentences, targetSentences, device):
		"""
		Generates the predicates and arguments from inputSentences

		[Input]
		encoder: FOLEncoder to be used while encoding the input sentence
		decoder: FOLDecoder to be used while decoding and generating the output
		inputSentences: Input to be given to the encoder shape: (batchSize, seqlen)
		targetSentences: Target sentences of the shape : (batchSize, seqlen)
		device: device on which the model is : cpu or cuda
		"""


		# Getting the encoder and decoder
		self.encoder = encoder
		self.decoder = decoder
		self.encoder.eval()
		self.decoder.eval()

		numInstances = inputSentences.shape[0]
		outputs = []
		self.stacks = [[] for _ in range(numInstances)]
		past = []



		
		encoderOut, encoderHidden = encoder(inputSentences.to(device).long())
		# Making bidirectional hidden state of encoder compatible with unidirection hidden state of decoder
		hidden = (torch.cat((encoderHidden[0][-2], encoderHidden[0][-1]), 1).unsqueeze(0),torch.cat((encoderHidden[1][-2], encoderHidden[1][-1]), 1).unsqueeze(0))
		output = [torch.zeros(numInstances,1).fill_(SOS_token) for i in range(5)] # 5 - no of types
		for decoderState in range(self.maxSeqLen): # Basically the length of sequence of output
			output, hidden, _ = decoder(torch.stack(output, dim = -1).to(device).long(), hidden, encoderOut, past)
			output = [torch.argmax(output[t], dim=-1) for t in range(5)]

			output = self.adaptedOutput(output) # Adapting the output to remove decoder confusion
			past.append(hidden[0].transpose(0,1))
			outputs.append(torch.stack(output, dim=-1))

		outputs = torch.stack(outputs, dim=-1).transpose(0, 1).squeeze(0) # Converting outputs into a tensor and adapting it
		for index, sentence in enumerate(inputSentences):
			sentence = self.input.detokenize(sentence.unsqueeze(0)) # Unsqueezing as detokenizer expects a list of sentences
			output = self.detokenizeOutput(outputs[index])
			target = self.detokenizeOutput(targetSentences.transpose(1,2)[index])
			self.outputFile.write('Input Sentence: \n')
			self.outputFile.write(''.join(sentence)+'\n')
			self.outputFile.write('Target FOL: \n')
			self.outputFile.write(''.join(target)+'\n')
			self.outputFile.write('Output FOL: \n')
			self.outputFile.write(''.join(output)+'\n')
			self.outputFile.write("**********************************************************\n\n")
		return outputs

	def detokenizeOutput(self, output):
		"""
		Detokenizes the output and merges into a sentential form
		"""
		detokenizedOutput = ''
		unary = output[0]
		binary = output[1]
		variables = output[2]
		pointers = output[3]
		types = output[4]

		for idx, typ in enumerate(types):


			if self.types.index2word[typ.item()] == 'unaryPredicate': # Unary
				if unary[idx] == EOS_token:
					return detokenizedOutput
				detokenizedOutput += self.unary.index2word[unary[idx].item()] + ' '
			elif self.types.index2word[typ.item()] == 'binaryPredicate': # Binary
				detokenizedOutput += self.binary.index2word[binary[idx].item()] + ' '
			elif self.types.index2word[typ.item()] == 'variable': # Variable of Unary
				detokenizedOutput += self.variables.index2word[variables[idx].item()] + ' '
			elif self.types.index2word[typ.item()] == 'pointer':
				detokenizedOutput += self.pointers.index2word[pointers[idx].item()] + ' '

		return detokenizedOutput


	def adaptedOutput(self, output):
		"""
		Adapts the output of the decoder from testing stage so as to maintain the same conditions
		as during training thus removing unnecessary confusion for the decoder

		[Input]
		output: Output list of one step of shape [noTypes (=5), batchSize]

		[Output]
		outputA: adapted Output list of one step of shape [noTypes (=5), batchSize]
		"""
		unary = output[0]
		binary = output[1]
		variables = output[2]
		pointers = output[3]
		types = output[4]
		numInstances = len(unary)

		for index in range(numInstances):
			if self.types.index2word[types[index].item()] == 'unaryPredicate': # Unary
				binary[index] = PAD_token
				variables[index] = PAD_token
				pointers[index] = PAD_token


			elif self.types.index2word[types[index].item()] == 'binaryPredicate': # Binary
				unary[index] = PAD_token
				variables[index] = PAD_token
				pointers[index] = PAD_token

			elif self.types.index2word[types[index].item()] == 'variable': # Variable of Unary
				unary[index] = PAD_token
				binary[index] = PAD_token
				pointers[index] = PAD_token
			elif self.types.index2word[types[index].item()] == 'pointer': # Pointers
				unary[index] = PAD_token
				binary[index] = PAD_token
				variables[index] = PAD_token

		outputC = [unary, binary, variables, pointers, types]
		return outputC