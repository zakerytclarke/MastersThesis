"""
Code By Hrituraj Singh
"""
import operator
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F



# Default word tokens
PAD_token = 0  # Used for padding short sentences
UNK_token = 1  # Unknown or OOV word
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence



class UnaryEmbedding(nn.Module):
	"""
	Embedding Module for the unary predicates
	"""
	def __init__(self, embedDim, vocabSize):
		"""
		Initializer for the embeddings

		[Input]
		embedDim: Dimensions for the Embeddings
		vocabSize: Size of the vocabulary
		"""
		super(UnaryEmbedding, self).__init__()
		self.embedDim = embedDim
		self.vocabSize = vocabSize
		self.embedding = nn.Embedding(vocabSize, embedDim, padding_idx = PAD_token)

	def forward(self, x):
		"""
		Feedforward for the embedding
		"""
		return self.embedding(x)

class BinaryEmbedding(nn.Module):
	"""
	Embedding Module for the Binary predicates
	"""
	def __init__(self, embedDim, vocabSize):
		"""
		Initializer for the embeddings

		[Input]
		embedDim: Dimensions for the Embeddings
		vocabSize: Size of the vocabulary
		"""
		super(BinaryEmbedding, self).__init__()
		self.embedDim  = embedDim
		self.vocabSize = vocabSize
		self.embedding = nn.Embedding(vocabSize, embedDim, padding_idx = PAD_token)

	def forward(self, x):
		"""
		Feedforward for the embedding
		"""
		return self.embedding(x)

class PointerEmbedding(nn.Module):
	"""
	Embedding Module for the Binary predicates
	"""
	def __init__(self, embedDim, vocabSize):
		"""
		Initializer for the embeddings

		[Input]
		embedDim: Dimensions for the Embeddings
		vocabSize: Size of the vocabulary
		"""
		super(PointerEmbedding, self).__init__()
		self.embedDim  = embedDim
		self.vocabSize = vocabSize
		self.embedding = nn.Embedding(vocabSize, embedDim, padding_idx = PAD_token)

	def forward(self, x):
		"""
		Feedforward for the embedding
		"""
		return self.embedding(x)


class VariableEmbedding(nn.Module):
	"""
	Embedding Module for the Unary Variables
	"""
	def __init__(self, vocabSize):
		"""
		Initializer for the embeddings

		[Input]
		vocabSize: Size of the vocabulary
		"""
		super(VariableEmbedding, self).__init__()
		self.vocabSize = vocabSize


		#Initializing weights
		weights = torch.eye(vocabSize, vocabSize)
		self.embedding = nn.Embedding.from_pretrained(weights, freeze = True)

	def forward(self, x):
		"""
		Feedforward for the embedding
		"""
		return self.embedding(x) 


class TypesEmbedding(nn.Module):
	"""
	Embedding Module for variable 2 binary
	"""
	def __init__(self, vocabSize):
		"""
		Initializer for the embeddings

		[Input]
		vocabSize: Size of the vocabulary
		"""
		super(TypesEmbedding, self).__init__()
		self.vocabSize = vocabSize

		#Initializing weights
		weights = torch.eye(vocabSize, vocabSize)
		self.embedding = nn.Embedding.from_pretrained(weights, freeze = True)

	def forward(self, x):
		"""
		Feedforward for the embedding
		"""
		return self.embedding(x) 

	
