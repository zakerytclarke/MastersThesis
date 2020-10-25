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
UNK_token = 0  # Unknown or OOV word
PAD_token = 1  # Used for padding short sentences
SOS_token = 2  # Start of the sequence
EOS_token = 3  # End of the sequence



class FOLEncoder(nn.Module):
	"""
	FOLEncoder for the seq to seq model. This encoder is LSTM based only. Not
	recommended to be overridden with BERT/CNN. Create new one instead
	"""

	def __init__(self, vocabSize = 30000, embeddingSize = 100, hiddenSize = 200, numLayers = 1, dropout = 0.1, pretrained = False, pretrain = None):
		"""
		Initializer for the FOLEncoder class

		[Inputs]
		vocabSize: Size of the vocabulary
		embeddingSize: Size of the embeddings
		maxLen: Maximum length of the sequence
		numLayers: Number of layers in the LSTM (defaults to 1)
		dropout: Dropout rate for the network
		pretrained: if pretrained embeddings need to be loaded
		pretrain: Weights of the pretrained embeddings, if pretrained = True, must be passed as arg
		"""
		super(FOLEncoder, self).__init__()
		self.vocabSize = vocabSize
		self.embeddingSize = embeddingSize
		self.numLayers = numLayers
		self.hiddenSize = hiddenSize

		self.embeddings = nn.Embedding(vocabSize, embeddingSize, padding_idx = PAD_token)
		if pretrained:
			self.embeddings.weights = pretrain

		self.dropout = nn.Dropout(dropout)
		self.lstm = nn.LSTM(embeddingSize, hiddenSize, num_layers=numLayers, bidirectional = True, batch_first = True)

	def forward(self, x):
		"""
		Performs the feed forwards on LSTM

		[Input]
		x: Input for the encoder of shape: (batchSize, seqLen)

		[Output]
		output: Output hidden states for all the time steps in the encoder of shape: (batchSize, seqLen, 2*hiddenSize)
		"""

		x = self.embeddings(x)

		if self.dropout:
			x = self.dropout(x)

		output, hidden = self.lstm(x)

		return output, hidden




